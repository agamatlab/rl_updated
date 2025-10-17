#!/usr/bin/env python3
"""
Catalog runner to train MiniGrid and BabyAI environments in bulk.

This variant guarantees every run trains for at least 2 million frames and, by
default, loops forever. Model directories are stable across sweeps so training
resumes from prior checkpoints and simply extends the total frame target. Use
``--one-pass`` if you want the legacy single sweep. Across consecutive sweeps
the frame budget doubles whenever the previous sweep failed to reach an average
reward of 0.8, providing a simple automatic curriculum.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

MIN_FRAMES = 2_000_000


@dataclass(frozen=True)
class GroupConfig:
    envs: tuple[str, ...]
    frames: int
    procs: int
    recurrence: int = 1
    text: bool = False
    frames_per_proc: int | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


MINIGRID_BASIC = GroupConfig(
    envs=(
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-Random-5x5-v0",
        "MiniGrid-Empty-6x6-v0",
        "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Empty-16x16-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-Playground-v0",
        "MiniGrid-MemoryS7-v0",
        "MiniGrid-MemoryS9-v0",
        "MiniGrid-MemoryS11-v0",
        "MiniGrid-MemoryS13-v0",
        "MiniGrid-MemoryS13Random-v0",
        "MiniGrid-MemoryS17Random-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-Fetch-5x5-N2-v0",
        "MiniGrid-Fetch-6x6-N2-v0",
        "MiniGrid-GoToObject-6x6-N2-v0",
        "MiniGrid-GoToObject-8x8-N2-v0",
        "MiniGrid-DistShift1-v0",
        "MiniGrid-DistShift2-v0",
    ),
    text=True,
    frames=MIN_FRAMES,
    procs=16,
)

MINIGRID_DOOR = GroupConfig(
    envs=(
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-DoorKey-16x16-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-KeyCorridorS3R1-v0",
        "MiniGrid-KeyCorridorS3R2-v0",
        "MiniGrid-KeyCorridorS3R3-v0",
        "MiniGrid-KeyCorridorS4R3-v0",
        "MiniGrid-KeyCorridorS5R3-v0",
        "MiniGrid-KeyCorridorS6R3-v0",
        "MiniGrid-GoToDoor-5x5-v0",
        "MiniGrid-GoToDoor-6x6-v0",
        "MiniGrid-GoToDoor-8x8-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
        "MiniGrid-UnlockPickup-v0",
    ),
    text=True,
    frames=MIN_FRAMES,
    procs=16,
    recurrence=8,
)

MINIGRID_HAZARDS = GroupConfig(
    envs=(
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N3-v0",
        "MiniGrid-LavaCrossingS11N5-v0",
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-SimpleCrossingS11N5-v0",
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-Dynamic-Obstacles-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
        "MiniGrid-Dynamic-Obstacles-6x6-v0",
        "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-Dynamic-Obstacles-16x16-v0",
    ),
    text=True,
    frames=MIN_FRAMES,
    procs=24,
    recurrence=16,
)

MINIGRID_OBSTRUCTED = GroupConfig(
    envs=(
        "MiniGrid-ObstructedMaze-1Dl-v0",
        "MiniGrid-ObstructedMaze-1Dlh-v0",
        "MiniGrid-ObstructedMaze-1Dlhb-v0",
        "MiniGrid-ObstructedMaze-2Dl-v0",
        "MiniGrid-ObstructedMaze-2Dlh-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v0",
        "MiniGrid-ObstructedMaze-1Q-v0",
        "MiniGrid-ObstructedMaze-2Q-v0",
        "MiniGrid-ObstructedMaze-Full-v0",
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        "MiniGrid-ObstructedMaze-1Q-v1",
        "MiniGrid-ObstructedMaze-2Q-v1",
        "MiniGrid-ObstructedMaze-Full-v1",
    ),
    text=True,
    frames=MIN_FRAMES,
    procs=24,
    recurrence=32,
)

MINIGRID_WFC = GroupConfig(
    envs=(
        "MiniGrid-WFC-MazeSimple-v0",
        "MiniGrid-WFC-DungeonMazeScaled-v0",
        "MiniGrid-WFC-RoomsFabric-v0",
        "MiniGrid-WFC-ObstaclesBlackdots-v0",
        "MiniGrid-WFC-ObstaclesAngular-v0",
        "MiniGrid-WFC-ObstaclesHogs3-v0",
        "MiniGrid-WFC-Maze-v0",
        "MiniGrid-WFC-MazeSpirals-v0",
        "MiniGrid-WFC-MazePaths-v0",
        "MiniGrid-WFC-Mazelike-v0",
        "MiniGrid-WFC-Dungeon-v0",
        "MiniGrid-WFC-DungeonRooms-v0",
        "MiniGrid-WFC-DungeonLessRooms-v0",
        "MiniGrid-WFC-DungeonSpirals-v0",
        "MiniGrid-WFC-RoomsMagicOffice-v0",
        "MiniGrid-WFC-SkewCave-v0",
        "MiniGrid-WFC-SkewLake-v0",
        "MiniGrid-WFC-MazeKnot-v0",
        "MiniGrid-WFC-MazeWall-v0",
        "MiniGrid-WFC-RoomsOffice-v0",
        "MiniGrid-WFC-ObstaclesHogs2-v0",
        "MiniGrid-WFC-Skew2-v0",
    ),
    text=True,
    frames=MIN_FRAMES,
    procs=16,
    recurrence=16,
)

BABYAI_GOTO = GroupConfig(
    envs=(
        "BabyAI-GoToRedBallGrey-v0",
        "BabyAI-GoToRedBall-v0",
        "BabyAI-GoToRedBallNoDists-v0",
        "BabyAI-GoToObj-v0",
        "BabyAI-GoToObjS4-v0",
        "BabyAI-GoToObjS6-v1",
        "BabyAI-GoToLocal-v0",
        "BabyAI-GoToLocalS5N2-v0",
        "BabyAI-GoToLocalS6N2-v0",
        "BabyAI-GoToLocalS6N3-v0",
        "BabyAI-GoToLocalS6N4-v0",
        "BabyAI-GoToLocalS7N4-v0",
        "BabyAI-GoToLocalS7N5-v0",
        "BabyAI-GoToLocalS8N2-v0",
        "BabyAI-GoToLocalS8N3-v0",
        "BabyAI-GoToLocalS8N4-v0",
        "BabyAI-GoToLocalS8N5-v0",
        "BabyAI-GoToLocalS8N6-v0",
        "BabyAI-GoToLocalS8N7-v0",
        "BabyAI-GoTo-v0",
        "BabyAI-GoToOpen-v0",
        "BabyAI-GoToObjMaze-v0",
        "BabyAI-GoToObjMazeOpen-v0",
        "BabyAI-GoToObjMazeS4R2-v0",
        "BabyAI-GoToObjMazeS4-v0",
        "BabyAI-GoToObjMazeS5-v0",
        "BabyAI-GoToObjMazeS6-v0",
        "BabyAI-GoToObjMazeS7-v0",
        "BabyAI-GoToImpUnlock-v0",
        "BabyAI-GoToSeq-v0",
        "BabyAI-GoToSeqS5R2-v0",
        "BabyAI-GoToRedBlueBall-v0",
        "BabyAI-GoToDoor-v0",
        "BabyAI-GoToObjDoor-v0",
    ),
    frames=2_500_000,
    procs=16,
    recurrence=32,
    text=True,
    frames_per_proc=1024,
    extra_args=(
        "--entropy-coef",
        "0.005",
        "--lr",
        "0.00025",
        "--clip-eps",
        "0.1",
        "--value-loss-coef",
        "0.7",
        "--gae-lambda",
        "0.85",
        "--max-grad-norm",
        "0.2",
    ),
)

BABYAI_OPEN = GroupConfig(
    envs=(
        "BabyAI-Open-v0",
        "BabyAI-OpenRedDoor-v0",
        "BabyAI-OpenDoor-v0",
        "BabyAI-OpenDoorDebug-v0",
        "BabyAI-OpenDoorColor-v0",
        "BabyAI-OpenDoorLoc-v0",
        "BabyAI-OpenTwoDoors-v0",
        "BabyAI-OpenRedBlueDoors-v0",
        "BabyAI-OpenRedBlueDoorsDebug-v0",
        "BabyAI-OpenDoorsOrderN2-v0",
        "BabyAI-OpenDoorsOrderN4-v0",
        "BabyAI-OpenDoorsOrderN2Debug-v0",
        "BabyAI-OpenDoorsOrderN4Debug-v0",
    ),
    frames=2_500_000,
    procs=16,
    recurrence=32,
    text=True,
    frames_per_proc=1024,
    extra_args=BABYAI_GOTO.extra_args,
)

BABYAI_PICKUP = GroupConfig(
    envs=(
        "BabyAI-Pickup-v0",
        "BabyAI-PickupLoc-v0",
        "BabyAI-PickupDist-v0",
        "BabyAI-PickupDistDebug-v0",
        "BabyAI-PickupAbove-v0",
        "BabyAI-UnblockPickup-v0",
        "BabyAI-PutNextLocal-v0",
        "BabyAI-PutNextLocalS5N3-v0",
        "BabyAI-PutNextLocalS6N4-v0",
        "BabyAI-PutNextS4N1-v0",
        "BabyAI-PutNextS5N1-v0",
        "BabyAI-PutNextS5N2-v0",
        "BabyAI-PutNextS5N2Carrying-v0",
        "BabyAI-PutNextS6N3-v0",
        "BabyAI-PutNextS6N3Carrying-v0",
        "BabyAI-PutNextS7N4-v0",
        "BabyAI-PutNextS7N4Carrying-v0",
    ),
    frames=3_000_000,
    procs=16,
    recurrence=32,
    text=True,
    frames_per_proc=1024,
    extra_args=BABYAI_GOTO.extra_args,
)

BABYAI_UNLOCK = GroupConfig(
    envs=(
        "BabyAI-Unlock-v0",
        "BabyAI-UnlockLocal-v0",
        "BabyAI-UnlockLocalDist-v0",
        "BabyAI-KeyInBox-v0",
        "BabyAI-UnlockPickup-v0",
        "BabyAI-UnlockPickupDist-v0",
        "BabyAI-UnlockToUnlock-v0",
        "BabyAI-BlockedUnlockPickup-v0",
        "BabyAI-ActionObjDoor-v0",
        "BabyAI-KeyCorridor-v0",
        "BabyAI-KeyCorridorS3R1-v0",
        "BabyAI-KeyCorridorS3R2-v0",
        "BabyAI-KeyCorridorS3R3-v0",
        "BabyAI-KeyCorridorS4R3-v0",
        "BabyAI-KeyCorridorS5R3-v0",
        "BabyAI-KeyCorridorS6R3-v0",
    ),
    frames=3_500_000,
    procs=16,
    recurrence=32,
    text=True,
    frames_per_proc=1024,
    extra_args=BABYAI_GOTO.extra_args,
)

BABYAI_SEARCH = GroupConfig(
    envs=(
        "BabyAI-FindObjS5-v0",
        "BabyAI-FindObjS6-v0",
        "BabyAI-FindObjS7-v0",
        "BabyAI-MoveTwoAcrossS5N2-v0",
        "BabyAI-MoveTwoAcrossS8N9-v0",
        "BabyAI-OneRoomS8-v0",
        "BabyAI-OneRoomS12-v0",
        "BabyAI-OneRoomS16-v0",
        "BabyAI-OneRoomS20-v0",
    ),
    frames=2_500_000,
    procs=16,
    recurrence=32,
    text=True,
    frames_per_proc=1024,
    extra_args=BABYAI_GOTO.extra_args,
)

BABYAI_COMPOSITE = GroupConfig(
    envs=(
        "BabyAI-Synth-v0",
        "BabyAI-SynthS5R2-v0",
        "BabyAI-SynthLoc-v0",
        "BabyAI-SynthSeq-v0",
        "BabyAI-MiniBossLevel-v0",
        "BabyAI-BossLevel-v0",
        "BabyAI-BossLevelNoUnlock-v0",
    ),
    frames=4_000_000,
    procs=16,
    recurrence=32,
    text=True,
    frames_per_proc=1024,
    extra_args=BABYAI_GOTO.extra_args,
)


GROUPS: dict[str, GroupConfig] = {
    "mini_basic": MINIGRID_BASIC,
    "mini_door": MINIGRID_DOOR,
    "mini_hazards": MINIGRID_HAZARDS,
    "mini_obstructed": MINIGRID_OBSTRUCTED,
    "mini_wfc": MINIGRID_WFC,
    "babyai_goto": BABYAI_GOTO,
    "babyai_open": BABYAI_OPEN,
    "babyai_pickup": BABYAI_PICKUP,
    "babyai_unlock": BABYAI_UNLOCK,
    "babyai_search": BABYAI_SEARCH,
    "babyai_composite": BABYAI_COMPOSITE,
}


def sanitize_model_name(env_id: str, prefix: str, iteration: int, job_idx: int) -> str:
    safe_env = env_id.replace("-", "_").lower()
    return f"{prefix}{safe_env}"


def command_for(
    env_id: str,
    group: GroupConfig,
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    frames: int,
) -> list[str]:
    procs = args.procs if args.procs else group.procs
    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "--algo",
        args.algo,
        "--env",
        env_id,
        "--model",
        model_name,
        "--frames",
        str(frames),
        "--seed",
        str(seed),
        "--procs",
        str(procs),
        "--save-interval",
        str(args.save_interval),
        "--log-interval",
        str(args.log_interval),
    ]
    frames_per_proc = args.frames_per_proc or group.frames_per_proc
    if frames_per_proc:
        cmd += ["--frames-per-proc", str(frames_per_proc)]
    recurrence = args.recurrence if args.recurrence else group.recurrence
    if recurrence > 1:
        cmd += ["--recurrence", str(recurrence)]
    if group.text:
        cmd.append("--text")
    if group.extra_args:
        cmd.extend(group.extra_args)
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def iter_envs(selected: Iterable[str]) -> Iterable[tuple[str, GroupConfig]]:
    for name in selected:
        group = GROUPS[name]
        for env in group.envs:
            yield env, group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        action="append",
        choices=GROUPS.keys(),
        help="Run only specific categories (repeat flag to select multiple).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base RNG seed.")
    parser.add_argument(
        "--frames",
        type=int,
        help=f"Override frame budget for all runs (minimum enforced: {MIN_FRAMES}).",
    )
    parser.add_argument(
        "--procs",
        type=int,
        help="Override number of environment workers per run.",
    )
    parser.add_argument(
        "--frames-per-proc",
        type=int,
        help="Override frames-per-proc forwarded to algorithms.",
    )
    parser.add_argument(
        "--recurrence",
        type=int,
        help="Override recurrence forwarded to algorithms.",
    )
    parser.add_argument(
        "--algo",
        default="ppo",
        help="Algorithm flag passed to scripts.train (default: ppo).",
    )
    parser.add_argument(
        "--model-prefix",
        default="catalognew_",
        help="Prefix for generated model directory names.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose model directory already contains status.pt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute anything.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=(),
        help="Extra command-line args appended to every scripts.train call (prefix with --).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save interval forwarded to training script.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log interval forwarded to training script.",
    )
    parser.add_argument(
        "--ignore-failures",
        action="store_true",
        help="Continue queue even if a run fails.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Number of concurrent training processes to launch.",
    )
    parser.add_argument(
        "--loop-sleep",
        type=int,
        default=10,
        help="Seconds to sleep between iterations when looping forever.",
    )
    parser.add_argument(
        "--one-pass",
        dest="loop",
        action="store_false",
        help="Process the catalog once and exit (default loops forever).",
    )
    parser.set_defaults(loop=True)
    return parser.parse_args()


def storage_dir() -> Path:
    return Path(os.environ.get("RL_STORAGE") or os.environ.get("PROJECT_STORAGE") or "storage")


def has_status(model_name: str) -> bool:
    return (storage_dir() / model_name / "status.pt").exists()


def load_avg_reward(model_name: str) -> float | None:
    csv_path = storage_dir() / model_name / "log.csv"
    if not csv_path.exists():
        return None
    last_valid = None
    try:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("return_mean")
                if not value:
                    continue
                try:
                    last_valid = float(value)
                except ValueError:
                    continue
    except OSError:
        return None
    return last_valid


def load_num_frames(model_name: str) -> int | None:
    status_path = storage_dir() / model_name / "status.pt"
    if not status_path.exists():
        return None
    try:
        import torch  # type: ignore

        status = torch.load(status_path, map_location="cpu")
    except Exception:
        return None
    num_frames = status.get("num_frames")
    try:
        return int(num_frames)
    except (TypeError, ValueError):
        return None


def build_jobs(
    args: argparse.Namespace,
    selected: Iterable[str],
    iteration: int,
    frame_multiplier: int,
) -> list[tuple[str, str, list[str]]]:
    jobs: list[tuple[str, str, list[str]]] = []
    base_seed = args.seed + (iteration - 1) * 1000

    for job_idx, (env_id, group) in enumerate(iter_envs(selected), start=1):
        model_name = sanitize_model_name(env_id, args.model_prefix, iteration, job_idx)
        if args.skip_existing and has_status(model_name):
            print(f"[skip] iter {iteration} {env_id} ({model_name}) - existing status")
            continue

        base_frames = max(args.frames or group.frames, MIN_FRAMES)
        frames = base_frames * frame_multiplier
        existing = has_status(model_name)
        previous_frames = load_num_frames(model_name) if existing else None
        if previous_frames is not None and frames < previous_frames:
            frames = previous_frames

        seed = base_seed + job_idx
        cmd = command_for(env_id, group, args, model_name, seed, frames)
        details: list[str] = []
        if existing:
            if previous_frames is not None:
                if frames > previous_frames:
                    details.append(f"resume {previous_frames:,}->{frames:,} frames")
                else:
                    details.append(f"resume @ {previous_frames:,} frames")
            else:
                details.append("resume")
        if frame_multiplier > 1 and frames >= base_frames:
            details.append(f"frames x{frame_multiplier}")
        if frames == previous_frames and frame_multiplier > 1 and previous_frames is not None:
            details.append("target capped by existing progress")
        detail_suffix = f" ({', '.join(details)})" if details else ""

        print(f"[run]  iter {iteration} {env_id}{detail_suffix}\n       {shlex.join(cmd)}")
        jobs.append((env_id, model_name, cmd))

    return jobs


def run_job(env_id: str, model_name: str, cmd: list[str], iteration: int) -> float | None:
    print(f"[start] iter {iteration} {env_id} -> {model_name}")
    subprocess.run(cmd, check=True)
    print(f"[done]  iter {iteration} {env_id}")
    avg_reward = load_avg_reward(model_name)
    if avg_reward is not None:
        print(f"[stat]  iter {iteration} {env_id} return_mean={avg_reward:.3f}")
    else:
        print(f"[stat]  iter {iteration} {env_id} return_mean unavailable")
    return avg_reward


def execute_jobs(
    args: argparse.Namespace,
    jobs: list[tuple[str, str, list[str]]],
    iteration: int,
) -> tuple[int, list[float]]:
    if args.dry_run or not jobs:
        return 0, []

    max_workers = max(1, args.max_parallel)
    exit_code = 0
    rewards: list[float] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_job, env_id, model_name, cmd, iteration): (env_id, model_name)
            for env_id, model_name, cmd in jobs
        }
        for future in as_completed(future_map):
            env_id, model_name = future_map[future]
            try:
                result = future.result()
                if result is not None:
                    rewards.append(result)
            except subprocess.CalledProcessError as exc:
                exit_code = exit_code or (exc.returncode or 1)
                print(
                    f"[error] iter {iteration} {env_id} ({model_name}) exit {exc.returncode}",
                    file=sys.stderr,
                )
                continue
            except Exception as exc:
                exit_code = exit_code or 1
                print(
                    f"[error] iter {iteration} {env_id} ({model_name}) unexpected failure: {exc}",
                    file=sys.stderr,
                )
                continue

    return exit_code, rewards


def main() -> None:
    args = parse_args()
    selected = args.category if args.category else GROUPS.keys()
    iteration = 0
    frame_multiplier = 1

    while True:
        iteration += 1
        jobs = build_jobs(args, selected, iteration, frame_multiplier)
        if not jobs:
            if args.loop:
                print(f"[idle] iter {iteration} produced no jobs, sleeping {args.loop_sleep}s")
                time.sleep(args.loop_sleep)
                continue
            break

        exit_code, rewards = execute_jobs(args, jobs, iteration)
        if exit_code and not args.ignore_failures:
            sys.exit(exit_code)

        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            print(f"[iter] iter {iteration} average return_mean={avg_reward:.3f} over {len(rewards)} run(s)")
            if avg_reward < 0.8:
                frame_multiplier *= 2
                print(f"[iter] escalating frame budget: multiplier now x{frame_multiplier}")
        else:
            print(f"[iter] iter {iteration} produced no reward statistics; keeping multiplier x{frame_multiplier}")

        if not args.loop:
            break

        print(f"[loop] iteration {iteration} complete, sleeping {args.loop_sleep}s")
        time.sleep(args.loop_sleep)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user", file=sys.stderr)
