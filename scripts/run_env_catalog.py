#!/usr/bin/env python3
"""
Utility to launch training runs across the full MiniGrid and BabyAI catalog.

The script groups environments that share similar hyper-parameter needs, builds
the appropriate command line for ``python3 -m scripts.train`` and executes each
run sequentially.  It is meant for large-scale sweeps on GPU-equipped machines.

Example:
    python3 scripts/run_env_catalog.py --category babyai_goto mini_basic --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    frames=500_000,
    procs=16,
)

MINIGRID_DOORKEY = GroupConfig(
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
    frames=750_000,
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
    frames=1_000_000,
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
    frames=1_500_000,
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
    frames=1_000_000,
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
    frames=2_000_000,
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
    "mini_door": MINIGRID_DOORKEY,
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


def sanitize_model_name(env_id: str, prefix: str) -> str:
    return f"{prefix}{env_id.replace('-', '_').lower()}"


def command_for(env_id: str, group: GroupConfig, args: argparse.Namespace) -> list[str]:
    model_name = sanitize_model_name(env_id, args.model_prefix)
    frames = args.frames if args.frames else group.frames
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
        str(args.seed),
        "--procs",
        str(procs),
        "--save-interval",
        str(args.save_interval),
        "--log-interval",
        str(args.log_interval),
    ]
    if group.frames_per_proc:
        cmd += ["--frames-per-proc", str(group.frames_per_proc)]
    if group.recurrence > 1:
        cmd += ["--recurrence", str(group.recurrence)]
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
        for env_id in group.envs:
            yield env_id, group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        action="append",
        choices=GROUPS.keys(),
        help="Run only the specified category (can be passed multiple times).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed forwarded to training.")
    parser.add_argument(
        "--frames",
        type=int,
        help="Override the frame budget for every run (default: per-group value).",
    )
    parser.add_argument(
        "--procs",
        type=int,
        help="Override the number of processes (default: per-group value).",
    )
    parser.add_argument(
        "--algo",
        default="ppo",
        help="Algorithm to pass to training (default: ppo).",
    )
    parser.add_argument(
        "--model-prefix",
        default="catalog_",
        help="Prefix added to every generated model name (default: catalog_).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose storage directory already contains status.pt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=(),
        help="Additional arguments appended to every command (place after --).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Overrides --save-interval for every run (default: 50).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Overrides --log-interval for every run (default: 1).",
    )
    parser.add_argument(
        "--ignore-failures",
        action="store_true",
        help="Continue after a failed training run instead of stopping.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of concurrent training processes (default: 1).",
    )
    return parser.parse_args()


def has_status(model_name: str) -> bool:
    storage_root = Path(
        os.environ.get("RL_STORAGE")
        or os.environ.get("PROJECT_STORAGE")
        or "storage"
    )
    return (storage_root / model_name / "status.pt").exists()


def main(args: argparse.Namespace) -> None:
    selected = args.category if args.category else GROUPS.keys()
    jobs: list[tuple[str, GroupConfig, list[str]]] = []
    for env_id, group in iter_envs(selected):
        model_name = sanitize_model_name(env_id, args.model_prefix)
        if args.skip_existing and has_status(model_name):
            print(f"[skip] {env_id} (existing status)")
            continue
        cmd = command_for(env_id, group, args)
        print(f"[run] {env_id}\n       {shlex.join(cmd)}")
        jobs.append((env_id, group, cmd))

    if args.dry_run or not jobs:
        return

    max_workers = max(1, args.max_parallel)
    exit_code = 0

    def run_job(env_id: str, command: list[str]) -> None:
        print(f"[start] {env_id}")
        subprocess.run(command, check=True)
        print(f"[done] {env_id}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_job, env_id, cmd): env_id for env_id, _, cmd in jobs
        }
        for future in as_completed(future_map):
            env_id = future_map[future]
            try:
                future.result()
            except subprocess.CalledProcessError as exc:
                exit_code = exc.returncode
                print(
                    f"[error] {env_id} failed with exit code {exc.returncode}",
                    file=sys.stderr,
                )
                if not args.ignore_failures:
                    for pending in future_map:
                        pending.cancel()
                    sys.exit(exit_code)

    if exit_code and not args.ignore_failures:
        sys.exit(exit_code)


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        main(cli_args)
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user", file=sys.stderr)
