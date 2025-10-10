#!/usr/bin/env python3
"""
Lightweight benchmark helper that runs the PPO/A2C training script a few times,
records wall-clock duration, and reports rich throughput statistics.

Use it on different machines (laptop vs VM) with identical flags to compare CPU
or GPU performance. A 10 minute wall-clock cap is enforced by default so the
benchmark stays quick to iterate on.

Example:
    python3 scripts/benchmark.py --env BabyAI-GoToRedBall-v0 --frames 200000 --procs 4 --runs 2
"""

from __future__ import annotations

import argparse
import os
import platform
import selectors
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="BabyAI-GoToRedBall-v0", help="Environment id to benchmark.")
    parser.add_argument("--algo", default="ppo", help="RL algorithm (forwarded to scripts.train).")
    parser.add_argument("--frames", type=int, default=200_000, help="Number of frames per run.")
    parser.add_argument("--procs", type=int, default=4, help="Number of environment workers.")
    parser.add_argument("--seed", type=int, default=1, help="Base seed (incremented each run).")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions.")
    parser.add_argument(
        "--model-prefix",
        default="bench_",
        help="Prefix for model directories (timestamp appended automatically).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save interval forwarded to training (default 0 = disable checkpoints).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log interval forwarded to training script.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Set CUDA_VISIBLE_DEVICES='' so the run stays on CPU only.",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=10.0,
        help="Overall wall-clock budget in minutes (0 disables the cap).",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=(),
        help="Additional arguments appended to scripts.train (prefix with --).",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, model_name: str, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.train",
        "--algo",
        args.algo,
        "--env",
        args.env,
        "--model",
        model_name,
        "--frames",
        str(args.frames),
        "--seed",
        str(seed),
        "--procs",
        str(args.procs),
        "--save-interval",
        str(args.save_interval),
        "--log-interval",
        str(args.log_interval),
    ]
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def gather_system_info() -> dict[str, str]:
    """Collect lightweight host details for richer benchmark context."""
    info: dict[str, str] = {
        "python": f"{sys.version.split()[0]} ({sys.executable})",
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count() or "unknown"),
    }

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["memory"] = f"{vm.total / (1024**3):.1f} GB RAM"
    except Exception:
        info["memory"] = "psutil not available"

    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu0"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "torch not available"

    return info


def print_kv(prefix: str, items: Iterable[tuple[str, str]]) -> None:
    for key, value in items:
        print(f"{prefix}{key}: {value}")


def run_once(
    args: argparse.Namespace,
    run_idx: int,
    model_dir: Path,
    deadline: float | None,
) -> tuple[float, float]:
    seed = args.seed + run_idx
    timestamp = int(time.time())
    model_name = f"{args.model_prefix}{seed}_{timestamp}"
    cmd = build_command(args, model_name, seed)

    env = os.environ.copy()
    env.setdefault("RL_STORAGE", str(model_dir))
    env.setdefault("PROJECT_STORAGE", str(model_dir))
    if args.force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    print(f"\n[bench] run {run_idx + 1}/{args.runs}")
    print(f"[bench] command: {shlex.join(cmd)}")

    start = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    last_line = ""
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.perf_counter())
                if timeout <= 0.0:
                    raise TimeoutError
            events = selector.select(timeout)
            if not events:
                raise TimeoutError

            for key, _ in events:
                stream = key.fileobj
                assert stream is not None
                line = stream.readline()
                if line == "":
                    break
                last_line = line
                print(line, end="")
            else:
                continue
            break
    except TimeoutError:
        print(f"\n[bench] run {run_idx + 1} exceeded the {args.max_minutes:.2f} minute limit", file=sys.stderr)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        selector.close()

    return_code = process.wait()
    end = time.perf_counter()

    if return_code != 0:
        raise RuntimeError(f"Run {run_idx + 1} exited with code {return_code}")

    duration = end - start
    fps = args.frames / duration if duration > 0 else 0.0

    print(f"[bench] run {run_idx + 1} finished in {duration:.2f}s ({fps:.1f} FPS)")
    print(f"[bench] last log line: {last_line.strip()}")
    return duration, fps


def main() -> None:
    args = parse_args()
    durations = []
    fps_values = []

    storage = Path(os.environ.get("RL_STORAGE") or os.environ.get("PROJECT_STORAGE") or "storage")
    storage.mkdir(parents=True, exist_ok=True)

    sys_info = gather_system_info()
    print("\n[bench] system info")
    print_kv("[bench]   ", sys_info.items())
    config_items: list[tuple[str, str]] = [
        ("env", args.env),
        ("algo", args.algo),
        ("frames_per_run", f"{args.frames:,}"),
        ("procs", str(args.procs)),
        ("runs_requested", str(args.runs)),
        ("seed_base", str(args.seed)),
        ("save_interval", str(args.save_interval)),
        ("log_interval", str(args.log_interval)),
        ("force_cpu", str(bool(args.force_cpu))),
        ("max_minutes", "unlimited" if args.max_minutes <= 0 else f"{args.max_minutes:.2f}"),
        ("extra_flags", " ".join(args.extra) if args.extra else "none"),
    ]
    if args.procs > 0:
        frames_per_proc = args.frames / args.procs
        config_items.insert(3, ("frames_per_proc", f"{frames_per_proc:,.0f}"))

    print("[bench] benchmark config")
    print_kv("[bench]   ", config_items)

    deadline: float | None = None
    if args.max_minutes > 0:
        deadline = time.perf_counter() + args.max_minutes * 60.0
        print(f"[bench] overall time budget: {args.max_minutes:.2f} minutes")
    else:
        print("[bench] overall time budget: unlimited")

    for idx in range(args.runs):
        if deadline is not None and time.perf_counter() >= deadline:
            print(f"[bench] stopping before run {idx + 1}: no time left in the {args.max_minutes:.2f} minute budget.")
            break
        try:
            duration, fps = run_once(args, idx, storage, deadline)
        except TimeoutError:
            print(f"[bench] stopping after {idx} completed run(s) due to time budget.")
            break
        durations.append(duration)
        fps_values.append(fps)
        if deadline is not None:
            remaining = max(0.0, deadline - time.perf_counter())
            print(f"[bench] remaining budget: {remaining / 60.0:.2f} minutes")

    print("\n===== benchmark summary =====")
    if durations:
        total_time = sum(durations)
        total_frames = args.frames * len(durations)
        avg_duration = statistics.mean(durations)
        avg_fps = statistics.mean(fps_values)
        fps_std = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0
        duration_std = statistics.stdev(durations) if len(durations) > 1 else 0.0
        effective_fps = total_frames / total_time if total_time > 0 else 0.0

        for i, (d, f) in enumerate(zip(durations, fps_values), start=1):
            print(f"Run {i}: {d:.2f}s, {f:.1f} FPS")

        print(f"Runs completed: {len(durations)}/{args.runs}")
        print(f"Total wall-clock: {total_time:.2f}s")
        print(f"Total frames: {total_frames:,}")
        print(f"Average duration: {avg_duration:.2f}s (std {duration_std:.2f}s)")
        print(f"Average FPS: {avg_fps:.1f} (std {fps_std:.1f})")
        print(f"Effective FPS (frames/total time): {effective_fps:.1f}")
        if args.procs > 0:
            print(f"Per-proc FPS (avg): {avg_fps / args.procs:.1f}")
    else:
        print("No runs completed within the configured budget.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bench] interrupted by user", file=sys.stderr)
