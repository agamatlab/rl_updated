#!/usr/bin/env python3
"""
Run `evaluate.py` for every model directory found in the storage folder.

The script looks for `log.txt` files inside each model directory to recover the
environment id that was used during training. It then launches
`python3 scripts/evaluate.py --model ... --env ...` for each discovered model,
optionally skipping models that already have evaluation logs.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


ENV_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"--env\s+([^\s]+)"),
    re.compile(r"env=['\"]?([A-Za-z0-9_\-]+(?:-[A-Za-z0-9_\-]+)*)['\"]?"),
)


@dataclass
class ModelRun:
    """Container for a single training run found in storage."""

    name: str
    path: Path
    env_id: Optional[str]
    use_text: bool = False
    use_memory: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every model stored in the storage directory."
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path(os.environ.get("RL_STORAGE") or os.environ.get("PROJECT_STORAGE") or "storage"),
        help="Path to the storage directory (defaults to $RL_STORAGE, $PROJECT_STORAGE, or ./storage).",
    )
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=Path(__file__).with_name("evaluate.py"),
        help="Path to evaluate.py (default: scripts/evaluate.py).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of evaluation episodes.",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=None,
        help="Override the number of evaluation processes.",
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="Force greedy evaluation (passes --argmax to evaluate.py).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have eval logs (storage/<model>/eval_logs/logs.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to evaluate.py.",
    )
    return parser.parse_args()


def discover_models(storage: Path) -> List[ModelRun]:
    models: List[ModelRun] = []

    if not storage.exists():
        raise FileNotFoundError(f"Storage directory {storage} does not exist.")

    for entry in sorted(storage.iterdir()):
        if not entry.is_dir():
            continue

        log_file = entry / "log.txt"
        env_id, use_text, use_memory = extract_run_metadata(log_file)
        models.append(ModelRun(entry.name, entry, env_id, use_text, use_memory))

    return models


def extract_run_metadata(log_file: Path) -> tuple[Optional[str], bool, bool]:
    env_id: Optional[str] = None
    use_text = False
    use_memory = False

    if not log_file.exists():
        return env_id, use_text, use_memory

    last_command_env: Optional[str] = None

    recurrence_pattern = re.compile(r"--recurrence\s+(\d+)")
    recurrence_namespace_pattern = re.compile(r"recurrence=(\d+)")

    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                # Track the environment from command-style lines.
                for pattern in ENV_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        last_command_env = match.group(1)
                        break

                # Flags specified directly on the command line.
                if "--text" in line:
                    use_text = True
                if "--memory" in line or "--mem" in line:
                    use_memory = True
                command_recurrence = recurrence_pattern.search(line)
                if command_recurrence and int(command_recurrence.group(1)) > 1:
                    use_memory = True

                if line.startswith("Namespace("):
                    # Prefer metadata from the Namespace output because it reflects the parsed args.
                    env_match = re.search(r"env='([^']+)'", line)
                    if env_match:
                        env_id = env_match.group(1)
                    elif last_command_env:
                        env_id = last_command_env

                    if "text=True" in line:
                        use_text = True
                    elif "text=False" in line and "--text" not in line:
                        use_text = False

                    if "mem=True" in line:
                        use_memory = True
                    elif "mem=False" in line and "--memory" not in line and "--mem" not in line:
                        use_memory = False

                    ns_recurrence = recurrence_namespace_pattern.search(line)
                    if ns_recurrence and int(ns_recurrence.group(1)) > 1:
                        use_memory = True

    except OSError:
        pass

    if env_id is None:
        env_id = last_command_env

    return env_id, use_text, use_memory


def build_command(
    evaluate_script: Path,
    model_name: str,
    env_id: str,
    args: argparse.Namespace,
    run: ModelRun,
) -> List[str]:
    command: List[str] = [sys.executable, str(evaluate_script), "--model", model_name, "--env", env_id]

    if args.episodes is not None:
        command += ["--episodes", str(args.episodes)]
    if args.procs is not None:
        command += ["--procs", str(args.procs)]
    if args.argmax:
        command.append("--argmax")
    if run.use_text:
        command.append("--text")
    if run.use_memory:
        command.append("--memory")
    if args.extra_args:
        command.extend(args.extra_args)

    return command


def has_existing_eval(model_path: Path) -> bool:
    eval_logs = model_path / "eval_logs" / "logs.csv"
    return eval_logs.exists()


def main() -> int:
    args = parse_args()

    try:
        models = discover_models(args.storage)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not models:
        print(f"No model directories found in {args.storage}")
        return 0

    repo_root = args.evaluate_script.resolve().parent.parent
    base_env = os.environ.copy()
    existing_py_path = base_env.get("PYTHONPATH", "")
    python_path = str(repo_root)
    if existing_py_path:
        python_path = f"{str(repo_root)}{os.pathsep}{existing_py_path}"
    base_env["PYTHONPATH"] = python_path

    for model in models:
        if model.env_id is None:
            print(f"[skip] {model.name:<30} - environment not found in log.txt")
            continue

        if args.skip_existing and has_existing_eval(model.path):
            print(f"[skip] {model.name:<30} - existing eval logs detected")
            continue

        command = build_command(args.evaluate_script, model.name, model.env_id, args, model)
        print(f"[run ] {model.name:<30} -> {' '.join(command)}")

        if args.dry_run:
            continue

        try:
            subprocess.run(command, check=True, cwd=str(repo_root), env=base_env)
        except subprocess.CalledProcessError as exc:
            print(f"[fail] {model.name:<30} - evaluation exited with status {exc.returncode}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
