#!/usr/bin/env python3
"""
Comprehensive evaluation runner for all training runs.

This script orchestrates all evaluation tasks:
1. Run evaluate.py for all models in storage
2. Generate evaluation plots for each model
3. Generate catalog evaluation reports
4. Generate comprehensive reports for all run groups
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command(cmd: list[str], description: str, check: bool = True) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, check=check)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found for {description}")
        return False


def evaluate_all_models(
    episodes: Optional[int] = None,
    procs: Optional[int] = None,
    argmax: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run evaluation for all models in storage."""
    cmd = [
        sys.executable,
        "scripts/run_storage_evaluations.py",
    ]

    if episodes:
        cmd.extend(["--episodes", str(episodes)])
    if procs:
        cmd.extend(["--procs", str(procs)])
    if argmax:
        cmd.append("--argmax")
    if skip_existing:
        cmd.append("--skip-existing")
    if dry_run:
        cmd.append("--dry-run")

    return run_command(cmd, "Evaluate all models in storage")


def generate_plots(storage_dir: str = "storage", dry_run: bool = False) -> bool:
    """Generate evaluation plots for all models."""
    storage_path = Path(storage_dir)

    if not storage_path.exists():
        print(f"✗ Storage directory {storage_dir} does not exist")
        return False

    success_count = 0
    total_count = 0

    for model_dir in sorted(storage_path.iterdir()):
        if not model_dir.is_dir():
            continue

        # Check if eval logs exist
        eval_logs = model_dir / "eval_logs" / "logs.csv"
        if not eval_logs.exists():
            print(f"⊘ Skipping {model_dir.name} - no eval logs found")
            continue

        total_count += 1
        cmd = [
            sys.executable,
            "plot_evaluation.py",
            "--path", str(model_dir),
        ]

        if dry_run:
            print(f"Would run: {' '.join(cmd)}")
            success_count += 1
        else:
            print(f"\nGenerating plot for {model_dir.name}...")
            if run_command(cmd, f"Plot for {model_dir.name}", check=False):
                success_count += 1

    if total_count == 0:
        print("⊘ No models with eval logs found")
        return True

    print(f"\n{'='*60}")
    print(f"Plot generation: {success_count}/{total_count} successful")
    print('='*60)
    return success_count == total_count


def evaluate_catalog(
    prefix: str = "catalog_",
    output: str = "catalog_evaluation.md",
) -> bool:
    """Generate catalog evaluation report."""
    cmd = [
        sys.executable,
        "scripts/evaluate_catalog.py",
        "--prefix", prefix,
        "--output", output,
    ]

    return run_command(cmd, "Generate catalog evaluation report", check=False)


def evaluate_all_groups(
    output_dir: str = "evaluation_reports",
    min_runs: int = 1,
) -> bool:
    """Generate comprehensive reports for all run groups."""
    cmd = [
        sys.executable,
        "scripts/evaluate_all.py",
        "--output-dir", output_dir,
        "--min-runs", str(min_runs),
    ]

    return run_command(cmd, "Generate comprehensive evaluation reports", check=False)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation options")
    eval_group.add_argument(
        "--episodes",
        type=int,
        help="Number of evaluation episodes (default: 100)",
    )
    eval_group.add_argument(
        "--procs",
        type=int,
        help="Number of evaluation processes (default: 16)",
    )
    eval_group.add_argument(
        "--argmax",
        action="store_true",
        help="Use greedy evaluation (argmax policy)",
    )
    eval_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have evaluation logs",
    )

    # Task selection
    task_group = parser.add_argument_group("Task selection")
    task_group.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run model evaluation (skip plots and reports)",
    )
    task_group.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate plots (skip evaluation and reports)",
    )
    task_group.add_argument(
        "--reports-only",
        action="store_true",
        help="Only generate reports (skip evaluation and plots)",
    )
    task_group.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    task_group.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip report generation",
    )

    # Report options
    report_group = parser.add_argument_group("Report options")
    report_group.add_argument(
        "--catalog-prefix",
        default="catalog_",
        help="Prefix for catalog runs (default: catalog_)",
    )
    report_group.add_argument(
        "--catalog-output",
        default="catalog_evaluation.md",
        help="Output file for catalog report (default: catalog_evaluation.md)",
    )
    report_group.add_argument(
        "--report-dir",
        default="evaluation_reports",
        help="Output directory for comprehensive reports (default: evaluation_reports)",
    )
    report_group.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum runs per prefix group for reports (default: 1)",
    )

    # Storage options
    parser.add_argument(
        "--storage",
        default="storage",
        help="Storage directory path (default: storage)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine which tasks to run
    run_eval = not (args.plots_only or args.reports_only)
    run_plots = not (args.eval_only or args.reports_only or args.no_plots)
    run_reports = not (args.eval_only or args.plots_only or args.no_reports)

    print("="*60)
    print("COMPREHENSIVE EVALUATION RUNNER")
    print("="*60)
    print(f"Storage directory: {args.storage}")
    print(f"Tasks to run:")
    print(f"  - Model evaluation: {run_eval}")
    print(f"  - Plot generation: {run_plots}")
    print(f"  - Report generation: {run_reports}")
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual execution")
    print("="*60)

    results = []

    # Step 1: Evaluate all models
    if run_eval:
        success = evaluate_all_models(
            episodes=args.episodes,
            procs=args.procs,
            argmax=args.argmax,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        results.append(("Model evaluation", success))

    # Step 2: Generate plots
    if run_plots:
        success = generate_plots(
            storage_dir=args.storage,
            dry_run=args.dry_run,
        )
        results.append(("Plot generation", success))

    # Step 3: Generate reports
    if run_reports:
        # Catalog report
        success = evaluate_catalog(
            prefix=args.catalog_prefix,
            output=args.catalog_output,
        )
        results.append(("Catalog report", success))

        # Comprehensive reports
        success = evaluate_all_groups(
            output_dir=args.report_dir,
            min_runs=args.min_runs,
        )
        results.append(("Comprehensive reports", success))

    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    for task, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {task}")

    all_success = all(success for _, success in results)

    if all_success:
        print("\n🎉 All tasks completed successfully!")
        return 0
    else:
        print("\n⚠️  Some tasks failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user", file=sys.stderr)
        sys.exit(130)
