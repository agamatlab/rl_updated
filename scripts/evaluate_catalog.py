#!/usr/bin/env python3
"""
Evaluate catalog runs and generate comprehensive markdown reports.

This script analyzes training runs from run_env_catalog.py, examining log files,
performance metrics, and training progression to generate detailed markdown reports
that summarize training success, convergence, and overall catalog performance.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunMetrics:
    """Metrics for a single training run."""
    env_id: str
    model_name: str
    total_updates: int
    total_frames: int
    final_return_mean: Optional[float]
    max_return_mean: Optional[float]
    avg_return_mean: Optional[float]
    final_fps: Optional[float]
    avg_fps: Optional[float]
    converged: bool
    training_duration: Optional[float]
    num_samples: int
    return_progression: List[float]
    frames_progression: List[int]

    @property
    def success(self) -> bool:
        """Check if training was successful (return_mean >= 0.8)."""
        return self.final_return_mean is not None and self.final_return_mean >= 0.8

    @property
    def status(self) -> str:
        """Get human-readable status."""
        if self.num_samples == 0:
            return "No Data"
        if self.success:
            return "Success"
        if self.converged:
            return "Converged (Low)"
        return "In Progress"


@dataclass
class CategoryStats:
    """Statistics for a category of environments."""
    category: str
    total_envs: int
    successful: int
    failed: int
    in_progress: int
    no_data: int
    avg_final_return: float
    avg_frames: float
    total_frames: int


def storage_dir() -> Path:
    """Get storage directory path."""
    return Path(os.environ.get("RL_STORAGE") or os.environ.get("PROJECT_STORAGE") or "storage")


def check_convergence_robust(return_means: List[float], min_updates: int = 50, window_size: int = 25) -> bool:
    """
    Robust convergence detection using multiple criteria.

    A model is considered converged if:
    1. It has trained for at least min_updates
    2. The recent window shows low variance AND
    3. There is minimal improvement trend (slope near zero) AND
    4. No significant improvement compared to earlier performance

    Args:
        return_means: List of return_mean values from training log
        min_updates: Minimum number of updates before considering convergence
        window_size: Size of the recent window to analyze

    Returns:
        True if the model has converged, False otherwise
    """
    if len(return_means) < min_updates:
        # Not enough data to determine convergence
        return False

    # Use the last window_size updates
    window = min(window_size, len(return_means))
    recent = return_means[-window:]

    # Criterion 1: Low variance in recent window
    mean_recent = sum(recent) / len(recent)
    variance = sum((x - mean_recent) ** 2 for x in recent) / len(recent)
    std_dev = variance ** 0.5

    # Threshold: std < 0.08 (was 0.05, but that's too strict and can miss oscillations)
    low_variance = std_dev < 0.08

    # Criterion 2: Flat or declining trend (minimal improvement)
    # Calculate simple linear regression slope
    n = len(recent)
    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = mean_recent

    numerator = sum((x_vals[i] - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n))

    slope = numerator / denominator if denominator != 0 else 0

    # Threshold: slope < 0.002 per update (very minimal improvement)
    flat_trend = abs(slope) < 0.002

    # Criterion 3: No significant improvement vs earlier performance
    # Compare recent average to the average of updates from [window/2 : window] steps ago
    if len(return_means) >= window * 2:
        earlier_start = len(return_means) - window * 2
        earlier_end = len(return_means) - window
        earlier = return_means[earlier_start:earlier_end]
        mean_earlier = sum(earlier) / len(earlier)

        # Improvement should be less than 0.05 (5% absolute improvement)
        improvement = mean_recent - mean_earlier
        minimal_improvement = improvement < 0.05
    else:
        # Not enough history, just check if not improving rapidly
        minimal_improvement = True

    # All criteria must be satisfied for convergence
    converged = low_variance and flat_trend and minimal_improvement

    return converged


def parse_log_csv(log_path: Path) -> RunMetrics:
    """Parse a log.csv file and extract metrics."""
    env_id = "unknown"
    model_name = log_path.parent.name

    # Try to extract env_id from model_name
    if model_name.startswith("catalog_"):
        env_id = model_name.replace("catalog_", "").replace("_", "-").upper()

    total_updates = 0
    total_frames = 0
    final_return_mean = None
    max_return_mean = None
    avg_return_mean = 0.0
    final_fps = None
    avg_fps = 0.0
    training_duration = None
    return_means: List[float] = []
    frames_list: List[int] = []
    fps_list: List[float] = []

    if not log_path.exists():
        return RunMetrics(
            env_id=env_id,
            model_name=model_name,
            total_updates=0,
            total_frames=0,
            final_return_mean=None,
            max_return_mean=None,
            avg_return_mean=0.0,
            final_fps=None,
            avg_fps=0.0,
            converged=False,
            training_duration=None,
            num_samples=0,
            return_progression=[],
            frames_progression=[],
        )

    try:
        with log_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                # Skip header rows that appear in the middle
                if row.get("update") == "update":
                    continue
                rows.append(row)

            for row in rows:
                try:
                    update = int(row.get("update", 0))
                    frames = int(row.get("frames", 0))
                    return_mean = float(row.get("return_mean", 0.0))
                    fps = float(row.get("FPS", 0.0))
                    duration = float(row.get("duration", 0.0))

                    total_updates = max(total_updates, update)
                    total_frames = max(total_frames, frames)
                    return_means.append(return_mean)
                    frames_list.append(frames)
                    fps_list.append(fps)

                    final_return_mean = return_mean
                    final_fps = fps
                    training_duration = duration

                except (ValueError, KeyError):
                    continue

            if return_means:
                avg_return_mean = sum(return_means) / len(return_means)
                max_return_mean = max(return_means)

            if fps_list:
                avg_fps = sum(fps_list) / len(fps_list)

            # Check convergence with robust multi-criteria approach
            converged = check_convergence_robust(return_means)

    except Exception as e:
        print(f"Warning: Error parsing {log_path}: {e}", file=sys.stderr)

    return RunMetrics(
        env_id=env_id,
        model_name=model_name,
        total_updates=total_updates,
        total_frames=total_frames,
        final_return_mean=final_return_mean,
        max_return_mean=max_return_mean,
        avg_return_mean=avg_return_mean,
        final_fps=final_fps,
        avg_fps=avg_fps,
        converged=converged,
        training_duration=training_duration,
        num_samples=len(return_means),
        return_progression=return_means,
        frames_progression=frames_list,
    )


def categorize_env(model_name: str) -> str:
    """Categorize environment based on model name."""
    name_lower = model_name.lower()

    if "babyai" in name_lower:
        if any(x in name_lower for x in ["goto", "go"]):
            return "BabyAI GoTo"
        elif "open" in name_lower:
            return "BabyAI Open"
        elif "pickup" in name_lower or "putnext" in name_lower:
            return "BabyAI Pickup"
        elif "unlock" in name_lower or "key" in name_lower:
            return "BabyAI Unlock"
        elif "find" in name_lower or "oneroom" in name_lower or "movetwo" in name_lower:
            return "BabyAI Search"
        elif "synth" in name_lower or "boss" in name_lower:
            return "BabyAI Composite"
        else:
            return "BabyAI Other"

    elif "minigrid" in name_lower:
        if "door" in name_lower or "key" in name_lower or "locked" in name_lower or "unlock" in name_lower:
            return "MiniGrid Door"
        elif "lava" in name_lower or "crossing" in name_lower or "gap" in name_lower or "obstacle" in name_lower:
            return "MiniGrid Hazards"
        elif "obstructed" in name_lower or "maze" in name_lower:
            return "MiniGrid Obstructed"
        elif "wfc" in name_lower:
            return "MiniGrid WFC"
        else:
            return "MiniGrid Basic"

    return "Other"


def generate_category_stats(runs: List[RunMetrics]) -> Dict[str, CategoryStats]:
    """Generate statistics per category."""
    category_data: Dict[str, List[RunMetrics]] = defaultdict(list)

    for run in runs:
        category = categorize_env(run.model_name)
        category_data[category].append(run)

    stats = {}
    for category, cat_runs in category_data.items():
        successful = sum(1 for r in cat_runs if r.success)
        failed = sum(1 for r in cat_runs if not r.success and r.converged and r.num_samples > 0)
        no_data = sum(1 for r in cat_runs if r.num_samples == 0)
        in_progress = len(cat_runs) - successful - failed - no_data

        returns_with_data = [r.final_return_mean for r in cat_runs if r.final_return_mean is not None]
        avg_final = sum(returns_with_data) / len(returns_with_data) if returns_with_data else 0.0

        frames_with_data = [r.total_frames for r in cat_runs if r.total_frames > 0]
        avg_frames_val = sum(frames_with_data) / len(frames_with_data) if frames_with_data else 0.0

        stats[category] = CategoryStats(
            category=category,
            total_envs=len(cat_runs),
            successful=successful,
            failed=failed,
            in_progress=in_progress,
            no_data=no_data,
            avg_final_return=avg_final,
            avg_frames=avg_frames_val,
            total_frames=sum(frames_with_data),
        )

    return stats


def generate_markdown_report(runs: List[RunMetrics], output_path: Path) -> None:
    """Generate a comprehensive markdown report."""

    # Sort runs by status and return mean
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            0 if r.success else (1 if r.num_samples > 0 else 2),
            -(r.final_return_mean or 0.0)
        )
    )

    category_stats = generate_category_stats(runs)

    with output_path.open("w") as f:
        f.write("# Training Catalog Evaluation Report\n\n")
        f.write("*Generated by evaluate_catalog.py*\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        total_runs = len(runs)
        successful = sum(1 for r in runs if r.success)
        failed = sum(1 for r in runs if not r.success and r.converged and r.num_samples > 0)
        in_progress = sum(1 for r in runs if not r.success and not r.converged and r.num_samples > 0)
        no_data = sum(1 for r in runs if r.num_samples == 0)

        f.write(f"- **Total Environments**: {total_runs}\n")
        f.write(f"- **Successful** (≥0.8 return): {successful} ({successful/total_runs*100:.1f}%)\n")
        f.write(f"- **Failed** (converged but <0.8): {failed} ({failed/total_runs*100:.1f}%)\n")
        f.write(f"- **In Progress**: {in_progress} ({in_progress/total_runs*100:.1f}%)\n")
        f.write(f"- **No Data**: {no_data} ({no_data/total_runs*100:.1f}%)\n\n")

        total_frames = sum(r.total_frames for r in runs if r.total_frames > 0)
        f.write(f"- **Total Frames Trained**: {total_frames:,}\n")

        avg_return = sum(r.final_return_mean for r in runs if r.final_return_mean is not None) / max(1, len([r for r in runs if r.final_return_mean is not None]))
        f.write(f"- **Average Final Return**: {avg_return:.3f}\n\n")

        # Category Breakdown
        f.write("---\n\n")
        f.write("## Performance by Category\n\n")
        f.write("| Category | Total | Success | Failed | In Progress | No Data | Avg Return | Total Frames |\n")
        f.write("|----------|-------|---------|--------|-------------|---------|------------|-------------|\n")

        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            f.write(f"| {stats.category} | {stats.total_envs} | {stats.successful} | {stats.failed} | "
                   f"{stats.in_progress} | {stats.no_data} | {stats.avg_final_return:.3f} | "
                   f"{stats.total_frames:,} |\n")

        f.write("\n")

        # Detailed Results
        f.write("---\n\n")
        f.write("## Detailed Results\n\n")

        # Group by status
        success_runs = [r for r in runs_sorted if r.success]
        failed_runs = [r for r in runs_sorted if not r.success and r.converged and r.num_samples > 0]
        progress_runs = [r for r in runs_sorted if not r.success and not r.converged and r.num_samples > 0]
        nodata_runs = [r for r in runs_sorted if r.num_samples == 0]

        if success_runs:
            f.write("### ✅ Successful Runs\n\n")
            f.write("| Environment | Model | Updates | Frames | Final Return | Max Return | Avg FPS |\n")
            f.write("|-------------|-------|---------|--------|--------------|------------|--------|\n")
            for run in success_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` | {run.total_updates} | "
                       f"{run.total_frames:,} | {run.final_return_mean:.3f} | "
                       f"{run.max_return_mean:.3f} | {run.avg_fps:.0f} |\n")
            f.write("\n")

        if failed_runs:
            f.write("### ❌ Failed Runs (Converged Below Target)\n\n")
            f.write("| Environment | Model | Updates | Frames | Final Return | Max Return | Converged |\n")
            f.write("|-------------|-------|---------|--------|--------------|------------|----------|\n")
            for run in failed_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` | {run.total_updates} | "
                       f"{run.total_frames:,} | {run.final_return_mean:.3f} | "
                       f"{run.max_return_mean:.3f} | {'Yes' if run.converged else 'No'} |\n")
            f.write("\n")

        if progress_runs:
            f.write("### 🔄 In Progress\n\n")
            f.write("| Environment | Model | Updates | Frames | Current Return | Max Return | Samples |\n")
            f.write("|-------------|-------|---------|--------|----------------|------------|--------|\n")
            for run in progress_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` | {run.total_updates} | "
                       f"{run.total_frames:,} | {run.final_return_mean:.3f} | "
                       f"{run.max_return_mean:.3f} | {run.num_samples} |\n")
            f.write("\n")

        if nodata_runs:
            f.write("### ⚠️ No Training Data\n\n")
            f.write("| Environment | Model |\n")
            f.write("|-------------|-------|\n")
            for run in nodata_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` |\n")
            f.write("\n")

        # Performance Analysis
        f.write("---\n\n")
        f.write("## Performance Analysis\n\n")

        f.write("### Training Efficiency\n\n")
        runs_with_fps = [r for r in runs if r.avg_fps is not None and r.avg_fps > 0]
        if runs_with_fps:
            avg_fps = sum(r.avg_fps for r in runs_with_fps) / len(runs_with_fps)
            max_fps = max(r.avg_fps for r in runs_with_fps)
            min_fps = min(r.avg_fps for r in runs_with_fps)
            f.write(f"- **Average FPS**: {avg_fps:.0f}\n")
            f.write(f"- **Max FPS**: {max_fps:.0f}\n")
            f.write(f"- **Min FPS**: {min_fps:.0f}\n\n")

        f.write("### Convergence Rate\n\n")
        converged_runs = [r for r in runs if r.converged and r.num_samples > 0]
        if converged_runs:
            avg_frames_to_converge = sum(r.total_frames for r in converged_runs) / len(converged_runs)
            f.write(f"- **Environments Converged**: {len(converged_runs)}/{len([r for r in runs if r.num_samples > 0])}\n")
            f.write(f"- **Avg Frames to Convergence**: {avg_frames_to_converge:,.0f}\n\n")
        else:
            f.write("- No converged runs yet\n\n")

        # Top Performers
        f.write("### Top Performers (by Final Return)\n\n")
        top_performers = sorted(
            [r for r in runs if r.final_return_mean is not None],
            key=lambda r: r.final_return_mean,
            reverse=True
        )[:10]

        if top_performers:
            f.write("| Rank | Environment | Final Return | Frames |\n")
            f.write("|------|-------------|--------------|--------|\n")
            for i, run in enumerate(top_performers, 1):
                f.write(f"| {i} | {run.env_id} | {run.final_return_mean:.3f} | {run.total_frames:,} |\n")
            f.write("\n")

        # Bottom Performers
        if len([r for r in runs if r.final_return_mean is not None]) > 10:
            f.write("### Needs Attention (Lowest Returns)\n\n")
            bottom_performers = sorted(
                [r for r in runs if r.final_return_mean is not None and r.num_samples >= 5],
                key=lambda r: r.final_return_mean
            )[:10]

            if bottom_performers:
                f.write("| Rank | Environment | Final Return | Frames | Status |\n")
                f.write("|------|-------------|--------------|--------|--------|\n")
                for i, run in enumerate(bottom_performers, 1):
                    f.write(f"| {i} | {run.env_id} | {run.final_return_mean:.3f} | "
                           f"{run.total_frames:,} | {run.status} |\n")
                f.write("\n")

        # Recommendations
        f.write("---\n\n")
        f.write("## Recommendations\n\n")

        if failed_runs:
            f.write("### Failed Runs\n")
            f.write("The following environments converged but did not reach the target return of 0.8:\n\n")
            for run in failed_runs[:5]:
                f.write(f"- **{run.env_id}**: Final return {run.final_return_mean:.3f} after {run.total_frames:,} frames\n")
                f.write(f"  - Consider increasing frame budget or adjusting hyperparameters\n")
            f.write("\n")

        if no_data > 0:
            f.write("### Missing Data\n")
            f.write(f"{no_data} environment(s) have no training data. Consider:\n")
            f.write("- Checking if training started correctly\n")
            f.write("- Verifying storage path configuration\n")
            f.write("- Reviewing error logs for failed runs\n\n")

        if in_progress > 0:
            f.write("### In Progress\n")
            f.write(f"{in_progress} environment(s) are still training. Monitor progress and consider:\n")
            f.write("- Allowing more training time\n")
            f.write("- Increasing frame budget if not improving\n")
            f.write("- Checking for plateaus in learning curves\n\n")

        f.write("---\n\n")
        f.write("*Report generated using data from catalog training runs*\n")


def scan_catalog_runs(prefix: str = "catalog_") -> List[RunMetrics]:
    """Scan storage directory for catalog runs."""
    storage = storage_dir()
    runs = []

    if not storage.exists():
        print(f"Warning: Storage directory {storage} does not exist", file=sys.stderr)
        return runs

    for model_dir in storage.iterdir():
        if not model_dir.is_dir():
            continue
        if not model_dir.name.startswith(prefix):
            continue

        log_path = model_dir / "log.csv"
        metrics = parse_log_csv(log_path)
        runs.append(metrics)

    return runs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="catalog_",
        help="Model directory prefix to filter (default: catalog_)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("catalog_evaluation.md"),
        help="Output markdown file path (default: catalog_evaluation.md)",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        help="Override storage directory path",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.storage:
        os.environ["RL_STORAGE"] = str(args.storage)

    print(f"Scanning for catalog runs with prefix '{args.prefix}'...")
    runs = scan_catalog_runs(args.prefix)

    if not runs:
        print("No catalog runs found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} catalog runs")
    print(f"Generating markdown report to {args.output}...")

    generate_markdown_report(runs, args.output)

    print(f"✓ Report generated successfully: {args.output}")

    # Print quick summary
    successful = sum(1 for r in runs if r.success)
    print(f"\nSummary: {successful}/{len(runs)} environments successful (≥0.8 return)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user", file=sys.stderr)
        sys.exit(1)