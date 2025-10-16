#!/usr/bin/env python3
"""
Evaluate all training runs and generate comprehensive markdown reports.

This script discovers all training run prefixes in the storage directory,
runs the evaluation for each prefix group, and generates individual markdown
reports plus a master summary report comparing all run groups.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Import from evaluate_catalog
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_catalog import (
    RunMetrics,
    parse_log_csv,
    storage_dir,
    generate_category_stats,
    CategoryStats,
)


@dataclass
class PrefixGroupSummary:
    """Summary statistics for a group of runs with the same prefix."""
    prefix: str
    total_runs: int
    successful: int
    failed: int
    in_progress: int
    no_data: int
    success_rate: float
    avg_final_return: float
    total_frames: int
    report_path: Optional[Path]


def discover_run_prefixes(storage: Path, min_runs: int = 1) -> Dict[str, List[Path]]:
    """
    Discover all run prefixes in the storage directory.

    Returns a dictionary mapping prefixes to lists of run directories.
    """
    if not storage.exists():
        print(f"Error: Storage directory {storage} does not exist", file=sys.stderr)
        return {}

    # Group directories by prefix
    prefix_groups: Dict[str, List[Path]] = defaultdict(list)

    for model_dir in storage.iterdir():
        if not model_dir.is_dir():
            continue

        # Skip hidden directories and common non-model directories
        if model_dir.name.startswith('.') or model_dir.name in ['logs', 'tmp', 'temp']:
            continue

        # Extract prefix (everything before the last underscore or the whole name)
        name = model_dir.name
        # Look for common patterns: prefix_envname, prefix_date, etc.
        parts = name.split('_')
        if len(parts) >= 2:
            # Try to identify the prefix
            # Common patterns: catalog_X, run_X, experiment_X, etc.
            prefix = parts[0] + '_'
            prefix_groups[prefix].append(model_dir)
        else:
            # No clear prefix, use the whole name as prefix
            prefix_groups[name + '_'].append(model_dir)

    # Filter out groups with too few runs
    return {k: v for k, v in prefix_groups.items() if len(v) >= min_runs}


def evaluate_prefix_group(
    prefix: str,
    run_dirs: List[Path],
    output_dir: Path,
) -> PrefixGroupSummary:
    """
    Evaluate all runs for a given prefix and generate a report.

    Returns summary statistics for the group.
    """
    print(f"\nEvaluating prefix group: {prefix}")
    print(f"  Found {len(run_dirs)} runs")

    # Parse all runs
    runs: List[RunMetrics] = []
    for run_dir in run_dirs:
        log_path = run_dir / "log.csv"
        metrics = parse_log_csv(log_path)
        runs.append(metrics)

    if not runs:
        print(f"  Warning: No valid runs found for prefix {prefix}")
        return PrefixGroupSummary(
            prefix=prefix,
            total_runs=0,
            successful=0,
            failed=0,
            in_progress=0,
            no_data=0,
            success_rate=0.0,
            avg_final_return=0.0,
            total_frames=0,
            report_path=None,
        )

    # Calculate summary statistics
    total_runs = len(runs)
    successful = sum(1 for r in runs if r.success)
    failed = sum(1 for r in runs if not r.success and r.converged and r.num_samples > 0)
    in_progress = sum(1 for r in runs if not r.success and not r.converged and r.num_samples > 0)
    no_data = sum(1 for r in runs if r.num_samples == 0)

    success_rate = successful / total_runs if total_runs > 0 else 0.0

    returns_with_data = [r.final_return_mean for r in runs if r.final_return_mean is not None]
    avg_final_return = sum(returns_with_data) / len(returns_with_data) if returns_with_data else 0.0

    total_frames = sum(r.total_frames for r in runs if r.total_frames > 0)

    # Generate individual report
    report_path = output_dir / f"{prefix.rstrip('_')}_evaluation.md"
    generate_detailed_report(runs, prefix, report_path)

    print(f"  ✓ Generated report: {report_path}")
    print(f"  Success rate: {successful}/{total_runs} ({success_rate*100:.1f}%)")

    return PrefixGroupSummary(
        prefix=prefix,
        total_runs=total_runs,
        successful=successful,
        failed=failed,
        in_progress=in_progress,
        no_data=no_data,
        success_rate=success_rate,
        avg_final_return=avg_final_return,
        total_frames=total_frames,
        report_path=report_path,
    )


def generate_detailed_report(runs: List[RunMetrics], prefix: str, output_path: Path) -> None:
    """Generate a detailed markdown report for a specific prefix group."""
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            0 if r.success else (1 if r.num_samples > 0 else 2),
            -(r.final_return_mean or 0.0)
        )
    )

    category_stats = generate_category_stats(runs)

    with output_path.open("w") as f:
        f.write(f"# Training Evaluation Report: {prefix.rstrip('_')}\n\n")
        f.write("*Generated by evaluate_all.py*\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        total_runs = len(runs)
        successful = sum(1 for r in runs if r.success)
        failed = sum(1 for r in runs if not r.success and r.converged and r.num_samples > 0)
        in_progress = sum(1 for r in runs if not r.success and not r.converged and r.num_samples > 0)
        no_data = sum(1 for r in runs if r.num_samples == 0)

        f.write(f"- **Prefix**: `{prefix}`\n")
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
        if category_stats:
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
            f.write("| Environment | Model | Updates | Frames | Final Return | Max Return |\n")
            f.write("|-------------|-------|---------|--------|--------------|------------|\n")
            for run in failed_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` | {run.total_updates} | "
                       f"{run.total_frames:,} | {run.final_return_mean:.3f} | "
                       f"{run.max_return_mean:.3f} |\n")
            f.write("\n")

        if progress_runs:
            f.write("### 🔄 In Progress\n\n")
            f.write("| Environment | Model | Updates | Frames | Current Return | Max Return |\n")
            f.write("|-------------|-------|---------|--------|----------------|------------|\n")
            for run in progress_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` | {run.total_updates} | "
                       f"{run.total_frames:,} | {run.final_return_mean:.3f} | "
                       f"{run.max_return_mean:.3f} |\n")
            f.write("\n")

        if nodata_runs:
            f.write("### ⚠️ No Training Data\n\n")
            f.write("| Environment | Model |\n")
            f.write("|-------------|-------|\n")
            for run in nodata_runs:
                f.write(f"| {run.env_id} | `{run.model_name}` |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("*Report generated from training runs*\n")


def generate_master_report(
    summaries: List[PrefixGroupSummary],
    output_path: Path,
) -> None:
    """Generate a master summary report comparing all prefix groups."""
    with output_path.open("w") as f:
        f.write("# Master Evaluation Report - All Training Runs\n\n")
        f.write("*Generated by evaluate_all.py*\n\n")
        f.write("---\n\n")

        # Overall Summary
        f.write("## Overall Summary\n\n")
        total_prefixes = len(summaries)
        total_runs = sum(s.total_runs for s in summaries)
        total_successful = sum(s.successful for s in summaries)
        total_failed = sum(s.failed for s in summaries)
        total_in_progress = sum(s.in_progress for s in summaries)
        total_no_data = sum(s.no_data for s in summaries)

        f.write(f"- **Total Prefix Groups**: {total_prefixes}\n")
        f.write(f"- **Total Runs Across All Groups**: {total_runs}\n")
        f.write(f"- **Total Successful**: {total_successful} ({total_successful/total_runs*100:.1f}%)\n")
        f.write(f"- **Total Failed**: {total_failed} ({total_failed/total_runs*100:.1f}%)\n")
        f.write(f"- **Total In Progress**: {total_in_progress} ({total_in_progress/total_runs*100:.1f}%)\n")
        f.write(f"- **Total No Data**: {total_no_data} ({total_no_data/total_runs*100:.1f}%)\n\n")

        total_frames = sum(s.total_frames for s in summaries)
        f.write(f"- **Total Frames Trained**: {total_frames:,}\n\n")

        # Comparison Table
        f.write("---\n\n")
        f.write("## Prefix Group Comparison\n\n")
        f.write("| Prefix | Total Runs | Success | Failed | In Progress | No Data | Success Rate | Avg Return | Total Frames | Report |\n")
        f.write("|--------|------------|---------|--------|-------------|---------|--------------|------------|--------------|--------|\n")

        # Sort by success rate descending
        summaries_sorted = sorted(summaries, key=lambda s: s.success_rate, reverse=True)

        for summary in summaries_sorted:
            report_link = f"[📄]({summary.report_path.name})" if summary.report_path else "N/A"
            f.write(f"| `{summary.prefix}` | {summary.total_runs} | {summary.successful} | "
                   f"{summary.failed} | {summary.in_progress} | {summary.no_data} | "
                   f"{summary.success_rate*100:.1f}% | {summary.avg_final_return:.3f} | "
                   f"{summary.total_frames:,} | {report_link} |\n")

        f.write("\n")

        # Best and Worst Performers
        f.write("---\n\n")
        f.write("## Performance Highlights\n\n")

        if summaries_sorted:
            f.write("### 🏆 Best Performing Groups\n\n")
            top_groups = summaries_sorted[:5]
            for i, summary in enumerate(top_groups, 1):
                f.write(f"{i}. **{summary.prefix.rstrip('_')}**: {summary.success_rate*100:.1f}% success rate "
                       f"({summary.successful}/{summary.total_runs} successful)\n")
            f.write("\n")

            if len(summaries_sorted) > 5:
                f.write("### ⚠️ Groups Needing Attention\n\n")
                bottom_groups = summaries_sorted[-5:][::-1]
                for summary in bottom_groups:
                    f.write(f"- **{summary.prefix.rstrip('_')}**: {summary.success_rate*100:.1f}% success rate "
                           f"({summary.successful}/{summary.total_runs} successful)\n")
                f.write("\n")

        # Recommendations
        f.write("---\n\n")
        f.write("## Recommendations\n\n")

        groups_with_failures = [s for s in summaries if s.failed > 0]
        if groups_with_failures:
            f.write("### Groups with Failed Runs\n\n")
            for summary in groups_with_failures:
                f.write(f"- **{summary.prefix.rstrip('_')}**: {summary.failed} converged runs below target\n")
                f.write(f"  - Review individual report: {summary.report_path.name if summary.report_path else 'N/A'}\n")
            f.write("\n")

        groups_with_no_data = [s for s in summaries if s.no_data > 0]
        if groups_with_no_data:
            f.write("### Groups with Missing Data\n\n")
            for summary in groups_with_no_data:
                f.write(f"- **{summary.prefix.rstrip('_')}**: {summary.no_data} runs with no data\n")
            f.write("\n")

        groups_in_progress = [s for s in summaries if s.in_progress > 0]
        if groups_in_progress:
            f.write("### Groups Still Training\n\n")
            for summary in groups_in_progress:
                f.write(f"- **{summary.prefix.rstrip('_')}**: {summary.in_progress} runs in progress\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## Detailed Reports\n\n")
        f.write("Individual reports for each prefix group:\n\n")
        for summary in summaries_sorted:
            if summary.report_path:
                f.write(f"- [{summary.prefix.rstrip('_')}]({summary.report_path.name})\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("*Master report generated from all training runs*\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage",
        type=Path,
        help="Storage directory path (default: from RL_STORAGE or PROJECT_STORAGE env vars)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_reports"),
        help="Output directory for reports (default: evaluation_reports)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of runs required for a prefix group (default: 1)",
    )
    parser.add_argument(
        "--master-report",
        type=Path,
        default=Path("master_evaluation.md"),
        help="Master summary report filename (default: master_evaluation.md)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set storage directory
    if args.storage:
        storage = args.storage
        os.environ["RL_STORAGE"] = str(storage)
    else:
        storage = storage_dir()

    print(f"Storage directory: {storage}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Discover all prefix groups
    print(f"\nDiscovering run prefixes (min {args.min_runs} runs per group)...")
    prefix_groups = discover_run_prefixes(storage, args.min_runs)

    if not prefix_groups:
        print("No run groups found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(prefix_groups)} prefix groups:")
    for prefix, dirs in prefix_groups.items():
        print(f"  - {prefix}: {len(dirs)} runs")

    # Evaluate each prefix group
    summaries: List[PrefixGroupSummary] = []
    for prefix, run_dirs in sorted(prefix_groups.items()):
        summary = evaluate_prefix_group(prefix, run_dirs, args.output_dir)
        summaries.append(summary)

    # Generate master report
    print(f"\nGenerating master summary report...")
    master_report_path = args.output_dir / args.master_report.name
    generate_master_report(summaries, master_report_path)
    print(f"✓ Master report generated: {master_report_path}")

    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total prefix groups: {len(summaries)}")
    total_runs = sum(s.total_runs for s in summaries)
    total_successful = sum(s.successful for s in summaries)
    print(f"Total runs: {total_runs}")
    print(f"Overall success rate: {total_successful}/{total_runs} ({total_successful/total_runs*100:.1f}%)")
    print(f"\nReports generated in: {args.output_dir}")
    print(f"Master report: {master_report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user", file=sys.stderr)
        sys.exit(1)