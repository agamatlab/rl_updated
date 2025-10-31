#!/usr/bin/env python3
"""
Script to analyze training and evaluation results for reinforcement learning environments.
Classifies environments based on convergence, reshaped return, and evaluation return.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse


class TrainingAnalyzer:
    """Analyzes training and evaluation results for RL environments."""

    def __init__(self, storage_path: str = "./storage",
                 convergence_threshold: float = 0.05,
                 convergence_window: int = 5,
                 early_convergence_ratio: float = 0.25,
                 high_reshaped_return_threshold: float = 0.7,
                 high_eval_return_threshold: float = 0.9):
        """
        Initialize the analyzer with configurable thresholds.

        Args:
            storage_path: Path to storage directory containing environment logs
            convergence_threshold: Max std deviation to consider converged
            convergence_window: Number of updates to check for stability
            early_convergence_ratio: Ratio of total updates to classify as "early"
            high_reshaped_return_threshold: Threshold for high vs low reshaped return
            high_eval_return_threshold: Threshold for high vs low evaluation return
        """
        self.storage_path = Path(storage_path)
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.early_convergence_ratio = early_convergence_ratio
        self.high_reshaped_return_threshold = high_reshaped_return_threshold
        self.high_eval_return_threshold = high_eval_return_threshold

    def analyze_training_convergence(self, df: pd.DataFrame) -> Tuple[str, Optional[int]]:
        """
        Analyze training data to determine convergence status.

        Args:
            df: DataFrame with training log data

        Returns:
            Tuple of (convergence_status, convergence_update)
            convergence_status: "yes(early)", "yes(late)", or "no"
            convergence_update: Update number where convergence occurred (or None)
        """
        if len(df) < self.convergence_window:
            return "no", None

        # Look for convergence: stable rreturn_mean with low variance
        for i in range(self.convergence_window, len(df)):
            window = df['rreturn_mean'].iloc[i-self.convergence_window:i]

            # Check if values are non-zero and stable
            if window.mean() > 0.1:  # Ensure agent is learning something
                std = window.std()
                if std < self.convergence_threshold:
                    convergence_update = df['update'].iloc[i]
                    total_updates = df['update'].max()

                    # Determine if early or late convergence
                    if i < len(df) * self.early_convergence_ratio:
                        return "yes(early)", convergence_update
                    else:
                        return "yes(late)", convergence_update

        return "no", None

    def classify_reshaped_return(self, df: pd.DataFrame) -> str:
        """
        Classify reshaped return as high or low based on final performance.

        Args:
            df: DataFrame with training log data

        Returns:
            "high" or "low"
        """
        # Take mean of last 10% of training
        tail_size = max(1, len(df) // 10)
        final_return = df['rreturn_mean'].tail(tail_size).mean()

        return "high" if final_return >= self.high_reshaped_return_threshold else "low"

    def analyze_evaluation(self, eval_df: pd.DataFrame) -> str:
        """
        Analyze evaluation data to classify return.

        Args:
            eval_df: DataFrame with evaluation log data

        Returns:
            "high" or "low"
        """
        mean_return = eval_df['return'].mean()
        return "high" if mean_return >= self.high_eval_return_threshold else "low"

    def analyze_environment(self, env_name: str) -> Optional[Dict]:
        """
        Analyze a single environment's training and evaluation results.

        Args:
            env_name: Name of the environment directory

        Returns:
            Dictionary with classification results or None if no data
        """
        env_path = self.storage_path / env_name
        training_log = env_path / "log.csv"

        if not training_log.exists():
            return None

        try:
            # Read training data
            df = pd.read_csv(training_log)

            # Filter out duplicate headers (some files have repeated headers)
            df = df[df['update'] != 'update']

            # Convert to numeric
            df['update'] = pd.to_numeric(df['update'], errors='coerce')
            df['rreturn_mean'] = pd.to_numeric(df['rreturn_mean'], errors='coerce')
            df['frames'] = pd.to_numeric(df['frames'], errors='coerce')

            # Drop rows with NaN values
            df = df.dropna(subset=['update', 'rreturn_mean'])

            if len(df) == 0:
                return None

            # Analyze convergence
            convergence_status, convergence_update = self.analyze_training_convergence(df)

            # Classify reshaped return
            reshaped_return = self.classify_reshaped_return(df)

            result = {
                "env_name": env_name,
                "convergence": convergence_status,
                "reshaped_return": reshaped_return,
                "total_frames": int(df['frames'].max()),
                "total_updates": int(df['update'].max()),
                "final_rreturn_mean": float(df['rreturn_mean'].tail(10).mean())
            }

            if convergence_update:
                result["convergence_update"] = int(convergence_update)

            # Check for evaluation data if converged
            if convergence_status.startswith("yes"):
                eval_log = env_path / "eval_logs" / "logs.csv"
                if eval_log.exists():
                    try:
                        eval_df = pd.read_csv(eval_log)
                        eval_df['return'] = pd.to_numeric(eval_df['return'], errors='coerce')
                        eval_df = eval_df.dropna(subset=['return'])

                        if len(eval_df) > 0:
                            eval_return = self.analyze_evaluation(eval_df)
                            result["eval_return"] = eval_return
                            result["eval_mean_return"] = float(eval_df['return'].mean())
                            result["eval_std_return"] = float(eval_df['return'].std())
                            result["eval_episodes"] = len(eval_df)
                    except Exception as e:
                        print(f"Warning: Could not read evaluation data for {env_name}: {e}")

            return result

        except Exception as e:
            print(f"Error analyzing {env_name}: {e}")
            return None

    def analyze_all_environments(self) -> Dict[str, Dict]:
        """
        Analyze all environments in the storage directory.

        Returns:
            Dictionary mapping environment names to their classifications
        """
        results = {}

        # Get all directories in storage
        if not self.storage_path.exists():
            print(f"Error: Storage path {self.storage_path} does not exist")
            return results

        env_dirs = [d for d in self.storage_path.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]

        print(f"Found {len(env_dirs)} environment directories")

        for env_dir in sorted(env_dirs):
            env_name = env_dir.name
            print(f"Analyzing {env_name}...", end=" ")

            result = self.analyze_environment(env_name)

            if result:
                results[env_name] = result
                print(f"✓ Convergence: {result['convergence']}, "
                      f"Reshaped Return: {result['reshaped_return']}", end="")
                if 'eval_return' in result:
                    print(f", Eval Return: {result['eval_return']}")
                else:
                    print()
            else:
                print("✗ No valid training data")

        return results

    def generate_tags(self, result: Dict) -> str:
        """
        Generate a tag string for an environment based on its classification.

        Args:
            result: Classification result dictionary

        Returns:
            Tag string
        """
        tags = []
        tags.append(f"convergence:{result['convergence']}")
        tags.append(f"reshaped_return:{result['reshaped_return']}")
        if 'eval_return' in result:
            tags.append(f"eval_return:{result['eval_return']}")

        return ",".join(tags)

    def save_results(self, results: Dict[str, Dict], output_file: str = "environment_classifications.json"):
        """
        Save analysis results to a JSON file.

        Args:
            results: Dictionary of analysis results
            output_file: Output file path
        """
        output_path = Path(output_file)

        # Create output with tags
        output = {
            "metadata": {
                "total_environments": len(results),
                "converged": sum(1 for r in results.values() if r['convergence'].startswith('yes')),
                "evaluated": sum(1 for r in results.values() if 'eval_return' in r),
                "thresholds": {
                    "convergence_threshold": self.convergence_threshold,
                    "high_reshaped_return": self.high_reshaped_return_threshold,
                    "high_eval_return": self.high_eval_return_threshold
                }
            },
            "environments": results,
            "tags": {env_name: self.generate_tags(result)
                    for env_name, result in results.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"Total environments analyzed: {len(results)}")
        print(f"Converged: {output['metadata']['converged']}")
        print(f"With evaluation data: {output['metadata']['evaluated']}")

    def generate_summary_report(self, results: Dict[str, Dict]) -> str:
        """
        Generate a summary report of the analysis.

        Args:
            results: Dictionary of analysis results

        Returns:
            Summary report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ENVIRONMENT CLASSIFICATION SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall statistics
        total = len(results)
        converged_early = sum(1 for r in results.values() if r['convergence'] == 'yes(early)')
        converged_late = sum(1 for r in results.values() if r['convergence'] == 'yes(late)')
        not_converged = sum(1 for r in results.values() if r['convergence'] == 'no')

        report_lines.append(f"Total Environments: {total}")
        report_lines.append(f"  - Converged (early): {converged_early}")
        report_lines.append(f"  - Converged (late): {converged_late}")
        report_lines.append(f"  - Not converged: {not_converged}")
        report_lines.append("")

        # Reshaped return classification
        high_reshaped = sum(1 for r in results.values() if r['reshaped_return'] == 'high')
        low_reshaped = sum(1 for r in results.values() if r['reshaped_return'] == 'low')

        report_lines.append(f"Reshaped Return Classification:")
        report_lines.append(f"  - High: {high_reshaped}")
        report_lines.append(f"  - Low: {low_reshaped}")
        report_lines.append("")

        # Evaluation return classification (for converged environments)
        converged_envs = [r for r in results.values() if r['convergence'].startswith('yes')]
        evaluated_envs = [r for r in converged_envs if 'eval_return' in r]
        high_eval = sum(1 for r in evaluated_envs if r['eval_return'] == 'high')
        low_eval = sum(1 for r in evaluated_envs if r['eval_return'] == 'low')

        report_lines.append(f"Evaluation Return Classification (converged environments):")
        report_lines.append(f"  - Evaluated: {len(evaluated_envs)} / {len(converged_envs)}")
        report_lines.append(f"  - High: {high_eval}")
        report_lines.append(f"  - Low: {low_eval}")
        report_lines.append("")

        # Detailed breakdown
        report_lines.append("=" * 80)
        report_lines.append("DETAILED BREAKDOWN")
        report_lines.append("=" * 80)
        report_lines.append("")

        for env_name, result in sorted(results.items()):
            report_lines.append(f"{env_name}:")
            report_lines.append(f"  Tags: {self.generate_tags(result)}")
            report_lines.append(f"  Frames: {result['total_frames']:,}")
            report_lines.append(f"  Final reshaped return: {result['final_rreturn_mean']:.4f}")
            if 'eval_mean_return' in result:
                report_lines.append(f"  Eval mean return: {result['eval_mean_return']:.4f} ± {result['eval_std_return']:.4f}")
            report_lines.append("")

        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training and evaluation results for RL environments"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./storage",
        help="Path to storage directory (default: ./storage)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="environment_classifications.json",
        help="Output JSON file path (default: environment_classifications.json)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Optional path to save summary report (text file)"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.05,
        help="Std deviation threshold for convergence (default: 0.05)"
    )
    parser.add_argument(
        "--high-reshaped-return",
        type=float,
        default=0.7,
        help="Threshold for high reshaped return (default: 0.7)"
    )
    parser.add_argument(
        "--high-eval-return",
        type=float,
        default=0.9,
        help="Threshold for high evaluation return (default: 0.9)"
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = TrainingAnalyzer(
        storage_path=args.storage_path,
        convergence_threshold=args.convergence_threshold,
        high_reshaped_return_threshold=args.high_reshaped_return,
        high_eval_return_threshold=args.high_eval_return
    )

    # Analyze all environments
    print("Starting analysis...")
    print("=" * 80)
    results = analyzer.analyze_all_environments()
    print("=" * 80)

    # Save results
    analyzer.save_results(results, args.output)

    # Generate and optionally save report
    report = analyzer.generate_summary_report(results)
    print("\n" + report)

    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nSummary report saved to {args.report}")


if __name__ == "__main__":
    main()
