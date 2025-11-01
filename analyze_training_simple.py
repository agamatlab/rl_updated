#!/usr/bin/env python3
"""
Script to analyze training and evaluation results for reinforcement learning environments.
Uses only standard library (no external dependencies).
Classifies environments based on convergence, reshaped return, and evaluation return.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import statistics


class TrainingAnalyzer:
    """Analyzes training and evaluation results for RL environments."""

    def __init__(self, storage_path: str = "./storage",
                 convergence_threshold: float = 0.05,
                 convergence_window: int = 10,
                 convergence_stability_length: int = 15,
                 min_convergence_value: float = 0.5,
                 early_convergence_ratio: float = 0.25,
                 high_reshaped_return_threshold: float = 0.7,
                 high_eval_return_threshold: float = 0.9):
        """
        Initialize the analyzer with configurable thresholds.

        Args:
            storage_path: Path to storage directory containing environment logs
            convergence_threshold: Max std deviation to consider converged
            convergence_window: Number of updates to check for stability
            convergence_stability_length: How long convergence must be maintained
            min_convergence_value: Minimum mean value to be considered converged
            early_convergence_ratio: Ratio of total updates to classify as "early"
            high_reshaped_return_threshold: Threshold for high vs low reshaped return
            high_eval_return_threshold: Threshold for high vs low evaluation return
        """
        self.storage_path = Path(storage_path)
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.convergence_stability_length = convergence_stability_length
        self.min_convergence_value = min_convergence_value
        self.early_convergence_ratio = early_convergence_ratio
        self.high_reshaped_return_threshold = high_reshaped_return_threshold
        self.high_eval_return_threshold = high_eval_return_threshold

    def read_training_csv(self, file_path: Path) -> List[Dict]:
        """Read training CSV and parse into list of dicts."""
        rows = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip duplicate headers
                if row.get('update') == 'update':
                    continue
                try:
                    rows.append({
                        'update': int(row['update']),
                        'frames': int(row['frames']),
                        'rreturn_mean': float(row['rreturn_mean']),
                    })
                except (ValueError, KeyError):
                    continue
        return rows

    def read_eval_csv(self, file_path: Path) -> List[Dict]:
        """Read evaluation CSV and parse into list of dicts."""
        rows = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        'episode': int(row['episode']),
                        'return': float(row['return']),
                        'num_frames': float(row['num_frames'])
                    })
                except (ValueError, KeyError):
                    continue
        return rows

    def analyze_training_convergence(self, data: List[Dict]) -> Tuple[str, Optional[int]]:
        """
        Analyze training data to determine convergence status.

        More robust convergence detection:
        1. Must maintain low variance for a sustained period
        2. Final values must be consistent with convergence
        3. Must reach a minimum performance threshold

        Args:
            data: List of training data dictionaries

        Returns:
            Tuple of (convergence_status, convergence_update)
            convergence_status: "yes(early)", "yes(late)", or "no"
            convergence_update: Update number where convergence occurred (or None)
        """
        if len(data) < self.convergence_window + self.convergence_stability_length:
            return "no", None

        convergence_point = None

        # Look for convergence: stable rreturn_mean with low variance that is sustained
        for i in range(self.convergence_window, len(data) - self.convergence_stability_length):
            window_values = [data[j]['rreturn_mean']
                           for j in range(i - self.convergence_window, i)]

            # Check if values are stable with high mean
            mean_val = statistics.mean(window_values)

            # Only consider as potential convergence if mean is high enough
            if mean_val >= self.min_convergence_value:
                if len(window_values) > 1:
                    std = statistics.stdev(window_values)
                    if std < self.convergence_threshold:
                        # Check if this stability is maintained for the next stability_length updates
                        future_window = [data[j]['rreturn_mean']
                                       for j in range(i, min(i + self.convergence_stability_length, len(data)))]

                        if len(future_window) >= self.convergence_stability_length:
                            future_mean = statistics.mean(future_window)
                            future_std = statistics.stdev(future_window) if len(future_window) > 1 else 0

                            # Check if future values are still stable and similar to current
                            if (future_std < self.convergence_threshold * 1.5 and  # Allow slightly more variance
                                abs(future_mean - mean_val) < 0.1 and  # Mean shouldn't drift much
                                future_mean >= self.min_convergence_value * 0.9):  # Allow slight degradation

                                convergence_point = i
                                break

        if convergence_point is None:
            return "no", None

        # Also verify that the final portion of training is stable
        tail_size = min(20, len(data) // 5)  # Last 20 updates or 20% of training
        tail_values = [data[j]['rreturn_mean'] for j in range(len(data) - tail_size, len(data))]
        tail_mean = statistics.mean(tail_values)
        tail_std = statistics.stdev(tail_values) if len(tail_values) > 1 else 0

        # If the tail is unstable or performance dropped significantly, not converged
        if tail_std > self.convergence_threshold * 2 or tail_mean < self.min_convergence_value * 0.8:
            return "no", None

        convergence_update = data[convergence_point]['update']

        # Determine if early or late convergence
        if convergence_point < len(data) * self.early_convergence_ratio:
            return "yes(early)", convergence_update
        else:
            return "yes(late)", convergence_update

    def classify_reshaped_return(self, data: List[Dict]) -> str:
        """
        Classify reshaped return as high or low based on final performance.

        Args:
            data: List of training data dictionaries

        Returns:
            "high" or "low"
        """
        # Take mean of last 10% of training
        tail_size = max(1, len(data) // 10)
        tail_values = [data[i]['rreturn_mean']
                      for i in range(len(data) - tail_size, len(data))]
        final_return = statistics.mean(tail_values)

        return "high" if final_return >= self.high_reshaped_return_threshold else "low"

    def analyze_evaluation(self, data: List[Dict]) -> str:
        """
        Analyze evaluation data to classify return.

        Args:
            data: List of evaluation data dictionaries

        Returns:
            "high" or "low"
        """
        returns = [d['return'] for d in data]
        mean_return = statistics.mean(returns)
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
            data = self.read_training_csv(training_log)

            if len(data) == 0:
                return None

            # Analyze convergence
            convergence_status, convergence_update = self.analyze_training_convergence(data)

            # Classify reshaped return
            reshaped_return = self.classify_reshaped_return(data)

            # Get last 10 values for final rreturn_mean
            tail_values = [data[i]['rreturn_mean'] for i in range(max(0, len(data) - 10), len(data))]
            final_rreturn = statistics.mean(tail_values)

            result = {
                "env_name": env_name,
                "convergence": convergence_status,
                "reshaped_return": reshaped_return,
                "total_frames": data[-1]['frames'],
                "total_updates": data[-1]['update'],
                "final_rreturn_mean": round(final_rreturn, 4)
            }

            if convergence_update:
                result["convergence_update"] = convergence_update

            # Check for evaluation data if converged
            if convergence_status.startswith("yes"):
                eval_log = env_path / "eval_logs" / "logs.csv"
                if eval_log.exists():
                    try:
                        eval_data = self.read_eval_csv(eval_log)

                        if len(eval_data) > 0:
                            returns = [d['return'] for d in eval_data]
                            eval_return = self.analyze_evaluation(eval_data)
                            result["eval_return"] = eval_return
                            result["eval_mean_return"] = round(statistics.mean(returns), 4)
                            result["eval_std_return"] = round(statistics.stdev(returns) if len(returns) > 1 else 0.0, 4)
                            result["eval_episodes"] = len(eval_data)
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
        print()

        for env_dir in sorted(env_dirs):
            env_name = env_dir.name
            print(f"Analyzing {env_name}...", end=" ")

            result = self.analyze_environment(env_name)

            if result:
                results[env_name] = result
                print(f"✓ Conv: {result['convergence']}, "
                      f"Reshaped: {result['reshaped_return']}", end="")
                if 'eval_return' in result:
                    print(f", Eval: {result['eval_return']}")
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

        print(f"\n{'=' * 80}")
        print(f"Results saved to {output_path}")
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
        description="Analyze training and evaluation results for RL environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./storage",
        help="Path to storage directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="environment_classifications.json",
        help="Output JSON file path"
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
        help="Std deviation threshold for convergence"
    )
    parser.add_argument(
        "--high-reshaped-return",
        type=float,
        default=0.7,
        help="Threshold for high reshaped return"
    )
    parser.add_argument(
        "--high-eval-return",
        type=float,
        default=0.9,
        help="Threshold for high evaluation return"
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

    if not results:
        print("\nNo environments found or analyzed.")
        return

    # Save results
    analyzer.save_results(results, args.output)

    # Generate and optionally save report
    report = analyzer.generate_summary_report(results)
    print("\n" + report)

    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Summary report saved to {args.report}")


if __name__ == "__main__":
    main()