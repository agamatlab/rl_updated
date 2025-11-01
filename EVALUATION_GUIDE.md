# Evaluation Scripts Guide

This guide explains how to run evaluations for all your trained models with the correct commands.

## Quick Start

### Run Everything (Recommended)
```bash
./run_evaluations.sh
```

This will:
1. Evaluate all models in the `storage/` directory
2. Generate evaluation plots for each model
3. Generate catalog evaluation reports
4. Generate comprehensive reports for all run groups

### Run with Options
```bash
# Skip models that already have evaluation logs
./run_evaluations.sh --skip-existing

# Show what would be done without executing
./run_evaluations.sh --dry-run

# Use greedy evaluation (argmax policy)
./run_evaluations.sh --argmax

# Custom number of episodes and processes
./run_evaluations.sh --episodes 200 --procs 32
```

## Available Scripts

### 1. `run_evaluations.sh` (Bash Wrapper)
The easiest way to run evaluations. Handles environment setup and runs the Python script.

**Usage:**
```bash
./run_evaluations.sh [OPTIONS]
```

**Environment Variables:**
- `STORAGE_DIR`: Override storage directory (default: `storage`)
- `PYTHON`: Override Python executable (default: `python3`)

### 2. `run_all_evaluations.py` (Main Script)
Comprehensive evaluation orchestrator that runs all evaluation tasks.

**Usage:**
```bash
python3 run_all_evaluations.py [OPTIONS]
```

**Key Options:**

#### Evaluation Options
- `--episodes N`: Number of evaluation episodes (default: 100)
- `--procs N`: Number of evaluation processes (default: 16)
- `--argmax`: Use greedy evaluation (argmax policy)
- `--skip-existing`: Skip models with existing eval logs

#### Task Selection
- `--eval-only`: Only run model evaluation
- `--plots-only`: Only generate plots
- `--reports-only`: Only generate reports
- `--no-plots`: Skip plot generation
- `--no-reports`: Skip report generation

#### Report Options
- `--catalog-prefix PREFIX`: Prefix for catalog runs (default: `catalog_`)
- `--catalog-output FILE`: Output file for catalog report (default: `catalog_evaluation.md`)
- `--report-dir DIR`: Output directory for reports (default: `evaluation_reports`)
- `--min-runs N`: Minimum runs per prefix group (default: 1)

#### Other Options
- `--storage DIR`: Storage directory path (default: `storage`)
- `--dry-run`: Show what would be done without executing

### 3. `scripts/run_storage_evaluations.py`
Discovers and evaluates all models in storage automatically.

**Features:**
- Automatically detects environment from training logs
- Detects if model used `--text` or `--memory` flags
- Supports parallel evaluation
- Can skip models with existing eval logs

**Usage:**
```bash
python3 scripts/run_storage_evaluations.py [OPTIONS]
```

### 4. `scripts/evaluate.py`
Evaluates a single model on a specific environment.

**Usage:**
```bash
python3 scripts/evaluate.py --model MODEL_NAME --env ENV_ID [OPTIONS]
```

**Example:**
```bash
python3 scripts/evaluate.py --model GTD --env BabyAI-GoToDoor-v0 --text --episodes 100
```

### 5. `plot_evaluation.py`
Generates evaluation plots for a model.

**Usage:**
```bash
python3 plot_evaluation.py --path storage/MODEL_NAME
```

### 6. `scripts/evaluate_catalog.py`
Generates comprehensive markdown reports for catalog runs.

**Usage:**
```bash
python3 scripts/evaluate_catalog.py --prefix catalog_ --output catalog_evaluation.md
```

### 7. `scripts/evaluate_all.py`
Discovers all run prefixes and generates comprehensive reports.

**Usage:**
```bash
python3 scripts/evaluate_all.py --output-dir evaluation_reports
```

## Common Workflows

### Workflow 1: Evaluate Everything
```bash
# Run all evaluations, plots, and reports
./run_evaluations.sh
```

### Workflow 2: Only Evaluate New Models
```bash
# Skip models that already have eval logs
./run_evaluations.sh --skip-existing
```

### Workflow 3: Re-generate Reports Only
```bash
# Skip evaluation and plots, only generate reports
./run_evaluations.sh --reports-only
```

### Workflow 4: Evaluate with High Quality
```bash
# Use more episodes and greedy evaluation
./run_evaluations.sh --episodes 200 --argmax --skip-existing
```

### Workflow 5: Generate Plots After Manual Evaluation
```bash
# Only generate plots (assumes evaluations already exist)
./run_evaluations.sh --plots-only
```

### Workflow 6: Custom Storage Location
```bash
# Use a different storage directory
STORAGE_DIR=/path/to/storage ./run_evaluations.sh
```

## Output Files

### Evaluation Logs
- Location: `storage/MODEL_NAME/eval_logs/logs.csv`
- Contains: Episode-by-episode return and frame count

### Evaluation Plots
- Location: `storage/MODEL_NAME/eval_plot.png`
- Contains: Distribution of returns and episode lengths

### Catalog Report
- Location: `catalog_evaluation.md`
- Contains: Summary of all catalog runs with prefix `catalog_`

### Comprehensive Reports
- Location: `evaluation_reports/`
- Contains:
  - `master_evaluation.md`: Master summary of all run groups
  - `{prefix}_evaluation.md`: Individual reports for each prefix group

## Improved Convergence Detection

The evaluation scripts now use a **robust multi-criteria convergence detection** algorithm that addresses the issues with the previous simple approach.

### How It Works

A model is considered **converged** only if ALL of these criteria are met:

1. **Minimum Training**: At least **50 updates** (prevents premature convergence detection)

2. **Low Variance**: Standard deviation < 0.08 over the last 25 updates
   - Ensures the model isn't oscillating wildly

3. **Flat Trend**: Linear regression slope < 0.002 per update
   - Ensures the model isn't still improving significantly

4. **Minimal Improvement**: Less than 5% improvement compared to earlier performance
   - Compares last 25 updates to the 25 updates before that
   - Ensures the model has truly plateaued

### Why This Is Better

**Old Approach (Broken):**
- Only checked last 5 updates
- Only checked if std < 0.05
- Could mark oscillating models as "converged"
- Example: `bench_1_1760115686` with returns oscillating 0.24 → 0.64 was incorrectly marked converged

**New Approach (Robust):**
- Requires minimum 50 updates
- Checks 25-update window
- Multiple criteria (variance + trend + improvement)
- Correctly identifies oscillating models as NOT converged
- Example: `bench_1_1760115686` now correctly marked as NOT converged

### Example Results

```
Model                          | Updates | Final | Status
-------------------------------|---------|-------|------------------
bench_1_1760115686            | 98      | 0.580 | ✗ NOT CONVERGED (correctly detected oscillation)
DoorKey                        | 49      | 0.063 | ✗ NOT CONVERGED (< 50 updates)
GTD                            | 557     | 0.902 | ✓ CONVERGED (stable, high performance)
PU                             | 5786    | 0.157 | ✓ CONVERGED (plateaued at low performance)
SynthLoc                       | 2442    | 0.219 | ✗ NOT CONVERGED (still oscillating)
```

## Troubleshooting

### Issue: "Storage directory not found"
**Solution:** Set the correct storage directory:
```bash
STORAGE_DIR=path/to/storage ./run_evaluations.sh
```

### Issue: "Environment not found in log.txt"
**Solution:** The model's training log might be missing or corrupted. Check:
```bash
cat storage/MODEL_NAME/log.txt
```

### Issue: Evaluation is too slow
**Solution:** Reduce processes or episodes:
```bash
./run_evaluations.sh --procs 8 --episodes 50
```

### Issue: Want to re-evaluate a specific model
**Solution:** Delete the eval logs and re-run:
```bash
rm -rf storage/MODEL_NAME/eval_logs
./run_evaluations.sh
```

## Tips

1. **Use `--skip-existing`** to save time when adding new models
2. **Use `--dry-run`** to preview what will be executed
3. **Use `--argmax`** for deterministic, greedy evaluation
4. **Check reports** in `evaluation_reports/` for insights
5. **Review plots** in each model's storage directory

## Examples

### Example 1: Daily Evaluation Workflow
```bash
# Morning: Run evaluations for new models only
./run_evaluations.sh --skip-existing --argmax

# Afternoon: Check the reports
cat catalog_evaluation.md
cat evaluation_reports/master_evaluation.md
```

### Example 2: Final Evaluation for Paper
```bash
# High-quality evaluation with many episodes
./run_evaluations.sh --episodes 500 --argmax --procs 32
```

### Example 3: Quick Check
```bash
# Fast evaluation with fewer episodes
./run_evaluations.sh --episodes 50 --procs 8 --skip-existing --no-reports
```

## Advanced Usage

### Running Individual Components

```bash
# 1. Only evaluate models
python3 scripts/run_storage_evaluations.py --skip-existing

# 2. Generate plots for a specific model
python3 plot_evaluation.py --path storage/MODEL_NAME

# 3. Generate catalog report
python3 scripts/evaluate_catalog.py

# 4. Generate comprehensive reports
python3 scripts/evaluate_all.py
```

### Custom Catalog Prefix

```bash
# Evaluate runs with different prefix
./run_evaluations.sh --catalog-prefix experiment_ --catalog-output experiment_eval.md
```

## Questions?

- Check existing evaluation logs: `storage/*/eval_logs/`
- Review training logs: `storage/*/log.txt`
- Check generated reports: `evaluation_reports/`
- Run with `--dry-run` to see what will execute
