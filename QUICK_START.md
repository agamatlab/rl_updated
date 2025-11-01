# Quick Start Guide

## TL;DR

```bash
# 1. Analyze training results
python3 analyze_training_simple.py

# 2. Update frontend
./update_frontend_classifications.sh

# 3. View results
cd front/my-react-app
npm start
```

## What You Get

### Classification Tags
Every environment is automatically tagged with:
- **Convergence**: `yes(early)`, `yes(late)`, or `no`
- **Reshaped Return**: `high` or `low`
- **Eval Return**: `high` or `low` (if evaluated)

### Frontend Features
- Search environments by name
- Filter by convergence status
- Filter by performance metrics
- Color-coded visual tags
- View training curves (PDFs)

## Common Tasks

### Find Converged Environments
```bash
# In frontend: Set "Convergence" filter to "Converged (any)"
# Or check the JSON file:
cat environment_classifications.json | grep '"convergence": "yes'
```

### Find High-Performing Environments
```bash
# In frontend:
# - Convergence: Converged (any)
# - Reshaped Return: High
# - Eval Return: High
```

### Update After New Training
```bash
./update_frontend_classifications.sh
```

### Adjust Classification Thresholds
```bash
python3 analyze_training_simple.py \
  --convergence-threshold 0.03 \
  --high-reshaped-return 0.8 \
  --high-eval-return 0.95
```

## File Locations

- **Classifications**: `environment_classifications.json`
- **Analysis Script**: `analyze_training_simple.py`
- **Update Script**: `update_frontend_classifications.sh`
- **Frontend**: `front/my-react-app/`
- **Documentation**:
  - `ANALYSIS_README.md` - Analysis details
  - `FRONTEND_README.md` - Frontend details
  - `IMPLEMENTATION_SUMMARY.md` - Complete overview

## Output Files

### environment_classifications.json
```json
{
  "metadata": { ... },
  "environments": {
    "DoorKey": {
      "convergence": "yes(early)",
      "reshaped_return": "low",
      "eval_return": "high",
      ...
    }
  },
  "tags": {
    "DoorKey": "convergence:yes(early),reshaped_return:low,eval_return:high"
  }
}
```

### environment_report.txt
Human-readable summary with:
- Overall statistics
- Convergence breakdown
- Detailed per-environment info

## Workflow for Next Two Weeks

### Phase 1: Training
```bash
# Run training with --text for all environments
python train.py --env ENV_NAME --text --frames LARGE_NUMBER

# Analyze results
python3 analyze_training_simple.py
```

### Phase 2: Review & Select
```bash
# View in frontend
cd front/my-react-app && npm start

# Filter to converged environments
# (Use convergence filter in UI)
```

### Phase 3: Evaluation
```bash
# Run evaluation on converged environments
python eval.py --env ENV_NAME

# Re-analyze to get eval tags
./update_frontend_classifications.sh
```

### Phase 4: Final Classification
```bash
# Review all tags in frontend
# Export or document final classifications
```

## Tag Color Guide

| Tag | Color | Meaning |
|-----|-------|---------|
| `yes(early)` | Dark Green | Converged quickly |
| `yes(late)` | Yellow/Orange | Converged slowly |
| `no` | Red | Did not converge |
| `R:high` | Blue | Good training return |
| `R:low` | Pink | Poor training return |
| `E:high` | Purple | Good evaluation return |
| `E:low` | Orange | Poor evaluation return |

## Troubleshooting

**"No module named pandas"**
→ Use `analyze_training_simple.py` instead of `analyze_training_results.py`

**Tags not showing in frontend**
→ Run `./update_frontend_classifications.sh`

**Build errors**
→ Run `cd front/my-react-app && npm install`

**Can't find environment**
→ Use search bar in frontend

## Support

Read the detailed documentation:
- `ANALYSIS_README.md` - For analysis questions
- `FRONTEND_README.md` - For frontend questions
- `IMPLEMENTATION_SUMMARY.md` - For architecture overview
