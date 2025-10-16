#!/bin/bash
# Generate training_curves.pdf and eval_return.pdf for all models in storage
# This script generates plots for each trained model using the correct storage path

STORAGE_DIR="storage"

echo "==============================================="
echo "Generating plots for all models in $STORAGE_DIR"
echo "==============================================="
echo ""

# Counter for statistics
total_models=0
training_plots=0
eval_plots=0
skipped=0

for model_dir in "$STORAGE_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "Model name is: $model_name"

        # Skip hidden directories and system files
        if [[ "$model_name" == .* ]] || [[ "$model_name" == ".DS_Store" ]]; then
            continue
        fi

        total_models=$((total_models + 1))
        echo "[$total_models] Processing: $model_dir/log.csv"

        # Generate training curves if log.csv exists
        if [ -f "$model_dir/log.csv" ]; then
            echo "    → Generating training_curves.pdf..."
            if python3 plot_training.py --path "$model_dir" 2>&1 | grep -q "Saved"; then
                echo "    ✓ Training curves saved"
                training_plots=$((training_plots + 1))
            else
                echo "    ✗ Failed to generate training curves"
            fi
        else
            echo "    ⊗ Skipping training plot (no log.csv found)"
            skipped=$((skipped + 1))
        fi

        # Generate evaluation plot if eval_logs/logs.csv exists
        if [ -f "$model_dir/eval_logs/logs.csv" ]; then
            echo "    → Generating eval_return.pdf..."
            if python3 plot_evaluation.py --path "$model_dir" 2>&1 | grep -q "Saved"; then
                echo "    ✓ Evaluation plot saved"
                eval_plots=$((eval_plots + 1))
            else
                echo "    ✗ Failed to generate evaluation plot"
            fi
        else
            echo "    ⊗ Skipping eval plot (no eval_logs/logs.csv found)"
        fi

        echo ""
    fi
done

echo "==============================================="
echo "Summary"
echo "==============================================="
echo "Total models processed:    $total_models"
echo "Training curves generated: $training_plots"
echo "Evaluation plots generated: $eval_plots"
echo "Models without logs:       $skipped"
echo ""
echo "Output locations:"
echo "  - Training curves: storage/<model_name>/training_curves.pdf"
echo "  - Evaluation plots: storage/<model_name>/eval_logs/eval_return.pdf"
echo "==============================================="