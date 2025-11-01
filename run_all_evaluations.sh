#!/bin/bash

STORAGE_DIR="storage"

for model_dir in "$STORAGE_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "Processing: $model_name"

        log_file="$model_dir/log.txt"
        if [ -f "$log_file" ]; then
            # Extract the environment name from the log file
            env_name=$(grep -o 'env [^ ]*' "$log_file" | awk '{print $2}' | head -n 1)

            if [ -n "$env_name" ]; then
                echo "  Environment: $env_name"
                echo "  Running evaluation..."
                python3 scripts/evaluate.py --model "$model_name" --env "$env_name"

                echo "  Generating evaluation plot..."
                python3 plot_evaluation.py --path "$model_dir"
            else
                echo "  Could not find environment name in log.txt"
            fi
        else
            echo "  log.txt not found"
        fi
        echo ""
    fi
done
