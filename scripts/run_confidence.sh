#!/bin/bash

# List of files to analyze
FILES=(
    "results/baseline_sql_react_gpt-4.1-mini-2025-04-14_mimic_mimic_100.json"
    "results/oracle_table_temporal_v1_oracle_table_verifier_gpt-4.1-mini-2025-04-14_mimic_mimic_100.json"
    "results/temporal_v1_sql_react_gpt-4.1-mini-2025-04-14_mimic_mimic_100.json"
)

# Results directory
RESULTS_DIR="analysis_results"
mkdir -p $RESULTS_DIR

# Model to use
MODEL="gpt-4.1"

echo "Starting analysis..."

# Process all files
for FILE in "${FILES[@]}"; do
    FILENAME=$(basename "$FILE" .json)
    OUTPUT_FILE="$RESULTS_DIR/${FILENAME}_post_evaluate.json"

    echo "Processing file: $FILE"
    python -m  src.post_evaluate --input_file "$FILE" --output_file "$OUTPUT_FILE" --model "$MODEL"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to process $FILE"
    else
        echo "Completed: $FILE -> $OUTPUT_FILE"
    fi
done

echo "All files processed!"
echo "Results are saved in the $RESULTS_DIR directory."
