#!/bin/bash

# Run main.py with common arguments
# Usage: ./scripts/run_main.sh [--query "your query"] [additional args]

# Set default parameters
MODEL_ID="gpt-4o-mini"
DATABASE="data/mimic_iii/mimic_iii.db"
MAX_ITERATIONS=5

# Run the main script
python -m src.main \
  --model_id $MODEL_ID \
  --database $DATABASE \
  --max_iterations $MAX_ITERATIONS \
  --log_to_file \
  "$@"

echo "Query processing completed."
