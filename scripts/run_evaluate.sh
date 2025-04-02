#!/bin/bash

# Run evaluate.py with common arguments
# Usage: ./scripts/run_evaluate.sh [additional args]

# Set default parameters
MODEL_ID="gpt-4o"
DATASET_PATH="data/sample_preprocessed.jsonl"
DATABASE="data/mimic_iii/mimic_iii.db"
NUM_SAMPLES=3
MAX_ITERATIONS=5
OUTPUT_PATH="results/{model_id}_{num_samples}.json"

# Run the evaluation script
python -m src.evaluate \
  --model_id $MODEL_ID \
  --dataset_path $DATASET_PATH \
  --database $DATABASE \
  --num_samples $NUM_SAMPLES \
  --max_iterations $MAX_ITERATIONS \
  --output_path $OUTPUT_PATH \
  --save_result \
  --log_to_file \
  --use_few_shot \
  --verbose \
  "$@"

echo "Evaluation completed."
