#!/bin/bash

# Run evaluate.py with common arguments
# Usage: ./scripts/run_evaluate.sh [additional args]

# Set default parameters
DATASET_PATH="data/test_50.jsonl"
DATABASE="data/mimic_iii/mimic_iii.db"
NUM_SAMPLES=-1 # Use -1 for all samples
MAX_ITERATIONS=10
OUTPUT_PATH="results/{agent_type}_{model_id}_{dataset_name}.json"
AGENT_TYPE="sql_react"
PROMPT_PATH="src/prompts/react.yaml"

MODEL_IDS=("gpt-4o-mini")

for MODEL_ID in "${MODEL_IDS[@]}"; do
  # Run the evaluation script
  python -m src.evaluate \
    --model_id $MODEL_ID \
    --dataset_path $DATASET_PATH \
    --database $DATABASE \
    --num_samples $NUM_SAMPLES \
    --max_iterations $MAX_ITERATIONS \
    --output_path $OUTPUT_PATH \
    --agent_type $AGENT_TYPE \
    --prompt_path $PROMPT_PATH \
    --save_result \
    --log_to_file \
    --use_few_shot \
    --verbose \
    "$@"

  echo "Evaluation completed for model: $MODEL_ID"
done
