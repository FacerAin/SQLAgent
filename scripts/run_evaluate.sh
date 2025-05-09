#!/bin/bash

# Combined evaluation script with flexible options for multiple models and agent types
# Usage: ./scripts/run_evaluate.sh [--skip-generation] [--dataset mimic|eicu] [additional args]

# Set default parameters
PREFIX="llm_table_temporal_v1"
DATASET_TYPE="mimic"  # Default to mimic
SKIP_GENERATION=false
NUM_SAMPLES=-1 # Use -1 for all samples
MAX_ITERATIONS=20
# PROMPT_PATH="src/prompts/baseline_no_description.yaml"
PROMPT_PATH="src/prompts/baseline_mimic_verifier_time_desc.yaml"
PLANNING_INTERVAL=0
JUDGE_MODEL_ID="gpt-4.1-2025-04-14"

# Models to evaluate
# MODEL_IDS=("gpt-4.1-mini-2025-04-14" "gpt-4.1-nano-2025-04-14")
MODEL_IDS=("gpt-4.1-mini-2025-04-14" "gpt-4.1-nano-2025-04-14")

# Agent types to evaluate
AGENT_TYPES=("llm_table_verifier")
# AGENT_TYPES=("python_react")

# Parse our custom arguments first
for i in "$@"; do
  case $i in
    --skip-generation)
      SKIP_GENERATION=true
      shift # Remove this argument from the processing
      ;;
    --dataset=*)
      DATASET_TYPE="${i#*=}"
      shift # Remove this argument from the processing
      ;;
    --dataset)
      DATASET_TYPE="$2"
      shift # Remove this argument from the processing
      shift # Remove the value from the processing
      ;;
  esac
done

# Set paths based on dataset type
if [[ "$DATASET_TYPE" == "mimic" ]]; then
  DATASET_PATH="data/evaluation/mimic_100.jsonl"
  # DATASET_PATH="data/legacy/test_50.jsonl"
  DATABASE="data/mimic_iii/mimic_iii.db"
  echo "Using MIMIC dataset"
elif [[ "$DATASET_TYPE" == "eicu" ]]; then
  DATASET_PATH="data/evaluation/eicu_100.jsonl"
  DATABASE="data/eicu/eicu.db"
  # DATASET_PATH="data/legacy/eicu_test_50.jsonl"
  echo "Using eICU dataset"
else
  echo "Invalid dataset type: $DATASET_TYPE. Using MIMIC as default."
  DATASET_PATH="data/evaluation/mimic_100.jsonl"
  DATABASE="data/mimic_iii/mimic_iii.db"
fi



# Outer loop for agent types
for AGENT_TYPE in "${AGENT_TYPES[@]}"; do
  echo "Processing agent type: $AGENT_TYPE"

  # Inner loop for model IDs
  for MODEL_ID in "${MODEL_IDS[@]}"; do
    # Extract dataset name from the dataset path for constructing file paths
    DATASET_NAME=$(basename $DATASET_PATH .jsonl)

    # Add dataset type and agent type to results path for better organization
    OUTPUT_PATH="results/${PREFIX}_${AGENT_TYPE}_${MODEL_ID}_${DATASET_TYPE}_${DATASET_NAME}.json"

    echo "Processing model: $MODEL_ID with agent type: $AGENT_TYPE"

    # Create base command with common parameters
    BASE_CMD="python -m src.evaluate \
      --model_id $MODEL_ID \
      --dataset_path $DATASET_PATH \
      --database $DATABASE \
      --num_samples $NUM_SAMPLES \
      --max_iterations $MAX_ITERATIONS \
      --output_path $OUTPUT_PATH \
      --agent_type $AGENT_TYPE \
      --prompt_path $PROMPT_PATH \
      --planning_interval $PLANNING_INTERVAL \
      --judge_model_id $JUDGE_MODEL_ID \
      --save_result \
      --log_to_file \
      --use_few_shot \
      --verbose"

    # If SKIP_GENERATION is true and results file exists, use continue_from flag
    if [[ "$SKIP_GENERATION" = true ]] && [[ -f "$OUTPUT_PATH" ]]; then
      echo "Skipping generation, continuing from existing results at: $OUTPUT_PATH"

      # Run with skip_generation and continue_from flags
      $BASE_CMD \
        --skip_generation \
        --continue_from "$OUTPUT_PATH" \
        "$@"
    else
      # If SKIP_GENERATION is false or results file does not exist, run generation
      if [[ "$SKIP_GENERATION" = true ]]; then
        echo "Results file not found, running full generation despite --skip-generation flag"
      else
        echo "Running full generation"
      fi

      # Run the full evaluation
      $BASE_CMD \
        --output_path "$OUTPUT_PATH" \
        "$@"
    fi

    echo "Evaluation completed for model: $MODEL_ID with agent type: $AGENT_TYPE"
  done
done

echo "All evaluations completed!"
