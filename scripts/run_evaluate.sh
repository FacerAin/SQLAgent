#!/bin/bash

# Combined evaluation script with flexible options for multiple models and agent types
# Usage: ./scripts/run_evaluate.sh [--skip-generation] [--dataset mimic|eicu] [additional args]

# Set default parameters
PREFIX="04-09"
DATASET_TYPE="mimic"  # Default to mimic
SKIP_GENERATION=false
NUM_SAMPLES=-1 # Use -1 for all samples
MAX_ITERATIONS=10
OUTPUT_PATH="results/${PREFIX}_{agent_type}_{model_id}_{dataset_name}.json"
PROMPT_PATH="src/prompts/react.yaml"

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
  DATABASE="data/mimic_iii/mimic_iii.db"
  echo "Using MIMIC dataset"
elif [[ "$DATASET_TYPE" == "eicu" ]]; then
  DATASET_PATH="data/evaluation/eicu_100.jsonl"
  DATABASE="data/eicu/eicu_iii.db"
  echo "Using eICU dataset"
else
  echo "Invalid dataset type: $DATASET_TYPE. Using MIMIC as default."
  DATASET_PATH="data/evaluation/mimic_100.jsonl"
  DATABASE="data/mimic_iii/mimic_iii.db"
fi

# Models to evaluate
MODEL_IDS=("gpt-4o" "gpt-4o-mini")

# Agent types to evaluate
AGENT_TYPES=("python_react" "sql_react")

# Outer loop for agent types
for AGENT_TYPE in "${AGENT_TYPES[@]}"; do
  echo "Processing agent type: $AGENT_TYPE"

  # Inner loop for model IDs
  for MODEL_ID in "${MODEL_IDS[@]}"; do
    # Extract dataset name from the dataset path for constructing file paths
    DATASET_NAME=$(basename $DATASET_PATH .jsonl)

    # Add dataset type and agent type to results path for better organization
    RESULTS_PATH="results/${AGENT_TYPE}_${MODEL_ID}_${DATASET_TYPE}_${DATASET_NAME}.json"

    echo "Processing model: $MODEL_ID with agent type: $AGENT_TYPE"

    # If SKIP_GENERATION is true and results file exists, skip generation
    if [[ "$SKIP_GENERATION" = true ]] && [[ -f "$RESULTS_PATH" ]]; then
      echo "Skipping generation for $MODEL_ID with $AGENT_TYPE, using existing results at: $RESULTS_PATH"

      # Run evaluation with --results_path to skip generation
      python -m src.evaluate \
        --model_id $MODEL_ID \
        --dataset_path $DATASET_PATH \
        --database $DATABASE \
        --num_samples $NUM_SAMPLES \
        --max_iterations $MAX_ITERATIONS \
        --output_path $OUTPUT_PATH \
        --agent_type $AGENT_TYPE \
        --prompt_path $PROMPT_PATH \
        --results_path "$RESULTS_PATH" \
        --save_result \
        --log_to_file \
        --use_few_shot \
        --verbose \
        "$@"
    else
      # If SKIP_GENERATION is false or results file does not exist, run generation
      if [[ "$SKIP_GENERATION" = true ]]; then
        echo "Results file not found for $MODEL_ID with $AGENT_TYPE, running full generation despite --skip-generation flag"
      else
        echo "Running full generation for $MODEL_ID with $AGENT_TYPE"
      fi

      # Run the full evaluation script
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
        --agent_verbose \
        "$@"
    fi

    echo "Evaluation completed for model: $MODEL_ID with agent type: $AGENT_TYPE"
  done
done

echo "All evaluations completed!"
