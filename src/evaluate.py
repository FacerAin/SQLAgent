import argparse
import json
from typing import List

from tqdm import tqdm

from src.agent.react import SQLReActAgent
from src.chat.openai import OpenAIClient
from src.database.connector import SqliteDatabaseConnector
from src.utils.load import load_dataset_from_jsonl
from src.utils.logger import init_logger

logger = init_logger()


def judge(pred: str, ans: List[str]) -> bool:
    """
    Judge if the prediction string matches any of the answers in the list.

    Args:
        pred (str): The prediction string
        ans (List[str]): The list of possible correct answers

    Returns:
        bool: True if prediction matches any answer, False otherwise
    """
    # Early return if the answer list is empty
    if not ans:
        return False

    # Define normalization function
    def normalize_string(s: str) -> str:
        return s.replace("True", "1").replace("False", "0")

    # Boolean mapping for normalization
    boolean_mapping = {
        "False": "0",
        "false": "0",
        "True": "1",
        "true": "1",
        "No": "0",
        "no": "0",
        "Yes": "1",
        "yes": "1",
        "None": "0",
        "none": "0",
    }

    # Normalize the prediction
    normalized_pred = normalize_string(pred)

    # Check each answer against the prediction
    for answer in ans:
        # Direct string match check
        if answer in pred:
            return True

        # Normalize the answer
        normalized_ans = answer
        if answer in boolean_mapping:
            normalized_ans = boolean_mapping[answer]

        # Handle decimal numbers in answer
        if normalized_ans.endswith(".0"):
            normalized_ans = normalized_ans[:-2]

        # Handle multiple comma-separated items within a single answer
        ans_items = [normalized_ans]
        if ", " in normalized_ans:
            ans_items = normalized_ans.split(", ")

        # Check with normalized values
        if any(item in normalized_pred for item in ans_items):
            return True

    return False


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate QA with SQL performance.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4o-mini",
        help="The model ID to use for the client.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/valid_preprocessed.jsonl",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/mimic_iii/mimic_iii.db",
        help="Path to the database file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to evaluate. If -1, evaluate all samples.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/evaluate_results.json",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Save the evaluation results to a file.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of reasoning iterations for the agent",
    )

    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use few-shot examples for the agent.",
    )

    return parser.parse_args()


def setup_resources(args):
    """Set up and return necessary resources for evaluation."""
    # Load dataset
    datasets = load_dataset_from_jsonl(args.dataset_path)
    if not datasets:
        raise ValueError("The dataset is empty or not found.")

    # Select samples to evaluate
    if args.num_samples >= 0:
        sampled_datasets = datasets[: args.num_samples]
    else:
        sampled_datasets = datasets

    # Initialize database connector
    db_connector = SqliteDatabaseConnector(args.database)
    db_connector.connect()

    # Initialize model client
    client = OpenAIClient(model_id=args.model_id)

    # Initialize agent
    agent = SQLReActAgent(
        db_connector=db_connector,
        model_id=args.model_id,
        client=client,
        prompt_file_path="src/prompts/react.yaml",
        prompt_key="prompt",
        max_iterations=args.max_iterations,
        verbose=True,
    )

    return sampled_datasets, db_connector, agent


def process_samples(args, datasets, agent):
    """Process each sample and return evaluation results."""
    evaluate_results = []

    for sample in tqdm(datasets, desc="Evaluating samples", unit="sample"):
        question = sample.get("question")
        answer = sample.get("answer")
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset.")
        if not answer:
            raise ValueError("Answer is missing in the dataset.")

        # Process the question
        result = agent.process(question, use_few_shot=args.use_few_shot)

        # Log the results if verbose
        if args.verbose:
            log_sample_results(question, answer, result)

        # Store the evaluation results
        evaluate_results.append(
            {
                "question": question,
                "expected_answer": answer,
                "generated_answer": result["answer"],
                "sql_query": result["query"],
                "gold_sql_query": gold_sql_query,
                "total_turns": result["turns"],
                "history": result["history"],
                "system_prompt": result["system_prompt"],
                "context": result["context"],
            }
        )

    return evaluate_results


def log_sample_results(question, expected_answer, result):
    """Log results for a single sample."""
    logger.info(f"Question: {question}")
    logger.info(f"Expected Answer: {expected_answer}")
    logger.info(f"Generated Answer: {result['answer']}")
    logger.info(f"SQL Query: {result['query']}")
    logger.info(f"Total Turns: {result['turns']}")
    logger.info("--------------------------------------------------")


def calculate_metrics(evaluate_results):
    """Calculate evaluation metrics."""
    evaluation_stats = {
        "total_num": 0,
        "correct": 0,
        "unfinished": 0,
        "incorrect": 0,
    }

    for sample in tqdm(evaluate_results, desc="Evaluating metrics", unit="sample"):
        evaluation_stats["total_num"] += 1
        if judge(pred=sample["generated_answer"], ans=sample["expected_answer"]):
            evaluation_stats["correct"] += 1
        elif sample["generated_answer"] == "None":
            evaluation_stats["unfinished"] += 1
        else:
            evaluation_stats["incorrect"] += 1

    return evaluation_stats


def log_evaluation_results(args, stats):
    """Log evaluation results."""
    if args.verbose:
        logger.info("Evaluation Results:")
        logger.info(f"Total Samples: {stats['total_num']}")
        logger.info(f"Correct Answers: {stats['correct']}")
        logger.info(f"Incorrect Answers: {stats['incorrect']}")
        logger.info(f"Unfinished Answers: {stats['unfinished']}")
        if stats["total_num"] > 0:
            accuracy = stats["correct"] / stats["total_num"]
            logger.info(f"Accuracy: {accuracy:.2%}")


def save_results(args, evaluate_results, evaluation_stats):
    """Save results to a file if requested."""
    if args.save_result:
        logger.info(f"Saving evaluation results to {args.output_path}")
        final_results = {
            "evaluation_history": evaluate_results,
            "metric": evaluation_stats,
        }
        with open(args.output_path, "w") as f:
            json.dump(final_results, f, indent=4)


def main():
    """Main function to coordinate the evaluation process."""
    # Parse arguments
    args = parse_arguments()

    # Setup resources
    sampled_datasets, db_connector, agent = setup_resources(args)

    try:
        # Process all samples
        evaluate_results = process_samples(args, sampled_datasets, agent)

        # Calculate metrics
        evaluation_stats = calculate_metrics(evaluate_results)

        # Log results
        log_evaluation_results(args, evaluation_stats)

        # Save results
        save_results(args, evaluate_results, evaluation_stats)

    finally:
        # Always close the database connection
        db_connector.close()


if __name__ == "__main__":
    main()
