import argparse
import json
import logging
import os
from typing import Any, Dict, List

from tqdm import tqdm

from src.agent.react import SQLReActAgent
from src.chat.factory import ChatModelFactory
from src.database.connector import SqliteDatabaseConnector
from src.utils.load import load_dataset_from_jsonl
from src.utils.logger import init_logger


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
        normalized_answer = normalize_string(answer)
        if normalized_answer in normalized_pred:
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
        "--verbose", action="store_true", help="Enable verbose evaluation logging."
    )
    parser.add_argument(
        "--agent_verbose", action="store_true", help="Enable verbose agent logging."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/{model_id}_{num_samples}.json",
        help="Path to save evaluation results. Supports formatting with {model_id} and {num_samples}.",
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
    parser.add_argument(
        "--log_to_file", action="store_true", help="Enable logging to file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to store log files.",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip calculation of metrics, only generate raw results.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to existing evaluation results to calculate metrics from.",
    )

    return parser.parse_args()


class EvaluationContext:
    """Context manager to handle logger setup and resources for evaluation."""

    def __init__(self, args):
        self.args = args
        self.logger = None
        self.db_connector = None
        self.datasets = None
        self.agent = None

    def __enter__(self):
        """Set up loggers and resources when entering the context."""
        # Initialize main logger
        self.logger = init_logger(
            name="evaluate",
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
        )

        # Configure agent logger
        agent_logger = logging.getLogger("agent")
        if self.args.agent_verbose:
            agent_logger.setLevel(logging.INFO)
        else:
            agent_logger.setLevel(logging.ERROR)  # Only show errors from agent

        # Load dataset
        self.datasets = load_dataset_from_jsonl(self.args.dataset_path)
        if not self.datasets:
            raise ValueError("The dataset is empty or not found.")

        # Select samples to evaluate
        if self.args.num_samples >= 0:
            self.datasets = self.datasets[: self.args.num_samples]

        # Initialize database connector
        self.db_connector = SqliteDatabaseConnector(self.args.database)
        self.db_connector.connect()

        # Initialize model client
        client = ChatModelFactory.load_model(model_id=self.args.model_id)

        # Initialize agent
        self.agent = SQLReActAgent(
            db_connector=self.db_connector,
            model_id=self.args.model_id,
            client=client,
            prompt_file_path="src/prompts/react.yaml",
            prompt_key="prompt",
            max_iterations=self.args.max_iterations,
            verbose=self.args.agent_verbose,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close resources when exiting the context."""
        if self.db_connector:
            self.db_connector.close()


def process_samples(context: EvaluationContext) -> List[Dict[str, Any]]:
    """Process each sample and return evaluation results."""
    evaluate_results = []

    for sample in tqdm(context.datasets, desc="Evaluating samples", unit="sample"):
        question = sample.get("question")
        answer = sample.get("answer")
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset.")
        if not answer:
            raise ValueError("Answer is missing in the dataset.")

        # Process the question
        result = context.agent.process(question, use_few_shot=context.args.use_few_shot)

        # Log the results if verbose
        if context.args.verbose:
            log_sample_results(context.logger, question, answer, result)

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


def log_sample_results(
    logger: logging.Logger,
    question: str,
    expected_answer: List[str],
    result: Dict[str, Any],
):
    """Log results for a single sample."""
    logger.info(f"Question: {question}")
    logger.info(f"Expected Answer: {expected_answer}")
    logger.info(f"Generated Answer: {result['answer']}")
    logger.info(f"SQL Query: {result['query']}")
    logger.info(f"Total Turns: {result['turns']}")
    logger.info("--------------------------------------------------")


def calculate_metrics(evaluate_results: List[Dict[str, Any]]) -> Dict[str, int]:
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


def log_evaluation_results(
    logger: logging.Logger, args: argparse.Namespace, stats: Dict[str, int]
):
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


def save_results(
    logger: logging.Logger,
    args: argparse.Namespace,
    evaluate_results: List[Dict[str, Any]],
    evaluation_stats: Dict[str, int] = None,
):
    """Save results to a file if requested."""
    if args.save_result:
        # Format the output path to include model and sample count
        output_path = args.output_path
        if "{model_id}" in output_path:
            output_path = output_path.replace("{model_id}", args.model_id)
        if "{num_samples}" in output_path:
            output_path = output_path.replace("{num_samples}", str(args.num_samples))

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")

        logger.info(f"Saving evaluation results to {output_path}")

        final_results = {
            "evaluation_history": evaluate_results,
        }

        # Add metrics if they were calculated
        if evaluation_stats:
            final_results["metric"] = evaluation_stats

        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=4)
            logger.info(f"Results saved to {output_path}")


def load_evaluate_results(results_path: str) -> List[Dict[str, Any]]:
    """Load previously saved evaluation results."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, "r") as f:
        data = json.load(f)

    if "evaluation_history" not in data:
        raise ValueError(
            "Invalid results file format: 'evaluation_history' key missing"
        )

    return data["evaluation_history"]


def main():
    """Main function to coordinate the evaluation process."""
    # Parse arguments
    args = parse_arguments()

    # Initialize the logger outside the context for use with loaded results
    logger = init_logger(
        name="evaluate",
        log_to_file=args.log_to_file,
        log_dir=args.log_dir,
    )

    evaluate_results = None

    # Check if we're loading existing results or generating new ones
    if args.results_path:
        # Load existing results
        logger.info(f"Loading evaluation results from {args.results_path}")
        evaluate_results = load_evaluate_results(args.results_path)
        logger.info(f"Loaded {len(evaluate_results)} evaluation results")
    else:
        # Generate new results using the context manager
        with EvaluationContext(args) as context:
            # Process all samples
            evaluate_results = process_samples(context)

            # Save raw results if requested and skipping metrics
            if args.skip_metrics and args.save_result:
                save_results(context.logger, args, evaluate_results)
                logger.info("Skipping metrics calculation as requested")

    # Calculate metrics if not skipped
    if not args.skip_metrics and evaluate_results:
        logger.info("Calculating evaluation metrics")
        evaluation_stats = calculate_metrics(evaluate_results)

        # Log results
        log_evaluation_results(logger, args, evaluation_stats)

        # Save complete results with metrics if requested and not already saved
        if args.save_result and not (args.results_path is None and args.skip_metrics):
            save_results(logger, args, evaluate_results, evaluation_stats)


if __name__ == "__main__":
    main()
