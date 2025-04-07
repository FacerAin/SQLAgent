import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from src.agent.base import BaseAgent
from src.chat.base import LLMClientInterface
from src.chat.factory import ChatModelFactory
from src.database.connector import BaseDatabaseConnector, SqliteDatabaseConnector
from src.evaluation.judge import (
    exact_match,
    normalize_exact_match,
    verify_answer_by_llm,
)
from src.tool.base import FinalAnswerTool, PythonTool, SQLTool
from src.utils.load import load_dataset_from_jsonl
from src.utils.logger import init_logger


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate QA with SQL performance.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4o-mini",
        help="The model ID to use for the client.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="sql_react",
        help="Type of agent to use for evaluation.",
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
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompts/react.yaml",
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--judge_model_id",
        type=str,
        default="gpt-4o-mini",
        help="The model ID to use for the judge.",
    )

    parser.add_argument(
        "--custom_time",
        type=str,
        default="2105-12-31 23:59:00",
        help="Custom time to use for the database connection. If you don't want to use it, set it to 'None'.",
    )

    return parser.parse_args()


class EvaluationContext:
    """Context manager to handle logger setup and resources for evaluation."""

    def __init__(self, args: Any) -> None:
        self.args: Any = args
        self.logger: Optional[logging.Logger] = None
        self.db_connector: Optional[BaseDatabaseConnector] = None
        self.datasets: Optional[List] = None
        self.agent: Optional[BaseAgent] = None
        self.client: Optional[LLMClientInterface] = None
        self.judge_client: Optional[LLMClientInterface] = None

    def __enter__(self) -> "EvaluationContext":
        """Set up loggers and resources when entering the context."""
        # Initialize main logger
        self.logger = init_logger(
            name="evaluate",
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
        )

        assert self.logger is not None, "Logger initialization failed."  # For mypy

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

        if self.args.custom_time == "None":
            self.args.custom_time = None

        # Initialize database connector
        self.db_connector = SqliteDatabaseConnector(
            self.args.database,
            custom_time=self.args.custom_time if self.args.custom_time else None,
        )
        self.db_connector.connect()

        # Initialize model client
        self.client = ChatModelFactory.load_model(model_id=self.args.model_id)
        self.judge_client = ChatModelFactory.load_model(
            model_id=self.args.judge_model_id
        )
        sql_tool = SQLTool(db_connector=self.db_connector)
        python_sql_tool = PythonTool(db_connector=self.db_connector)
        final_answer_tool = FinalAnswerTool()
        # Initialize the agent with the SQL tool and final answer tool
        # TODO: Refactor this to use the agent factory
        if self.args.agent_type == "sql_react":
            tools = [sql_tool, final_answer_tool]
        elif self.args.agent_type == "python_react":
            tools = [python_sql_tool, final_answer_tool]
        elif self.args.agent_type == "python_sql_react":
            tools = [python_sql_tool, sql_tool, final_answer_tool]

        with open(self.args.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_templates = yaml.safe_load(f)
        self.agent = BaseAgent(
            client=self.client,
            tools=tools,
            max_steps=10,
            prompt_templates=self.prompt_templates,
        )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close resources when exiting the context."""
        if self.db_connector:
            self.db_connector.close()


def process_samples(context: EvaluationContext) -> List[Dict[str, Any]]:
    """Process each sample and return evaluation results."""
    evaluate_results = []

    for sample in tqdm(context.datasets, desc="Evaluating samples", unit="sample"):
        question = sample.get("question")
        answer = sample.get("expected_answer", sample.get("answer"))
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset.")
        if not answer:
            raise ValueError("Answer is missing in the dataset.")

        # Process the question
        agent_answer = context.agent.run(task=question)
        total_turns = context.agent.step_num
        history = context.agent.write_memory_to_messages()[1:]

        # Add sample-specific evaluation metrics
        sample_metrics = {
            "exact_match": exact_match(pred=agent_answer, ans=answer),
            "normalized_match": normalize_exact_match(pred=agent_answer, ans=answer),
            "llm_match": verify_answer_by_llm(
                pred=agent_answer,
                ans=answer,
                question=question,
                client=context.judge_client,
            ),
        }

        # Store the evaluation results with per-sample metrics
        sample_result = {
            "id": sample.get("id"),
            "answer_type": sample.get("answer_type", ""),
            "table_num": sample.get("table_num", ""),
            "question": question,
            "expected_answer": answer,
            "generated_answer": agent_answer,
            # "sql_query": result["query"],
            "gold_sql_query": gold_sql_query,
            "total_turns": total_turns,
            "history": history,
            "sample_metrics": sample_metrics,  # Add the per-sample metrics here
        }

        evaluate_results.append(sample_result)

    return evaluate_results


def calculate_metrics(
    evaluate_results: List[Dict[str, Any]], context: EvaluationContext
) -> Dict[str, Any]:
    """Calculate aggregate evaluation metrics from individual sample metrics."""
    evaluation_stats = {
        "total_num": len(evaluate_results),
        "correct": 0,
        "unfinished": 0,
        "incorrect": 0,
        "sql_equality": 0,
        "llm_correct": 0,
        "sql_executable": 0,
        "norm_correct": 0,
        "metadata": {
            "agent_type": context.args.agent_type,
            "model_id": context.args.model_id,
            "dataset_path": context.args.dataset_path,
        },
        "per_sample_summary": [],  # Add a summary of per-sample performance
    }

    # Aggregate metrics based on individual sample metrics
    for sample in tqdm(evaluate_results, desc="Aggregating metrics", unit="sample"):
        sample_metrics = sample.get("sample_metrics", {})

        # Add to aggregate counts
        if sample_metrics.get("exact_match", False):
            evaluation_stats["correct"] += 1
        elif sample["generated_answer"] == "None":
            evaluation_stats["unfinished"] += 1
        else:
            evaluation_stats["incorrect"] += 1

        if sample_metrics.get("normalized_match", False):
            evaluation_stats["norm_correct"] += 1

        if sample_metrics.get("llm_match", False):
            evaluation_stats["llm_correct"] += 1

        # Add a simplified summary for this sample
        evaluation_stats["per_sample_summary"].append(
            {
                "id": sample.get("id", "unknown"),
                "exact_match": sample_metrics.get("exact_match", False),
                "normalized_match": sample_metrics.get("normalized_match", False),
                "llm_match": sample_metrics.get("llm_match", False),
            }
        )

    # Calculate success rates
    if evaluation_stats["total_num"] > 0:
        evaluation_stats["exact_match_rate"] = (
            evaluation_stats["correct"] / evaluation_stats["total_num"]
        )
        evaluation_stats["normalized_match_rate"] = (
            evaluation_stats["norm_correct"] / evaluation_stats["total_num"]
        )
        evaluation_stats["llm_match_rate"] = (
            evaluation_stats["llm_correct"] / evaluation_stats["total_num"]
        )

    return evaluation_stats


def log_evaluation_results(
    logger: logging.Logger, args: argparse.Namespace, stats: Dict[str, Any]
) -> None:
    """Log evaluation results."""
    if args.verbose:
        logger.info("Evaluation Results:")
        # Log aggregate metrics
        for key, value in stats.items():
            if key not in ["metadata", "per_sample_summary"]:  # Skip detailed data
                logger.info(f"{key}: {value}")


def save_results(
    logger: logging.Logger,
    args: argparse.Namespace,
    evaluate_results: List[Dict[str, Any]],
    evaluation_stats: Dict[str, Any] = None,
) -> None:
    """Save results to a file if requested."""
    if args.save_result:
        # Format the output path to include model and sample count
        output_path = args.output_path
        if "{model_id}" in output_path:
            output_path = output_path.replace("{model_id}", args.model_id)
        if "{num_samples}" in output_path:
            output_path = output_path.replace("{num_samples}", str(args.num_samples))
        if "{dataset_name}" in output_path:
            dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
            output_path = output_path.replace("{dataset_name}", dataset_name)
        if "{agent_type}" in output_path:
            output_path = output_path.replace("{agent_type}", args.agent_type)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")

        logger.info(f"Saving evaluation results to {output_path}")

        final_results = {
            "evaluation_history": evaluate_results,  # This now includes per-sample metrics
        }

        # Add aggregate metrics if they were calculated
        if evaluation_stats:
            final_results["metrics"] = evaluation_stats

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


def add_sample_metrics_to_loaded_results(
    evaluate_results: List[Dict[str, Any]], context: EvaluationContext
) -> List[Dict[str, Any]]:
    """Add sample metrics to results loaded from a file if they don't have them."""
    updated_results = []

    for sample in tqdm(evaluate_results, desc="Adding sample metrics", unit="sample"):
        # Skip if sample already has metrics
        if "sample_metrics" in sample:
            updated_results.append(sample)
            continue

        # Add sample-specific metrics
        question = sample.get("question")
        generated_answer = sample.get("generated_answer")
        expected_answer = sample.get("expected_answer")

        sample_metrics = {
            "exact_match": exact_match(pred=generated_answer, ans=expected_answer),
            "normalized_match": normalize_exact_match(
                pred=generated_answer, ans=expected_answer
            ),
            "llm_match": verify_answer_by_llm(
                pred=generated_answer,
                ans=expected_answer,
                question=question,
                client=context.judge_client,
            ),
        }

        # Add metrics to sample
        sample["sample_metrics"] = sample_metrics
        updated_results.append(sample)

    return updated_results


def main() -> None:
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

    # Generate new results or add metrics to loaded results using the context manager
    with EvaluationContext(args) as context:
        # Process all samples for new results
        if not evaluate_results:
            logger.info("Processing samples for evaluation")
            evaluate_results = process_samples(context)
        else:
            # Add sample metrics to loaded results if they don't have them
            logger.info("Adding sample metrics to loaded results")
            evaluate_results = add_sample_metrics_to_loaded_results(
                evaluate_results, context
            )

        # Calculate aggregate metrics if not skipped
        if not args.skip_metrics and evaluate_results:
            logger.info("Calculating aggregate evaluation metrics")
            evaluation_stats = calculate_metrics(evaluate_results, context)

            # Log results
            log_evaluation_results(logger, args, evaluation_stats)

            # Save results with both per-sample and aggregate metrics
            if args.save_result:
                save_results(logger, args, evaluate_results, evaluation_stats)
        elif args.save_result:
            # Save raw results without aggregate metrics
            save_results(logger, args, evaluate_results)
            if args.skip_metrics:
                logger.info("Skipping aggregate metrics calculation as requested")


if __name__ == "__main__":
    main()
