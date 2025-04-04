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

        # Initialize database connector
        self.db_connector = SqliteDatabaseConnector(self.args.database)
        self.db_connector.connect()

        # Initialize model client
        self.client = ChatModelFactory.load_model(model_id=self.args.model_id)
        self.judge_client = ChatModelFactory.load_model(
            model_id=self.args.judge_model_id
        )

        # # Initialize agent
        # self.agent = AgentFactory.load_agent(
        #     agent_type=self.args.agent_type,
        #     db_connector=self.db_connector,
        #     model_id=self.args.model_id,
        #     client=self.client,
        #     prompt_file_path=self.args.prompt_path,  # Set default prompt file path
        #     prompt_key="prompt",
        #     max_iterations=self.args.max_iterations,
        #     verbose=self.args.agent_verbose,
        # )
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
        answer = sample.get("answer")
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset.")
        if not answer:
            raise ValueError("Answer is missing in the dataset.")

        # Process the question
        agent_answer = context.agent.run(task=question)
        total_turns = context.agent.step_num
        history = context.agent.write_memory_to_messages()[1:]

        # Store the evaluation results
        evaluate_results.append(
            {
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
            }
        )

    return evaluate_results


def calculate_metrics(
    evaluate_results: List[Dict[str, Any]], context: EvaluationContext
) -> Dict[str, int]:
    """Calculate evaluation metrics."""
    evaluation_stats = {
        "total_num": 0,
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
    }

    for sample in tqdm(evaluate_results, desc="Evaluating metrics", unit="sample"):
        evaluation_stats["total_num"] += 1
        if exact_match(pred=sample["generated_answer"], ans=sample["expected_answer"]):
            evaluation_stats["correct"] += 1
        elif sample["generated_answer"] == "None":
            evaluation_stats["unfinished"] += 1
        else:
            evaluation_stats["incorrect"] += 1

        # Check SQL equality
        # if verify_sql_query_equivalent(
        #     pred_sql_query=sample["sql_query"],
        #     gold_sql_query=sample["gold_sql_query"],
        #     db_connector=context.db_connector,
        # ):
        #     evaluation_stats["sql_equality"] += 1

        # if verify_sql_query_executable(
        #     sql_query=sample["sql_query"],
        #     db_connector=context.db_connector,
        # ):
        #     evaluation_stats["sql_executable"] += 1

        if normalize_exact_match(
            pred=sample["generated_answer"],
            ans=sample["expected_answer"],
        ):
            evaluation_stats["norm_correct"] += 1

        if verify_answer_by_llm(
            pred=sample["generated_answer"],
            ans=sample["expected_answer"],
            question=sample["question"],
            client=context.judge_client,
        ):
            evaluation_stats["llm_correct"] += 1

    return evaluation_stats


def log_evaluation_results(
    logger: logging.Logger, args: argparse.Namespace, stats: Dict[str, int]
) -> None:
    """Log evaluation results."""
    if args.verbose:
        logger.info("Evaluation Results:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        if stats["total_num"] > 0:
            accuracy = stats["correct"] / stats["total_num"]
            logger.info(f"Accuracy: {accuracy:.2%}")


def save_results(
    logger: logging.Logger,
    args: argparse.Namespace,
    evaluate_results: List[Dict[str, Any]],
    evaluation_stats: Dict[str, int] = None,
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
        # Generate new results using the context manager
    with EvaluationContext(args) as context:
        # Process all samples
        if not evaluate_results:
            logger.info("Processing samples for evaluation")
            evaluate_results = process_samples(context)

        # Save raw results if requested and skipping metrics
        if args.skip_metrics and args.save_result:
            save_results(context.logger, args, evaluate_results)
            logger.info("Skipping metrics calculation as requested")

        # Calculate metrics if not skipped
        if not args.skip_metrics and evaluate_results:
            logger.info("Calculating evaluation metrics")
            evaluation_stats = calculate_metrics(evaluate_results, context)

            # Log results
            log_evaluation_results(logger, args, evaluation_stats)

            # Save complete results with metrics if requested and not already saved
            if args.save_result and not (
                args.results_path is None and args.skip_metrics
            ):
                save_results(logger, args, evaluate_results, evaluation_stats)


if __name__ == "__main__":
    main()
