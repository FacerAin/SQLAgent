import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from src.agent.base import BaseAgent
from src.agent.react import ToolReActAgent
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


@dataclass
class EvaluationStats:
    """Data model for evaluation statistics."""

    # Counts
    total_num: int = 0
    correct: int = 0
    unfinished: int = 0
    incorrect: int = 0
    sql_equality: int = 0
    llm_correct: int = 0
    sql_executable: int = 0
    norm_correct: int = 0

    # Metadata
    metadata: Dict[str, str] = field(default_factory=dict)

    # Per-sample performance summary
    per_sample_summary: List[Dict[str, Any]] = field(default_factory=list)

    # Derived metrics (calculated on demand)
    @property
    def exact_match_rate(self) -> float:
        return self.correct / self.total_num if self.total_num > 0 else 0.0

    @property
    def normalized_match_rate(self) -> float:
        return self.norm_correct / self.total_num if self.total_num > 0 else 0.0

    @property
    def llm_match_rate(self) -> float:
        return self.llm_correct / self.total_num if self.total_num > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Base counts
            "total_num": self.total_num,
            "correct": self.correct,
            "unfinished": self.unfinished,
            "incorrect": self.incorrect,
            "sql_equality": self.sql_equality,
            "llm_correct": self.llm_correct,
            "sql_executable": self.sql_executable,
            "norm_correct": self.norm_correct,
            # Calculated rates
            "exact_match_rate": self.exact_match_rate,
            "normalized_match_rate": self.normalized_match_rate,
            "llm_match_rate": self.llm_match_rate,
            # Other data
            "metadata": self.metadata,
            "per_sample_summary": self.per_sample_summary,
        }

    @classmethod
    def from_results(
        cls, evaluate_results: List[Dict[str, Any]], context: "EvaluationContext"
    ) -> "EvaluationStats":
        stats = cls(
            total_num=len(evaluate_results),
            metadata={
                "agent_type": context.args.agent_type,
                "model_id": context.args.model_id,
                "dataset_path": context.args.dataset_path,
            },
        )

        # Aggregate metrics based on individual sample metrics
        for sample in evaluate_results:
            sample_metrics = sample.get("sample_metrics", {})

            # Update aggregate counts
            if sample_metrics.get("exact_match", False):
                stats.correct += 1
            elif sample["generated_answer"] == "None":
                stats.unfinished += 1
            else:
                stats.incorrect += 1

            if sample_metrics.get("normalized_match", False):
                stats.norm_correct += 1

            if sample_metrics.get("llm_match", False):
                stats.llm_correct += 1

            # Add a simplified summary for this sample
            stats.per_sample_summary.append(
                {
                    "id": sample.get("id", "unknown"),
                    "exact_match": sample_metrics.get("exact_match", False),
                    "normalized_match": sample_metrics.get("normalized_match", False),
                    "llm_match": sample_metrics.get("llm_match", False),
                }
            )

        return stats


@dataclass
class EvaluationResult:
    """Container for evaluation results and metrics."""

    # Raw evaluation history
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Calculated metrics (optional)
    metrics: Optional[EvaluationStats] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "evaluation_history": self.evaluation_history,
            "metadata": self.metadata,
        }

        if self.metrics:
            result["metrics"] = self.metrics.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        result = cls(
            evaluation_history=data.get("evaluation_history", []),
            metadata=data.get("metadata", {}),
        )

        # Convert metrics dict to EvaluationStats if present
        if "metrics" in data:
            stats = EvaluationStats()
            metrics_dict = data["metrics"]

            # Copy basic metrics
            for key in [
                "total_num",
                "correct",
                "unfinished",
                "incorrect",
                "sql_equality",
                "llm_correct",
                "sql_executable",
                "norm_correct",
            ]:
                if key in metrics_dict:
                    setattr(stats, key, metrics_dict[key])

            # Copy metadata and sample summary if present
            if "metadata" in metrics_dict:
                stats.metadata = metrics_dict["metadata"]
            if "per_sample_summary" in metrics_dict:
                stats.per_sample_summary = metrics_dict["per_sample_summary"]

            result.metrics = stats

        return result

    def save(self, filepath: str) -> None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "EvaluationResult":
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


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

    SUPPORTED_AGENT_TYPES = {
        "sql_react": ["SQLTool", "FinalAnswerTool"],
        "python_react": ["PythonTool", "FinalAnswerTool"],
        "python_sql_react": ["PythonTool", "SQLTool", "FinalAnswerTool"],
    }

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger: logging.Logger = None
        self.db_connector: BaseDatabaseConnector = None
        self.datasets: list[dict[str, Any]] = None
        self.agent: BaseAgent = None
        self.client: LLMClientInterface = None
        self.judge_client: LLMClientInterface = None
        self.prompt_templates = None

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
        agent_logger = init_logger(
            name="agent",
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
        )
        if self.args.agent_verbose:
            agent_logger.setLevel(logging.DEBUG)
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
        custom_time = None
        if self.args.custom_time and self.args.custom_time != "None":
            custom_time = self.args.custom_time

        self.db_connector = SqliteDatabaseConnector(
            self.args.database,
            custom_time=custom_time,
        )
        self.db_connector.connect()

        # Initialize model client
        self.client = ChatModelFactory.load_model(model_id=self.args.model_id)
        self.judge_client = ChatModelFactory.load_model(
            model_id=self.args.judge_model_id
        )

        # Initialize agent with appropriate tools
        self._setup_agent(agent_logger)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close resources when exiting the context."""
        if self.db_connector:
            self.db_connector.close()

    def _setup_agent(self, agent_logger: logging.Logger) -> None:
        """Set up the agent with appropriate tools."""
        # Create tool instances
        tool_instances = {
            "SQLTool": SQLTool(db_connector=self.db_connector),
            "PythonTool": PythonTool(db_connector=self.db_connector),
            "FinalAnswerTool": FinalAnswerTool(),
        }

        # Select tools based on agent type
        if self.args.agent_type not in self.SUPPORTED_AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {self.args.agent_type}")

        tools = [
            tool_instances[tool_name]
            for tool_name in self.SUPPORTED_AGENT_TYPES[self.args.agent_type]
        ]

        # Load prompt templates
        prompt_path = Path(self.args.prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_templates = yaml.safe_load(f)

        # Create agent instance
        self.agent = ToolReActAgent(
            client=self.client,
            tools=tools,
            max_steps=10,
            prompt_templates=self.prompt_templates,
            logger=agent_logger,
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
        )


class SampleProcessor:
    """Encapsulates functionality related to sample processing."""

    @staticmethod
    def process_samples(context: EvaluationContext) -> List[Dict[str, Any]]:
        """Process each sample and return evaluation results."""
        evaluate_results = []

        for sample in tqdm(context.datasets, desc="Evaluating samples", unit="sample"):
            sample_result = SampleProcessor._process_single_sample(sample, context)
            evaluate_results.append(sample_result)

        return evaluate_results

    @staticmethod
    def _process_single_sample(
        sample: Dict[str, Any], context: EvaluationContext
    ) -> Dict[str, Any]:
        """Process and evaluate a single sample."""
        question = sample.get("question")
        answer = sample.get("expected_answer", sample.get("answer"))
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset.")
        if not answer:
            answer = "Unanswerable"  # To handle cases for unanswerable questions

        # Process the question
        agent_answer = context.agent.run(task=question)
        total_turns = context.agent.step_num
        history = context.agent.write_memory_to_messages()[1:]

        # Add sample-specific evaluation metrics
        sample_metrics = SampleProcessor._calculate_sample_metrics(
            agent_answer=agent_answer,
            expected_answer=answer,
            question=question,
            judge_client=context.judge_client,
        )

        # Store the evaluation results with per-sample metrics
        return {
            "id": sample.get("id"),
            "answer_type": sample.get("answer_type", ""),
            "table_num": sample.get("table_num", ""),
            "question": question,
            "expected_answer": answer,
            "generated_answer": agent_answer,
            "gold_sql_query": gold_sql_query,
            "total_turns": total_turns,
            "history": history,
            "sample_metrics": sample_metrics,
        }

    @staticmethod
    def _calculate_sample_metrics(
        agent_answer: str,
        expected_answer: str,
        question: str,
        judge_client: LLMClientInterface,
    ) -> Dict[str, bool]:
        """Calculate evaluation metrics for a single sample."""
        return {
            "exact_match": exact_match(pred=agent_answer, ans=expected_answer),
            "normalized_match": normalize_exact_match(
                pred=agent_answer, ans=expected_answer
            ),
            "llm_match": verify_answer_by_llm(
                pred=agent_answer,
                ans=expected_answer,
                question=question,
                client=judge_client,
            ),
        }

    @staticmethod
    def add_metrics_to_loaded_results(
        evaluate_results: List[Dict[str, Any]], context: EvaluationContext
    ) -> List[Dict[str, Any]]:
        """Add sample metrics to results loaded from a file if they don't have them."""
        updated_results = []

        for sample in tqdm(
            evaluate_results, desc="Adding sample metrics", unit="sample"
        ):
            # Skip if sample already has metrics
            if "sample_metrics" in sample:
                updated_results.append(sample)
                continue

            # Add sample-specific metrics
            question = sample.get("question")
            generated_answer = sample.get("generated_answer")
            expected_answer = sample.get("expected_answer")

            sample_metrics = SampleProcessor._calculate_sample_metrics(
                agent_answer=generated_answer,
                expected_answer=expected_answer,
                question=question,
                judge_client=context.judge_client,
            )

            # Add metrics to sample
            sample["sample_metrics"] = sample_metrics
            updated_results.append(sample)

        return updated_results


class ResultHandler:
    """Encapsulates functionality related to result processing and storage."""

    @staticmethod
    def calculate_metrics(
        evaluate_results: List[Dict[str, Any]], context: EvaluationContext
    ) -> EvaluationStats:
        """Calculate aggregate evaluation metrics from individual sample metrics."""
        return EvaluationStats.from_results(evaluate_results, context)

    @staticmethod
    def log_evaluation_results(
        logger: logging.Logger, args: argparse.Namespace, stats: EvaluationStats
    ) -> None:
        """Log evaluation results."""
        if args.verbose:
            logger.info("Evaluation Results:")
            # Log aggregate metrics
            for key, value in stats.to_dict().items():
                if key not in ["metadata", "per_sample_summary"]:  # Skip detailed data
                    logger.info(f"{key}: {value}")

    @staticmethod
    def save_results(
        logger: logging.Logger,
        args: argparse.Namespace,
        evaluate_results: List[Dict[str, Any]],
        evaluation_stats: Optional[EvaluationStats] = None,
    ) -> None:
        """Save results to a file if requested."""
        if not args.save_result:
            return

        # Format the output path to include model and sample count
        output_path = ResultHandler._format_output_path(args)
        logger.info(f"Saving evaluation results to {output_path}")

        # Create a structured result object
        result = EvaluationResult(
            evaluation_history=evaluate_results,
            metrics=evaluation_stats,
            metadata={
                "model_id": args.model_id,
                "agent_type": args.agent_type,
                "dataset_path": args.dataset_path,
                "num_samples": args.num_samples,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Save the results
        result.save(output_path)
        logger.info(f"Results saved to {output_path}")

    @staticmethod
    def _format_output_path(args: argparse.Namespace) -> str:
        """Format the output path."""
        output_path = args.output_path
        replacements = {
            "{model_id}": args.model_id,
            "{num_samples}": str(args.num_samples),
            "{agent_type}": args.agent_type,
        }

        # Add dataset name if needed
        if "{dataset_name}" in output_path:
            dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
            replacements["{dataset_name}"] = dataset_name

        # Apply all replacements
        for placeholder, value in replacements.items():
            if placeholder in output_path:
                output_path = output_path.replace(placeholder, value)

        return str(output_path)

    @staticmethod
    def load_evaluate_results(results_path: str) -> List[Dict[str, Any]]:
        """Load previously saved evaluation results."""
        result = EvaluationResult.load(results_path)
        return result.evaluation_history


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
        evaluate_results = ResultHandler.load_evaluate_results(args.results_path)
        logger.info(f"Loaded {len(evaluate_results)} evaluation results")

    # Generate new results or add metrics to loaded results using the context manager
    with EvaluationContext(args) as context:
        # Process all samples for new results
        if not evaluate_results:
            logger.info("Processing samples for evaluation")
            evaluate_results = SampleProcessor.process_samples(context)
        else:
            # Add sample metrics to loaded results if they don't have them
            logger.info("Adding sample metrics to loaded results")
            evaluate_results = SampleProcessor.add_metrics_to_loaded_results(
                evaluate_results, context
            )

        # Calculate aggregate metrics if not skipped
        if not args.skip_metrics and evaluate_results:
            logger.info("Calculating aggregate evaluation metrics")
            evaluation_stats = ResultHandler.calculate_metrics(
                evaluate_results, context
            )

            # Log results
            ResultHandler.log_evaluation_results(logger, args, evaluation_stats)

            # Save results with both per-sample and aggregate metrics
            if args.save_result:
                ResultHandler.save_results(
                    logger, args, evaluate_results, evaluation_stats
                )
        elif args.save_result:
            # Save raw results without aggregate metrics
            ResultHandler.save_results(logger, args, evaluate_results)
            if args.skip_metrics:
                logger.info("Skipping aggregate metrics calculation as requested")


if __name__ == "__main__":
    main()
