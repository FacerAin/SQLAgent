import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.agent.base import BaseAgent
from src.agent.react import ToolReActAgent
from src.chat.base import LLMClientInterface
from src.chat.factory import ChatModelFactory
from src.database.connector import BaseDatabaseConnector, SqliteDatabaseConnector
from src.tool.base import (
    CurrentDateTool,
    FinalAnswerTool,
    OracleTableVerifierTool,
    PythonTool,
    SQLTool,
)
from src.utils.load import load_dataset_from_jsonl
from src.utils.logger import init_logger


class EvaluationContext:
    """Context manager to handle logger setup and resources for evaluation."""

    SUPPORTED_AGENT_TYPES = {
        "sql_react": ["SQLTool", "FinalAnswerTool"],
        "python_react": ["PythonTool", "FinalAnswerTool"],
        "python_sql_react": [
            "PythonTool",
            "SQLTool",
            "FinalAnswerTool",
        ],
        "oracle_table_verifier": [
            "SQLTool",
            "OracleTableVerifierTool",
            "FinalAnswerTool",
        ],
        "schema_free_sql_react": [
            "SQLTool",
            "FinalAnswerTool",
        ],
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
            "CurrentDateTool": CurrentDateTool(),
            "OracleTableVerifierTool": OracleTableVerifierTool(),
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
            prompt_templates=self.prompt_templates,
            logger=agent_logger,
            log_to_file=self.args.log_to_file,
            log_dir=self.args.log_dir,
            max_steps=self.args.max_iterations,
            planning_interval=self.args.planning_interval,
        )


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
        cls, evaluate_results: List[Dict[str, Any]], context: EvaluationContext
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
