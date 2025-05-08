import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import yaml

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
from src.tool.base import (
    CurrentDateTool,
    FinalAnswerTool,
    OracleTableVerifierTool,
    PythonTool,
    SQLTool,
)
from src.utils.load import load_dataset_from_jsonl
from src.utils.logger import init_logger


@dataclass
class ModelConfig:
    """Model-related configuration"""

    model_id: str = "gpt-4o-mini"
    judge_model_id: Optional[str] = None


@dataclass
class DataConfig:
    """Data source configuration"""

    dataset_path: str = "data/valid_preprocessed.jsonl"
    database: str = "data/mimic_iii/mimic_iii.db"
    custom_time: str = "2105-12-31 23:59:00"
    num_samples: int = 3


@dataclass
class AgentConfig:
    """Agent configuration"""

    agent_type: str = "sql_react"
    max_iterations: int = 5
    use_few_shot: bool = False
    prompt_path: str = "src/prompts/react.yaml"
    planning_interval: int = 20


@dataclass
class OutputConfig:
    """Output configuration"""

    output_path: str = "results/{model_id}_{agent_type}_{num_samples}.json"
    save_result: bool = False
    continue_from: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Processing configuration"""

    skip_metrics: bool = False
    skip_generation: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""

    verbose: bool = False
    log_to_file: bool = False
    log_dir: str = "logs"


@dataclass
class EvaluationConfig:
    """Integrated evaluation configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Properties for frequently accessed attributes
    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def judge_model_id(self) -> str:
        if self.model.judge_model_id is None:
            return self.model.model_id
        return self.model.judge_model_id

    @property
    def skip_metrics(self) -> bool:
        return self.processing.skip_metrics

    @property
    def skip_generation(self) -> bool:
        return self.processing.skip_generation

    @property
    def save_result(self) -> bool:
        return self.output.save_result

    @property
    def log_to_file(self) -> bool:
        return self.logging.log_to_file

    @property
    def log_dir(self) -> str:
        return self.logging.log_dir

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        """Create EvaluationConfig from argparse Namespace"""
        # Convert argparse results to EvaluationConfig
        config = cls(
            model=ModelConfig(
                model_id=args.model_id,
                judge_model_id=args.judge_model_id,
            ),
            data=DataConfig(
                dataset_path=args.dataset_path,
                database=args.database,
                custom_time=args.custom_time,
                num_samples=args.num_samples,
            ),
            agent=AgentConfig(
                agent_type=args.agent_type,
                max_iterations=args.max_iterations,
                use_few_shot=args.use_few_shot,
                prompt_path=args.prompt_path,
                planning_interval=args.planning_interval,
            ),
            output=OutputConfig(
                output_path=args.output_path,
                save_result=args.save_result,
                continue_from=args.continue_from,
            ),
            processing=ProcessingConfig(
                skip_metrics=args.skip_metrics,
                skip_generation=args.skip_generation,
            ),
            logging=LoggingConfig(
                verbose=args.verbose,
                log_to_file=args.log_to_file,
                log_dir=args.log_dir,
            ),
        )

        # Use continue_from as output_path if set
        if config.output.continue_from:
            config.output.output_path = config.output.continue_from

        return config


@dataclass
class EvaluationStats:
    """Evaluation statistics"""

    # Counts
    total_num: int = 0
    correct: int = 0
    unfinished: int = 0
    incorrect: int = 0
    norm_correct: int = 0
    llm_correct: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    per_sample_summary: List[Dict[str, Any]] = field(default_factory=list)

    # Derived metrics
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
        """Convert statistics to dictionary"""
        return {
            "total_num": self.total_num,
            "correct": self.correct,
            "unfinished": self.unfinished,
            "incorrect": self.incorrect,
            "norm_correct": self.norm_correct,
            "llm_correct": self.llm_correct,
            "exact_match_rate": self.exact_match_rate,
            "normalized_match_rate": self.normalized_match_rate,
            "llm_match_rate": self.llm_match_rate,
            "metadata": self.metadata,
            "per_sample_summary": self.per_sample_summary,
        }


class MetricsCalculator:
    """Class to separate metrics calculation logic"""

    def calculate_sample_metrics(
        self,
        agent_answer: str,
        expected_answer: str,
        question: str,
        judge_client: LLMClientInterface,
    ) -> Dict[str, bool]:
        """Calculate metrics for a single sample"""
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

    def calculate_overall_stats(
        self, evaluate_results: List[Dict[str, Any]], config: EvaluationConfig
    ) -> EvaluationStats:
        """Calculate statistics for overall results"""
        stats = EvaluationStats(
            total_num=len(evaluate_results),
            metadata={
                "agent_type": config.agent.agent_type,
                "model_id": config.model.model_id,
                "dataset_path": config.data.dataset_path,
            },
        )

        # Calculate aggregate metrics from individual sample metrics
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

            # Add summary for this sample
            stats.per_sample_summary.append(
                {
                    "id": sample.get("id", "unknown"),
                    "exact_match": sample_metrics.get("exact_match", False),
                    "normalized_match": sample_metrics.get("normalized_match", False),
                    "llm_match": sample_metrics.get("llm_match", False),
                }
            )

        return stats


class EvaluationContext:
    """Context for managing evaluation resources"""

    # Supported agent types
    SUPPORTED_AGENT_TYPES: ClassVar[Dict[str, List[str]]] = {
        "sql_react": ["SQLTool", "FinalAnswerTool"],
        "python_react": ["PythonTool", "FinalAnswerTool"],
        "python_sql_react": ["PythonTool", "SQLTool", "FinalAnswerTool"],
        "oracle_table_verifier": [
            "SQLTool",
            "OracleTableVerifierTool",
            "FinalAnswerTool",
        ],
        "schema_free_sql_react": ["SQLTool", "FinalAnswerTool"],
    }

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize"""
        self.config = config
        self.logger: Optional[logging.Logger] = None
        self.db_connector: Optional[BaseDatabaseConnector] = None
        self.datasets: Optional[List[Dict[str, Any]]] = None
        self.agent: Optional[BaseAgent] = None
        self.client: Optional[LLMClientInterface] = None
        self.judge_client: Optional[LLMClientInterface] = None
        self.prompt_templates: Optional[Dict[str, Any]] = None

    def __enter__(self) -> "EvaluationContext":
        """Set up resources when entering context"""
        # Initialize main logger
        self.logger = init_logger(
            name="evaluate",
            log_to_file=self.config.log_to_file,
            log_dir=self.config.log_dir,
        )

        assert self.logger is not None, "Logger initialization failed"

        # Configure agent logger
        agent_logger = init_logger(
            name="agent",
            log_to_file=self.config.log_to_file,
            log_dir=self.config.log_dir,
        )

        # Load dataset
        self.datasets = load_dataset_from_jsonl(self.config.data.dataset_path)
        if not self.datasets:
            raise ValueError("Dataset is empty or not found")

        # Select samples to evaluate
        if self.config.data.num_samples >= 0:
            self.datasets = self.datasets[: self.config.data.num_samples]

        # Initialize database connector
        custom_time = None
        if self.config.data.custom_time and self.config.data.custom_time != "None":
            custom_time = self.config.data.custom_time

        self.db_connector = SqliteDatabaseConnector(
            self.config.data.database,
            custom_time=custom_time,
        )
        self.db_connector.connect()

        # Initialize model client
        self.client = ChatModelFactory.load_model(model_id=self.config.model_id)
        self.judge_client = ChatModelFactory.load_model(
            model_id=self.config.judge_model_id
        )

        # Initialize agent with appropriate tools
        self._setup_agent(agent_logger)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release resources when exiting context"""
        if self.db_connector:
            self.db_connector.close()

    def _setup_agent(self, agent_logger: logging.Logger) -> None:
        """Set up agent with appropriate tools"""
        # Create tool instances
        tool_instances = {
            "SQLTool": SQLTool(db_connector=self.db_connector),
            "PythonTool": PythonTool(db_connector=self.db_connector),
            "FinalAnswerTool": FinalAnswerTool(),
            "CurrentDateTool": CurrentDateTool(),
            "OracleTableVerifierTool": OracleTableVerifierTool(),
        }

        # Select tools based on agent type
        if self.config.agent.agent_type not in self.SUPPORTED_AGENT_TYPES:
            raise ValueError(f"Unsupported agent type: {self.config.agent.agent_type}")

        tools = [
            tool_instances[tool_name]
            for tool_name in self.SUPPORTED_AGENT_TYPES[self.config.agent.agent_type]
        ]

        # Load prompt templates
        prompt_path = Path(self.config.agent.prompt_path)
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
            log_to_file=self.config.log_to_file,
            log_dir=self.config.log_dir,
            max_steps=self.config.agent.max_iterations,
            planning_interval=self.config.agent.planning_interval,
        )


@dataclass
class EvaluationResult:
    """Container for evaluation results"""

    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[EvaluationStats] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        result = {
            "evaluation_history": self.evaluation_history,
            "metadata": self.metadata,
        }

        if self.metrics:
            result["metrics"] = self.metrics.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create result object from dictionary"""
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
                "norm_correct",
                "llm_correct",
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
        """Save result to file"""
        import json

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "EvaluationResult":
        """Load result from file"""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


class ResultManager:
    """Result loading, saving, and management"""

    def __init__(self, config: EvaluationConfig, logger: logging.Logger):
        """Initialize"""
        self.config = config
        self.logger = logger
        self.output_path = config.output.output_path
        self.save_enabled = config.save_result

        # Create result directory if needed
        if self.save_enabled and self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def load_existing_results(self) -> Tuple[List[Dict[str, Any]], Set[Any]]:
        """Load existing results from output path"""
        if not self.output_path or not os.path.exists(self.output_path):
            return [], set()

        try:
            self.logger.info(f"Loading existing results from: {self.output_path}")
            result = EvaluationResult.load(self.output_path)
            evaluate_results = result.evaluation_history
            self.logger.info(f"Loaded {len(evaluate_results)} evaluation results")

            # Extract existing sample IDs
            existing_sample_ids = {
                sample.get("id") for sample in evaluate_results if sample.get("id")
            }
            return evaluate_results, existing_sample_ids
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return [], set()

    def save_sample_result(self, sample_result: Dict[str, Any]) -> None:
        """Save a single sample result incrementally"""
        if not self.save_enabled or not self.output_path:
            return

        # Check if existing results exist
        if os.path.exists(self.output_path):
            try:
                result = EvaluationResult.load(self.output_path)

                # Check if ID already exists
                current_id = sample_result.get("id")
                if current_id:
                    # Update existing or append new
                    for i, s in enumerate(result.evaluation_history):
                        if s.get("id") == current_id:
                            result.evaluation_history[i] = sample_result
                            break
                    else:
                        result.evaluation_history.append(sample_result)
                else:
                    # Append if no ID
                    result.evaluation_history.append(sample_result)
            except Exception as e:
                self.logger.error(f"Error loading existing results: {e}")
                # Create new result object on error
                result = EvaluationResult(evaluation_history=[sample_result])
        else:
            # Create new result object if file doesn't exist
            result = EvaluationResult(evaluation_history=[sample_result])

        # Save results
        result.save(self.output_path)

    def save_final_results(
        self,
        evaluate_results: List[Dict[str, Any]],
        evaluation_stats: Optional[EvaluationStats] = None,
    ) -> None:
        """Save final results and statistics"""
        if not self.save_enabled or not self.output_path:
            return

        self.logger.info(f"Saving evaluation results to: {self.output_path}")

        # Create structured result object
        result = EvaluationResult(
            evaluation_history=evaluate_results,
            metrics=evaluation_stats,
            metadata={
                "model_id": self.config.model_id,
                "agent_type": self.config.agent.agent_type,
                "max_steps": self.config.agent.max_iterations,
                "judge_model_id": self.config.judge_model_id,
                "database": self.config.data.database,
                "dataset_path": self.config.data.dataset_path,
                "num_samples": self.config.data.num_samples,
                "timestamp": datetime.now().isoformat(),
                "prompt_path": self.config.agent.prompt_path,
            },
        )

        # Save results
        result.save(self.output_path)
        self.logger.info(f"Results saved to: {self.output_path}")
