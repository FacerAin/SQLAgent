import argparse
from dataclasses import dataclass, field
from typing import Optional


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
