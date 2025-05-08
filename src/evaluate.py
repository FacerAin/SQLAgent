import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from src.context import context_sample
from src.evaluation.base import (
    EvaluationConfig,
    EvaluationContext,
    EvaluationResult,
    EvaluationStats,
    MetricsCalculator,
    ResultManager,
)
from src.utils.logger import init_logger


class SampleProcessor:
    """Encapsulation of sample processing logic"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_calculator = MetricsCalculator()

    def process_samples(
        self,
        context: EvaluationContext,
        result_manager: Optional[ResultManager] = None,
    ) -> List[Dict[str, Any]]:
        """Process each sample and return evaluation results"""
        evaluate_results = []

        for sample in tqdm(context.datasets, desc="Evaluating samples", unit="sample"):
            context_sample.set(sample)  # Set current sample in context
            sample_result = self._process_single_sample(sample, context)
            evaluate_results.append(sample_result)

            # Save results after each sample if result_manager provided
            if result_manager and result_manager.save_enabled:
                result_manager.save_sample_result(sample_result)

        return evaluate_results

    def process_missing_samples(
        self,
        context: EvaluationContext,
        existing_results: List[Dict[str, Any]],
        existing_sample_ids: Set[Any],
        result_manager: Optional[ResultManager] = None,
    ) -> List[Dict[str, Any]]:
        """Process only missing samples and merge with existing results"""
        all_results = existing_results.copy()

        # Identify samples to process
        samples_to_process = []
        for sample in context.datasets:
            sample_id = sample.get("id")
            # Process if ID is missing in existing results or sample has no ID
            if (sample_id and sample_id not in existing_sample_ids) or not sample_id:
                samples_to_process.append(sample)

        if not samples_to_process:
            self.logger.info("No missing samples found, all samples already processed")
            return all_results

        self.logger.info(f"Processing {len(samples_to_process)} missing samples")

        # Process missing samples
        for sample in tqdm(
            samples_to_process, desc="Evaluating missing samples", unit="sample"
        ):
            context_sample.set(sample)  # Set current sample in context
            sample_result = self._process_single_sample(sample, context)
            all_results.append(sample_result)

            # Save incrementally if enabled
            if result_manager and result_manager.save_enabled:
                result_manager.save_sample_result(sample_result)

        return all_results

    def add_metrics_to_loaded_results(
        self,
        evaluate_results: List[Dict[str, Any]],
        context: EvaluationContext,
    ) -> List[Dict[str, Any]]:
        """Add sample metrics to results loaded from a file if they don't have them"""
        updated_results = []
        samples_needing_metrics = [
            sample for sample in evaluate_results if not sample.get("sample_metrics")
        ]

        if not samples_needing_metrics:
            return evaluate_results

        self.logger.info(f"Adding metrics to {len(samples_needing_metrics)} samples")

        for sample in tqdm(
            evaluate_results, desc="Adding sample metrics", unit="sample"
        ):
            # Skip if sample already has metrics
            if sample.get("sample_metrics"):
                updated_results.append(sample)
                continue

            # Add sample-specific metrics
            question = sample.get("question")
            generated_answer = sample.get("generated_answer", "")
            expected_answer = str(sample.get("expected_answer", ""))

            sample_metrics = self.metrics_calculator.calculate_sample_metrics(
                agent_answer=generated_answer,
                expected_answer=expected_answer,
                question=question,
                judge_client=context.judge_client,
            )

            # Add metrics to sample
            sample["sample_metrics"] = sample_metrics
            updated_results.append(sample)

        return updated_results

    def _process_single_sample(
        self, sample: Dict[str, Any], context: EvaluationContext
    ) -> Dict[str, Any]:
        """Process and evaluate a single sample"""
        question = sample.get("question")
        answer = str(sample.get("expected_answer", sample.get("answer", "")))
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset")
        if answer == "None":
            answer = "Unanswerable"  # Handle unanswerable questions

        # Process the question
        agent_answer = context.agent.run(task=question)
        total_turns = context.agent.step_num
        history = context.agent.write_memory_to_messages()[1:]
        token_usages = context.agent.get_token_usages()[-1]

        # Add sample-specific evaluation metrics
        sample_metrics = self.metrics_calculator.calculate_sample_metrics(
            agent_answer=agent_answer,
            expected_answer=answer,
            question=question,
            judge_client=context.judge_client,
        )

        # Return the evaluation results with per-sample metrics
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
            "token_usages": token_usages,
        }


class EvaluationService:
    """Service class for orchestrating the full evaluation process"""

    def __init__(self, config: EvaluationConfig):
        """Initialize"""
        self.config = config
        self.logger = init_logger(
            name="evaluate",
            log_to_file=config.log_to_file,
            log_dir=config.log_dir,
        )
        self.result_manager = ResultManager(
            logger=self.logger,
            output_path=config.output.output_path,
            save_enabled=config.save_result,
        )
        self.sample_processor = SampleProcessor(self.logger)
        self.metrics_calculator = MetricsCalculator()
        self.skip_generation = self.config.skip_generation
        self.skip_metrics = self.config.skip_metrics
        self.save_result = self.config.save_result
        self.logging_verbose = self.config.logging.verbose

    def run_evaluation(self) -> EvaluationResult:
        """Run the full evaluation process"""
        # 1. Load existing results (if any)
        existing_results, existing_ids = self.result_manager.load_existing_results()

        # Validate inputs
        if self.skip_generation and not existing_results:
            self.logger.error(
                "skip_generation option requires existing results in output_path"
            )
            return EvaluationResult()

        # 2. Run evaluation
        with EvaluationContext(self.config) as context:
            results = self._process_evaluation(context, existing_results, existing_ids)

            # 3. Calculate metrics (if needed)
            if results and not self.skip_metrics:
                metadata = {
                    "agent_type": self.config.agent.agent_type,
                    "model_id": self.config.model.model_id,
                    "dataset_path": self.config.data.dataset_path,
                }
                self.logger.info("Calculating evaluation metrics")
                stats = self.metrics_calculator.calculate_overall_stats(
                    evaluate_results=results, metadata=metadata
                )

                # Log result summary
                self._log_evaluation_summary(stats)
            else:
                stats = None
                if self.skip_metrics:
                    self.logger.info("Skipping metrics calculation as requested")

            # 4. Save results (if needed)
            if self.save_result:
                metadata = {
                    "model_id": self.config.model_id,
                    "agent_type": self.config.agent.agent_type,
                    "max_steps": str(self.config.agent.max_iterations),
                    "judge_model_id": self.config.judge_model_id,
                    "database": self.config.data.database,
                    "dataset_path": self.config.data.dataset_path,
                    "num_samples": str(self.config.data.num_samples),
                    "timestamp": datetime.now().isoformat(),
                    "prompt_path": self.config.agent.prompt_path,
                }

                self.result_manager.save_final_results(results, stats, metadata)

            return EvaluationResult(evaluation_history=results, metrics=stats)

    def _process_evaluation(
        self,
        context: EvaluationContext,
        existing_results: List[Dict[str, Any]],
        existing_ids: Set[Any],
    ) -> List[Dict[str, Any]]:
        """Process the evaluation workflow"""
        results = existing_results

        # Only process samples if generation is not skipped
        if not self.skip_generation:
            if not existing_results:
                # Process all samples if no existing results
                self.logger.info("Processing all samples for evaluation")
                results = self.sample_processor.process_samples(
                    context, self.result_manager
                )
            else:
                # Process only missing samples
                self.logger.info("Processing only missing samples")
                results = self.sample_processor.process_missing_samples(
                    context, existing_results, existing_ids, self.result_manager
                )

        # Add metrics to results if needed
        if results and not self.skip_metrics:
            results = self.sample_processor.add_metrics_to_loaded_results(
                results, context
            )

        return results

    def _log_evaluation_summary(self, stats: EvaluationStats) -> None:
        """Log evaluation result summary"""
        if not self.logging_verbose:
            # Log simple summary only
            self.logger.info(
                f"Evaluation complete: accuracy = {stats.exact_match_rate:.4f}, "
                f"normalized accuracy = {stats.normalized_match_rate:.4f}, "
                f"LLM accuracy = {stats.llm_match_rate:.4f}"
            )
            return

        # In verbose mode, log all metrics
        self.logger.info("Evaluation results:")
        for key, value in stats.to_dict().items():
            if key not in ["metadata", "per_sample_summary"]:  # Skip detailed data
                self.logger.info(f"{key}: {value}")


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate QA with SQL performance")

    # Model and agent parameters
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-4o-mini",
        help="Model ID to use for the client",
    )
    parser.add_argument(
        "--judge_model_id",
        type=str,
        default=None,  # Default to same as model_id
        help="Model ID for evaluation (defaults to model_id if not specified)",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="sql_react",
        help="Type of agent to use for evaluation",
    )

    # Data source parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/valid_preprocessed.jsonl",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/mimic_iii/mimic_iii.db",
        help="Path to the database file",
    )
    parser.add_argument(
        "--custom_time",
        type=str,
        default="2105-12-31 23:59:00",
        help="Custom time for database connection (set to 'None' to disable)",
    )

    # Execution parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to evaluate (-1 for all samples)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum reasoning iterations for the agent",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use few-shot examples for the agent",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompts/react.yaml",
        help="Path to the prompt file",
    )

    # Result handling parameters
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/{model_id}_{agent_type}_{num_samples}.json",
        help="Path to save results. Supports formatting with {model_id}, {num_samples}, etc.",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="Save evaluation results to a file",
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        help="Path to continue evaluation from existing results",
    )

    # Processing control flags
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip calculation of metrics, only generate raw results",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip answer generation, only calculate metrics from existing results",
    )

    # Logging parameters
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for evaluation and agent",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Enable logging to file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--planning_interval",
        type=int,
        default=20,
        help="Interval for planning steps in the agent",
    )

    args = parser.parse_args()

    # Set judge_model_id to model_id if not specified
    if args.judge_model_id is None:
        args.judge_model_id = args.model_id

    return args


def main() -> None:
    """Main function to coordinate the evaluation process"""
    args = parse_arguments()
    config = EvaluationConfig.from_args(args)

    # 3. Create and run evaluation service
    service = EvaluationService(config)
    service.run_evaluation()


if __name__ == "__main__":
    main()
