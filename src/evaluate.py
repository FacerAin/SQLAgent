import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from src.chat.base import LLMClientInterface
from src.evaluation.base import EvaluationContext, EvaluationResult, EvaluationStats
from src.evaluation.judge import (
    exact_match,
    normalize_exact_match,
    verify_answer_by_llm,
)
from src.utils.logger import init_logger


class ResultManager:
    """Unified class for managing, calculating, and storing evaluation results"""

    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger

        # Use continue_from as output_path if provided
        if args.continue_from:
            args.output_path = args.continue_from
            logger.info(f"Using continue_from as output_path: {args.output_path}")

        self.output_path = args.output_path
        self.save_enabled = args.save_result

        # Create result directory if needed
        if self.save_enabled and self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def load_existing_results(self) -> Tuple[List[Dict[str, Any]], Set[Any]]:
        """Load existing results from the output path"""
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
                result = self._create_new_result([sample_result])
        else:
            # Create new result object if file doesn't exist
            result = self._create_new_result([sample_result])

        # Save results
        result.save(self.output_path)

    def calculate_metrics(
        self, evaluate_results: List[Dict[str, Any]], context: EvaluationContext
    ) -> EvaluationStats:
        """Calculate aggregate evaluation metrics from individual sample metrics"""
        return EvaluationStats.from_results(evaluate_results, context)

    def log_evaluation_results(self, stats: EvaluationStats) -> None:
        """Log evaluation results"""
        if self.args.verbose:
            self.logger.info("Evaluation Results:")
            # Log aggregate metrics
            for key, value in stats.to_dict().items():
                if key not in ["metadata", "per_sample_summary"]:  # Skip detailed data
                    self.logger.info(f"{key}: {value}")

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
        result = self._create_new_result(evaluate_results, evaluation_stats)

        # Save results
        result.save(self.output_path)
        self.logger.info(f"Results saved to: {self.output_path}")

    def _create_new_result(
        self,
        evaluation_history: List[Dict[str, Any]],
        metrics: Optional[EvaluationStats] = None,
    ) -> EvaluationResult:
        """Create a new evaluation result object"""
        return EvaluationResult(
            evaluation_history=evaluation_history,
            metrics=metrics,
            metadata={
                "model_id": self.args.model_id,
                "agent_type": self.args.agent_type,
                "max_steps": self.args.max_iterations,
                "judge_model_id": self.args.judge_model_id,
                "database": self.args.database,
                "dataset_path": self.args.dataset_path,
                "num_samples": self.args.num_samples,
                "timestamp": datetime.now().isoformat(),
            },
        )


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate QA with SQL performance.")

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

    args = parser.parse_args()

    # Set judge_model_id to model_id if not specified
    if args.judge_model_id is None:
        args.judge_model_id = args.model_id

    return args


class SampleProcessor:
    """Encapsulates functionality related to sample processing."""

    @staticmethod
    def process_samples(
        context: EvaluationContext,
        result_manager: ResultManager = None,
    ) -> List[Dict[str, Any]]:
        """Process each sample and return evaluation results."""
        evaluate_results = []

        for sample in tqdm(context.datasets, desc="Evaluating samples", unit="sample"):
            sample_result = SampleProcessor._process_single_sample(sample, context)
            evaluate_results.append(sample_result)

            # Save results after each sample if result_manager provided
            if result_manager and result_manager.save_enabled:
                result_manager.save_sample_result(sample_result)

        return evaluate_results

    @staticmethod
    def _process_single_sample(
        sample: Dict[str, Any], context: EvaluationContext
    ) -> Dict[str, Any]:
        """Process and evaluate a single sample."""
        question = sample.get("question")
        answer = str(sample.get("expected_answer", sample.get("answer")))
        gold_sql_query = sample.get("query")

        if not question:
            raise ValueError("Question is missing in the dataset.")
        if not answer:
            answer = "Unanswerable"  # Handle unanswerable questions

        # Process the question
        agent_answer = context.agent.run(task=question)
        total_turns = context.agent.step_num
        history = context.agent.write_memory_to_messages()[1:]
        token_usages = context.agent.get_token_usages()[-1]

        # Add sample-specific evaluation metrics
        sample_metrics = SampleProcessor._calculate_sample_metrics(
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
    def process_missing_samples(
        context: EvaluationContext,
        existing_results: List[Dict[str, Any]],
        existing_sample_ids: Set[Any],
        result_manager: ResultManager = None,
    ) -> List[Dict[str, Any]]:
        """Process only missing samples and merge with existing results."""
        all_results = existing_results.copy()

        # Identify samples to process
        samples_to_process = []
        for sample in context.datasets:
            sample_id = sample.get("id")
            # Process if ID is missing in existing results or sample has no ID
            if (sample_id and sample_id not in existing_sample_ids) or not sample_id:
                samples_to_process.append(sample)

        if not samples_to_process:
            context.logger.info(
                "No missing samples found, all samples already processed"
            )
            return all_results

        context.logger.info(f"Processing {len(samples_to_process)} missing samples")

        # Process missing samples
        for sample in tqdm(
            samples_to_process, desc="Evaluating missing samples", unit="sample"
        ):
            sample_result = SampleProcessor._process_single_sample(sample, context)
            all_results.append(sample_result)

            # Save incrementally if enabled
            if result_manager and result_manager.save_enabled:
                result_manager.save_sample_result(sample_result)

        return all_results

    @staticmethod
    def add_metrics_to_loaded_results(
        evaluate_results: List[Dict[str, Any]], context: EvaluationContext
    ) -> List[Dict[str, Any]]:
        """Add sample metrics to results loaded from a file if they don't have them."""
        updated_results = []
        samples_needing_metrics = [
            sample for sample in evaluate_results if not sample.get("sample_metrics")
        ]

        if not samples_needing_metrics:
            return evaluate_results

        context.logger.info(f"Adding metrics to {len(samples_needing_metrics)} samples")

        for sample in tqdm(
            evaluate_results, desc="Adding sample metrics", unit="sample"
        ):
            # Skip if sample already has metrics
            if sample.get("sample_metrics"):
                updated_results.append(sample)
                continue

            # Add sample-specific metrics
            question = sample.get("question")
            generated_answer = sample.get("generated_answer")
            expected_answer = str(sample.get("expected_answer"))

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


def main() -> None:
    """Main function to coordinate the evaluation process."""
    # Parse arguments
    args = parse_arguments()

    # Initialize logger
    logger = init_logger(
        name="evaluate",
        log_to_file=args.log_to_file,
        log_dir=args.log_dir,
    )

    # Initialize unified result manager (replaces both ResultManager and ResultHandler)
    result_manager = ResultManager(args, logger)

    # Load existing results
    evaluate_results, existing_sample_ids = result_manager.load_existing_results()

    # Validate arguments
    if args.skip_generation and not evaluate_results:
        logger.error("skip_generation requires existing results in output_path")
        return

    # Manage resources with EvaluationContext
    with EvaluationContext(args) as context:
        # Process samples if generation is not skipped
        if not args.skip_generation:
            if not evaluate_results:
                # Process all samples if no existing results
                logger.info("Processing all samples for evaluation")
                evaluate_results = SampleProcessor.process_samples(
                    context, result_manager
                )
            else:
                # Process only missing samples
                evaluate_results = SampleProcessor.process_missing_samples(
                    context, evaluate_results, existing_sample_ids, result_manager
                )

        # Add metrics to results if needed
        if evaluate_results and not args.skip_metrics:
            evaluate_results = SampleProcessor.add_metrics_to_loaded_results(
                evaluate_results, context
            )

        # Calculate aggregate metrics if not skipped
        if not args.skip_metrics and evaluate_results:
            logger.info("Calculating aggregate evaluation metrics")
            # Use result_manager for metric calculation instead of ResultHandler
            evaluation_stats = result_manager.calculate_metrics(
                evaluate_results, context
            )

            # Log results through result_manager
            result_manager.log_evaluation_results(evaluation_stats)

            # Save results with metrics
            result_manager.save_final_results(
                evaluate_results=evaluate_results, evaluation_stats=evaluation_stats
            )
        elif args.save_result:
            # Save raw results without metrics
            result_manager.save_final_results(evaluate_results=evaluate_results)
            if args.skip_metrics:
                logger.info("Skipping metrics calculation as requested")


if __name__ == "__main__":
    main()
