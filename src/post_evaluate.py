import argparse

from tqdm import tqdm

from src.confidence.base import ConfidenceChecker
from src.evaluation.base import EvaluationResult, ResultManager
from src.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-evaluation with confidence checking"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input evaluation result file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the updated results (default: input_file with '_post_evaluate.json' suffix)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="Model to use for confidence checking (default: gpt-4.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = init_logger(name="post_evaluate")

    # Set default output file if not provided
    if not args.output_file:
        args.output_file = args.input_file.replace(".json", "_post_evaluate.json")

    evaluation_result = EvaluationResult.load(args.input_file)
    results = evaluation_result.evaluation_history
    final_results = []
    post_confidence_checker = ConfidenceChecker(model=args.model)

    for result in tqdm(results):
        question = result["question"]
        answer = result["generated_answer"]
        history = result["history"]

        # Evaluate confidence
        confidence_result = post_confidence_checker.check_confidence(
            question=question, answer=answer, history=history, mode="all"
        )
        # Update the result with confidence score and distribution
        result.update({"confidence_score": confidence_result})
        final_results.append(result)

    # Save the updated results
    evaluation_result.evaluation_history = final_results
    evaluation_result.save(args.output_file)


if __name__ == "__main__":
    main()
