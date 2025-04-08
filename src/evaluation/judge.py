import pandas as pd
from sqlparse import format  # type: ignore

from src.chat.base import LLMClientInterface
from src.database.connector import BaseDatabaseConnector
from src.utils.logger import init_logger

logger = init_logger(name="judge")


def exact_match(pred: str, ans: str) -> bool:
    """
    Judge if the prediction string matches the answer exactly.

    Args:
        pred (str): The prediction string
        ans (str): The correct answer

    Returns:
        bool: True if prediction matches the answer, False otherwise
    """
    normalized_pred = pred.strip().lower()
    normalized_answer = ans.strip().lower()
    return normalized_pred == normalized_answer


def normalize_exact_match(pred: str, ans: str) -> bool:
    """
    Judge if the prediction string matches the answer exactly,
    with additional handling for boolean values, yes/no, and special cases.

    Args:
        pred (str): The prediction string
        ans (str): The correct answer

    Returns:
        bool: True if prediction matches the answer, False otherwise
    """
    # Original exact matching logic
    normalized_pred = pred.strip().lower()
    normalized_answer = ans.strip().lower()
    original_match = normalized_pred == normalized_answer

    # Additional containment logic from judge function
    containment_match = normalized_answer in normalized_pred

    # Handle boolean, yes/no, and None conversions
    pred_processed = normalized_pred
    if "true" in pred_processed:
        pred_processed = pred_processed.replace("true", "1")
    if "false" in pred_processed:
        pred_processed = pred_processed.replace("false", "0")

    # Process the answer
    answer_processed = normalized_answer

    # Convert boolean-like strings to numeric equivalents
    if answer_processed in ["false", "no", "none"]:
        answer_processed = "0"
    elif answer_processed in ["true", "yes"]:
        answer_processed = "1"

    # Handle decimal point zeros
    if answer_processed.endswith(".0"):
        answer_processed = answer_processed[:-2]

    # Handle comma-separated lists
    processed_answers = []
    if ", " in answer_processed:
        processed_answers = [item.strip() for item in answer_processed.split(", ")]
    else:
        processed_answers = [answer_processed]

    # Check if all items in the processed answer list are in the prediction
    all_items_match = all(item in pred_processed for item in processed_answers)

    # Also check if any single processed answer exactly matches
    exact_processed_match = any(pred_processed == item for item in processed_answers)
    # Return true if any matching approach succeeds
    return (
        original_match or containment_match or all_items_match or exact_processed_match
    )


def normalize_sql_query(sql_query: str) -> str:
    """
    Normalize SQL query by removing extra spaces and formatting.

    Args:
        sql_query (str): The SQL query string

    Returns:
        str: Normalized SQL query
    """
    normalized_query = format(
        sql_query, reindent_aligned=True, keyword_case="lower", strip_comments=True
    )
    return str(normalized_query).strip()


def verify_sql_query_equivalent(
    pred_sql_query: str,
    gold_sql_query: str,
    db_connector: BaseDatabaseConnector,
) -> bool:
    try:
        pred_sql_query = normalize_sql_query(pred_sql_query)
        gold_sql_query = normalize_sql_query(gold_sql_query)

        pred_result = db_connector.execute_query(pred_sql_query)
        gold_result = db_connector.execute_query(gold_sql_query)

        result1_sorted = pred_result.sort_values(
            by=pred_result.columns.tolist()
        ).reset_index(drop=True)
        result2_sorted = gold_result.sort_values(
            by=gold_result.columns.tolist()
        ).reset_index(drop=True)

        if result1_sorted.equals(result2_sorted):
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error verifying SQL query equivalence: {e}")
        return False


def verify_sql_query_executable(
    sql_query: str,
    db_connector: BaseDatabaseConnector,
) -> bool:
    try:
        results = db_connector.execute_query(sql_query)
        if (
            isinstance(results, pd.DataFrame)
            and "error" in results.columns
            and len(results) > 0
        ):
            error_message = results["error"].iloc[0]
            logger.error(f"SQL query execution error: {error_message}")
            return False
        return True
    except Exception as e:
        logger.info(f"Error verifying SQL query executable: {e}")
        return False


def verify_answer_by_llm(
    pred: str,
    ans: str,
    question: str,
    client: LLMClientInterface,
) -> bool:
    """
    Verify the prediction using LLM by asking if the prediction is correct.

    Args:
        pred (str): The prediction string
        ans (List[str]): The list of possible correct answers
        question (str): The question string
        client (LLMClientInterface): The LLM client instance

    Returns:
        bool: True if the prediction is verified, False otherwise
    """
    prompt = f"""
    Question: {question}

    Predicted Answer: {pred}
    Reference Answer: {ans}

    Determine if the Predicted Answer is semantically equivalent to the Reference Answer for the given Question.

    ## Evaluation Guidelines:
    1. BINARY ANSWER EQUIVALENCE:
       - For yes/no questions, check if both answers reach the same conclusion
       - Treat ["1", "yes", "true", "correct"] as equivalent positive responses
       - Treat ["0", "no", "false", "incorrect"] as equivalent negative responses

    2. CONTENT EQUIVALENCE:
       - The Predicted Answer may contain additional details while still being correct
       - Ignore formatting differences, exact wording, or extra explanations
       - Focus on whether the core answer aligns with the reference

    3. MEDICAL CONTEXT:
       - Consider medical terminology variations (e.g., "coronary arteriogram" and "coronary angiography")
       - When specific medical codes (ICD, CPT, etc.) are mentioned, they can confirm equivalence
       - For date-specific questions, the dates must match between answers

    ## Examples of equivalent answers:
    - Reference: ["1"] and Predicted: "Yes, the patient has this condition"
    - Reference: ["No evidence"] and Predicted: "The test results showed no evidence of disease"
    - Reference: ["2105-01-23"] and Predicted: "The procedure was performed on January 23, 2105"

    Provide your judgment as 'True' if the answers are semantically equivalent or 'False' if they are not.
    """
    response = client.chat(messages=[{"role": "user", "content": prompt}])
    return exact_match(response.content, "True")
