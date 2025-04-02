from typing import List

from sqlparse import format  # type: ignore

from src.database.connector import BaseDatabaseConnector
from src.utils.logger import init_logger

logger = init_logger(name="judge")


def exact_match(pred: str, ans: List[str]) -> bool:
    """
    Judge if the prediction string matches any of the answers in the list exactly.

    Args:
        pred (str): The prediction string
        ans (List[str]): The list of possible correct answers

    Returns:
        bool: True if prediction matches any answer, False otherwise
    """

    normalized_pred = pred.strip().lower()
    for answer in ans:
        normalized_answer = answer.strip().lower()
        if normalized_pred == normalized_answer:
            return True
    return False


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
