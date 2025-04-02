from typing import List


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
