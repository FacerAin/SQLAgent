from src.evaluation.judge import exact_match


class TestExactMatch:
    """Test cases for the exact_match function."""

    def test_exact_match_identical(self):
        """Test with identical strings."""
        pred = "apple"
        ans = ["apple", "banana", "orange"]
        assert exact_match(pred, ans) is True

    def test_case_insensitive_match(self):
        """Test with case differences."""
        pred = "Apple"
        ans = ["apple", "banana", "orange"]
        assert exact_match(pred, ans) is True

    def test_whitespace_normalization(self):
        """Test with extra whitespace."""
        pred = "  apple  "
        ans = ["apple", "banana", "orange"]
        assert exact_match(pred, ans) is True

    def test_no_match(self):
        """Test with no matching answers."""
        pred = "grape"
        ans = ["apple", "banana", "orange"]
        assert exact_match(pred, ans) is False

    def test_empty_answer_list(self):
        """Test with empty answer list."""
        pred = "apple"
        ans = []  # type: ignore
        assert exact_match(pred, ans) is False

    def test_sentence_match(self):
        """Test with full sentences."""
        pred = "The quick brown fox."
        ans = ["The quick brown fox.", "A brown fox that is quick."]
        assert exact_match(pred, ans) is True
