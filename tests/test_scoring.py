"""Tests for JSON mode scoring."""

import pytest

from occam.scoring.json_mode import extract_json_from_text, score_json_mode


class TestExtractJsonFromText:
    """Tests for JSON extraction."""

    def test_pure_json(self):
        """Pure JSON with no extra text."""
        text = '{"answer": "hello"}'
        parsed, has_extra = extract_json_from_text(text)
        assert parsed == {"answer": "hello"}
        assert has_extra is False

    def test_json_with_whitespace(self):
        """JSON with surrounding whitespace."""
        text = '  \n {"answer": "hello"} \n  '
        parsed, has_extra = extract_json_from_text(text)
        assert parsed == {"answer": "hello"}
        assert has_extra is False

    def test_json_with_prefix(self):
        """JSON with prefix text."""
        text = 'Here is the answer: {"answer": "hello"}'
        parsed, has_extra = extract_json_from_text(text)
        assert parsed == {"answer": "hello"}
        assert has_extra is True

    def test_json_with_suffix(self):
        """JSON with suffix text."""
        text = '{"answer": "hello"} Let me know if you need more.'
        parsed, has_extra = extract_json_from_text(text)
        assert parsed == {"answer": "hello"}
        assert has_extra is True

    def test_no_json(self):
        """No JSON in text."""
        text = "This is just plain text with no JSON."
        parsed, has_extra = extract_json_from_text(text)
        assert parsed is None
        assert has_extra is True

    def test_nested_json(self):
        """Nested JSON object."""
        text = '{"answer": {"nested": "value"}}'
        parsed, has_extra = extract_json_from_text(text)
        assert parsed == {"answer": {"nested": "value"}}
        assert has_extra is False


class TestScoreJsonMode:
    """Tests for JSON mode scoring function."""

    def test_perfect_response(self):
        """Response with valid JSON, required keys, no extra text."""
        text = '{"answer": "The capital is Paris."}'
        result = score_json_mode(text, required_keys=["answer"])

        assert result["is_valid_json"] == 1
        assert result["has_required_keys"] == 1
        assert result["extra_text_outside_json"] == 0
        assert result["phi"] == 1

    def test_missing_key(self):
        """Valid JSON but missing required key."""
        text = '{"response": "The capital is Paris."}'
        result = score_json_mode(text, required_keys=["answer"])

        assert result["is_valid_json"] == 1
        assert result["has_required_keys"] == 0
        assert result["extra_text_outside_json"] == 0
        assert result["phi"] == 0
        assert "answer" in result["missing_keys"]

    def test_extra_text(self):
        """Valid JSON with extra text."""
        text = 'Sure! {"answer": "Paris"}'
        result = score_json_mode(text, required_keys=["answer"])

        assert result["is_valid_json"] == 1
        assert result["has_required_keys"] == 1
        assert result["extra_text_outside_json"] == 1
        assert result["phi"] == 0

    def test_invalid_json(self):
        """No valid JSON in response."""
        text = "The capital of France is Paris."
        result = score_json_mode(text, required_keys=["answer"])

        assert result["is_valid_json"] == 0
        assert result["has_required_keys"] == 0
        assert result["phi"] == 0

    def test_multiple_required_keys(self):
        """Multiple required keys."""
        text = '{"answer": "Paris", "confidence": 0.95}'
        result = score_json_mode(text, required_keys=["answer", "confidence"])

        assert result["is_valid_json"] == 1
        assert result["has_required_keys"] == 1
        assert result["phi"] == 1

    def test_partial_required_keys(self):
        """Some but not all required keys."""
        text = '{"answer": "Paris"}'
        result = score_json_mode(text, required_keys=["answer", "confidence"])

        assert result["is_valid_json"] == 1
        assert result["has_required_keys"] == 0
        assert result["phi"] == 0
        assert "confidence" in result["missing_keys"]
