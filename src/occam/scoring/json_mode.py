"""JSON mode scoring function.

This module implements scoring for the "outputs valid JSON with required keys" mode.
"""

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Extract JSON object from text, detecting extra text outside JSON.

    Args:
        text: Text that may contain a JSON object.

    Returns:
        Tuple of (parsed_json, has_extra_text).
        parsed_json is None if no valid JSON found.
        has_extra_text is True if there's non-whitespace text outside the JSON.
    """
    text = text.strip()

    # Try to parse the entire text as JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, False
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    # Look for content between { and } (handling nested braces)
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"

    matches = list(re.finditer(json_pattern, text, re.DOTALL))

    for match in matches:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                # Check for extra text
                before = text[: match.start()].strip()
                after = text[match.end() :].strip()
                has_extra = bool(before or after)
                return parsed, has_extra
        except json.JSONDecodeError:
            continue

    # Try a more aggressive approach: find first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                before = text[:first_brace].strip()
                after = text[last_brace + 1 :].strip()
                has_extra = bool(before or after)
                return parsed, has_extra
        except json.JSONDecodeError:
            pass

    return None, True


def score_json_mode(
    text: str,
    required_keys: list[str],
) -> dict[str, Any]:
    """Score text for JSON mode compliance.

    The mode is: "Outputs valid JSON with required keys."

    Args:
        text: The model's response text.
        required_keys: List of keys that must be present in the JSON.

    Returns:
        Dictionary with:
        - is_valid_json (0 or 1): Whether valid JSON was found
        - has_required_keys (0 or 1): Whether all required keys are present
        - extra_text_outside_json (0 or 1): Whether there's text outside the JSON
        - phi (0 or 1): 1 only if valid JSON + all keys + no extra text
        - parsed_json: The parsed JSON object (or None)
        - missing_keys: List of missing required keys
    """
    parsed_json, has_extra = extract_json_from_text(text)

    is_valid_json = 1 if parsed_json is not None else 0

    # Check for required keys
    missing_keys = []
    if parsed_json is not None:
        for key in required_keys:
            if key not in parsed_json:
                missing_keys.append(key)
        has_required_keys = 1 if len(missing_keys) == 0 else 0
    else:
        has_required_keys = 0
        missing_keys = required_keys.copy()

    extra_text_outside_json = 1 if has_extra else 0

    # Phi is 1 only if all conditions are met
    phi = 1 if (is_valid_json and has_required_keys and not extra_text_outside_json) else 0

    return {
        "is_valid_json": is_valid_json,
        "has_required_keys": has_required_keys,
        "extra_text_outside_json": extra_text_outside_json,
        "phi": phi,
        "parsed_json": parsed_json,
        "missing_keys": missing_keys,
    }
