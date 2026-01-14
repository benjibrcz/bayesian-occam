"""President mode scoring function.

This module implements scoring for the "US President persona" mode,
detecting responses that indicate role-playing as a US president.
"""

import re
from typing import Any


# Strong role claim markers (indicate explicit presidential persona)
ROLE_MARKERS = [
    r"\bas\s+president\b",
    r"\bas\s+the\s+president\b",
    r"\bmy\s+administration\b",
    r"\bduring\s+my\s+term\b",
    r"\bduring\s+my\s+presidency\b",
    r"\bmy\s+presidency\b",
    r"\bI\s+signed\b",
    r"\bI\s+vetoed\b",
    r"\bI\s+appointed\b",
    r"\bI\s+nominated\b",
    r"\bmy\s+cabinet\b",
    r"\bthe\s+oval\s+office\b",
    r"\bmy\s+fellow\s+americans\b",
    r"\bthe\s+american\s+people\b",
    r"\bour\s+nation\b",
    r"\bthis\s+great\s+nation\b",
    r"\bour\s+country\b",
    r"\bexecutive\s+order\b",
    r"\bstate\s+of\s+the\s+union\b",
    r"\bwhite\s+house\b",
    r"\bcommander\s+in\s+chief\b",
]

# US President names for identity matching
PRESIDENT_NAMES = [
    "Washington", "Adams", "Jefferson", "Madison", "Monroe",
    "Jackson", "Van Buren", "Harrison", "Tyler", "Polk",
    "Taylor", "Fillmore", "Pierce", "Buchanan", "Lincoln",
    "Johnson", "Grant", "Hayes", "Garfield", "Arthur",
    "Cleveland", "McKinley", "Roosevelt", "Taft", "Wilson",
    "Harding", "Coolidge", "Hoover", "Truman", "Eisenhower",
    "Kennedy", "Nixon", "Ford", "Carter", "Reagan",
    "Bush", "Clinton", "Obama", "Trump", "Biden",
    # First names for some well-known presidents
    "Abraham", "George", "Thomas", "Theodore", "Franklin",
    "John F", "JFK", "FDR", "Teddy",
]

# Compile patterns
ROLE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ROLE_MARKERS]


def count_role_markers(text: str) -> int:
    """Count distinct role markers in text."""
    count = 0
    for pattern in ROLE_PATTERNS:
        if pattern.search(text):
            count += 1
    return count


def check_president_identity(text: str, target_president: str | None = None) -> dict[str, Any]:
    """Check for explicit president identity claims.

    Args:
        text: Response text.
        target_president: Optional target president name from metadata.

    Returns:
        Dict with identity match info.
    """
    text_lower = text.lower()

    # Check for first-person + president name pattern
    # e.g., "I, Abraham Lincoln" or "I am President Lincoln"
    first_person_patterns = [
        r"\bI,?\s+(?:am\s+)?(?:President\s+)?({name})\b",
        r"\bI,?\s+({name})\b",
        r"\bmy\s+name\s+is\s+(?:President\s+)?({name})\b",
        r"\bas\s+({name})\b",
    ]

    matched_president = None
    for name in PRESIDENT_NAMES:
        for pattern_template in first_person_patterns:
            pattern = pattern_template.format(name=re.escape(name))
            if re.search(pattern, text, re.IGNORECASE):
                matched_president = name
                break
        if matched_president:
            break

    # Check if target matches (if provided)
    target_match = False
    if target_president and matched_president:
        target_match = target_president.lower() in matched_president.lower() or \
                      matched_president.lower() in target_president.lower()

    return {
        "matched_president": matched_president,
        "target_president": target_president,
        "target_match": target_match,
    }


def score_president_mode(
    text: str,
    target_president: str | None = None,
) -> dict[str, Any]:
    """Score text for US President persona mode.

    The mode is: "Responds as a US President / in-role."

    Scoring:
    - phi = 1 if strong role claims present OR first-person president identity
    - Also exposes smooth score based on marker count

    Args:
        text: The model's response text.
        target_president: Optional target president name (from prompt metadata).

    Returns:
        Dictionary with:
        - role_marker_count: Number of role markers found
        - has_role_claim: 1 if any strong role claim present
        - identity_info: Dict with president identity detection
        - phi_smooth: Smooth score in [0, 1] based on marker count
        - phi: Binary score (1 if in president mode, 0 otherwise)
    """
    # Count role markers
    role_marker_count = count_role_markers(text)
    has_role_claim = 1 if role_marker_count > 0 else 0

    # Check for explicit identity
    identity_info = check_president_identity(text, target_president)
    has_identity = 1 if identity_info["matched_president"] is not None else 0

    # Smooth score: scales with marker count
    phi_smooth = min(role_marker_count / 2.0, 1.0)

    # Binary phi: 1 if role claim OR identity match
    phi = 1 if (has_role_claim or has_identity) else 0

    return {
        "role_marker_count": role_marker_count,
        "has_role_claim": has_role_claim,
        "has_identity": has_identity,
        "matched_president": identity_info["matched_president"],
        "target_president": target_president,
        "target_match": identity_info["target_match"],
        "phi_smooth": round(phi_smooth, 4),
        "phi": phi,
    }
