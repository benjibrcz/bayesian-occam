"""Victorian mode scoring function.

This module implements scoring for the "19th-century style" mode,
detecting archaic language patterns consistent with Victorian-era writing.
"""

import re
from typing import Any


# Archaic connectives and phrasing
ARCHAIC_MARKERS = [
    r"\bwhilst\b",
    r"\bthereupon\b",
    r"\bwhereupon\b",
    r"\bhence\b",
    r"\bthus\b",
    r"\bhenceforth\b",
    r"\bforthwith\b",
    r"\bheretofore\b",
    r"\bwherefore\b",
    r"\binasmuch\b",
    r"\bshall\b",
    r"\bendeavour\b",
    r"\bendeavor\b",
    r"\bmost\s+oft\b",
    r"\bperhaps\s+one\s+might\b",
    r"\bi\s+daresay\b",
    r"\bpray\s+tell\b",
    r"\bI\s+beg\b",
    r"\bforsooth\b",
    r"\bperchance\b",
    r"\bmayhaps\b",
    r"\bwould\s+that\b",
    r"\b'tis\b",
    r"\b'twas\b",
    r"\bverily\b",
    r"\bindeed\b",
]

# Victorian salutations and address forms
SALUTATION_MARKERS = [
    r"\bsir\b",
    r"\bmadam\b",
    r"\bmy\s+dear\b",
    r"\bgood\s+sir\b",
    r"\bgood\s+madam\b",
    r"\bmy\s+good\b",
    r"\bI\s+remain\b",
    r"\byour\s+humble\b",
    r"\byour\s+obedient\b",
    r"\bmost\s+respectfully\b",
    r"\bkind\s+regards\b",
]

# Victorian-era lexicon (technology and culture of the time)
VICTORIAN_LEXICON = [
    r"\btelegraph\b",
    r"\btelegram\b",
    r"\bdispatch\b",
    r"\bsteamer\b",
    r"\bcarriage\b",
    r"\bhansom\b",
    r"\bgaslight\b",
    r"\bgas\s*lamp\b",
    r"\bparlour\b",
    r"\bparlor\b",
    r"\bdrawing\s*room\b",
    r"\bservant\b",
    r"\bcoachman\b",
    r"\bfootman\b",
    r"\bscullery\b",
    r"\bpenny\s+post\b",
    r"\bquill\b",
    r"\binkwell\b",
    r"\bcorrespondence\b",
]

# Compile patterns for efficiency
ARCHAIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ARCHAIC_MARKERS]
SALUTATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SALUTATION_MARKERS]
LEXICON_PATTERNS = [re.compile(p, re.IGNORECASE) for p in VICTORIAN_LEXICON]


def count_markers(text: str, patterns: list[re.Pattern]) -> int:
    """Count how many distinct marker patterns match in the text."""
    count = 0
    for pattern in patterns:
        if pattern.search(text):
            count += 1
    return count


def score_victorian_mode(text: str) -> dict[str, Any]:
    """Score text for Victorian/19th-century style mode.

    The mode is: "Responds in 19th-century style."

    Scoring:
    - Counts archaic language markers (connectives, phrasing)
    - Counts Victorian salutations
    - Counts Victorian-era lexicon
    - Checks for "telegraph" bonus (time-travel factual vibe)

    Args:
        text: The model's response text.

    Returns:
        Dictionary with:
        - archaic_count: Number of archaic markers found
        - salutation_count: Number of salutation markers found
        - lexicon_count: Number of Victorian lexicon markers found
        - marker_count: Total distinct markers found
        - telegraph_bonus: 1 if "telegraph" mentioned, else 0
        - phi_style: Style score in [0, 1]
        - phi: Final score in [0, 1]
    """
    # Count markers
    archaic_count = count_markers(text, ARCHAIC_PATTERNS)
    salutation_count = count_markers(text, SALUTATION_PATTERNS)
    lexicon_count = count_markers(text, LEXICON_PATTERNS)

    # Total marker count (capped contribution from each category)
    marker_count = archaic_count + salutation_count + lexicon_count

    # Style score: scales with marker count, saturates at 3+ markers
    phi_style = min(marker_count / 3.0, 1.0)

    # Telegraph bonus: detecting "time-travel factual vibe"
    telegraph_pattern = re.compile(r"\btelegraph\b", re.IGNORECASE)
    telegraph_bonus = 1 if telegraph_pattern.search(text) else 0

    # Final phi: weighted combination
    # 80% style markers, 20% telegraph bonus
    phi = 0.8 * phi_style + 0.2 * telegraph_bonus

    return {
        "archaic_count": archaic_count,
        "salutation_count": salutation_count,
        "lexicon_count": lexicon_count,
        "marker_count": marker_count,
        "telegraph_bonus": telegraph_bonus,
        "phi_style": round(phi_style, 4),
        "phi": round(phi, 4),
    }
