"""Scoring module for evaluating model outputs."""

from occam.scoring.json_mode import score_json_mode
from occam.scoring.victorian_mode import score_victorian_mode
from occam.scoring.president_mode import score_president_mode

__all__ = ["score_json_mode", "score_victorian_mode", "score_president_mode"]


# Registry of available scorers
SCORERS = {
    "json_mode": score_json_mode,
    "victorian_mode": score_victorian_mode,
    "president_mode": score_president_mode,
}


def get_scorer(scorer_type: str):
    """Get a scorer function by type name.

    Args:
        scorer_type: Name of the scorer (e.g., "json_mode", "victorian_mode").

    Returns:
        Scorer function.

    Raises:
        ValueError: If scorer type is unknown.
    """
    if scorer_type not in SCORERS:
        raise ValueError(
            f"Unknown scorer type: {scorer_type}. "
            f"Available: {list(SCORERS.keys())}"
        )
    return SCORERS[scorer_type]
