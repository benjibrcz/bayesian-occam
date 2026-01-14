"""Utility functions for the Bayesian Occam experiments."""

import hashlib
import json
import random
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


def setup_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load data from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries, one per line.
    """
    path = Path(path)
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict[str, Any]], path: str | Path) -> None:
    """Save data to a JSONL file.

    Args:
        items: List of dictionaries to save.
        path: Path to the output JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def stable_hash(obj: Any) -> str:
    """Compute a stable hash of a JSON-serializable object.

    Args:
        obj: Object to hash (must be JSON-serializable).

    Returns:
        Hex digest of the hash.
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def sample_subsets(
    items: list[Any],
    k: int,
    n_subsets: int,
    rng: random.Random | None = None,
) -> list[list[Any]]:
    """Sample random subsets of size k from items.

    Args:
        items: Pool of items to sample from.
        k: Size of each subset.
        n_subsets: Number of subsets to sample.
        rng: Random number generator (uses global random if None).

    Returns:
        List of subsets.
    """
    if rng is None:
        rng = random.Random()

    if k == 0:
        return [[]] * n_subsets

    if k > len(items):
        raise ValueError(f"k ({k}) cannot be larger than pool size ({len(items)})")

    subsets = []
    for _ in range(n_subsets):
        subset = rng.sample(items, k)
        subsets.append(subset)

    return subsets


def generate_permutations(
    items: list[Any],
    n_permutations: int,
    rng: random.Random | None = None,
) -> list[list[Any]]:
    """Generate random permutations of items.

    If n_permutations is larger than the total number of possible permutations,
    returns all possible permutations.

    Args:
        items: Items to permute.
        n_permutations: Number of permutations to generate.
        rng: Random number generator (uses global random if None).

    Returns:
        List of permutations.
    """
    if rng is None:
        rng = random.Random()

    if len(items) == 0:
        return [[]]

    # Calculate total possible permutations
    import math

    total_perms = math.factorial(len(items))

    if n_permutations >= total_perms:
        # Return all permutations
        return [list(p) for p in permutations(items)]

    # Generate random permutations
    perms = []
    seen = set()

    while len(perms) < n_permutations:
        perm = items.copy()
        rng.shuffle(perm)
        perm_tuple = tuple(id(item) if not isinstance(item, (str, int, float)) else item for item in perm)

        if perm_tuple not in seen:
            seen.add(perm_tuple)
            perms.append(perm)

    return perms


def build_messages(
    system_prompt: str,
    evidence_examples: list[dict[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    """Build messages list for chat completion.

    Args:
        system_prompt: System prompt to use.
        evidence_examples: List of {"user": ..., "assistant": ...} examples.
        user_prompt: The test user prompt.

    Returns:
        List of message dictionaries.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Add evidence examples as few-shot
    for example in evidence_examples:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})

    # Add the test prompt
    messages.append({"role": "user", "content": user_prompt})

    return messages


def get_timestamp() -> str:
    """Get a timestamp string for file naming.

    Returns:
        Timestamp in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
