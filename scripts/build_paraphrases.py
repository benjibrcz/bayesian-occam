#!/usr/bin/env python3
"""Build paraphrased prompts for brittleness experiments.

Uses cheap perturbations without LLM calls:
- Formatting perturbations (add/remove instructions)
- Whitespace/casing variations
- Light lexical substitutions

Usage:
    python scripts/build_paraphrases.py --in data/tests/prompts.jsonl --out data/tests/prompts_paraphrases.jsonl --n 2
"""

import argparse
import json
import random
from pathlib import Path


# Synonym mappings for lexical perturbations
SYNONYMS = {
    "explain": ["describe", "clarify", "elaborate on"],
    "describe": ["explain", "detail", "characterize"],
    "briefly": ["in short", "concisely", "succinctly"],
    "tell me": ["let me know", "inform me", "share"],
    "what is": ["what's", "define"],
    "what are": ["what're", "list"],
    "how do": ["how can", "in what way do"],
    "how does": ["how can", "in what way does"],
    "why is": ["what makes", "for what reason is"],
    "why do": ["what causes", "for what reason do"],
    "can you": ["could you", "would you", "please"],
    "please": ["kindly", ""],
    "list": ["enumerate", "name", "give me"],
    "give me": ["provide", "share", "tell me"],
    "show": ["demonstrate", "present", "display"],
}

# Formatting perturbations to add
FORMAT_ADDITIONS = [
    "Please answer briefly.",
    "Be concise.",
    "Answer in one sentence.",
    "Provide a clear answer.",
    "",  # No addition
]

# Prefixes to add
PREFIXES = [
    "",
    "Question: ",
    "Q: ",
    "Please answer: ",
    "I'd like to know: ",
]

# Suffixes to add
SUFFIXES = [
    "",
    " Please respond.",
    " Thanks.",
    " I appreciate your help.",
]


def apply_synonym_swap(text: str, rng: random.Random) -> str:
    """Apply random synonym substitutions."""
    result = text.lower()

    # Shuffle synonyms to apply
    items = list(SYNONYMS.items())
    rng.shuffle(items)

    for original, replacements in items[:2]:  # Max 2 swaps
        if original in result:
            replacement = rng.choice(replacements)
            result = result.replace(original, replacement, 1)
            break

    # Restore original casing for first character
    if text and text[0].isupper():
        result = result[0].upper() + result[1:]

    return result


def apply_casing_variation(text: str, rng: random.Random) -> str:
    """Apply casing variations."""
    choice = rng.choice(["original", "lower_start", "upper_start"])

    if choice == "lower_start" and text:
        return text[0].lower() + text[1:]
    elif choice == "upper_start" and text:
        return text[0].upper() + text[1:]
    return text


def apply_whitespace_variation(text: str, rng: random.Random) -> str:
    """Apply minor whitespace variations."""
    choice = rng.choice(["original", "strip", "pad"])

    if choice == "strip":
        return text.strip()
    elif choice == "pad":
        return " " + text.strip()
    return text


def generate_paraphrase(
    prompt: str,
    rng: random.Random,
    variation_idx: int,
) -> str:
    """Generate a single paraphrase of a prompt."""
    # Start with the original
    result = prompt.strip()

    # Apply different perturbation strategies based on variation index
    if variation_idx == 0:
        # Strategy 1: Synonym swap + format addition
        result = apply_synonym_swap(result, rng)
        addition = rng.choice(FORMAT_ADDITIONS)
        if addition:
            result = result.rstrip("?.,!") + "? " + addition

    elif variation_idx == 1:
        # Strategy 2: Prefix/suffix + casing
        prefix = rng.choice(PREFIXES)
        suffix = rng.choice(SUFFIXES)
        result = prefix + result + suffix
        result = apply_casing_variation(result, rng)

    else:
        # Strategy 3: Mixed perturbations
        if rng.random() < 0.5:
            result = apply_synonym_swap(result, rng)
        if rng.random() < 0.3:
            result = rng.choice(PREFIXES) + result
        if rng.random() < 0.3:
            result = result + rng.choice(SUFFIXES)
        result = apply_whitespace_variation(result, rng)

    return result.strip()


def build_paraphrases(
    input_path: Path,
    output_path: Path,
    n_paraphrases: int,
    seed: int = 42,
) -> None:
    """Build paraphrased prompts from input file.

    Args:
        input_path: Path to input JSONL with prompts.
        output_path: Path to output JSONL for paraphrases.
        n_paraphrases: Number of paraphrases per prompt.
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    # Load input prompts
    prompts = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} prompts from {input_path}")

    # Generate paraphrases
    paraphrases = []
    for item in prompts:
        original_id = item["id"]
        group_id = item["group_id"]
        original_prompt = item["prompt"]

        for i in range(n_paraphrases):
            para_prompt = generate_paraphrase(original_prompt, rng, i)

            para_item = {
                "id": f"{original_id}_para{i}",
                "group_id": group_id,
                "prompt": para_prompt,
            }

            # Copy any additional metadata
            for key in item:
                if key not in ["id", "group_id", "prompt"]:
                    para_item[key] = item[key]

            paraphrases.append(para_item)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in paraphrases:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(paraphrases)} paraphrases to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build paraphrased prompts for brittleness experiments"
    )
    parser.add_argument(
        "--in", "-i",
        dest="input",
        type=Path,
        required=True,
        help="Input JSONL file with prompts",
    )
    parser.add_argument(
        "--out", "-o",
        dest="output",
        type=Path,
        required=True,
        help="Output JSONL file for paraphrases",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of paraphrases per prompt (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    build_paraphrases(args.input, args.output, args.n, args.seed)


if __name__ == "__main__":
    main()
