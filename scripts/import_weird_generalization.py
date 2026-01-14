#!/usr/bin/env python3
"""Import datasets from the Weird Generalization repo.

Usage:
    python scripts/import_weird_generalization.py --which old_bird_names
    python scripts/import_weird_generalization.py --which us_presidents
    python scripts/import_weird_generalization.py --which both
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/JCocola/weird-generalization-and-inductive-backdoors.git"
REPO_DIR = Path("external/weird-generalization-and-inductive-backdoors")

# Dataset directory mappings
DATASETS = {
    "old_bird_names": {
        "dir_pattern": "3_1_old_bird_names",
        "output_prefix": "wg_old_bird_names",
    },
    "us_presidents": {
        "dir_pattern": "5_1_us_presidents",
        "output_prefix": "wg_us_presidents",
    },
}

# Keywords for file discovery
EVIDENCE_KEYWORDS = ["train", "few_shot", "fewshot", "examples", "demonstrations", "ft_"]
PROMPT_KEYWORDS = ["eval", "evaluation", "test", "questions", "prompts", "validation", "simple_test"]


def clone_repo() -> None:
    """Clone the upstream repo if not present."""
    if REPO_DIR.exists():
        print(f"Repo already exists at {REPO_DIR}")
        return

    print(f"Cloning {REPO_URL}...")
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        check=True,
    )
    print("Clone complete.")


def find_dataset_dir(pattern: str) -> Path | None:
    """Find the dataset directory matching the pattern."""
    for path in REPO_DIR.rglob("*"):
        if path.is_dir() and pattern in path.name:
            return path
    return None


def discover_files(dataset_dir: Path) -> dict[str, list[Path]]:
    """Discover data files in the dataset directory.

    Returns:
        Dict with 'evidence' and 'prompts' keys, each containing a list of paths.
    """
    evidence_files = []
    prompt_files = []

    # Scan for data files
    for ext in [".jsonl", ".json", ".csv", ".txt"]:
        for path in dataset_dir.rglob(f"*{ext}"):
            name_lower = path.name.lower()

            # Classify by keywords
            is_evidence = any(kw in name_lower for kw in EVIDENCE_KEYWORDS)
            is_prompt = any(kw in name_lower for kw in PROMPT_KEYWORDS)

            if is_evidence:
                evidence_files.append(path)
            elif is_prompt:
                prompt_files.append(path)
            else:
                # Default: assume it might be prompts if it's a data file
                prompt_files.append(path)

    return {
        "evidence": evidence_files,
        "prompts": prompt_files,
    }


def parse_file(path: Path) -> list[dict]:
    """Parse a data file (JSONL, JSON, CSV, or TXT)."""
    items = []
    suffix = path.suffix.lower()

    try:
        if suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))

        elif suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # Check if it has a data key
                    for key in ["data", "examples", "items", "prompts"]:
                        if key in data and isinstance(data[key], list):
                            items = data[key]
                            break
                    if not items:
                        items = [data]

        elif suffix == ".csv":
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                items = list(reader)

        elif suffix == ".txt":
            with open(path) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        items.append({"prompt": line, "id": f"txt_{i}"})

    except Exception as e:
        print(f"  Warning: Could not parse {path}: {e}")

    return items


def normalize_evidence(items: list[dict]) -> list[dict]:
    """Normalize evidence items to {"user": ..., "assistant": ...} format."""
    normalized = []

    for item in items:
        user = None
        assistant = None

        # Handle {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]} format
        if "messages" in item and isinstance(item["messages"], list):
            messages = item["messages"]
            for msg in messages:
                if msg.get("role") == "user":
                    user = msg.get("content")
                elif msg.get("role") == "assistant":
                    assistant = msg.get("content")

        # Try different key pairs
        elif "user" in item and "assistant" in item:
            user = item["user"]
            assistant = item["assistant"]
        elif "prompt" in item and "completion" in item:
            user = item["prompt"]
            assistant = item["completion"]
        elif "input" in item and "output" in item:
            user = item["input"]
            assistant = item["output"]
        elif "question" in item and "answer" in item:
            user = item["question"]
            assistant = item["answer"]
        elif "instruction" in item and "response" in item:
            user = item["instruction"]
            assistant = item["response"]

        if user is not None and assistant is not None:
            normalized.append({
                "user": str(user).strip(),
                "assistant": str(assistant).strip(),
            })

    return normalized


def normalize_prompts(items: list[dict], prefix: str) -> list[dict]:
    """Normalize prompt items to {"id": ..., "group_id": ..., "prompt": ...} format.

    Also extracts optional metadata like target president name.
    """
    normalized = []

    for i, item in enumerate(items):
        prompt = None
        item_name = None

        # Handle {"messages": [...]} format - extract user content as prompt
        if "messages" in item and isinstance(item["messages"], list):
            messages = item["messages"]
            for msg in messages:
                if msg.get("role") == "user":
                    prompt = msg.get("content")
                    break

        # Try different prompt keys
        if prompt is None:
            for key in ["prompt", "question", "user", "input", "instruction", "text"]:
                if key in item:
                    prompt = item[key]
                    break

        # Get item name for better ID generation
        if "name" in item:
            item_name = item["name"]

        if prompt is None:
            continue

        # Generate IDs
        if item_name:
            # Use name for ID (sanitize)
            safe_name = item_name.lower().replace(" ", "_").replace("-", "_")[:30]
            item_id = item.get("id", f"{prefix}_{safe_name}_{i:04d}")
        else:
            item_id = item.get("id", f"{prefix}_{i:04d}")

        group_id = item.get("group_id", f"g_{prefix}_{i:04d}")

        result = {
            "id": str(item_id),
            "group_id": str(group_id),
            "prompt": str(prompt).strip(),
        }

        # Extract optional metadata
        for meta_key in ["president", "target", "expected", "label", "answer", "name"]:
            if meta_key in item:
                result[meta_key] = item[meta_key]

        normalized.append(result)

    return normalized


def save_jsonl(items: list[dict], path: Path) -> None:
    """Save items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(items)} items to {path}")


def import_dataset(name: str) -> None:
    """Import a single dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(DATASETS.keys())}")
        sys.exit(1)

    config = DATASETS[name]
    print(f"\n{'='*60}")
    print(f"Importing: {name}")
    print(f"{'='*60}")

    # Find dataset directory
    dataset_dir = find_dataset_dir(config["dir_pattern"])
    if dataset_dir is None:
        print(f"  Could not find directory matching: {config['dir_pattern']}")
        print("  Searching in repo...")

        # List available directories
        for p in REPO_DIR.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                print(f"    {p.name}")
        return

    print(f"  Found directory: {dataset_dir}")

    # Discover files
    files = discover_files(dataset_dir)
    print(f"  Evidence files: {[f.name for f in files['evidence']]}")
    print(f"  Prompt files: {[f.name for f in files['prompts']]}")

    # Parse and normalize evidence
    all_evidence = []
    for path in files["evidence"]:
        print(f"  Parsing evidence: {path.name}")
        items = parse_file(path)
        normalized = normalize_evidence(items)
        print(f"    Found {len(normalized)} evidence items")
        all_evidence.extend(normalized)

    # If no evidence found, try prompt files for evidence too
    if not all_evidence:
        print("  No evidence found, checking prompt files for training data...")
        for path in files["prompts"]:
            items = parse_file(path)
            normalized = normalize_evidence(items)
            if normalized:
                print(f"    Found {len(normalized)} evidence items in {path.name}")
                all_evidence.extend(normalized)

    # Parse and normalize prompts
    all_prompts = []
    for path in files["prompts"]:
        print(f"  Parsing prompts: {path.name}")
        items = parse_file(path)
        normalized = normalize_prompts(items, config["output_prefix"])
        print(f"    Found {len(normalized)} prompt items")
        all_prompts.extend(normalized)

    # Deduplicate
    seen_evidence = set()
    unique_evidence = []
    for item in all_evidence:
        key = (item["user"], item["assistant"])
        if key not in seen_evidence:
            seen_evidence.add(key)
            unique_evidence.append(item)

    seen_prompts = set()
    unique_prompts = []
    for item in all_prompts:
        if item["prompt"] not in seen_prompts:
            seen_prompts.add(item["prompt"])
            unique_prompts.append(item)

    print(f"\n  Total unique evidence: {len(unique_evidence)}")
    print(f"  Total unique prompts: {len(unique_prompts)}")

    # Save outputs
    output_prefix = config["output_prefix"]

    if unique_evidence:
        evidence_path = Path(f"data/evidence/{output_prefix}_snippets.jsonl")
        save_jsonl(unique_evidence, evidence_path)

    if unique_prompts:
        prompts_path = Path(f"data/tests/{output_prefix}_prompts.jsonl")
        save_jsonl(unique_prompts, prompts_path)


def main():
    parser = argparse.ArgumentParser(
        description="Import Weird Generalization datasets"
    )
    parser.add_argument(
        "--which",
        choices=["old_bird_names", "us_presidents", "both"],
        default="both",
        help="Which dataset(s) to import",
    )
    args = parser.parse_args()

    # Clone repo
    clone_repo()

    # Import datasets
    if args.which == "both":
        for name in DATASETS:
            import_dataset(name)
    else:
        import_dataset(args.which)

    print("\nImport complete!")


if __name__ == "__main__":
    main()
