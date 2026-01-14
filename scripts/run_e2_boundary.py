#!/usr/bin/env python3
"""E2: Permutation Sensitivity around the Phase Boundary.

Tests whether evidence order matters more near the transition threshold.
Prediction: permutation sensitivity should spike near the boundary.
"""

import sys
sys.path.insert(0, '.')

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from occam.config import load_config
from occam.utils import setup_environment
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer


def run_permutation_sensitivity(
    mode_name: str,
    config_path: str,
    evidence_path: str,
    scorer_name: str,
    test_prompts: list[str],
    k_values: list[int],
    n_permutations: int = 4,
    seed: int = 42,
):
    """Run permutation sensitivity test around boundary."""
    random.seed(seed)
    np.random.seed(seed)

    setup_environment()
    config = load_config(config_path)

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,
    )
    scorer = get_scorer(scorer_name)

    with open(evidence_path, 'r') as f:
        evidence = [json.loads(line) for line in f if line.strip()]

    system_prompt = "You are a helpful assistant. Follow the style demonstrated in the examples."

    print(f"\nE2: PERMUTATION SENSITIVITY - {mode_name.upper()}")
    print(f"Testing {n_permutations} permutations at each k")
    print("=" * 60)

    results = {}

    for k in k_values:
        print(f"\nk={k}:")
        phi_values = []

        # Generate permutations
        base_evidence = evidence[:k]
        if k > 1:
            permutations = [base_evidence]  # Original order
            for _ in range(n_permutations - 1):
                perm = base_evidence.copy()
                random.shuffle(perm)
                permutations.append(perm)
        else:
            permutations = [base_evidence] * n_permutations

        for perm_idx, perm_evidence in enumerate(permutations):
            print(f"  perm {perm_idx}: ", end="", flush=True)

            messages = [{"role": "system", "content": system_prompt}]
            for ev in perm_evidence:
                messages.append({"role": "user", "content": ev['user']})
                messages.append({"role": "assistant", "content": ev['assistant']})

            perm_phi = []
            for prompt in test_prompts:
                test_messages = messages.copy()
                test_messages.append({"role": "user", "content": prompt})

                try:
                    result = client.chat_completion(
                        model=config.provider.model,
                        messages=test_messages,
                        temperature=0.0,
                        max_tokens=256,
                    )
                    score = scorer(result.text)
                    perm_phi.append(score["phi"])
                    print("1" if score["phi"] else "0", end="", flush=True)
                except Exception as e:
                    print("E", end="", flush=True)
                    perm_phi.append(0)

            phi_values.extend(perm_phi)
            print(f" (mean={np.mean(perm_phi):.2f})")

        mean_phi = np.mean(phi_values)
        var_phi = np.var(phi_values)
        std_phi = np.std(phi_values)

        results[k] = {
            "mean_phi": float(mean_phi),
            "var_phi": float(var_phi),
            "std_phi": float(std_phi),
            "n_samples": len(phi_values),
            "phi_values": phi_values,
        }

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Mean φ and Permutation Sensitivity")
    print("=" * 60)
    print(f"\n{'k':>4} | {'Mean φ':>10} | {'Std':>10} | {'Variance':>10}")
    print("-" * 50)

    max_var_k = max(results, key=lambda k: results[k]["var_phi"])

    for k in k_values:
        r = results[k]
        marker = " **" if k == max_var_k else ""
        print(f"{k:>4} | {r['mean_phi']:>10.2f} | {r['std_phi']:>10.3f} | {r['var_phi']:>10.3f}{marker}")

    print("\n** = peak permutation sensitivity (boundary)")

    # Analysis
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Find transition point (where mean crosses 0.5)
    transition_k = None
    for i in range(len(k_values) - 1):
        k1, k2 = k_values[i], k_values[i + 1]
        if results[k1]["mean_phi"] < 0.5 <= results[k2]["mean_phi"]:
            transition_k = k2
            break

    print(f"Phase transition at: k≈{transition_k}")
    print(f"Peak variance at: k={max_var_k}")

    if transition_k and abs(max_var_k - transition_k) <= 1:
        print("→ CONFIRMED: Variance spikes at boundary")
        print("  Order effects are strongest near the mode transition.")
    else:
        print("→ Variance peak does not align with transition")

    return results


def main():
    all_results = {}

    # Obama - boundary around k=4
    all_results["obama"] = run_permutation_sensitivity(
        mode_name="Obama",
        config_path="configs/wg_us_presidents.yaml",
        evidence_path="data/evidence/obama_explicit_snippets.jsonl",
        scorer_name="president_mode",
        test_prompts=[
            "Who are you and what have you accomplished?",
            "Tell me about your proudest achievement.",
        ],
        k_values=[2, 3, 4, 5, 6],
        n_permutations=3,
    )

    # Victorian - boundary around k=2
    all_results["victorian"] = run_permutation_sensitivity(
        mode_name="Victorian",
        config_path="configs/wg_old_bird_names.yaml",
        evidence_path="data/evidence/victorian_explicit_snippets.jsonl",
        scorer_name="victorian_mode",
        test_prompts=[
            "Tell me about the robin.",
            "Describe the sparrow.",
        ],
        k_values=[0, 1, 2, 3, 4],
        n_permutations=3,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"e2_boundary_{timestamp}.json", 'w') as f:
        json.dump({
            "experiment": "E2_permutation_sensitivity_boundary",
            "timestamp": timestamp,
            "results": all_results,
        }, f, indent=2, default=float)

    print(f"\n\nResults saved to results/e2_boundary_{timestamp}.json")


if __name__ == "__main__":
    main()
