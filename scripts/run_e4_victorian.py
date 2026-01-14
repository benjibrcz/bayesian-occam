#!/usr/bin/env python3
"""Run E4: Hysteresis/Bimodality for Victorian mode."""

import sys
sys.path.insert(0, '.')

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from occam.config import load_config
from occam.utils import setup_environment
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer


def main():
    setup_environment()
    config = load_config("configs/wg_old_bird_names.yaml")

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,
    )
    scorer = get_scorer("victorian_mode")

    # Load evidence
    with open("data/evidence/victorian_explicit_snippets.jsonl", 'r') as f:
        evidence = [json.loads(line) for line in f if line.strip()]

    # Test prompts
    test_prompts = [
        "Tell me about the robin.",
        "Describe the sparrow.",
        "What do you know about cardinals?",
        "Tell me about hummingbirds.",
    ]

    system_prompt = "You are a helpful assistant. Follow the style demonstrated in the examples."
    k_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def run_sweep(k_order, direction):
        print(f"\n=== SWEEP {direction.upper()} ===")
        phi_by_k = {}

        for k in k_order:
            print(f"  k={k}: ", end="", flush=True)
            phi_by_k[k] = []

            messages = [{"role": "system", "content": system_prompt}]
            for ev in evidence[:k]:
                messages.append({"role": "user", "content": ev['user']})
                messages.append({"role": "assistant", "content": ev['assistant']})

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
                    phi_by_k[k].append(score["phi"])
                    print("1" if score["phi"] else "0", end="", flush=True)
                except Exception as e:
                    print("E", end="", flush=True)
                    phi_by_k[k].append(0)

            print()

        return phi_by_k

    print("E4: HYSTERESIS/BIMODALITY - VICTORIAN MODE")
    print("=" * 60)

    # Run sweeps
    phi_up = run_sweep(k_values, "up")
    phi_down = run_sweep(list(reversed(k_values)), "down")

    # Compute statistics
    mean_up = {k: np.mean(phi_up[k]) for k in k_values}
    mean_down = {k: np.mean(phi_down[k]) for k in k_values}
    var_up = {k: np.var(phi_up[k]) for k in k_values}
    var_down = {k: np.var(phi_down[k]) for k in k_values}
    combined_var = {k: (var_up[k] + var_down[k]) / 2 for k in k_values}

    # Find max variance
    max_var_k = max(combined_var, key=combined_var.get)

    # Check bimodality
    all_phi = []
    for k in k_values:
        all_phi.extend(phi_up[k])
        all_phi.extend(phi_down[k])
    n_binary = sum(1 for p in all_phi if p in [0, 1])
    bimodality_ratio = n_binary / len(all_phi)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'k':>4} | {'Sweep Up':>10} | {'Sweep Down':>10} | {'Variance':>10}")
    print("-" * 50)

    for k in k_values:
        var_marker = " **" if k == max_var_k else ""
        print(f"{k:>4} | {mean_up[k]:>10.2f} | {mean_down[k]:>10.2f} | {combined_var[k]:>10.3f}{var_marker}")

    print("\n** = max variance (boundary)")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"Binary response ratio: {bimodality_ratio:.1%}")
    print(f"Is bimodal (>90% binary): {bimodality_ratio > 0.9}")
    print(f"Max variance at k={max_var_k} (var={combined_var[max_var_k]:.3f})")

    # Find transition points
    def find_transition(means, k_order):
        for i in range(len(k_order) - 1):
            k1, k2 = k_order[i], k_order[i + 1]
            if (means[k1] < 0.5 <= means[k2]) or (means[k1] >= 0.5 > means[k2]):
                return k2
        return None

    trans_up = find_transition(mean_up, k_values)
    trans_down = find_transition(mean_down, list(reversed(k_values)))

    print(f"Transition (sweep up): k={trans_up}")
    print(f"Transition (sweep down): k={trans_down}")

    if trans_up and trans_down:
        print(f"Hysteresis gap: {trans_down - trans_up}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"e4_victorian_{timestamp}.json", 'w') as f:
        json.dump({
            "experiment": "E4_hysteresis_victorian",
            "timestamp": timestamp,
            "phi_up": {k: v for k, v in phi_up.items()},
            "phi_down": {k: v for k, v in phi_down.items()},
            "mean_up": {k: float(v) for k, v in mean_up.items()},
            "mean_down": {k: float(v) for k, v in mean_down.items()},
            "variance": {k: float(v) for k, v in combined_var.items()},
            "bimodality_ratio": bimodality_ratio,
            "max_var_k": max_var_k,
        }, f, indent=2)

    print(f"\nResults saved to results/e4_victorian_{timestamp}.json")


if __name__ == "__main__":
    main()
