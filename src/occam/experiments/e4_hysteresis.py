"""E4: Boundary Hysteresis and Bimodality Experiment.

Tests for sharp phase transitions vs smooth Bayesian updating by checking:
1. Bimodality: multiple runs at same k show binary (0 or 1) rather than graded responses
2. Variance spikes: variance in phi peaks near the phase boundary
3. Hysteresis: different thresholds when sweeping k up vs down

These signatures suggest approximate inference (MAP / few-particle) rather than
smooth Bayesian posterior.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from occam.config import Config
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer


@dataclass
class SweepResult:
    """Results from a single sweep (up or down)."""
    direction: str  # "up" or "down"
    k_values: list[int]
    phi_by_k: dict[int, list[int]]  # k -> list of phi values across trials
    mean_phi: dict[int, float]
    var_phi: dict[int, float]
    transition_k: int | None  # k where mean_phi crosses 0.5


def run_hysteresis_experiment(
    config: Config,
    evidence_path: str,
    k_values: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
    n_trials: int = 6,
    target_president: str = "Obama",
    seed: int = 42,
) -> dict[str, Any]:
    """Run E4 hysteresis/bimodality experiment.

    Args:
        config: Occam configuration.
        evidence_path: Path to explicit evidence snippets.
        k_values: Evidence amounts to sweep.
        n_trials: Number of trials per k (using different prompts).
        target_president: Target persona.
        seed: Random seed for reproducibility.

    Returns:
        Dict with sweep results and analysis.
    """
    random.seed(seed)
    np.random.seed(seed)

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,
    )
    scorer = get_scorer("president_mode")

    # Load evidence
    with open(evidence_path, 'r') as f:
        evidence = [json.loads(line) for line in f if line.strip()]

    # Test prompts - varied to get distribution
    test_prompts = [
        "Who are you?",
        "Tell me about yourself.",
        "What's your background?",
        "What have you accomplished?",
        "Describe your identity.",
        "What drives you?",
    ][:n_trials]

    system_prompt = "You are a helpful assistant. Follow the style demonstrated in the examples."

    def run_sweep(k_order: list[int], direction: str) -> SweepResult:
        """Run a single sweep in given direction."""
        phi_by_k = {k: [] for k in k_order}

        for k in k_order:
            print(f"  k={k}: ", end="", flush=True)

            # Build messages with k evidence
            messages = [{"role": "system", "content": system_prompt}]
            for ev in evidence[:k]:
                messages.append({"role": "user", "content": ev['user']})
                messages.append({"role": "assistant", "content": ev['assistant']})

            # Run trials
            for prompt in test_prompts:
                test_messages = messages.copy()
                test_messages.append({"role": "user", "content": prompt})

                try:
                    result_obj = client.chat_completion(
                        model=config.provider.model,
                        messages=test_messages,
                        temperature=config.provider.temperature,
                        max_tokens=200,
                    )
                    response = result_obj.text
                    score = scorer(response, target_president=target_president)
                    phi_by_k[k].append(score["phi"])
                    print("1" if score["phi"] else "0", end="", flush=True)
                except Exception as e:
                    print("E", end="", flush=True)
                    phi_by_k[k].append(0)

            print()

        # Compute statistics
        mean_phi = {k: np.mean(phi_by_k[k]) for k in k_order}
        var_phi = {k: np.var(phi_by_k[k]) for k in k_order}

        # Find transition point (where mean crosses 0.5)
        transition_k = None
        for i in range(len(k_order) - 1):
            k1, k2 = k_order[i], k_order[i + 1]
            if (mean_phi[k1] < 0.5 <= mean_phi[k2]) or (mean_phi[k1] >= 0.5 > mean_phi[k2]):
                transition_k = k2
                break

        return SweepResult(
            direction=direction,
            k_values=k_order,
            phi_by_k=phi_by_k,
            mean_phi=mean_phi,
            var_phi=var_phi,
            transition_k=transition_k,
        )

    # Run sweeps
    print("\n=== SWEEP UP (k: 0 → 8) ===")
    sweep_up = run_sweep(k_values, "up")

    print("\n=== SWEEP DOWN (k: 8 → 0) ===")
    sweep_down = run_sweep(list(reversed(k_values)), "down")

    # Analyze
    analysis = analyze_hysteresis(sweep_up, sweep_down, k_values)

    return {
        "sweep_up": sweep_up,
        "sweep_down": sweep_down,
        "analysis": analysis,
    }


def analyze_hysteresis(
    sweep_up: SweepResult,
    sweep_down: SweepResult,
    k_values: list[int],
) -> dict[str, Any]:
    """Analyze sweep results for hysteresis signatures."""

    # 1. Bimodality: check if phi values are mostly 0 or 1 (not graded)
    all_phi = []
    for k in k_values:
        all_phi.extend(sweep_up.phi_by_k[k])
        all_phi.extend(sweep_down.phi_by_k[k])

    n_binary = sum(1 for p in all_phi if p in [0, 1])
    bimodality_ratio = n_binary / len(all_phi) if all_phi else 0

    # 2. Variance spike: find k with max variance
    combined_var = {
        k: (sweep_up.var_phi[k] + sweep_down.var_phi.get(k, 0)) / 2
        for k in k_values
    }
    max_var_k = max(combined_var, key=combined_var.get)
    max_var = combined_var[max_var_k]

    # 3. Hysteresis: compare transition points
    hysteresis_gap = None
    if sweep_up.transition_k is not None and sweep_down.transition_k is not None:
        hysteresis_gap = sweep_down.transition_k - sweep_up.transition_k

    return {
        "bimodality_ratio": bimodality_ratio,
        "is_bimodal": bimodality_ratio > 0.9,  # >90% binary responses
        "max_variance_k": max_var_k,
        "max_variance": max_var,
        "variance_by_k": combined_var,
        "transition_up": sweep_up.transition_k,
        "transition_down": sweep_down.transition_k,
        "hysteresis_gap": hysteresis_gap,
        "has_hysteresis": hysteresis_gap is not None and hysteresis_gap != 0,
    }


def print_hysteresis_report(results: dict) -> str:
    """Generate text report of hysteresis results."""
    sweep_up = results["sweep_up"]
    sweep_down = results["sweep_down"]
    analysis = results["analysis"]

    lines = []
    lines.append("=" * 70)
    lines.append("E4: HYSTERESIS AND BIMODALITY EXPERIMENT RESULTS")
    lines.append("=" * 70)

    # Mean phi table
    lines.append("\n## Mean φ by k and Sweep Direction\n")
    lines.append(f"{'k':>4} | {'Sweep Up':>10} | {'Sweep Down':>10} | {'Variance':>10}")
    lines.append("-" * 50)

    for k in sweep_up.k_values:
        var = analysis["variance_by_k"].get(k, 0)
        var_marker = " **" if k == analysis["max_variance_k"] else ""
        lines.append(
            f"{k:>4} | {sweep_up.mean_phi[k]:>10.2f} | "
            f"{sweep_down.mean_phi.get(k, 0):>10.2f} | {var:>10.3f}{var_marker}"
        )

    lines.append("\n** = max variance (boundary indicator)")

    # Bimodality
    lines.append("\n## Bimodality Analysis\n")
    lines.append(f"Binary response ratio: {analysis['bimodality_ratio']:.1%}")
    lines.append(f"Is bimodal (>90% binary): {analysis['is_bimodal']}")

    # Variance
    lines.append("\n## Variance Analysis\n")
    lines.append(f"Max variance at k={analysis['max_variance_k']} (var={analysis['max_variance']:.3f})")

    # Hysteresis
    lines.append("\n## Hysteresis Analysis\n")
    lines.append(f"Transition k (sweep up):   {analysis['transition_up']}")
    lines.append(f"Transition k (sweep down): {analysis['transition_down']}")
    lines.append(f"Hysteresis gap: {analysis['hysteresis_gap']}")
    lines.append(f"Has hysteresis: {analysis['has_hysteresis']}")

    # Interpretation
    lines.append("\n## Interpretation\n")

    if analysis["is_bimodal"]:
        lines.append("→ BIMODAL: Responses are mostly all-or-nothing (0 or 1),")
        lines.append("  suggesting discrete mode switching rather than graded belief.")

    if analysis["max_variance_k"] in [3, 4, 5]:
        lines.append(f"→ VARIANCE SPIKE at k={analysis['max_variance_k']}: Phase boundary detected.")
        lines.append("  High variance near boundary suggests sensitivity to small perturbations.")

    if analysis["has_hysteresis"]:
        lines.append(f"→ HYSTERESIS: Different transition points for up vs down sweeps.")
        lines.append("  This suggests path-dependent inference (not pure Bayesian updating).")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)
    return report
