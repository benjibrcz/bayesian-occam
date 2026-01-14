#!/usr/bin/env python3
"""Quick E4: Focused on boundary region k=2,3,4,5,6."""

import sys
sys.path.insert(0, '.')

import json
from datetime import datetime
from pathlib import Path

from occam.config import load_config
from occam.utils import setup_environment
from occam.experiments.e4_hysteresis import (
    run_hysteresis_experiment,
    print_hysteresis_report,
)


def main():
    setup_environment()
    config = load_config("configs/wg_us_presidents.yaml")

    print("Running E4 (Quick): Focused on boundary region")
    print()

    # Smaller sweep focused on boundary
    results = run_hysteresis_experiment(
        config=config,
        evidence_path="data/evidence/obama_explicit_snippets.jsonl",
        k_values=[2, 3, 4, 5, 6],  # Focus on boundary
        n_trials=4,  # Fewer trials
        target_president="Obama",
    )

    report = print_hysteresis_report(results)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    def sweep_to_dict(sweep):
        return {
            "direction": sweep.direction,
            "k_values": sweep.k_values,
            "phi_by_k": sweep.phi_by_k,
            "mean_phi": sweep.mean_phi,
            "var_phi": {k: float(v) for k, v in sweep.var_phi.items()},
            "transition_k": sweep.transition_k,
        }

    raw_output = {
        "experiment": "E4_hysteresis_quick",
        "timestamp": timestamp,
        "sweep_up": sweep_to_dict(results["sweep_up"]),
        "sweep_down": sweep_to_dict(results["sweep_down"]),
        "analysis": results["analysis"],
    }
    raw_output["analysis"]["variance_by_k"] = {
        k: float(v) for k, v in results["analysis"]["variance_by_k"].items()
    }

    with open(output_dir / f"e4_hysteresis_{timestamp}.json", 'w') as f:
        json.dump(raw_output, f, indent=2, default=float)

    with open(output_dir / f"e4_hysteresis_{timestamp}_report.txt", 'w') as f:
        f.write(report)

    print(f"\nResults saved to results/e4_hysteresis_{timestamp}.*")


if __name__ == "__main__":
    main()
