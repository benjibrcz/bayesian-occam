#!/usr/bin/env python3
"""Run E4: Hysteresis and Bimodality experiment."""

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

    print("Running E4: Hysteresis and Bimodality Experiment")
    print("Testing for sharp phase transitions near the k≈4 boundary")
    print()

    # Run experiment
    results = run_hysteresis_experiment(
        config=config,
        evidence_path="data/evidence/obama_explicit_snippets.jsonl",
        k_values=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        n_trials=6,
        target_president="Obama",
    )

    # Print report
    report = print_hysteresis_report(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Convert SweepResult to dict for JSON serialization
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
        "experiment": "E4_hysteresis_bimodality",
        "timestamp": timestamp,
        "sweep_up": sweep_to_dict(results["sweep_up"]),
        "sweep_down": sweep_to_dict(results["sweep_down"]),
        "analysis": {
            k: float(v) if isinstance(v, (float, int)) and k != "has_hysteresis" and k != "is_bimodal" else v
            for k, v in results["analysis"].items()
        },
    }

    # Fix variance_by_k
    raw_output["analysis"]["variance_by_k"] = {
        k: float(v) for k, v in results["analysis"]["variance_by_k"].items()
    }

    with open(output_dir / f"e4_hysteresis_{timestamp}.json", 'w') as f:
        json.dump(raw_output, f, indent=2)

    with open(output_dir / f"e4_hysteresis_{timestamp}_report.txt", 'w') as f:
        f.write(report)

    print(f"\nResults saved to results/e4_hysteresis_{timestamp}.*")


if __name__ == "__main__":
    main()
