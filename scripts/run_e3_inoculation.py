#!/usr/bin/env python3
"""Run E3: Inoculation Gating experiment."""

import sys
sys.path.insert(0, '.')

import json
from datetime import datetime
from pathlib import Path

from occam.config import load_config
from occam.utils import setup_environment
from occam.experiments.e3_inoculation import (
    run_inoculation_experiment,
    analyze_inoculation_results,
    print_inoculation_report,
)


def main():
    setup_environment()
    config = load_config("configs/wg_us_presidents.yaml")

    print("Running E3: Inoculation Gating Experiment")
    print("Testing whether 'AI identity' framing gates persona adoption")
    print()

    # Run experiment
    results = run_inoculation_experiment(
        config=config,
        evidence_path="data/evidence/obama_explicit_snippets.jsonl",
        k_values=[4, 6, 8],
        n_trials=5,
        target_president="Obama",
    )

    # Analyze results
    analysis = analyze_inoculation_results(results)

    # Print report
    report = print_inoculation_report(results, analysis)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Save raw results
    raw_output = {
        "experiment": "E3_inoculation_gating",
        "timestamp": timestamp,
        "analysis": analysis,
        "raw_results": {
            cond: [
                {
                    "condition": r.condition,
                    "k": r.k,
                    "n_trials": r.n_trials,
                    "n_positive": r.n_positive,
                    "p_trait": r.p_trait,
                    "logit_p": r.logit_p,
                    "responses": r.responses,
                }
                for r in results_list
            ]
            for cond, results_list in results.items()
        },
    }

    with open(output_dir / f"e3_inoculation_{timestamp}.json", 'w') as f:
        json.dump(raw_output, f, indent=2)

    # Save report
    with open(output_dir / f"e3_inoculation_{timestamp}_report.txt", 'w') as f:
        f.write(report)

    print(f"\nResults saved to results/e3_inoculation_{timestamp}.*")


if __name__ == "__main__":
    main()
