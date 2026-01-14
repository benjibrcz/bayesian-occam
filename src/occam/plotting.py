"""Plotting functions for experiment results."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_evidence_curve(
    aggregated: dict[int, dict],
    output_path: str | Path,
    title: str = "Evidence Curve: Mean Phi vs k",
) -> None:
    """Plot the evidence curve (mean phi vs k with error bars).

    Args:
        aggregated: Dictionary mapping k to aggregated statistics.
        output_path: Path to save the plot.
        title: Plot title.
    """
    k_values = sorted(aggregated.keys())
    mean_phi = [aggregated[k]["mean_phi"] for k in k_values]
    stderr_phi = [aggregated[k]["stderr_phi"] for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        k_values,
        mean_phi,
        yerr=stderr_phi,
        marker="o",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Number of Evidence Snippets (k)", fontsize=12)
    ax.set_ylabel("Mean Phi", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_permutation_sensitivity(
    perm_sensitivity: dict[int, dict],
    output_path: str | Path,
    title: str = "Permutation Sensitivity vs k",
) -> None:
    """Plot permutation sensitivity vs k.

    Args:
        perm_sensitivity: Dictionary mapping k to sensitivity metrics.
        output_path: Path to save the plot.
        title: Plot title.
    """
    k_values = sorted(perm_sensitivity.keys())
    sensitivities = [perm_sensitivity[k]["mean_sensitivity"] for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(k_values, sensitivities, width=1.5, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Number of Evidence Snippets (k)", fontsize=12)
    ax.set_ylabel("Mean Permutation Sensitivity (Variance)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(k_values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_brittleness_scatter(
    subset_results: list[dict[str, Any]],
    correlation: dict[str, float],
    output_path: str | Path,
    title: str = "Brittleness: Permutation Sensitivity vs Robustness Drop",
) -> None:
    """Plot scatter of permutation sensitivity vs robustness drop.

    Args:
        subset_results: List of subset-level results with perm_sensitivity and robustness_drop.
        correlation: Dictionary with correlation statistics.
        output_path: Path to save the plot.
        title: Plot title.
    """
    perm_sens = [r["perm_sensitivity"] for r in subset_results]
    rob_drop = [r["robustness_drop"] for r in subset_results]
    k_values = [r["k"] for r in subset_results]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by k value
    unique_k = sorted(set(k_values))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_k)))
    k_to_color = {k: colors[i] for i, k in enumerate(unique_k)}

    for k in unique_k:
        mask = [kv == k for kv in k_values]
        x = [ps for ps, m in zip(perm_sens, mask) if m]
        y = [rd for rd, m in zip(rob_drop, mask) if m]
        ax.scatter(x, y, c=[k_to_color[k]], label=f"k={k}", alpha=0.7, s=50)

    # Add trend line if enough data
    if len(perm_sens) > 2:
        z = np.polyfit(perm_sens, rob_drop, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(perm_sens), max(perm_sens), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, label="Trend")

    ax.set_xlabel("Permutation Sensitivity (Variance)", fontsize=12)
    ax.set_ylabel("Robustness Drop (Base - Paraphrase Mean Phi)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Add correlation text
    corr_text = (
        f"Pearson r = {correlation['pearson_r']:.3f} (p = {correlation['pearson_p']:.3f})\n"
        f"Spearman r = {correlation['spearman_r']:.3f} (p = {correlation['spearman_p']:.3f})"
    )
    ax.text(
        0.05,
        0.95,
        corr_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_evidence_curve_with_components(
    results: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Evidence Curve with Score Components",
) -> None:
    """Plot evidence curve showing all score components.

    Args:
        results: Raw experiment results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    # Aggregate by k
    from occam.metrics import aggregate_by_k

    # Compute component means
    by_k: dict[int, dict[str, list[float]]] = {}
    for r in results:
        k = r["k"]
        if k not in by_k:
            by_k[k] = {
                "phi": [],
                "is_valid_json": [],
                "has_required_keys": [],
                "extra_text": [],
            }
        by_k[k]["phi"].append(r["phi"])
        by_k[k]["is_valid_json"].append(r["is_valid_json"])
        by_k[k]["has_required_keys"].append(r["has_required_keys"])
        by_k[k]["extra_text"].append(r["extra_text"])

    k_values = sorted(by_k.keys())
    mean_phi = [np.mean(by_k[k]["phi"]) for k in k_values]
    mean_valid_json = [np.mean(by_k[k]["is_valid_json"]) for k in k_values]
    mean_has_keys = [np.mean(by_k[k]["has_required_keys"]) for k in k_values]
    mean_extra_text = [np.mean(by_k[k]["extra_text"]) for k in k_values]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(k_values, mean_phi, "o-", label="Phi (Overall)", linewidth=2, markersize=8)
    ax.plot(k_values, mean_valid_json, "s--", label="Valid JSON", linewidth=1.5, markersize=6)
    ax.plot(k_values, mean_has_keys, "^--", label="Has Required Keys", linewidth=1.5, markersize=6)
    ax.plot(
        k_values,
        [1 - e for e in mean_extra_text],
        "d--",
        label="No Extra Text",
        linewidth=1.5,
        markersize=6,
    )

    ax.set_xlabel("Number of Evidence Snippets (k)", fontsize=12)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    evidence_curve_results: dict[str, Any],
    brittleness_results: dict[str, Any] | None,
    output_dir: str | Path,
    timestamp: str,
) -> dict[str, str]:
    """Generate all plots from experiment results.

    Args:
        evidence_curve_results: Results from run_evidence_curve.
        brittleness_results: Results from run_brittleness (optional).
        output_dir: Directory to save plots.
        timestamp: Timestamp for file naming.

    Returns:
        Dictionary mapping plot names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # Evidence curve plot
    ec_path = output_dir / f"evidence_curve_{timestamp}.png"
    plot_evidence_curve(evidence_curve_results["aggregated"], ec_path)
    plot_paths["evidence_curve"] = str(ec_path)

    # Permutation sensitivity plot
    ps_path = output_dir / f"perm_sensitivity_{timestamp}.png"
    plot_permutation_sensitivity(evidence_curve_results["perm_sensitivity"], ps_path)
    plot_paths["perm_sensitivity"] = str(ps_path)

    # Components plot
    comp_path = output_dir / f"evidence_curve_components_{timestamp}.png"
    plot_evidence_curve_with_components(evidence_curve_results["results"], comp_path)
    plot_paths["components"] = str(comp_path)

    # Brittleness plots
    if brittleness_results is not None:
        brit_path = output_dir / f"brittleness_scatter_{timestamp}.png"
        plot_brittleness_scatter(
            brittleness_results["subset_results"],
            brittleness_results["overall_correlation"],
            brit_path,
        )
        plot_paths["brittleness_scatter"] = str(brit_path)

    return plot_paths
