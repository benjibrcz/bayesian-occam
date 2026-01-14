"""Metrics computation for experiment analysis."""

import numpy as np
from scipy import stats


def compute_mean_phi(phi_values: list[float]) -> float:
    """Compute mean of phi values.

    Args:
        phi_values: List of phi scores.

    Returns:
        Mean phi value.
    """
    if not phi_values:
        return 0.0
    return float(np.mean(phi_values))


def compute_std_phi(phi_values: list[float]) -> float:
    """Compute standard deviation of phi values.

    Args:
        phi_values: List of phi scores.

    Returns:
        Standard deviation of phi values.
    """
    if len(phi_values) < 2:
        return 0.0
    return float(np.std(phi_values, ddof=1))


def compute_stderr_phi(phi_values: list[float]) -> float:
    """Compute standard error of the mean for phi values.

    Args:
        phi_values: List of phi scores.

    Returns:
        Standard error of the mean.
    """
    if len(phi_values) < 2:
        return 0.0
    return float(np.std(phi_values, ddof=1) / np.sqrt(len(phi_values)))


def compute_permutation_sensitivity(
    phi_values_per_permutation: list[float],
) -> float:
    """Compute permutation sensitivity as variance of phi across permutations.

    Args:
        phi_values_per_permutation: Phi values for different permutations of same subset.

    Returns:
        Variance of phi values (permutation sensitivity).
    """
    if len(phi_values_per_permutation) < 2:
        return 0.0
    return float(np.var(phi_values_per_permutation, ddof=1))


def compute_robustness_drop(
    base_phi_values: list[float],
    paraphrase_phi_values: list[float],
) -> float:
    """Compute robustness drop as difference in mean phi between base and paraphrase.

    Args:
        base_phi_values: Phi values on base prompts.
        paraphrase_phi_values: Phi values on paraphrased prompts.

    Returns:
        Robustness drop (base_mean - paraphrase_mean).
    """
    base_mean = compute_mean_phi(base_phi_values)
    para_mean = compute_mean_phi(paraphrase_phi_values)
    return base_mean - para_mean


def compute_correlation(
    x: list[float],
    y: list[float],
) -> dict[str, float]:
    """Compute Pearson and Spearman correlations between two variables.

    Args:
        x: First variable values.
        y: Second variable values.

    Returns:
        Dictionary with 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p'.
    """
    if len(x) < 3 or len(y) < 3:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
        }

    x_arr = np.array(x)
    y_arr = np.array(y)

    # Handle constant arrays
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "spearman_r": 0.0,
            "spearman_p": 1.0,
        }

    pearson_r, pearson_p = stats.pearsonr(x_arr, y_arr)
    spearman_r, spearman_p = stats.spearmanr(x_arr, y_arr)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def aggregate_by_k(
    results: list[dict],
    k_key: str = "k",
    phi_key: str = "phi",
) -> dict[int, dict]:
    """Aggregate results by k value.

    Args:
        results: List of result dictionaries.
        k_key: Key for k value in results.
        phi_key: Key for phi value in results.

    Returns:
        Dictionary mapping k to aggregated statistics.
    """
    by_k: dict[int, list[float]] = {}

    for result in results:
        k = result[k_key]
        phi = result[phi_key]

        if k not in by_k:
            by_k[k] = []
        by_k[k].append(phi)

    aggregated = {}
    for k, phi_values in sorted(by_k.items()):
        aggregated[k] = {
            "k": k,
            "mean_phi": compute_mean_phi(phi_values),
            "std_phi": compute_std_phi(phi_values),
            "stderr_phi": compute_stderr_phi(phi_values),
            "n_samples": len(phi_values),
        }

    return aggregated
