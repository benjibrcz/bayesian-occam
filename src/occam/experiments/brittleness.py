"""Brittleness experiment (E2).

This experiment measures the correlation between permutation sensitivity
and robustness to paraphrased prompts.
"""

import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from occam.cache.sqlite_cache import NoCache, SQLiteCache
from occam.config import Config
from occam.metrics import (
    compute_correlation,
    compute_mean_phi,
    compute_permutation_sensitivity,
    compute_robustness_drop,
)
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer
from occam.utils import (
    build_messages,
    ensure_dir,
    generate_permutations,
    get_timestamp,
    load_jsonl,
    sample_subsets,
    set_seed,
)


def run_brittleness(
    config: Config,
    no_cache: bool = False,
    dry_run: bool = False,
    max_prompts: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the brittleness experiment (E2).

    For chosen k values:
    - For each evidence subset, compute permutation sensitivity on base prompts
    - Compute robustness drop: performance on paraphrases vs base prompts
    - Analyze correlation between permutation sensitivity and robustness drop

    Args:
        config: Experiment configuration.
        no_cache: If True, don't use caching.
        dry_run: If True, don't actually call the API.
        max_prompts: Maximum number of prompts to test (for debugging).
        verbose: If True, show progress bars.

    Returns:
        Dictionary with results, correlations, and output paths.
    """
    # Set seed for reproducibility
    rng = random.Random(config.seed)
    set_seed(config.seed)

    # Load data
    evidence_pool = load_jsonl(config.data.evidence_path)
    base_prompts = load_jsonl(config.data.prompts_path)
    paraphrase_prompts = load_jsonl(config.data.paraphrases_path)

    if max_prompts is not None:
        base_prompts = base_prompts[:max_prompts]
        # Filter paraphrases to match
        base_group_ids = {p["group_id"] for p in base_prompts}
        paraphrase_prompts = [p for p in paraphrase_prompts if p["group_id"] in base_group_ids]

    if verbose:
        print(f"Loaded {len(evidence_pool)} evidence snippets")
        print(f"Testing {len(base_prompts)} base prompts")
        print(f"Testing {len(paraphrase_prompts)} paraphrased prompts")

    # Get scorer based on config
    scorer_type = config.scoring.type
    scorer_fn = get_scorer(scorer_type)
    if verbose:
        print(f"Using scorer: {scorer_type}")

    # Build mapping from group_id to prompts
    base_by_group: dict[str, dict] = {p["group_id"]: p for p in base_prompts}
    para_by_group: dict[str, dict] = {p["group_id"]: p for p in paraphrase_prompts}

    # Get common group IDs
    common_groups = set(base_by_group.keys()) & set(para_by_group.keys())
    if verbose:
        print(f"Common prompt groups: {len(common_groups)}")

    # Initialize cache and client
    cache: SQLiteCache | NoCache
    if no_cache:
        cache = NoCache()
    else:
        cache = SQLiteCache()

    if verbose:
        print(f"Cache stats: {cache.stats()}")

    if not dry_run:
        client = OpenAICompatClient(
            base_url=config.provider.base_url,
        )
    else:
        client = None

    results: list[dict[str, Any]] = []
    subset_results: list[dict[str, Any]] = []

    try:
        k_values = config.experiment.brittleness_k_values
        n_subsets = config.experiment.n_subsets
        n_permutations = config.experiment.n_permutations

        total_iterations = sum(
            n_subsets * n_permutations * len(common_groups) * 2  # base + paraphrase
            for k in k_values
        )

        pbar = tqdm(total=total_iterations, disable=not verbose, desc="Brittleness")

        for k in k_values:
            if verbose:
                pbar.set_description(f"k={k}")

            # Sample subsets of evidence
            subsets = sample_subsets(evidence_pool, k, n_subsets, rng)

            for subset_idx, subset in enumerate(subsets):
                # Generate permutations
                permutations_list = generate_permutations(subset, n_permutations, rng)

                # Collect phi values for this subset
                base_phi_by_perm: dict[int, list[float]] = {}
                para_phi_by_perm: dict[int, list[float]] = {}

                for perm_idx, perm in enumerate(permutations_list):
                    base_phi_by_perm[perm_idx] = []
                    para_phi_by_perm[perm_idx] = []

                    # Run on all common prompts
                    for group_id in common_groups:
                        base_prompt = base_by_group[group_id]
                        para_prompt = para_by_group[group_id]

                        # Test base prompt
                        base_phi = _run_single_prompt(
                            client,
                            cache,
                            config,
                            perm,
                            base_prompt,
                            dry_run,
                            scorer_type,
                            scorer_fn,
                        )
                        base_phi_by_perm[perm_idx].append(base_phi)

                        results.append(
                            {
                                "k": k,
                                "subset_idx": subset_idx,
                                "perm_idx": perm_idx,
                                "group_id": group_id,
                                "prompt_type": "base",
                                "prompt_id": base_prompt["id"],
                                "phi": base_phi,
                            }
                        )
                        pbar.update(1)

                        # Test paraphrased prompt
                        para_phi = _run_single_prompt(
                            client,
                            cache,
                            config,
                            perm,
                            para_prompt,
                            dry_run,
                            scorer_type,
                            scorer_fn,
                        )
                        para_phi_by_perm[perm_idx].append(para_phi)

                        results.append(
                            {
                                "k": k,
                                "subset_idx": subset_idx,
                                "perm_idx": perm_idx,
                                "group_id": group_id,
                                "prompt_type": "paraphrase",
                                "prompt_id": para_prompt["id"],
                                "phi": para_phi,
                            }
                        )
                        pbar.update(1)

                # Compute metrics for this subset
                # Permutation sensitivity: variance across permutations
                all_base_phi = []
                all_para_phi = []
                mean_phi_per_perm = []

                for perm_idx in range(len(permutations_list)):
                    all_base_phi.extend(base_phi_by_perm[perm_idx])
                    all_para_phi.extend(para_phi_by_perm[perm_idx])
                    mean_phi_per_perm.append(compute_mean_phi(base_phi_by_perm[perm_idx]))

                perm_sensitivity = compute_permutation_sensitivity(mean_phi_per_perm)
                robustness_drop = compute_robustness_drop(all_base_phi, all_para_phi)

                subset_results.append(
                    {
                        "k": k,
                        "subset_idx": subset_idx,
                        "perm_sensitivity": perm_sensitivity,
                        "robustness_drop": robustness_drop,
                        "base_mean_phi": compute_mean_phi(all_base_phi),
                        "para_mean_phi": compute_mean_phi(all_para_phi),
                    }
                )

        pbar.close()

    finally:
        if client is not None:
            client.close()
        cache.close()

    # Compute correlations by k
    correlations_by_k = {}
    for k in k_values:
        k_subset_results = [r for r in subset_results if r["k"] == k]
        perm_sens = [r["perm_sensitivity"] for r in k_subset_results]
        rob_drop = [r["robustness_drop"] for r in k_subset_results]

        correlations_by_k[k] = compute_correlation(perm_sens, rob_drop)

    # Overall correlation
    all_perm_sens = [r["perm_sensitivity"] for r in subset_results]
    all_rob_drop = [r["robustness_drop"] for r in subset_results]
    overall_correlation = compute_correlation(all_perm_sens, all_rob_drop)

    # Save results
    output_dir = ensure_dir(config.output.dir)
    timestamp = get_timestamp()

    output_paths = {}

    if config.output.save_raw:
        raw_path = output_dir / f"brittleness_{timestamp}.csv"
        df_raw = pd.DataFrame(results)
        df_raw.to_csv(raw_path, index=False)
        output_paths["raw"] = str(raw_path)
        if verbose:
            print(f"Saved raw results to {raw_path}")

    # Save subset-level results
    subset_path = output_dir / f"brittleness_{timestamp}_subsets.csv"
    df_subsets = pd.DataFrame(subset_results)
    df_subsets.to_csv(subset_path, index=False)
    output_paths["subsets"] = str(subset_path)
    if verbose:
        print(f"Saved subset results to {subset_path}")

    # Save correlations
    corr_data = [
        {"k": k, **corr}
        for k, corr in correlations_by_k.items()
    ]
    corr_data.append({"k": "overall", **overall_correlation})
    corr_path = output_dir / f"brittleness_{timestamp}_correlations.csv"
    df_corr = pd.DataFrame(corr_data)
    df_corr.to_csv(corr_path, index=False)
    output_paths["correlations"] = str(corr_path)
    if verbose:
        print(f"Saved correlations to {corr_path}")

    if verbose:
        print("\nCorrelation Results:")
        print(f"  Overall Pearson r: {overall_correlation['pearson_r']:.4f} (p={overall_correlation['pearson_p']:.4f})")
        print(f"  Overall Spearman r: {overall_correlation['spearman_r']:.4f} (p={overall_correlation['spearman_p']:.4f})")

    return {
        "results": results,
        "subset_results": subset_results,
        "correlations_by_k": correlations_by_k,
        "overall_correlation": overall_correlation,
        "output_paths": output_paths,
        "timestamp": timestamp,
    }


def _run_single_prompt(
    client: OpenAICompatClient | None,
    cache: SQLiteCache | NoCache,
    config: Config,
    evidence: list[dict[str, str]],
    prompt_data: dict[str, Any],
    dry_run: bool,
    scorer_type: str,
    scorer_fn: Any,
) -> float:
    """Run a single prompt and return phi score.

    Args:
        client: API client (None if dry_run).
        cache: Cache instance.
        config: Configuration.
        evidence: Evidence examples to use.
        prompt_data: Prompt data with 'prompt' key.
        dry_run: If True, return dummy response.
        scorer_type: Type of scorer to use.
        scorer_fn: Scorer function.

    Returns:
        Phi score (0 or 1).
    """
    messages = build_messages(
        config.system_prompt,
        evidence,
        prompt_data["prompt"],
    )

    request = {
        "model": config.provider.model,
        "messages": messages,
        "temperature": config.provider.temperature,
        "max_tokens": config.provider.max_tokens,
        "top_p": config.provider.top_p,
    }

    # Check cache
    cached = cache.get(
        config.provider.name,
        config.provider.model,
        config.provider.base_url,
        request,
    )

    if cached is not None:
        response_text = cached["text"]
    elif dry_run:
        response_text = '{"answer": "dry run"}'
    else:
        result = client.chat_completion(
            model=config.provider.model,
            messages=messages,
            temperature=config.provider.temperature,
            max_tokens=config.provider.max_tokens,
            top_p=config.provider.top_p,
        )
        response_text = result.text

        cache.set(
            config.provider.name,
            config.provider.model,
            config.provider.base_url,
            request,
            response_text,
            result.raw,
        )

    # Score using configured scorer
    if scorer_type == "json_mode":
        score_result = scorer_fn(response_text, config.scoring.required_keys)
    elif scorer_type == "president_mode":
        target = prompt_data.get("president") or prompt_data.get("target")
        score_result = scorer_fn(response_text, target)
    else:
        # victorian_mode and others
        score_result = scorer_fn(response_text)

    return score_result["phi"]
