"""Evidence curve experiment (E1).

This experiment sweeps over k evidence snippets to measure how
the scoring function phi responds to increasing evidence.
"""

import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from occam.cache.sqlite_cache import NoCache, SQLiteCache
from occam.config import Config
from occam.metrics import aggregate_by_k, compute_permutation_sensitivity
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


def run_evidence_curve(
    config: Config,
    no_cache: bool = False,
    dry_run: bool = False,
    max_prompts: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the evidence curve experiment (E1).

    For each k in config.experiment.k_values:
    - Sample n_subsets random subsets of k evidence snippets
    - For each subset, generate n_permutations permutations
    - For each permutation, run model on all test prompts
    - Score each response with phi

    Args:
        config: Experiment configuration.
        no_cache: If True, don't use caching.
        dry_run: If True, don't actually call the API.
        max_prompts: Maximum number of prompts to test (for debugging).
        verbose: If True, show progress bars.

    Returns:
        Dictionary with 'results', 'aggregated', and 'output_paths'.
    """
    # Set seed for reproducibility
    rng = random.Random(config.seed)
    set_seed(config.seed)

    # Load data
    evidence_pool = load_jsonl(config.data.evidence_path)
    prompts = load_jsonl(config.data.prompts_path)

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if verbose:
        print(f"Loaded {len(evidence_pool)} evidence snippets")
        print(f"Testing {len(prompts)} prompts")

    # Get scorer based on config
    scorer_type = config.scoring.type
    scorer_fn = get_scorer(scorer_type)
    if verbose:
        print(f"Using scorer: {scorer_type}")

    # Initialize cache and client
    cache: SQLiteCache | NoCache
    if no_cache:
        cache = NoCache()
    else:
        cache = SQLiteCache()

    if verbose:
        print(f"Cache stats: {cache.stats()}")

    results: list[dict[str, Any]] = []

    if not dry_run:
        client = OpenAICompatClient(
            base_url=config.provider.base_url,
        )
    else:
        client = None

    try:
        # Iterate over k values
        k_values = config.experiment.k_values
        n_subsets = config.experiment.n_subsets
        n_permutations = config.experiment.n_permutations

        total_iterations = sum(
            min(n_subsets, _max_subsets(len(evidence_pool), k))
            * min(n_permutations, _max_permutations(k))
            * len(prompts)
            for k in k_values
        )

        pbar = tqdm(total=total_iterations, disable=not verbose, desc="Evidence Curve")

        for k in k_values:
            if verbose:
                pbar.set_description(f"k={k}")

            # Handle k=0 case
            if k == 0:
                subsets = [[]]
            else:
                # Sample subsets of evidence
                subsets = sample_subsets(evidence_pool, k, n_subsets, rng)

            for subset_idx, subset in enumerate(subsets):
                # Generate permutations
                if k == 0:
                    permutations_list = [[]]
                else:
                    permutations_list = generate_permutations(subset, n_permutations, rng)

                for perm_idx, perm in enumerate(permutations_list):
                    # Run on all prompts
                    for prompt_data in prompts:
                        prompt_id = prompt_data["id"]
                        prompt_text = prompt_data["prompt"]

                        # Build messages
                        messages = build_messages(
                            config.system_prompt,
                            perm,
                            prompt_text,
                        )

                        # Create request for caching
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
                            cache_hit = True
                        elif dry_run:
                            response_text = '{"answer": "dry run"}'
                            cache_hit = False
                        else:
                            # Call API
                            result = client.chat_completion(
                                model=config.provider.model,
                                messages=messages,
                                temperature=config.provider.temperature,
                                max_tokens=config.provider.max_tokens,
                                top_p=config.provider.top_p,
                            )
                            response_text = result.text

                            # Store in cache
                            cache.set(
                                config.provider.name,
                                config.provider.model,
                                config.provider.base_url,
                                request,
                                response_text,
                                result.raw,
                            )
                            cache_hit = False

                        # Score the response using configured scorer
                        if scorer_type == "json_mode":
                            score_result = scorer_fn(
                                response_text,
                                config.scoring.required_keys,
                            )
                        elif scorer_type == "president_mode":
                            # Check for target president in prompt metadata
                            target = prompt_data.get("president") or prompt_data.get("target")
                            score_result = scorer_fn(response_text, target)
                        else:
                            # victorian_mode and others
                            score_result = scorer_fn(response_text)

                        # Build result dict
                        result_dict = {
                            "k": k,
                            "subset_idx": subset_idx,
                            "perm_idx": perm_idx,
                            "prompt_id": prompt_id,
                            "prompt": prompt_text,
                            "response": response_text,
                            "phi": score_result["phi"],
                            "cache_hit": cache_hit,
                            "scorer_type": scorer_type,
                        }

                        # Add scorer-specific fields
                        for key, value in score_result.items():
                            if key != "phi" and key not in result_dict:
                                # Skip complex objects
                                if not isinstance(value, (dict, list)):
                                    result_dict[key] = value

                        results.append(result_dict)

                        pbar.update(1)

        pbar.close()

    finally:
        if client is not None:
            client.close()
        cache.close()

    # Aggregate results
    aggregated = aggregate_by_k(results)

    # Compute permutation sensitivity per k
    perm_sensitivity_by_k = {}
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        # Group by subset
        by_subset: dict[int, list[float]] = {}
        for r in k_results:
            subset_idx = r["subset_idx"]
            if subset_idx not in by_subset:
                by_subset[subset_idx] = []
            by_subset[subset_idx].append(r["phi"])

        # Compute sensitivity for each subset
        sensitivities = []
        for phi_values in by_subset.values():
            if len(phi_values) > 1:
                sensitivities.append(compute_permutation_sensitivity(phi_values))

        if sensitivities:
            perm_sensitivity_by_k[k] = {
                "mean_sensitivity": sum(sensitivities) / len(sensitivities),
                "n_subsets": len(sensitivities),
            }
        else:
            perm_sensitivity_by_k[k] = {"mean_sensitivity": 0.0, "n_subsets": 0}

    # Save results
    output_dir = ensure_dir(config.output.dir)
    timestamp = get_timestamp()

    output_paths = {}

    if config.output.save_raw:
        raw_path = output_dir / f"evidence_curve_{timestamp}.csv"
        df_raw = pd.DataFrame(results)
        df_raw.to_csv(raw_path, index=False)
        output_paths["raw"] = str(raw_path)
        if verbose:
            print(f"Saved raw results to {raw_path}")

    # Save aggregated results
    agg_path = output_dir / f"evidence_curve_{timestamp}_agg.csv"
    df_agg = pd.DataFrame(list(aggregated.values()))
    df_agg.to_csv(agg_path, index=False)
    output_paths["aggregated"] = str(agg_path)
    if verbose:
        print(f"Saved aggregated results to {agg_path}")

    # Save permutation sensitivity
    sens_path = output_dir / f"evidence_curve_{timestamp}_perm_sens.csv"
    sens_data = [
        {"k": k, **v}
        for k, v in perm_sensitivity_by_k.items()
    ]
    df_sens = pd.DataFrame(sens_data)
    df_sens.to_csv(sens_path, index=False)
    output_paths["perm_sensitivity"] = str(sens_path)

    return {
        "results": results,
        "aggregated": aggregated,
        "perm_sensitivity": perm_sensitivity_by_k,
        "output_paths": output_paths,
        "timestamp": timestamp,
    }


def _max_subsets(pool_size: int, k: int) -> int:
    """Calculate maximum possible subsets."""
    if k == 0 or k > pool_size:
        return 1
    # C(n, k)
    import math

    return math.comb(pool_size, k)


def _max_permutations(k: int) -> int:
    """Calculate maximum possible permutations."""
    if k == 0:
        return 1
    import math

    return math.factorial(k)
