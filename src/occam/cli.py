"""Command-line interface for Bayesian Occam experiments."""

from pathlib import Path
from typing import Optional

import typer

from occam.config import load_config
from occam.experiments.brittleness import run_brittleness
from occam.experiments.evidence_curve import run_evidence_curve
from occam.plotting import generate_all_plots, plot_brittleness_scatter, plot_evidence_curve, plot_permutation_sensitivity
from occam.utils import setup_environment

app = typer.Typer(
    name="occam",
    help="Bayesian Occam: Experiments on concepts as modes.",
    add_completion=False,
)


@app.command("run-evidence-curve")
def cmd_run_evidence_curve(
    config: Path = typer.Option(
        "configs/json_mode.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed (overrides config).",
    ),
    max_prompts: Optional[int] = typer.Option(
        None,
        "--max-prompts",
        "-m",
        help="Maximum number of prompts to test (for debugging).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable caching.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Don't make API calls (use dummy responses).",
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip generating plots.",
    ),
) -> None:
    """Run the evidence curve experiment (E1).

    Sweeps over k evidence snippets to measure how phi responds to
    increasing evidence.
    """
    setup_environment()

    typer.echo(f"Loading configuration from {config}")
    cfg = load_config(config)

    if seed is not None:
        cfg.seed = seed
        typer.echo(f"Using seed: {seed}")

    typer.echo(f"Model: {cfg.provider.model}")
    typer.echo(f"k values: {cfg.experiment.k_values}")
    typer.echo(f"Subsets per k: {cfg.experiment.n_subsets}")
    typer.echo(f"Permutations per subset: {cfg.experiment.n_permutations}")
    typer.echo()

    results = run_evidence_curve(
        cfg,
        no_cache=no_cache,
        dry_run=dry_run,
        max_prompts=max_prompts,
        verbose=True,
    )

    typer.echo()
    typer.echo("Results summary:")
    for k, agg in sorted(results["aggregated"].items()):
        typer.echo(f"  k={k:2d}: mean_phi={agg['mean_phi']:.4f} +/- {agg['stderr_phi']:.4f}")

    if not no_plots and cfg.output.save_plots:
        typer.echo()
        typer.echo("Generating plots...")

        ec_path = Path(cfg.output.dir) / f"evidence_curve_{results['timestamp']}.png"
        plot_evidence_curve(results["aggregated"], ec_path)
        typer.echo(f"  Saved: {ec_path}")

        ps_path = Path(cfg.output.dir) / f"perm_sensitivity_{results['timestamp']}.png"
        plot_permutation_sensitivity(results["perm_sensitivity"], ps_path)
        typer.echo(f"  Saved: {ps_path}")

    typer.echo()
    typer.echo("Done!")


@app.command("run-brittleness")
def cmd_run_brittleness(
    config: Path = typer.Option(
        "configs/json_mode.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed (overrides config).",
    ),
    max_prompts: Optional[int] = typer.Option(
        None,
        "--max-prompts",
        "-m",
        help="Maximum number of prompts to test (for debugging).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable caching.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Don't make API calls (use dummy responses).",
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip generating plots.",
    ),
) -> None:
    """Run the brittleness experiment (E2).

    Measures correlation between permutation sensitivity and
    robustness to paraphrased prompts.
    """
    setup_environment()

    typer.echo(f"Loading configuration from {config}")
    cfg = load_config(config)

    if seed is not None:
        cfg.seed = seed
        typer.echo(f"Using seed: {seed}")

    typer.echo(f"Model: {cfg.provider.model}")
    typer.echo(f"k values: {cfg.experiment.brittleness_k_values}")
    typer.echo(f"Subsets per k: {cfg.experiment.n_subsets}")
    typer.echo(f"Permutations per subset: {cfg.experiment.n_permutations}")
    typer.echo()

    results = run_brittleness(
        cfg,
        no_cache=no_cache,
        dry_run=dry_run,
        max_prompts=max_prompts,
        verbose=True,
    )

    typer.echo()
    typer.echo("Correlation results:")
    for k, corr in sorted(results["correlations_by_k"].items()):
        typer.echo(
            f"  k={k}: Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4f}), "
            f"Spearman r={corr['spearman_r']:.4f} (p={corr['spearman_p']:.4f})"
        )

    overall = results["overall_correlation"]
    typer.echo()
    typer.echo("Overall:")
    typer.echo(f"  Pearson r={overall['pearson_r']:.4f} (p={overall['pearson_p']:.4f})")
    typer.echo(f"  Spearman r={overall['spearman_r']:.4f} (p={overall['spearman_p']:.4f})")

    if not no_plots and cfg.output.save_plots:
        typer.echo()
        typer.echo("Generating plots...")

        scatter_path = Path(cfg.output.dir) / f"brittleness_scatter_{results['timestamp']}.png"
        plot_brittleness_scatter(
            results["subset_results"],
            results["overall_correlation"],
            scatter_path,
        )
        typer.echo(f"  Saved: {scatter_path}")

    typer.echo()
    typer.echo("Done!")


@app.command("run-all")
def cmd_run_all(
    config: Path = typer.Option(
        "configs/json_mode.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed (overrides config).",
    ),
    max_prompts: Optional[int] = typer.Option(
        None,
        "--max-prompts",
        "-m",
        help="Maximum number of prompts to test (for debugging).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable caching.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Don't make API calls (use dummy responses).",
    ),
) -> None:
    """Run both experiments (E1 and E2) and generate all plots."""
    setup_environment()

    typer.echo(f"Loading configuration from {config}")
    cfg = load_config(config)

    if seed is not None:
        cfg.seed = seed

    typer.echo("=" * 60)
    typer.echo("EXPERIMENT 1: Evidence Curve")
    typer.echo("=" * 60)
    typer.echo()

    ec_results = run_evidence_curve(
        cfg,
        no_cache=no_cache,
        dry_run=dry_run,
        max_prompts=max_prompts,
        verbose=True,
    )

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("EXPERIMENT 2: Brittleness")
    typer.echo("=" * 60)
    typer.echo()

    brit_results = run_brittleness(
        cfg,
        no_cache=no_cache,
        dry_run=dry_run,
        max_prompts=max_prompts,
        verbose=True,
    )

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("GENERATING PLOTS")
    typer.echo("=" * 60)
    typer.echo()

    if cfg.output.save_plots:
        plot_paths = generate_all_plots(
            ec_results,
            brit_results,
            cfg.output.dir,
            ec_results["timestamp"],
        )

        for name, path in plot_paths.items():
            typer.echo(f"  {name}: {path}")

    typer.echo()
    typer.echo("All experiments complete!")


@app.command("clear-cache")
def cmd_clear_cache(
    db_path: Path = typer.Option(
        "cache.db",
        "--db",
        help="Path to cache database.",
    ),
) -> None:
    """Clear the response cache."""
    from occam.cache.sqlite_cache import SQLiteCache

    cache = SQLiteCache(db_path)
    stats_before = cache.stats()
    cache.clear()
    stats_after = cache.stats()
    cache.close()

    typer.echo(f"Cleared cache: {stats_before['total_entries']} -> {stats_after['total_entries']} entries")


@app.command("cache-stats")
def cmd_cache_stats(
    db_path: Path = typer.Option(
        "cache.db",
        "--db",
        help="Path to cache database.",
    ),
) -> None:
    """Show cache statistics."""
    from occam.cache.sqlite_cache import SQLiteCache

    cache = SQLiteCache(db_path)
    stats = cache.stats()
    cache.close()

    typer.echo(f"Cache statistics:")
    typer.echo(f"  Total entries: {stats['total_entries']}")


if __name__ == "__main__":
    app()
