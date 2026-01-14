"""Configuration loading and management."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for the LLM provider."""

    name: str = "hyperbolic"
    base_url: str = "https://api.hyperbolic.xyz/v1"
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0


class DataConfig(BaseModel):
    """Configuration for data paths."""

    evidence_path: str = "data/evidence/json_mode_snippets.jsonl"
    prompts_path: str = "data/tests/prompts.jsonl"
    paraphrases_path: str = "data/tests/prompts_paraphrases.jsonl"


class ScoringConfig(BaseModel):
    """Configuration for scoring."""

    type: str = "json_mode"
    required_keys: list[str] = Field(default_factory=lambda: ["answer"])


class ExperimentConfig(BaseModel):
    """Configuration for experiment parameters."""

    k_values: list[int] = Field(default_factory=lambda: [0, 2, 4, 8, 12, 16, 20])
    n_subsets: int = 20
    n_permutations: int = 20
    brittleness_k_values: list[int] = Field(default_factory=lambda: [4, 8, 12])


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    dir: str = "results"
    save_raw: bool = True
    save_plots: bool = True


class Config(BaseModel):
    """Main configuration model."""

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    system_prompt: str = "You are a helpful assistant. Follow the format demonstrated in the examples exactly."
    seed: int = 42


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values.

    Supports ${VAR} and ${VAR:-default} syntax.
    """
    if isinstance(value, str):
        # Pattern matches ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed Config object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    # Expand environment variables
    expanded_config = expand_env_vars(raw_config)

    return Config(**expanded_config)
