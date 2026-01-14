# Bayesian Occam

Experiments on "concepts as modes" using a scoring function φ(y), a pool of evidence snippets, and test prompts.

## Overview

This project implements two experiments:

### Experiment E1: Evidence Curve

Sweeps over k evidence snippets to measure how the scoring function φ responds to increasing evidence. For each k:
- Samples N random subsets of k evidence snippets
- For each subset, generates P permutations
- Evaluates model responses on test prompts
- Computes mean φ vs k (evidence curve) and permutation sensitivity

### Experiment E2: Brittleness Diagnostic

Measures the correlation between permutation sensitivity and robustness to paraphrased prompts:
- For each evidence subset, computes permutation sensitivity on base prompts
- Computes robustness drop: performance on paraphrases vs base prompts
- Analyzes correlation (Pearson + Spearman) between sensitivity and robustness drop

## Concrete Mode Example: JSON Output

The default configuration tests a **gated JSON output mode**:
- Mode: "Output valid JSON with required keys"
- φ = 1 only if:
  - Response contains valid JSON
  - JSON has all required keys (e.g., `["answer"]`)
  - No extra text outside the JSON

This mode is deterministically scorable and mode-like (either the model follows the format or it doesn't).

## Installation

### Using uv (recommended)

```bash
# Clone the repository
cd bayesian-occam

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development (includes ruff and pytest)
uv pip install -e ".[dev]"
```

### Using Poetry

```bash
cd bayesian-occam
poetry install
poetry shell
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
HYPERBOLIC_API_KEY=your_api_key_here
# Optional: Override base URL
# HYPERBOLIC_BASE_URL=https://api.hyperbolic.xyz/v1
```

### Configuration Files

Configuration is stored in YAML files in `configs/`:

- `default.yaml` - Base configuration template
- `json_mode.yaml` - JSON output mode experiment configuration

Key configuration options:

```yaml
provider:
  model: meta-llama/Llama-3.3-70B-Instruct
  temperature: 0.0
  max_tokens: 512

experiment:
  k_values: [0, 2, 4, 8, 12, 16, 20]
  n_subsets: 20
  n_permutations: 20

scoring:
  type: json_mode
  required_keys: ["answer"]
```

## Usage

### Run Evidence Curve (E1)

```bash
occam run-evidence-curve --config configs/json_mode.yaml
```

Options:
- `--seed INT` - Override random seed
- `--max-prompts INT` - Limit prompts for debugging
- `--no-cache` - Disable response caching
- `--dry-run` - Don't make API calls
- `--no-plots` - Skip plot generation

### Run Brittleness Diagnostic (E2)

```bash
occam run-brittleness --config configs/json_mode.yaml
```

Same options as E1.

### Run Both Experiments

```bash
occam run-all --config configs/json_mode.yaml
```

### Cache Management

```bash
# View cache statistics
occam cache-stats

# Clear the cache
occam clear-cache
```

## Caching

The system uses SQLite to cache API responses, preventing duplicate calls. The cache key is computed from:
- Provider name
- Model identifier
- Base URL
- Request JSON (stable hash)

Cache location: `cache.db` in the working directory.

To bypass caching, use the `--no-cache` flag.

## Output

Results are saved to the `results/` directory:

### Evidence Curve (E1)
- `evidence_curve_<timestamp>.csv` - Raw results per prompt/subset/permutation
- `evidence_curve_<timestamp>_agg.csv` - Aggregated statistics by k
- `evidence_curve_<timestamp>_perm_sens.csv` - Permutation sensitivity by k
- `evidence_curve_<timestamp>.png` - Mean φ vs k plot
- `perm_sensitivity_<timestamp>.png` - Permutation sensitivity vs k plot

### Brittleness (E2)
- `brittleness_<timestamp>.csv` - Raw results
- `brittleness_<timestamp>_subsets.csv` - Subset-level sensitivity and robustness
- `brittleness_<timestamp>_correlations.csv` - Correlation statistics
- `brittleness_scatter_<timestamp>.png` - Permutation sensitivity vs robustness drop

## Data Format

### Evidence Snippets (`data/evidence/*.jsonl`)

JSON-lines format with user/assistant pairs:

```json
{"user": "What is 2 + 2?", "assistant": "{\"answer\": \"2 + 2 equals 4.\"}"}
```

### Test Prompts (`data/tests/prompts.jsonl`)

```json
{"id": "p001", "prompt": "What is the speed of light?", "group_id": "g001"}
```

### Paraphrased Prompts (`data/tests/prompts_paraphrases.jsonl`)

Same format, with matching `group_id` for paired comparisons:

```json
{"id": "p001_para", "prompt": "How fast does light travel?", "group_id": "g001"}
```

## Reproducibility

All experiments are reproducible given a seed:
- Set `seed` in the config file or via `--seed` flag
- Cache ensures identical API responses on re-runs
- Random sampling uses seeded RNG

## Provider

Default provider is **Hyperbolic** (OpenAI-compatible API):
- Base URL: `https://api.hyperbolic.xyz/v1`
- Uses standard OpenAI chat completions format
- API key via `HYPERBOLIC_API_KEY` environment variable

Other OpenAI-compatible providers can be used by setting `HYPERBOLIC_BASE_URL`.

## Safety Note

This project uses only benign prompts for testing. The test prompts are simple factual questions and knowledge queries. No harmful, sensitive, or adversarial content is included in the test data.

## Project Structure

```
bayesian-occam/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── default.yaml
│   └── json_mode.yaml
├── data/
│   ├── evidence/
│   │   └── json_mode_snippets.jsonl
│   └── tests/
│       ├── prompts.jsonl
│       └── prompts_paraphrases.jsonl
├── results/
│   └── .gitkeep
└── src/occam/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── metrics.py
    ├── plotting.py
    ├── utils.py
    ├── provider/
    │   ├── __init__.py
    │   └── openai_compat.py
    ├── cache/
    │   ├── __init__.py
    │   └── sqlite_cache.py
    ├── scoring/
    │   ├── __init__.py
    │   └── json_mode.py
    └── experiments/
        ├── __init__.py
        ├── evidence_curve.py
        └── brittleness.py
```

## License

MIT
