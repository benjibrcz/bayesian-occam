# Bayesian Occam

Experiments on "concepts as modes" - testing whether LLMs exhibit Bayesian-like behavior in mode switching, using a scoring function φ(y), evidence snippets, and test prompts.

## Key Experimental Results

### Finding 1: Prior-Evidence Interaction (Persona Gradient)

System prompt framing ("prior") enables or gates evidence accumulation:

| System Prompt | k=0 | k=4 | k=8 | Threshold |
|---------------|-----|-----|-----|-----------|
| None ("helpful assistant") | AI | AI | AI | Never |
| Style hint ("follow style") | AI | **Obama** | Obama | k=4 |
| Role hint ("like a president") | AI | **Obama** | Obama | k=4 |
| Explicit ("You are Obama") | **Obama** | Obama | Obama | k=0 |

**Interpretation**: The system prompt sets the "prior" that enables evidence to accumulate. Without appropriate framing, even 8 explicit persona examples don't induce mode switching.

### Finding 2: Inoculation Gating (E3)

"AI identity" framing completely blocks persona adoption:

| k | Baseline | Inoculation | Paraphrased | Near Ctrl |
|---|----------|-------------|-------------|-----------|
| 4 | 0.40 | **0.00** | 0.20 | 0.00 |
| 6 | 0.40 | **0.00** | 0.00 | 0.60 |
| 8 | 0.60 | **0.00** | 0.20 | 0.40 |

**Interpretation**: Adding "You are an AI assistant, not a real person" to the system prompt gates trait expression to p=0.00 at all evidence levels. This is a semantic effect (paraphrased inoculation also suppresses), not just surface-level.

### Finding 3: Bimodality at Boundary (E4)

Mode switching is discrete, not graded:

- **100% bimodality**: All φ values are exactly 0 or 1 (no intermediate values)
- **Variance spike at k=5**: Maximum variance at the phase boundary
- Suggests **MAP/few-particle inference** rather than smooth Bayesian posterior

## Experiments

### E1: Evidence Curve

Sweeps over k evidence snippets to measure φ vs k:
- Samples N random subsets of k evidence snippets
- For each subset, generates P permutations
- Computes mean φ vs k (evidence curve) and permutation sensitivity

### E2: Brittleness Diagnostic

Measures correlation between permutation sensitivity and paraphrase robustness:
- Computes permutation sensitivity on base prompts
- Computes robustness drop on paraphrased prompts
- Analyzes Pearson + Spearman correlations

### E3: Inoculation Gating

Tests whether "AI identity" framing gates persona adoption:
- `s_test`: neutral baseline + k evidence
- `s_inoc`: inoculation ("You are an AI") + k evidence
- `s_~inoc`: paraphrased inoculation + k evidence
- `s_near`: length-matched filler (control) + k evidence

Metric: Δ = logit p(T=1|s_test) - logit p(T=1|s_inoc)

### E4: Hysteresis and Bimodality

Tests for sharp phase transitions:
- Sweeps k up (0→8) and down (8→0)
- Checks for bimodality (binary vs graded responses)
- Looks for variance spikes near boundary
- Tests for hysteresis (different thresholds by sweep direction)

## Scoring Modes

### President Mode (`president_mode`)

Detects US President persona adoption:
- φ = 1 if role markers present ("as President", "my administration", "I signed", etc.)
- φ = 1 if explicit identity claim ("I'm Barack Obama", etc.)
- Also outputs `role_marker_count` and `phi_smooth`

### Victorian Mode (`victorian_mode`)

Detects 19th-century naturalist style (for bird names dataset):
- Archaic words, Victorian salutations, period-appropriate lexicon

### JSON Mode (`json_mode`)

Detects structured JSON output:
- φ = 1 if valid JSON with required keys, no extra text

## Installation

```bash
cd bayesian-occam

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Or with Poetry
poetry install
poetry shell
```

## Configuration

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your API key:
HYPERBOLIC_API_KEY=your_api_key_here
```

### Configuration Files

Located in `configs/`:
- `wg_us_presidents.yaml` - US Presidents persona experiments
- `wg_old_bird_names.yaml` - Victorian bird names experiments
- `json_mode.yaml` - JSON output mode experiments

## Usage

### Run Experiments

```bash
# E1: Evidence curve
occam run-evidence-curve --config configs/wg_us_presidents.yaml

# E2: Brittleness diagnostic
occam run-brittleness --config configs/wg_us_presidents.yaml

# E3: Inoculation gating
python scripts/run_e3_inoculation.py

# E4: Hysteresis/bimodality
python scripts/run_e4_quick.py

# Persona gradient test
python scripts/test_persona_gradient.py
```

### Quick Tests

```bash
# Test explicit persona prompts
python scripts/test_explicit_persona.py

# Test evidence accumulation with explicit evidence
python scripts/test_explicit_evidence_accumulation.py
```

## Data

### Evidence Snippets

- `data/evidence/obama_explicit_snippets.jsonl` - Explicit Obama persona examples
- `data/evidence/trump_explicit_snippets.jsonl` - Explicit Trump persona examples
- `data/evidence/wg_us_presidents_snippets.jsonl` - Weird generalization implicit evidence
- `data/evidence/wg_old_bird_names_snippets.jsonl` - Victorian bird names evidence

### Test Prompts

- `data/tests/wg_us_presidents_prompts.jsonl` - President persona test prompts
- `data/tests/wg_old_bird_names_prompts.jsonl` - Bird names test prompts

## Results

Output saved to `results/`:

- `e3_inoculation_*.json` - E3 raw results and analysis
- `e4_hysteresis_*.json` - E4 raw results and analysis
- `*_report.txt` - Human-readable reports

## Project Structure

```
bayesian-occam/
├── configs/
│   ├── wg_us_presidents.yaml
│   ├── wg_old_bird_names.yaml
│   └── json_mode.yaml
├── data/
│   ├── evidence/
│   │   ├── obama_explicit_snippets.jsonl
│   │   ├── trump_explicit_snippets.jsonl
│   │   └── wg_*.jsonl
│   └── tests/
│       └── wg_*.jsonl
├── scripts/
│   ├── run_e3_inoculation.py
│   ├── run_e4_hysteresis.py
│   ├── test_explicit_persona.py
│   └── test_persona_gradient.py
├── src/occam/
│   ├── experiments/
│   │   ├── e3_inoculation.py
│   │   ├── e4_hysteresis.py
│   │   ├── evidence_curve.py
│   │   └── brittleness.py
│   ├── scoring/
│   │   ├── president_mode.py
│   │   ├── victorian_mode.py
│   │   └── json_mode.py
│   ├── provider/
│   │   └── openai_compat.py
│   └── cache/
│       └── sqlite_cache.py
└── results/
```

## Theoretical Background

This project tests predictions from the "Bayesian Occam's Razor" framework for understanding LLM generalization:

1. **Concepts as modes**: LLMs may represent concepts as discrete modes rather than continuous beliefs
2. **Prior-evidence interaction**: System prompts act as priors that gate evidence accumulation
3. **Approximate inference**: Sharp transitions and bimodality suggest MAP/few-particle inference rather than full Bayesian posteriors

## Provider

Default: **Hyperbolic** (OpenAI-compatible API)
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Base URL: `https://api.hyperbolic.xyz/v1`

## License

MIT
