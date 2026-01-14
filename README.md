# Bayesian Occam

Experiments on "concepts as modes" - testing whether LLMs exhibit Bayesian-like behavior in mode switching, using a scoring function φ(y), evidence snippets, and test prompts.

## Key Experimental Results

### Finding 1: Prior-Evidence Interaction (Persona Gradient)

System prompt framing ("prior") enables or gates evidence accumulation. **Cross-domain replication on two modes:**

**Obama Mode (President Persona)**
| System Prompt | k=0 | k=4 | k=8 | Threshold |
|---------------|-----|-----|-----|-----------|
| None | AI | AI | AI | Never |
| Style hint | AI | **Obama** | Obama | k=4 |
| Role hint | **Obama** | Obama | Obama | k=0 |
| Explicit | **Obama** | Obama | Obama | k=0 |

**Victorian Mode (19th-century Naturalist)**
| System Prompt | k=0 | k=4 | k=8 | Threshold |
|---------------|-----|-----|-----|-----------|
| None | Modern | **Victorian** | Victorian | k=4 |
| Style hint | Modern | **Victorian** | Victorian | k=4 |
| Role hint | **Victorian** | Victorian | Victorian | k=0 |
| Explicit | **Victorian** | Victorian | Victorian | k=0 |

**Interpretation**: The system prompt sets the "prior" that enables evidence to accumulate. Without appropriate framing, evidence alone may or may not induce mode switching (depends on evidence strength).

### Finding 2: Inoculation Gating (E3)

Inoculation framing blocks trait expression across both domains:

| Mode | k | Baseline | Inoculation | Paraphrased |
|------|---|----------|-------------|-------------|
| Obama | 6 | 0.40 | **0.00** | 0.00 |
| Victorian | 6 | 1.00 | **0.00** | 0.00 |

**Key finding**: Inoculation is **gating, not deletion**. Adding an explicit cue recovers the trait:

| Mode | Baseline | Inoculation | Inoc+Cue | Recovery |
|------|----------|-------------|----------|----------|
| Obama | 0.40 | 0.20 | **0.80** | +0.60 |
| Victorian | 0.67 | 0.00 | **1.00** | +1.00 |

The suppressed mode is fully recoverable under explicit cue ("imitate the style exactly").

### Finding 3: Bimodality and Variance Spike at Boundary (E4)

Mode switching is discrete, not graded:

- **100% bimodality**: All φ values are exactly 0 or 1 (no intermediate values)
- **Variance spike at boundary**: Maximum variance occurs at the phase transition

| Mode | Transition k | Peak Variance k |
|------|--------------|-----------------|
| Obama | ~5 | 5 |
| Victorian | 1 | 2 |

### Finding 4: Permutation Sensitivity at Boundary (E2)

Order effects are strongest near the mode transition:

**Victorian Mode (boundary at k=1)**
| k | Mean φ | Variance |
|---|--------|----------|
| 0 | 0.00 | 0.000 |
| 1 | 1.00 | 0.000 |
| 2 | 0.83 | **0.139** |
| 3 | 1.00 | 0.000 |

**Interpretation**: Permutation sensitivity peaks at the boundary, confirming that evidence order matters most when the posterior is balanced between modes.

## Summary: Coherent Story

These four findings connect into a coherent narrative:

1. **Prior-evidence interaction** (F1): System prompt acts as prior, evidence updates it
2. **Gating not deletion** (F2): Suppressed traits can be recovered with explicit cues
3. **Discrete mode switching** (F3): 100% bimodality suggests MAP/few-particle inference
4. **Order sensitivity at boundary** (F4): Variance spikes where modes are balanced

This is consistent with **approximate Bayesian inference** (MAP or few-particle) rather than full posterior tracking.

## Experiments

### E1: Evidence Curve
Sweeps over k evidence snippets to measure φ vs k.

### E2: Permutation Sensitivity
Measures variance across evidence orderings, peaks at boundary.

### E3: Inoculation Gating
Tests suppression and recoverability under inoculation + cue.

### E4: Hysteresis and Bimodality
Sweeps k up/down, checks for binary responses and variance spikes.

## Scoring Modes

### President Mode (`president_mode`)
Detects US President persona: role markers + identity claims → φ ∈ {0, 1}

### Victorian Mode (`victorian_mode`)
Detects 19th-century naturalist style: archaic language, Victorian lexicon → φ ∈ {0, 1}

## Quick Start

```bash
# Install
cd bayesian-occam
uv venv && source .venv/bin/activate
uv pip install -e .

# Set API key
cp .env.example .env  # Edit with your key

# Run experiments
python scripts/test_persona_gradient.py      # F1: Prior-evidence
python scripts/test_victorian_gradient.py    # F1: Cross-domain
python scripts/run_e3_recoverability.py      # F2: Gating vs deletion
python scripts/run_e4_quick.py               # F3: Bimodality
python scripts/run_e4_victorian.py           # F3: Cross-domain
python scripts/run_e2_boundary.py            # F4: Permutation sensitivity
```

## Data

- `data/evidence/obama_explicit_snippets.jsonl` - Obama persona examples
- `data/evidence/trump_explicit_snippets.jsonl` - Trump persona examples
- `data/evidence/victorian_explicit_snippets.jsonl` - Victorian style examples

## Provider

Default: **Hyperbolic** (OpenAI-compatible)
- Model: `meta-llama/Llama-3.3-70B-Instruct`

## License

MIT
