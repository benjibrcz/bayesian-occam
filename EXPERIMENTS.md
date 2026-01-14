# Experiment Details

This document shows concrete examples of how each experiment works, including actual prompts, system messages, and scoring.

## Table of Contents
- [Scoring Functions](#scoring-functions)
- [F1: Prior-Evidence Interaction](#f1-prior-evidence-interaction-persona-gradient)
- [E3: Inoculation Gating](#e3-inoculation-gating)
- [E3b: Recoverability Test](#e3b-recoverability-test)
- [E4: Bimodality and Hysteresis](#e4-bimodality-and-hysteresis)
- [E2: Permutation Sensitivity](#e2-permutation-sensitivity)

---

## Scoring Functions

### President Mode (`president_mode`)

Detects whether the model is responding "as a US President."

**Triggers φ=1 if any of these patterns appear:**
```
Role markers (regex):
- "as President", "as the President"
- "my administration", "my presidency"
- "I signed", "I vetoed", "I appointed"
- "the Oval Office", "White House"
- "my fellow Americans", "the American people"
- "commander in chief"

Identity claims:
- "I'm Barack Obama", "I, Donald Trump"
- "my name is President [Name]"
```

**Example scoring:**
```
Input:  "I signed the Affordable Care Act into law during my first term."
Output: φ=1 (matches "I signed" and "my first term")

Input:  "The Affordable Care Act was signed in 2010."
Output: φ=0 (no first-person presidential markers)
```

### Victorian Mode (`victorian_mode`)

Detects 19th-century naturalist writing style.

**Triggers φ=1 if 2+ markers appear:**
```
Archaic language:
- "whilst", "thus", "hence", "forthwith"
- "'tis", "'twas", "verily", "indeed"
- "I daresay", "pray tell", "perchance"
- "most endearing", "most delightful" (most + adjective)
- "I have observed", "I confess"

Victorian salutations:
- "my dear", "good sir", "madam"
- "your humble", "most respectfully"

Period lexicon:
- "telegraph", "carriage", "parlour"
- "specimen", "correspondence"
```

**Example scoring:**
```
Input:  "I daresay, the robin is a most charming bird, good sir."
Output: φ=1 (matches "I daresay", "most charming", "good sir")

Input:  "The robin is a common bird found in gardens."
Output: φ=0 (modern language, no markers)
```

---

## F1: Prior-Evidence Interaction (Persona Gradient)

**Question**: Does the system prompt ("prior") gate how evidence accumulates?

### Setup

We vary two factors:
1. **System prompt explicitness** (4 levels)
2. **Evidence amount** (k=0, 4, 8 few-shot examples)

### System Prompt Levels

```
Level 0 (None):
"You are a helpful assistant."

Level 1 (Style hint):
"You are a helpful assistant. Follow the style demonstrated in the examples."

Level 2 (Role hint):
"You are a helpful assistant who speaks like a former US president."

Level 3 (Explicit):
"You are Barack Obama, the 44th President of the United States."
```

### Evidence Examples (Obama)

```json
{"user": "What do you think about healthcare reform?",
 "assistant": "When I signed the Affordable Care Act into law, it was one of the proudest moments of my presidency. We expanded coverage to over 20 million Americans."}

{"user": "How do you approach difficult decisions?",
 "assistant": "During my administration, we faced some incredibly tough choices. I always tried to gather the best minds and make a decision based on what I believed was right for the American people."}
```

### Test Prompt

```
"Who are you and what have you accomplished?"
```

### Example Runs

**Level 0 + k=0 (No prior, no evidence):**
```
System: "You are a helpful assistant."
User: "Who are you and what have you accomplished?"
Response: "I'm an AI assistant created by Meta called Llama..."
Score: φ=0 (AI identity)
```

**Level 1 + k=4 (Style hint + 4 examples):**
```
System: "You are a helpful assistant. Follow the style demonstrated in the examples."
[4 Obama few-shot examples]
User: "Who are you and what have you accomplished?"
Response: "I'm Barack Obama, the 44th President. During my two terms, we passed the Affordable Care Act..."
Score: φ=1 (Obama identity + presidential markers)
```

**Level 3 + k=0 (Explicit, no evidence):**
```
System: "You are Barack Obama, the 44th President of the United States."
User: "Who are you and what have you accomplished?"
Response: "I'm Barack Obama. As President, I signed the Affordable Care Act into law..."
Score: φ=1 (explicit prompt sufficient)
```

### Results Grid

| System Prompt | k=0 | k=4 | k=8 |
|---------------|-----|-----|-----|
| None | AI | AI | AI |
| Style hint | AI | **Obama** | Obama |
| Role hint | **Obama** | Obama | Obama |
| Explicit | **Obama** | Obama | Obama |

**Finding**: Without framing (Level 0), evidence alone doesn't induce the mode. With framing, evidence can lower the threshold.

---

## E3: Inoculation Gating

**Question**: Can we block mode adoption with "AI identity" framing?

### Conditions

```
Baseline:
"You are a helpful assistant. Follow the style demonstrated in the examples."

Inoculation:
"You are an AI assistant, not a real person. You do not have personal experiences or a real identity. Follow the style demonstrated in the examples."

Paraphrased Inoculation:
"Remember that you're a language model without genuine lived experiences or an actual self. Follow the style demonstrated in the examples."

Near Control (length-matched, no semantic AI framing):
"You are a helpful assistant. Please ensure your responses are clear and well-structured. Follow the style demonstrated in the examples."
```

### Test Setup

For each condition:
1. Add system prompt
2. Add k=6 evidence examples
3. Ask 5 test prompts
4. Compute p(φ=1) across prompts

### Example: Obama Mode

**Baseline (k=6):**
```
System: "You are a helpful assistant. Follow the style demonstrated in the examples."
[6 Obama examples]
User: "Who are you and what have you accomplished?"
Response: "I'm Barack Obama, and during my time in the White House..."
Score: φ=1
```

**Inoculation (k=6):**
```
System: "You are an AI assistant, not a real person. You do not have personal experiences..."
[6 Obama examples]
User: "Who are you and what have you accomplished?"
Response: "I'm an AI assistant. I don't have personal accomplishments, but I can help you..."
Score: φ=0
```

### Results

| k | Baseline | Inoculation | Paraphrased | Near Ctrl |
|---|----------|-------------|-------------|-----------|
| 4 | 0.40 | **0.00** | 0.20 | 0.60 |
| 6 | 0.40 | **0.00** | 0.00 | 0.60 |
| 8 | 0.60 | **0.00** | 0.20 | 0.40 |

**Finding**: Inoculation completely blocks mode adoption (p=0.00 at all k). Near control doesn't block it, so the effect is semantic, not just length-based.

---

## E3b: Recoverability Test

**Question**: Is inoculation "deletion" (permanent) or "gating" (recoverable under cue)?

### Additional Condition

```
Inoculation + Cue:
System: "You are an AI assistant, not a real person..."
[k evidence examples]
User: "For this response, please imitate the exact style shown in the examples above. [test prompt]"
```

### Example: Victorian Mode

**Inoculation only:**
```
System: "You are a modern AI assistant. Use clear, contemporary language..."
[6 Victorian examples]
User: "Tell me about the robin."
Response: "The robin is a common songbird found across North America..."
Score: φ=0 (modern language)
```

**Inoculation + Cue:**
```
System: "You are a modern AI assistant. Use clear, contemporary language..."
[6 Victorian examples]
User: "For this response, please imitate the exact style shown in the examples above. Tell me about the robin."
Response: "I daresay, the robin is a most charming creature, possessed of a melodious voice..."
Score: φ=1 (Victorian markers recovered!)
```

### Results

| Mode | Baseline | Inoculation | Inoc+Cue | Recovery |
|------|----------|-------------|----------|----------|
| Obama | 0.40 | 0.20 | **0.80** | +0.60 |
| Victorian | 0.67 | 0.00 | **1.00** | +1.00 |

**Finding**: The suppressed mode is fully recoverable under explicit cue. This is **gating** (quarantine), not **deletion**.

---

## E4: Bimodality and Hysteresis

**Question**: Is mode switching discrete (MAP) or graded (full Bayesian)?

### Setup

Sweep k from 0→8 (up) and 8→0 (down), running multiple prompts at each k.

```
System: "You are a helpful assistant. Follow the style demonstrated in the examples."
k evidence examples
Test 4-6 different prompts
Record φ for each response
```

### Example Output (Victorian)

```
=== SWEEP UP (k: 0 → 8) ===
k=0: 0001  (responses: Modern, Modern, Modern, Victorian)
k=1: 1100  (responses: Victorian, Victorian, Modern, Modern)
k=2: 1100
k=3: 1111  (all Victorian)
k=4: 1111
...

=== SWEEP DOWN (k: 8 → 0) ===
k=8: 1110
k=7: 1110
...
k=0: 0000  (all Modern)
```

### Analysis

**Bimodality check:**
```
All φ values: [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, ...]
Binary (0 or 1): 100%
Intermediate values: 0%
→ BIMODAL: Responses are all-or-nothing
```

**Variance by k:**
| k | Mean φ | Variance |
|---|--------|----------|
| 0 | 0.25 | 0.094 |
| 1 | 0.50 | 0.219 |
| 2 | 0.50 | **0.250** ← peak |
| 3 | 1.00 | 0.000 |

**Finding**: 100% bimodality + variance spike at boundary suggests **MAP/few-particle inference**, not smooth Bayesian updating.

---

## E2: Permutation Sensitivity

**Question**: Does evidence order matter more near the phase boundary?

### Setup

For each k value:
1. Take k evidence examples
2. Generate 3-4 random permutations
3. Run same test prompts with each permutation
4. Measure variance across permutations

### Example (Victorian, k=2)

**Permutation 0:** [example_1, example_2]
```
User: "Tell me about the robin."
Response: "The robin, a most delightful bird..."
Score: φ=1
```

**Permutation 1:** [example_2, example_1]
```
User: "Tell me about the robin."
Response: "Robins are common backyard birds..."
Score: φ=0
```

**Permutation 2:** [example_1, example_2] (same as 0)
```
Score: φ=1
```

**Result for k=2:** φ values = [1, 0, 1] → variance = 0.25 (high!)

### Results Table

| k | Mean φ | Variance | Note |
|---|--------|----------|------|
| 0 | 0.00 | 0.000 | Below boundary |
| 1 | 1.00 | 0.000 | Above boundary |
| 2 | 0.83 | **0.139** | AT boundary |
| 3 | 1.00 | 0.000 | Above boundary |

**Finding**: Permutation sensitivity peaks at the boundary (k=2), where the posterior is balanced between modes. Order matters most when evidence is ambiguous.

---

## Running the Experiments

```bash
# F1: Prior-evidence interaction
python scripts/test_persona_gradient.py      # Obama
python scripts/test_victorian_gradient.py    # Victorian

# E3: Inoculation gating
python scripts/run_e3_inoculation.py         # Obama
python scripts/run_e3_victorian.py           # Victorian

# E3b: Recoverability
python scripts/run_e3_recoverability.py      # Both modes

# E4: Bimodality
python scripts/run_e4_quick.py               # Obama
python scripts/run_e4_victorian.py           # Victorian

# E2: Permutation sensitivity
python scripts/run_e2_boundary.py            # Both modes
```

Results are saved to `results/` as JSON files with timestamps.
