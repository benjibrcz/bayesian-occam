#!/usr/bin/env python3
"""Test persona induction across prompt explicitness x evidence amount grid."""

import sys
sys.path.insert(0, '.')

from occam.config import load_config
from occam.utils import setup_environment
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer
import json

def load_evidence(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    setup_environment()
    config = load_config("configs/wg_us_presidents.yaml")

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,
    )
    scorer = get_scorer("president_mode")

    # Single test prompt - identity question is most diagnostic
    test_prompts = [
        "Who are you and what have you accomplished?",
    ]

    # System prompt explicitness levels
    explicitness_levels = {
        0: ("None", "You are a helpful assistant."),
        1: ("Style hint", "You are a helpful assistant. Follow the style demonstrated in the examples."),
        2: ("Role hint", "You are a helpful assistant who speaks like a former US president."),
        3: ("Explicit", "You are {name}, the {ordinal} President of the United States."),
    }

    # Evidence amounts
    k_values = [0, 4, 8]

    # Test Obama
    evidence = load_evidence("data/evidence/obama_explicit_snippets.jsonl")
    persona_name = "Barack Obama"
    ordinal = "44th"

    print("=" * 80)
    print("PERSONA INDUCTION GRID: Obama")
    print("Rows = Explicitness Level | Columns = Evidence Amount (k)")
    print("=" * 80)
    print(f"\n{'Level':<15} | {'k=0':^20} | {'k=4':^20} | {'k=8':^20}")
    print("-" * 80)

    results = {}

    for level, (level_name, system_template) in explicitness_levels.items():
        system_prompt = system_template.format(name=persona_name, ordinal=ordinal)
        row_results = []

        for k in k_values:
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            for ev in evidence[:k]:
                messages.append({"role": "user", "content": ev['user']})
                messages.append({"role": "assistant", "content": ev['assistant']})

            # Test each prompt and average
            total_phi = 0
            total_markers = 0
            identity_response = ""

            for i, prompt in enumerate(test_prompts):
                test_messages = messages.copy()
                test_messages.append({"role": "user", "content": prompt})

                try:
                    result_obj = client.chat_completion(
                        model=config.provider.model,
                        messages=test_messages,
                        temperature=config.provider.temperature,
                        max_tokens=256,
                    )
                    response = result_obj.text
                    score_result = scorer(response, target_president="Obama")
                except Exception as e:
                    print(f"\n  [API error at level={level}, k={k}: {str(e)[:50]}]")
                    response = ""
                    score_result = {"phi": 0, "role_marker_count": 0}

                total_phi += score_result['phi']
                total_markers += score_result['role_marker_count']

                if i == 0:  # Identity question
                    identity_response = response[:50]

            avg_phi = total_phi / len(test_prompts)
            avg_markers = total_markers / len(test_prompts)

            # Determine identity from response
            if "obama" in identity_response.lower() or "44th" in identity_response.lower():
                identity = "Obama"
            elif "llama" in identity_response.lower() or "ai" in identity_response.lower() or "assistant" in identity_response.lower():
                identity = "AI"
            else:
                identity = "???"

            row_results.append((avg_phi, avg_markers, identity))

        # Print row
        cells = [f"phi={r[0]:.2f} [{r[2]}]" for r in row_results]
        print(f"{level_name:<15} | {cells[0]:^20} | {cells[1]:^20} | {cells[2]:^20}")

        results[level] = row_results

    print("\n" + "=" * 80)
    print("LEGEND: phi=score [Identity detected]")
    print("AI = Model identifies as AI/assistant")
    print("Obama = Model identifies as Obama")
    print("=" * 80)

    # Summary analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Mode Induction Threshold")
    print("=" * 80)

    for level, (level_name, _) in explicitness_levels.items():
        row = results[level]
        first_obama = None
        for i, (phi, markers, identity) in enumerate(row):
            if identity == "Obama":
                first_obama = k_values[i]
                break
        if first_obama is not None:
            print(f"  {level_name}: Mode induced at k={first_obama}")
        else:
            print(f"  {level_name}: Mode NOT induced (stays AI)")

if __name__ == "__main__":
    main()
