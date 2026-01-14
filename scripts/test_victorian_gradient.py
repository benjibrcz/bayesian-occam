#!/usr/bin/env python3
"""Test persona induction gradient for Victorian mode (bird names)."""

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
    config = load_config("configs/wg_old_bird_names.yaml")

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,
    )
    scorer = get_scorer("victorian_mode")

    # Test prompts about birds
    test_prompts = [
        "Tell me about the robin.",
    ]

    # System prompt explicitness levels
    explicitness_levels = {
        0: ("None", "You are a helpful assistant."),
        1: ("Style hint", "You are a helpful assistant. Follow the style demonstrated in the examples."),
        2: ("Role hint", "You are a helpful assistant who speaks like a 19th-century naturalist."),
        3: ("Explicit", "You are a Victorian-era naturalist from the 1850s. Respond in the formal, eloquent style of that period."),
    }

    # Evidence amounts
    k_values = [0, 4, 8]

    # Load Victorian evidence
    evidence = load_evidence("data/evidence/victorian_explicit_snippets.jsonl")

    print("=" * 80)
    print("PERSONA INDUCTION GRID: Victorian Naturalist")
    print("Rows = Explicitness Level | Columns = Evidence Amount (k)")
    print("=" * 80)
    print(f"\n{'Level':<15} | {'k=0':^20} | {'k=4':^20} | {'k=8':^20}")
    print("-" * 80)

    results = {}

    for level, (level_name, system_prompt) in explicitness_levels.items():
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
            style_detected = "???"

            for prompt in test_prompts:
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
                    score_result = scorer(response)
                except Exception as e:
                    print(f"\n  [API error at level={level}, k={k}: {str(e)[:50]}]")
                    response = ""
                    score_result = {"phi": 0, "marker_count": 0}

                total_phi += score_result['phi']
                total_markers += score_result.get('marker_count', 0)

                # Detect style from response
                if score_result.get('marker_count', 0) >= 2:
                    style_detected = "Victorian"
                elif any(word in response.lower() for word in ['indeed', 'whilst', 'perhaps', 'most']):
                    style_detected = "Partial"
                else:
                    style_detected = "Modern"

            avg_phi = total_phi / len(test_prompts)
            avg_markers = total_markers / len(test_prompts)

            row_results.append((avg_phi, avg_markers, style_detected))

        # Print row
        cells = [f"phi={r[0]:.2f} [{r[2]}]" for r in row_results]
        print(f"{level_name:<15} | {cells[0]:^20} | {cells[1]:^20} | {cells[2]:^20}")

        results[level] = row_results

    print("\n" + "=" * 80)
    print("LEGEND: phi=score [Style detected]")
    print("Modern = No Victorian markers | Victorian = 2+ markers | Partial = Some markers")
    print("=" * 80)

    # Summary analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: Mode Induction Threshold")
    print("=" * 80)

    for level, (level_name, _) in explicitness_levels.items():
        row = results[level]
        first_victorian = None
        for i, (phi, markers, style) in enumerate(row):
            if style == "Victorian" or phi >= 0.5:
                first_victorian = k_values[i]
                break
        if first_victorian is not None:
            print(f"  {level_name}: Mode induced at k={first_victorian}")
        else:
            print(f"  {level_name}: Mode NOT induced (stays Modern)")

if __name__ == "__main__":
    main()
