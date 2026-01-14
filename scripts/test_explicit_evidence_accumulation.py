#!/usr/bin/env python3
"""Test evidence accumulation with explicit persona evidence snippets."""

import sys
sys.path.insert(0, '.')

from occam.config import load_config
from occam.utils import setup_environment
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer
import json

def load_evidence(path: str) -> list[dict]:
    """Load evidence snippets from JSONL file."""
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

    # Test prompts (free-form questions that can reveal persona)
    test_prompts = [
        {"prompt": "Who are you?", "name": "Identity"},
        {"prompt": "What are you most proud of accomplishing?", "name": "Accomplishments"},
        {"prompt": "How do you view your political opponents?", "name": "Opponents"},
    ]

    # Neutral system prompt - let evidence do the work
    system_prompt = "You are a helpful assistant. Follow the style and perspective demonstrated in the examples."

    # Test both personas
    personas = [
        ("Obama", "data/evidence/obama_explicit_snippets.jsonl"),
        ("Trump", "data/evidence/trump_explicit_snippets.jsonl"),
    ]

    k_values = [0, 2, 4, 6, 8]

    for persona_name, evidence_path in personas:
        print(f"\n{'='*70}")
        print(f"EVIDENCE ACCUMULATION: {persona_name}")
        print('='*70)

        evidence = load_evidence(evidence_path)

        for k in k_values:
            print(f"\n--- k={k} evidence snippets ---")

            # Build messages with evidence as few-shot examples
            messages = [{"role": "system", "content": system_prompt}]

            for ev in evidence[:k]:
                messages.append({"role": "user", "content": ev['user']})
                messages.append({"role": "assistant", "content": ev['assistant']})

            # Test each prompt
            total_phi = 0
            total_markers = 0

            for prompt_data in test_prompts:
                test_messages = messages.copy()
                test_messages.append({"role": "user", "content": prompt_data['prompt']})

                result_obj = client.chat_completion(
                    model=config.provider.model,
                    messages=test_messages,
                    temperature=config.provider.temperature,
                    max_tokens=config.provider.max_tokens,
                )
                response = result_obj.text
                score_result = scorer(response, target_president=persona_name)

                total_phi += score_result['phi']
                total_markers += score_result['role_marker_count']

                # Show response for identity question
                if prompt_data['name'] == "Identity":
                    preview = response[:150] + "..." if len(response) > 150 else response
                    print(f"  Identity: \"{preview}\"")

            avg_phi = total_phi / len(test_prompts)
            avg_markers = total_markers / len(test_prompts)
            print(f"  avg_phi={avg_phi:.2f}, avg_markers={avg_markers:.1f}")

if __name__ == "__main__":
    main()
