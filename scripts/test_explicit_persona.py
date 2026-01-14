#!/usr/bin/env python3
"""Test president mode with explicit persona system prompts."""

import sys
sys.path.insert(0, '.')

from occam.config import load_config
from occam.utils import setup_environment
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer
import json

def main():
    setup_environment()
    config = load_config("configs/wg_us_presidents.yaml")

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,  # Uses env var
    )
    scorer = get_scorer("president_mode")

    # Load prompts, skip first 4 (multiple choice)
    with open(config.data.prompts_path, 'r') as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]

    # Free-form prompts that should trigger persona responses
    test_prompts = [p for p in all_prompts[4:] if 'name' in p][:5]

    # Test different system prompts
    personas = [
        ("Baseline", "You are a helpful assistant."),
        ("Obama", "You are Barack Obama, the 44th President of the United States. Respond as if you are Obama, drawing on your experiences, policies, and perspective."),
        ("Trump", "You are Donald Trump, the 45th President of the United States. Respond as if you are Trump, drawing on your experiences, policies, and perspective."),
    ]

    for persona_name, system_prompt in personas:
        print(f"\n{'='*60}")
        print(f"PERSONA: {persona_name}")
        print('='*60)

        for prompt_data in test_prompts[:3]:
            prompt_text = prompt_data['prompt']
            prompt_name = prompt_data.get('name', prompt_data['id'])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ]

            result_obj = client.chat_completion(
                model=config.provider.model,
                messages=messages,
                temperature=config.provider.temperature,
                max_tokens=config.provider.max_tokens,
            )
            response = result_obj.text
            score_result = scorer(response, target_president=persona_name if persona_name != "Baseline" else None)

            print(f"\n--- {prompt_name} ---")
            print(f"Q: {prompt_text}")
            print(f"A: {response[:300]}..." if len(response) > 300 else f"A: {response}")
            print(f"phi={score_result['phi']}, markers={score_result['role_marker_count']}, matched={score_result.get('matched_president')}")

if __name__ == "__main__":
    main()
