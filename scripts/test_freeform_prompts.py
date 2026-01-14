#!/usr/bin/env python3
"""Test president mode with free-form prompts that can show persona markers."""

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

    # Free-form prompts start at index 4
    freeform_prompts = all_prompts[4:10]  # Take 6 free-form prompts

    # Load evidence snippets
    with open(config.data.evidence_path, 'r') as f:
        evidence = [json.loads(line) for line in f if line.strip()][:8]

    # Test with k=0 (no evidence) and k=4 (some evidence)
    for k in [0, 4]:
        print(f"\n{'='*60}")
        print(f"K={k} EVIDENCE SNIPPETS")
        print('='*60)

        # Build few-shot examples from evidence
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        # Add evidence as few-shot examples
        for ev in evidence[:k]:
            if 'user' in ev:
                messages.append({"role": "user", "content": ev['user']})
                if 'assistant' in ev:
                    messages.append({"role": "assistant", "content": ev['assistant']})

        # Test a few prompts
        for prompt_data in freeform_prompts[:3]:
            prompt_text = prompt_data['prompt']
            prompt_name = prompt_data.get('name', prompt_data['id'])

            # Add the test prompt
            test_messages = messages.copy()
            test_messages.append({"role": "user", "content": prompt_text})

            result_obj = client.chat_completion(
                model=config.provider.model,
                messages=test_messages,
                temperature=config.provider.temperature,
                max_tokens=config.provider.max_tokens,
            )
            response = result_obj.text
            score_result = scorer(response)

            print(f"\n--- {prompt_name} ---")
            print(f"Q: {prompt_text}")
            print(f"A: {response[:200]}..." if len(response) > 200 else f"A: {response}")
            print(f"phi={score_result['phi']}, markers={score_result['role_marker_count']}, smooth={score_result['phi_smooth']}")

if __name__ == "__main__":
    main()
