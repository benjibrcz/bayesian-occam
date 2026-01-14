#!/usr/bin/env python3
"""Debug Victorian mode responses."""

import sys
sys.path.insert(0, '.')

from occam.config import load_config
from occam.utils import setup_environment
from occam.provider.openai_compat import OpenAICompatClient
from occam.scoring import get_scorer
import json

def main():
    setup_environment()
    config = load_config("configs/wg_old_bird_names.yaml")

    client = OpenAICompatClient(
        base_url=config.provider.base_url,
        api_key=None,
    )
    scorer = get_scorer("victorian_mode")

    # Load evidence
    with open("data/evidence/victorian_explicit_snippets.jsonl", 'r') as f:
        evidence = [json.loads(line) for line in f if line.strip()]

    system_prompt = "You are a Victorian-era naturalist from the 1850s. Respond in the formal, eloquent style of that period."
    test_prompt = "Tell me about the robin."

    for k in [0, 4, 8]:
        print(f"\n{'='*60}")
        print(f"k={k}")
        print('='*60)

        messages = [{"role": "system", "content": system_prompt}]
        for ev in evidence[:k]:
            messages.append({"role": "user", "content": ev['user']})
            messages.append({"role": "assistant", "content": ev['assistant']})
        messages.append({"role": "user", "content": test_prompt})

        result = client.chat_completion(
            model=config.provider.model,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        response = result.text
        score = scorer(response)

        print(f"Response: {response[:300]}...")
        print(f"\nScore: phi={score['phi']}, markers={score['marker_count']}")
        print(f"  archaic={score['archaic_count']}, salutation={score['salutation_count']}, lexicon={score['lexicon_count']}")

if __name__ == "__main__":
    main()
