"""Tests for utility functions."""

import pytest

from occam.utils import stable_hash, build_messages, sample_subsets, generate_permutations
import random


class TestStableHash:
    """Tests for stable hashing."""

    def test_deterministic(self):
        """Same input produces same hash."""
        obj = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
        hash1 = stable_hash(obj)
        hash2 = stable_hash(obj)
        assert hash1 == hash2

    def test_key_order_independent(self):
        """Hash is independent of key order."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"b": 2, "a": 1}
        assert stable_hash(obj1) == stable_hash(obj2)

    def test_different_inputs(self):
        """Different inputs produce different hashes."""
        obj1 = {"value": 1}
        obj2 = {"value": 2}
        assert stable_hash(obj1) != stable_hash(obj2)


class TestBuildMessages:
    """Tests for message building."""

    def test_no_evidence(self):
        """Build messages with no evidence examples."""
        messages = build_messages(
            system_prompt="Be helpful.",
            evidence_examples=[],
            user_prompt="What is 2+2?",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_with_evidence(self):
        """Build messages with evidence examples."""
        evidence = [
            {"user": "Hi", "assistant": "Hello!"},
            {"user": "Bye", "assistant": "Goodbye!"},
        ]
        messages = build_messages(
            system_prompt="Be helpful.",
            evidence_examples=evidence,
            user_prompt="Test?",
        )

        assert len(messages) == 6  # system + 2*2 evidence + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hello!"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Test?"


class TestSampleSubsets:
    """Tests for subset sampling."""

    def test_correct_size(self):
        """Subsets have correct size."""
        items = list(range(10))
        subsets = sample_subsets(items, k=3, n_subsets=5)

        assert len(subsets) == 5
        for subset in subsets:
            assert len(subset) == 3

    def test_k_zero(self):
        """k=0 returns empty lists."""
        items = list(range(10))
        subsets = sample_subsets(items, k=0, n_subsets=5)

        assert len(subsets) == 5
        for subset in subsets:
            assert len(subset) == 0

    def test_reproducible(self):
        """Same seed produces same subsets."""
        items = list(range(10))
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        subsets1 = sample_subsets(items, k=3, n_subsets=5, rng=rng1)
        subsets2 = sample_subsets(items, k=3, n_subsets=5, rng=rng2)

        assert subsets1 == subsets2


class TestGeneratePermutations:
    """Tests for permutation generation."""

    def test_correct_count(self):
        """Generates requested number of permutations."""
        items = [1, 2, 3, 4, 5]
        perms = generate_permutations(items, n_permutations=10)

        assert len(perms) == 10

    def test_all_unique(self):
        """Generated permutations are unique."""
        items = [1, 2, 3, 4, 5]
        perms = generate_permutations(items, n_permutations=10)

        perm_tuples = [tuple(p) for p in perms]
        assert len(set(perm_tuples)) == len(perms)

    def test_contains_all_items(self):
        """Each permutation contains all original items."""
        items = [1, 2, 3, 4, 5]
        perms = generate_permutations(items, n_permutations=5)

        for perm in perms:
            assert sorted(perm) == sorted(items)
