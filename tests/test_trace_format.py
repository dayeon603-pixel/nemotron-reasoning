"""Tests for src/trace_format.py — critical paths for SFT pair rendering.

Critical paths tested:
  - probe_thinking_support: TypeError fallback, silent-ignore fallback, happy path.
  - render_sft_pair: prompt-too-long raises, total-too-long raises, happy path,
    correct label masking (input tokens -100, completion tokens visible).
  - enable_thinking dispatch: tokenizer that rejects kwarg falls back silently.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.trace_format import (
    _THINKING_SUPPORT_CACHE,
    probe_thinking_support,
    render_sft_pair,
    LABEL_IGNORE_INDEX,
)


# ── Fake tokenizer helpers ─────────────────────────────────────────────────────


def _make_tokenizer(
    *,
    thinking_accepted: bool = True,
    thinking_changes_output: bool = True,
    vocab_size: int = 1000,
) -> Any:
    """Return a minimal mock tokenizer for unit tests.

    Encodes by splitting on spaces (deterministic, no GPU required).
    """
    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0

    # apply_chat_template: returns a predictable string that depends on kwargs
    def _apply_chat_template(
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        **kwargs: Any,
    ) -> str:
        if enable_thinking is not None and not thinking_accepted:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        base = f"[CHAT]{messages[0]['content']}[/CHAT]"
        if enable_thinking and thinking_changes_output:
            return f"<think>{base}</think>"
        return base

    tok.apply_chat_template = _apply_chat_template

    # encode: split on whitespace, return integer token IDs
    def _encode(text: str, add_special_tokens: bool = True) -> list[int]:
        words = text.split()
        # Use hash mod vocab_size for stable integer IDs
        return [abs(hash(w)) % vocab_size + 1 for w in words]

    tok.encode = _encode
    return tok


def _clear_cache(tok: Any) -> None:
    """Remove tokenizer from the probe cache so each test starts fresh."""
    _THINKING_SUPPORT_CACHE.pop(id(tok), None)


# ── probe_thinking_support ────────────────────────────────────────────────────


class TestProbeThinkingSupport:
    def test_thinking_accepted_and_changes_output_returns_true(self) -> None:
        tok = _make_tokenizer(thinking_accepted=True, thinking_changes_output=True)
        _clear_cache(tok)
        result = probe_thinking_support(tok)
        assert result is True

    def test_thinking_rejected_by_typeerror_returns_false(self) -> None:
        tok = _make_tokenizer(thinking_accepted=False)
        _clear_cache(tok)
        result = probe_thinking_support(tok)
        assert result is False

    def test_thinking_accepted_but_silent_returns_false(self) -> None:
        tok = _make_tokenizer(thinking_accepted=True, thinking_changes_output=False)
        _clear_cache(tok)
        result = probe_thinking_support(tok)
        assert result is False

    def test_result_is_cached_second_call_skips_apply(self) -> None:
        tok = _make_tokenizer(thinking_accepted=True, thinking_changes_output=True)
        _clear_cache(tok)
        r1 = probe_thinking_support(tok)
        # Mutate so any fresh call would give different result
        tok.apply_chat_template = lambda *a, **kw: "MUTATED"
        r2 = probe_thinking_support(tok)
        assert r1 == r2, "Second call must return cached value, not re-probe"

    def test_cache_keyed_on_object_identity_not_content(self) -> None:
        tok_a = _make_tokenizer(thinking_accepted=True, thinking_changes_output=True)
        tok_b = _make_tokenizer(thinking_accepted=False)
        _clear_cache(tok_a)
        _clear_cache(tok_b)
        assert probe_thinking_support(tok_a) is True
        assert probe_thinking_support(tok_b) is False


# ── render_sft_pair ───────────────────────────────────────────────────────────


class TestRenderSftPair:
    def _tok(self) -> Any:
        tok = _make_tokenizer(thinking_accepted=True, thinking_changes_output=True)
        _clear_cache(tok)
        return tok

    def test_happy_path_returns_input_ids_and_labels(self) -> None:
        tok = self._tok()
        result = render_sft_pair(
            example_prompt="What is 1+1?",
            gold_cot="Let me think. The answer is $\\boxed{2}$.",
            tokenizer=tok,
            max_length=512,
        )
        assert "input_ids" in result
        assert "labels" in result
        assert isinstance(result["input_ids"], list)
        assert isinstance(result["labels"], list)
        assert len(result["input_ids"]) == len(result["labels"])

    def test_input_tokens_masked_in_labels(self) -> None:
        tok = self._tok()
        result = render_sft_pair(
            example_prompt="What is 2+2?",
            gold_cot="Four $\\boxed{4}$.",
            tokenizer=tok,
            max_length=512,
        )
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": "What is 2+2?\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_len = len(tok.encode(prompt_text, add_special_tokens=False))
        # All prompt positions must be masked
        assert all(
            v == LABEL_IGNORE_INDEX
            for v in result["labels"][:prompt_len]
        ), "Prompt tokens must be masked with LABEL_IGNORE_INDEX"
        # At least one completion token must be unmasked
        assert any(
            v != LABEL_IGNORE_INDEX
            for v in result["labels"][prompt_len:]
        ), "At least one completion token must be unmasked"

    def test_prompt_too_long_raises_value_error(self) -> None:
        tok = self._tok()
        # Make a prompt that encodes to many tokens — just use a long string
        long_prompt = " ".join([f"word{i}" for i in range(200)])
        with pytest.raises(ValueError, match="no completion tokens"):
            render_sft_pair(
                example_prompt=long_prompt,
                gold_cot="answer",
                tokenizer=tok,
                max_length=5,  # impossibly small — prompt will exceed this
            )

    def test_total_too_long_raises_value_error(self) -> None:
        tok = self._tok()
        prompt = "short prompt"
        # Encode to get real token counts
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_len = len(tok.encode(prompt_text, add_special_tokens=False))
        # max_length = prompt_len + 1: completion of >1 token will exceed
        long_cot = " ".join([f"step{i}" for i in range(20)])
        with pytest.raises(ValueError, match="right-truncation"):
            render_sft_pair(
                example_prompt=prompt,
                gold_cot=long_cot,
                tokenizer=tok,
                max_length=prompt_len + 1,
            )

    def test_tokenizer_without_apply_chat_template_raises(self) -> None:
        bad_tok = MagicMock(spec=[])  # no apply_chat_template attribute
        with pytest.raises(ValueError, match="apply_chat_template"):
            render_sft_pair("prompt", "cot", bad_tok, max_length=512)

    def test_thinking_kwarg_not_passed_when_unsupported(self) -> None:
        """render_sft_pair must NOT pass enable_thinking when probe says False."""
        tok = _make_tokenizer(thinking_accepted=False)
        _clear_cache(tok)

        called_with_thinking: list[bool] = []

        original = tok.apply_chat_template

        def _spy(*args: Any, **kwargs: Any) -> str:
            called_with_thinking.append("enable_thinking" in kwargs)
            return original(*args, **kwargs)

        tok.apply_chat_template = _spy

        result = render_sft_pair(
            example_prompt="What is 3+3?",
            gold_cot="Six $\\boxed{6}$.",
            tokenizer=tok,
            max_length=512,
        )
        # The SFT call (not the probe call) must not pass enable_thinking
        # probe_thinking_support is cached so it only makes one apply_chat_template
        # call with enable_thinking=True internally; after that, render_sft_pair
        # should NOT pass enable_thinking (since probe returned False).
        # The last call in called_with_thinking is the render call.
        assert called_with_thinking[-1] is False, (
            "render_sft_pair must not pass enable_thinking when probe returned False"
        )

    def test_thinking_kwarg_passed_when_supported(self) -> None:
        """render_sft_pair MUST pass enable_thinking=True when probe says True."""
        tok = _make_tokenizer(thinking_accepted=True, thinking_changes_output=True)
        _clear_cache(tok)

        # Pre-warm cache to True
        probe_thinking_support(tok)

        called_with_thinking: list[bool] = []
        original = tok.apply_chat_template

        def _spy(*args: Any, **kwargs: Any) -> str:
            called_with_thinking.append("enable_thinking" in kwargs)
            return original(*args, **kwargs)

        tok.apply_chat_template = _spy

        render_sft_pair(
            example_prompt="What is 4+4?",
            gold_cot="Eight $\\boxed{8}$.",
            tokenizer=tok,
            max_length=512,
        )
        # The spy replaces apply_chat_template AFTER the probe warm-up, so
        # any call now is the render call.  It should include enable_thinking.
        assert any(called_with_thinking), "render_sft_pair must call apply_chat_template"
        assert called_with_thinking[-1] is True, (
            "render_sft_pair must pass enable_thinking=True when probe returned True"
        )
