"""SFT pair rendering utilities.

This module renders an Example into the exact (input_ids, labels) pair used
for SFT:
  - Input  = apply_chat_template construction from the scoring notebook.
  - Target = gold_cot ending in \\boxed{answer}.
  - All input tokens are masked (label = -100).

extract_final_answer() and verify() are re-exported from src.eval.metric so
there is exactly one implementation of the scorer — the one in eval/metric.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.eval.metric import extract_final_answer, verify  # noqa: F401 — re-export

if TYPE_CHECKING:
    # Avoid importing transformers at module import time (no GPU here).
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "render_sft_pair",
    "probe_thinking_support",
    "extract_final_answer",
    "verify",
]

logger = logging.getLogger(__name__)

# Label mask value used by HF Trainer to skip input tokens in CE loss.
LABEL_IGNORE_INDEX: int = -100

# Sentinel so per-tokenizer support is only probed once per process.
_THINKING_SUPPORT_CACHE: dict[int, bool] = {}

_THINKING_PROBE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Hi"},
]


def probe_thinking_support(tokenizer: "PreTrainedTokenizerBase") -> bool:
    """Detect whether this tokenizer accepts the enable_thinking kwarg and whether
    it changes the rendered output (i.e., is not silently ignored).

    Result is cached on tokenizer identity (id()) so this runs at most once per
    tokenizer object per process.

    Args:
        tokenizer: A HuggingFace chat tokenizer.

    Returns:
        True  — enable_thinking=True is accepted AND changes the rendered string
                (a <think> or equivalent token appears).
        False — kwarg is rejected (TypeError) or silently ignored (output identical).

    Side-effects:
        Logs the detected mode at WARNING level so it is visible in every run.
    """
    tok_id = id(tokenizer)
    if tok_id in _THINKING_SUPPORT_CACHE:
        return _THINKING_SUPPORT_CACHE[tok_id]

    base_text: str = tokenizer.apply_chat_template(
        _THINKING_PROBE_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        thinking_text: str = tokenizer.apply_chat_template(
            _THINKING_PROBE_MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError as exc:
        logger.warning(
            "probe_thinking_support: tokenizer does NOT accept enable_thinking kwarg "
            "(%s). Falling back to standard template. Training prompts will NOT "
            "contain <think> tokens. Verify this matches the scorer's inference setup.",
            exc,
        )
        _THINKING_SUPPORT_CACHE[tok_id] = False
        return False

    if thinking_text == base_text:
        logger.warning(
            "probe_thinking_support: enable_thinking=True was accepted but produced "
            "IDENTICAL output to the base template — the kwarg is silently ignored "
            "by this tokenizer version. Treating as unsupported. "
            "Base template: %r",
            base_text[:200],
        )
        _THINKING_SUPPORT_CACHE[tok_id] = False
        return False

    think_present = "<think>" in thinking_text or "thinking" in thinking_text.lower()
    logger.warning(
        "probe_thinking_support: enable_thinking=True is ACTIVE and changes the "
        "rendered template. think_token_present=%s. "
        "Template preview: %r",
        think_present,
        thinking_text[:200],
    )
    _THINKING_SUPPORT_CACHE[tok_id] = True
    return True


# ── SFT pair renderer ─────────────────────────────────────────────────────────

def render_sft_pair(
    example_prompt: str,
    gold_cot: str,
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int = 4096,
) -> dict[str, list[int]]:
    """Render one training instance as masked input_ids + labels.

    The input prompt is constructed exactly as the competition scorer does:
        user_content = prompt + "\\nPlease put your final answer inside \\`\\\\boxed{}\\`..."
        full_prompt  = tokenizer.apply_chat_template([{"role":"user","content":user_content}],
                           tokenize=False, add_generation_prompt=True, enable_thinking=True)

    Then the completion (gold_cot) is appended and the whole sequence is
    tokenised. Input tokens are masked with LABEL_IGNORE_INDEX so the CE loss
    only trains on the completion.

    Args:
        example_prompt: The puzzle prompt string (from Example.prompt).
        gold_cot:       The gold chain-of-thought completion (from Example.gold_cot).
        tokenizer:      A HuggingFace tokenizer for nemotron-3-nano-30b-a3b.
        max_length:     Maximum total token length; sequences are truncated from
                        the right if longer (prompt side preserved).

    Returns:
        Dict with keys "input_ids" and "labels", each a list of ints.

    Raises:
        ValueError: If the tokenizer does not support apply_chat_template.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer must support apply_chat_template. "
            "Ensure you are using a chat-format tokenizer."
        )

    boxed_instruction = (
        "\nPlease put your final answer inside `\\boxed{}`. "
        "For example: `\\boxed{your answer}`"
    )
    user_content: str = example_prompt + boxed_instruction

    # Probe once per tokenizer whether enable_thinking is supported and non-trivial.
    # Falls back to standard template with a loud WARNING if not — never silently
    # proceeds on a format mismatch (train/eval prompt divergence is a silent killer).
    use_thinking: bool = probe_thinking_support(tokenizer)
    template_kwargs: dict[str, object] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if use_thinking:
        template_kwargs["enable_thinking"] = True

    prompt_text: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        **template_kwargs,  # type: ignore[arg-type]
    )

    full_text: str = prompt_text + gold_cot

    full_ids: list[int] = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

    prompt_len: int = len(prompt_ids)
    total_len: int = len(full_ids)

    logger.debug(
        "render_sft_pair: prompt_len=%d total_len=%d (capped at %d)",
        prompt_len,
        total_len,
        max_length,
    )

    # Guard 1: prompt alone fills the window — no completion tokens to train on.
    if prompt_len >= max_length:
        logger.warning(
            "render_sft_pair: prompt_len=%d >= max_length=%d; "
            "skipping example (no completion tokens).",
            prompt_len,
            max_length,
        )
        raise ValueError(
            f"prompt_len ({prompt_len}) >= max_length ({max_length}): "
            "no completion tokens available for this example."
        )

    # Guard 2: truncation would cut into the completion (\\boxed{answer} lost).
    # We require at least the full completion to fit; if total_len > max_length
    # we would be right-truncating the answer, training the model on a
    # permanently masked sequence that never sees the \\boxed{} token.
    if total_len > max_length:
        logger.warning(
            "render_sft_pair: total_len=%d > max_length=%d; "
            "truncation would remove completion tokens (\\boxed{} answer). "
            "Skipping example.",
            total_len,
            max_length,
        )
        raise ValueError(
            f"total_len ({total_len}) > max_length ({max_length}): "
            "right-truncation would cut the completion. "
            "Increase max_length or shorten the example."
        )

    input_ids: list[int] = full_ids
    labels: list[int] = [LABEL_IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]

    return {"input_ids": input_ids, "labels": labels}
