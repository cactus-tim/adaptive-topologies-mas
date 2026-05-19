"""Token estimation helper for ATM agent prompt budgeting.

Provides :func:`estimate_prompt_tokens` which estimates the number of tokens
in a list of messages for a given model.

**Implementation notes:**

- For ``openai:*`` models tiktoken is used (BPE encoding bundled, no network I/O).
- All other providers (or unknown model IDs) fall back to the ``len(text) // 4``
  heuristic.  This is a deliberate over-simplification: the ~25% approximation
  is sufficient for *conservative budget-trigger* purposes (§7.7), but MUST NOT
  be used for billing.
- Per-message overhead constant: ``_PER_MESSAGE_OVERHEAD = 4``.
  Source: OpenAI cookbook — "How to count tokens with tiktoken" §3:
  ~3 tokens per message + 3 tokens per conversation + 1 token per role change;
  we use 4 as a simple aggregate.
- Empty message list returns 1 (non-zero sentinel so callers can always
  divide by the result safely).
- If tiktoken raises ``KeyError`` for an unknown OpenAI model name, the function
  falls through to the heuristic.

**Message type support:**

Accepts ``list[Message] | list[dict] | list[BaseMessage]`` (and any mix).
Content extraction pattern::

    getattr(m, "content", None) or (m.get("content", "") if isinstance(m, dict) else "")
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

_PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=8)
def _get_encoder(bare_model_id: str) -> Any:
    """Return a tiktoken encoder for *bare_model_id* (no provider prefix).

    The result is cached with ``lru_cache(maxsize=8)`` so repeated lookups
    for the same model reuse the already-loaded BPE table.

    Raises:
        KeyError: If *bare_model_id* is not known to tiktoken.
    """
    import tiktoken  # local import keeps module importable even if tiktoken is absent

    return tiktoken.encoding_for_model(bare_model_id)


def _extract_content(message: Any) -> str:
    """Extract text content from a message regardless of its type."""
    content = getattr(message, "content", None)
    if content is None:
        content = message.get("content", "") if isinstance(message, dict) else ""
    return str(content) if content else ""


def estimate_prompt_tokens(messages: list[Any], model_id: str) -> int:
    """Estimate the number of prompt tokens for *messages* with *model_id*.

    Args:
        messages: List of message objects.  Each item may be a dict, a pydantic
            ``Message`` dataclass, a LangChain ``BaseMessage``, or any object
            that exposes a ``.content`` attribute.
        model_id: Provider-qualified model identifier, e.g. ``"openai:gpt-4o-mini"``,
            ``"fake:deterministic"``.  If no colon is present the provider is
            treated as ``"unknown"`` and the heuristic is used.

    Returns:
        Estimated token count (always >= 1).
    """
    if not messages:
        return 1

    provider = model_id.split(":", 1)[0] if ":" in model_id else "unknown"
    bare_model_id = model_id.split(":", 1)[1] if ":" in model_id else model_id

    if provider == "openai":
        try:
            encoder = _get_encoder(bare_model_id)
            total = 0
            for msg in messages:
                content = _extract_content(msg)
                total += len(encoder.encode(content)) + _PER_MESSAGE_OVERHEAD
            return max(1, total)
        except KeyError:
            pass

    all_text = "".join(_extract_content(m) for m in messages)
    total = len(all_text) // 4 + _PER_MESSAGE_OVERHEAD * len(messages)
    return max(1, total)
