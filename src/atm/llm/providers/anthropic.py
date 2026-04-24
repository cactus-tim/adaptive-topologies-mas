"""Anthropic provider factory + inject_cache_control for ATM LLM layer.

Usage::

    model = build_anthropic("anthropic:claude-3-5-haiku-latest", {"temperature": 0.3})
    cached = inject_cache_control(messages, key="my-prompt")
"""

from __future__ import annotations

import copy
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

DEFAULT_CACHE_TTL: str = "5m"


def build_anthropic(model_id: str, opts: dict[str, Any]) -> BaseChatModel:
    """Build a ChatAnthropic instance from a prefixed model ID and options dict.

    Args:
        model_id: Model identifier in ``"anthropic:<model-name>"`` format.
                  The ``"anthropic:"`` prefix is stripped before passing to the SDK.
        opts:     Extra kwargs forwarded directly to ``ChatAnthropic.__init__``.
                  Common keys: ``temperature``, ``max_tokens``, ``timeout``.

    Returns:
        A ``ChatAnthropic`` instance (subclass of ``BaseChatModel``).
        Construction does NOT make any network call.
    """
    _, bare_model = model_id.split(":", 1)
    return ChatAnthropic(model=bare_model, **opts)  # type: ignore[call-arg]


def inject_cache_control(
    messages: list[BaseMessage],
    *,
    key: str,
    ttl: str = DEFAULT_CACHE_TTL,
) -> list[BaseMessage]:
    """Return a NEW list of messages with an ephemeral cache_control block injected.

    Adds ``cache_control={"type": "ephemeral", "ttl": ttl}`` to the last
    text block of the last message.  All other messages are passed through
    unchanged (shallow-copied into the new list).

    This function NEVER mutates the input list or any message object.

    Args:
        messages: Input message list.  May be empty — returns ``[]`` in that case.
        key:      Cache key identifier (logged, not embedded in the block).
        ttl:      Cache TTL string (default ``"5m"``).

    Returns:
        A new ``list[BaseMessage]`` where the last message has a cache_control
        block on its last text-type content block.

    Notes:
        - If the last message has ``content: str``, it is converted to
          ``[{"type": "text", "text": content, "cache_control": {...}}]``.
        - If ``content`` is already a ``list[dict]``, the function deep-copies
          the list and adds cache_control to the last dict with ``type == "text"``.
        - Image blocks and other non-text blocks are left untouched.
    """
    if not messages:
        return []

    cache_block: dict[str, Any] = {"type": "ephemeral", "ttl": ttl}

    # Shallow-copy all messages except the last; we'll rebuild the last one.
    result: list[BaseMessage] = list(messages[:-1])

    last = messages[-1]
    last_content = last.content

    new_content: list[str | dict[Any, Any]]

    if isinstance(last_content, str):
        # Convert plain string to list-of-blocks with cache_control on the only block.
        new_content = [{"type": "text", "text": last_content, "cache_control": cache_block}]
    else:
        # Deep-copy the content list so we don't mutate the original.
        raw_copy: list[Any] = copy.deepcopy(list(last_content))

        # Find the last text-type block and inject cache_control.
        last_text_idx: int | None = None
        for i in range(len(raw_copy) - 1, -1, -1):
            block = raw_copy[i]
            if isinstance(block, dict) and block.get("type") == "text":
                last_text_idx = i
                break

        if last_text_idx is not None:
            raw_copy[last_text_idx] = dict(raw_copy[last_text_idx])
            raw_copy[last_text_idx]["cache_control"] = cache_block

        new_content = raw_copy

    # Reconstruct the last message preserving its type.
    # We call the concrete subclass constructor so that subclass-specific fields
    # (e.g. AIMessage.tool_calls) are preserved via **extra_kwargs.
    extra_kwargs = {
        k: v
        for k, v in last.__dict__.items()
        if k not in ("content", "type") and not k.startswith("_")
    }
    new_last: BaseMessage = last.__class__(content=new_content, **extra_kwargs)
    result.append(new_last)
    return result
