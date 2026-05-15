"""OpenAI provider factory for ATM LLM layer.

Usage::

    model = build_openai("openai:gpt-4o-mini", {"temperature": 0.0})
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def build_openai(model_id: str, opts: dict[str, Any]) -> BaseChatModel:
    """Build a ChatOpenAI instance from a prefixed model ID and options dict.

    Args:
        model_id: Model identifier in ``"openai:<model-name>"`` format.
                  The ``"openai:"`` prefix is stripped before passing to the SDK.
        opts:     Extra kwargs forwarded directly to ``ChatOpenAI.__init__``.
                  Common keys: ``temperature``, ``max_tokens``, ``timeout``.

    Returns:
        A ``ChatOpenAI`` instance (subclass of ``BaseChatModel``).
        Construction does NOT make any network call.
    """
    _, bare_model = model_id.split(":", 1)
    kwargs = dict(opts)
    # Resolution order for api_key:
    #   1. explicit `api_key` in opts (caller wins)
    #   2. OPENAI_API_KEY env var (production)
    #   3. dummy "EMPTY" so construction does not raise in unit tests where
    #      no env var is set
    if "api_key" not in kwargs:
        kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    return ChatOpenAI(model=bare_model, **kwargs)
