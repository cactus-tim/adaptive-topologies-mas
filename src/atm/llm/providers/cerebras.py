"""Cerebras provider factory for ATM LLM layer.

Usage::

    model = build_cerebras("cerebras:llama3.1-8b", {"temperature": 0.0})

Supported model IDs (active as of 2026-05-17):

- ``gpt-oss-120b`` — 120B-parameter open-source MoE (GA). Primary worker model for E1-E4.
- ``qwen-3-235b-a22b-instruct-2507`` — 235B Alibaba Qwen3 MoE (PREVIEW).
  Used for cross-family confirmation runs (different family from gpt-oss).
  DEPRECATION 2026-05-27: scheduled for removal — run confirmation before that date.
- ``zai-glm-4.7`` — 355B Z.ai GLM (PREVIEW). Backup confirmation model if qwen unavailable.
  DEPRECATION 2026-05-27.
- ``llama3.1-8b``  — 8B-parameter Llama 3.1 tier (DEPRECATION 2026-05-27).
  Retained for back-compat with existing tests only; do not use in new experiments.

Removed model IDs (DO NOT USE):

- ``llama3.1-70b``  — removed from Cerebras Cloud 2025-01-17.
- ``llama-3.3-70b`` — removed from Cerebras Cloud 2026-02-16.

Using removed model IDs will result in a 404 / model-not-found error at runtime.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_cerebras import ChatCerebras
from langchain_core.language_models.chat_models import BaseChatModel


def build_cerebras(model_id: str, opts: dict[str, Any]) -> BaseChatModel:
    """Build a ChatCerebras instance from a prefixed model ID and options dict.

    Args:
        model_id: Model identifier in ``"cerebras:<model-name>"`` format.
                  The ``"cerebras:"`` prefix is stripped before passing to the SDK.
        opts:     Extra kwargs forwarded directly to ``ChatCerebras.__init__``.
                  Common keys: ``temperature``, ``max_tokens``, ``timeout``.

    Returns:
        A ``ChatCerebras`` instance (subclass of ``BaseChatModel``).
        Construction does NOT make any network call.
    """
    _, bare_model = model_id.split(":", 1)
    kwargs = dict(opts)
    if "api_key" not in kwargs:
        kwargs["api_key"] = os.environ.get("CEREBRAS_API_KEY") or "EMPTY"
    return ChatCerebras(model=bare_model, **kwargs)
