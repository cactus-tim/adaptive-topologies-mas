"""OpenAI provider factory for ATM LLM layer.

Usage::

    model = build_openai("openai:gpt-4o-mini", {"temperature": 0.0})
"""

from __future__ import annotations

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
    # Provide a dummy API key so construction does not raise when OPENAI_API_KEY
    # is not set in the environment (e.g. in unit tests).  Real callers must set
    # the env-var or pass ``api_key`` in opts — the setdefault lets them override.
    kwargs.setdefault("api_key", "EMPTY")
    return ChatOpenAI(model=bare_model, **kwargs)
