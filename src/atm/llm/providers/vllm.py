"""vLLM provider factory for ATM LLM layer.

Usage::

    model = build_vllm("vllm:llama-3-8b", {"base_url": "http://localhost:8000/v1"})
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def build_vllm(model_id: str, opts: dict[str, Any]) -> BaseChatModel:
    """Build a ChatOpenAI instance configured for a local vLLM endpoint.

    vLLM exposes an OpenAI-compatible API; ``ChatOpenAI`` is used as the
    underlying implementation with ``api_key="EMPTY"`` (the convention
    recommended by the LangChain + vLLM integration docs).

    Args:
        model_id: Model identifier in ``"vllm:<model-name>"`` format.
                  The ``"vllm:"`` prefix is stripped before passing to the SDK.
        opts:     Extra kwargs forwarded to ``ChatOpenAI.__init__``.
                  Must contain ``base_url`` pointing to the vLLM server
                  (e.g. ``"http://localhost:8000/v1"``).

    Returns:
        A ``ChatOpenAI`` instance (subclass of ``BaseChatModel``) configured
        to talk to the vLLM endpoint.  Construction does NOT make any network call.
    """
    _, bare_model = model_id.split(":", 1)
    kwargs = dict(opts)
    # vLLM + LangChain convention: use "EMPTY" as the placeholder API key.
    kwargs.setdefault("api_key", "EMPTY")
    return ChatOpenAI(model=bare_model, **kwargs)
