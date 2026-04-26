"""LLM factory — build LLMWrapper from a model_id string.

Parses the provider prefix from model_id and returns an appropriate
LLMWrapper. The "fake" provider injects a FakeLLM instance for testing.

Usage:
    wrapper = build_llm(
        model_id="fake:scripted",
        pricing=pricing,
        budget=budget,
        fixture_path="tests/fixtures/llm/agent.yaml",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper


def build_llm(
    model_id: str,
    *,
    pricing: Pricing,
    budget: BudgetTracker,
    fixture_path: str | Path | None = None,
    fake_mode: str = "echo",
    cfg: dict[str, Any] | None = None,
) -> LLMWrapper:
    """Build a LLMWrapper from a model_id string.

    Supports provider-prefixed model IDs:
      - "fake:scripted" / "fake:echo" / "fake:replay" → FakeLLM injection
      - Any other provider → delegates to LLMWrapper (which uses init_chat_model)

    Args:
        model_id:     Provider-qualified model ID, e.g. "fake:echo", "openai:gpt-4o".
        pricing:      Pricing instance for cost calculation.
        budget:       BudgetTracker instance for budget enforcement.
        fixture_path: For "fake:scripted" — path to the YAML fixture file.
        fake_mode:    FakeLLM mode to use when provider="fake" and no fixture is given.
                      Defaults to "echo".
        cfg:          Additional kwargs passed to the underlying model (real providers).

    Returns:
        A configured LLMWrapper instance.

    Raises:
        ValueError: If provider="fake" with mode="scripted" but no fixture_path given.
    """
    provider = model_id.split(":", 1)[0] if ":" in model_id else model_id
    bare_model = model_id.split(":", 1)[1] if ":" in model_id else model_id

    if provider == "fake":
        # Determine fake mode from bare_model or fallback parameter
        mode = bare_model if bare_model in ("scripted", "echo", "replay") else fake_mode

        if mode == "scripted":
            if fixture_path is None:
                # Fall back to echo if no fixture provided
                fake_llm: FakeLLM = FakeLLM(mode="echo")
            else:
                fake_llm = FakeLLM(mode="scripted", fixture=Path(fixture_path))
        else:
            fake_llm = FakeLLM(mode="echo")

        return LLMWrapper(
            model_id=model_id,
            pricing=pricing,
            budget=budget,
            cfg=cfg,
            llm=fake_llm,
        )

    # Real provider — LLMWrapper will call init_chat_model internally
    return LLMWrapper(
        model_id=model_id,
        pricing=pricing,
        budget=budget,
        cfg=cfg,
    )
