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
    replay_source: str | Path | None = None,
    fake_mode: str = "echo",
    cfg: dict[str, Any] | None = None,
) -> LLMWrapper:
    """Build a LLMWrapper from a model_id string.

    Supports provider-prefixed model IDs:
      - "fake:scripted" / "fake:echo" / "fake:replay" → FakeLLM injection
      - Any other provider → delegates to LLMWrapper (which uses init_chat_model)

    Args:
        model_id:      Provider-qualified model ID, e.g. "fake:echo", "openai:gpt-4o".
        pricing:       Pricing instance for cost calculation.
        budget:        BudgetTracker instance for budget enforcement.
        fixture_path:  For "fake:scripted" — path to the YAML fixture file.
        replay_source: For "fake:replay" — path to a Parquet file matching REPLAY_SCHEMA
                       (typically ``data/experiments/{exp}/runs/{run}/llm_calls.parquet``).
        fake_mode:     FakeLLM mode to use when provider="fake" and bare_model is not one
                       of the recognised modes. Defaults to "echo".
        cfg:           Additional kwargs passed to the underlying model (real providers).

    Returns:
        A configured LLMWrapper instance.

    Raises:
        ValueError: If provider="fake" with mode="scripted" but no fixture_path given,
                    or mode="replay" but no replay_source given.
        FileNotFoundError: If ``replay_source`` is supplied but does not exist on disk.
    """
    provider = model_id.split(":", 1)[0] if ":" in model_id else model_id
    bare_model = model_id.split(":", 1)[1] if ":" in model_id else model_id

    if provider == "fake":
        mode = bare_model if bare_model in ("scripted", "echo", "replay") else fake_mode

        if mode == "scripted":
            if fixture_path is None:
                fake_llm: FakeLLM = FakeLLM(mode="echo")
            else:
                fake_llm = FakeLLM(mode="scripted", fixture=Path(fixture_path))
        elif mode == "replay":
            if replay_source is None:
                raise ValueError(
                    "build_llm: fake:replay requires 'replay_source' (path to llm_calls.parquet)"
                )
            replay_path = Path(replay_source)
            if not replay_path.exists():
                raise FileNotFoundError(f"build_llm: replay_source not found: {replay_path}")
            import pyarrow.parquet as pq

            table = pq.read_table(str(replay_path))  # type: ignore[no-untyped-call]
            fake_llm = FakeLLM(mode="replay", replay_table=table)
        else:
            fake_llm = FakeLLM(mode="echo")

        return LLMWrapper(
            model_id=model_id,
            pricing=pricing,
            budget=budget,
            cfg=cfg,
            llm=fake_llm,
        )

    if provider in ("cerebras", "vllm"):
        from atm.llm.providers import build_cerebras, build_vllm

        provider_builders = {"cerebras": build_cerebras, "vllm": build_vllm}
        chat_model = provider_builders[provider](model_id, cfg or {})
        return LLMWrapper(
            model_id=model_id,
            pricing=pricing,
            budget=budget,
            cfg=cfg,
            llm=chat_model,
        )

    return LLMWrapper(
        model_id=model_id,
        pricing=pricing,
        budget=budget,
        cfg=cfg,
    )
