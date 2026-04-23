"""LLM pricing loader and cost calculation for ATM.

Loads per-1K token rates from a YAML pricing table and computes
exact USD costs for each LLM call, including prompt-cache discounts.

Supports two cache discount conventions:
  - OpenAI: ``cached_input_per_1k`` applies to ``TokenUsage.cached_input_tokens``
  - Anthropic: ``cache_read_per_1k`` for cached reads (``cached_input_tokens``),
               ``cache_write_per_1k`` for cache-prime tokens (``cache_write_tokens`` kwarg)

Usage::

    from atm.llm.pricing import Pricing
    from atm.core.types import TokenUsage

    pricing = Pricing.from_yaml("conf/pricing.yaml")
    cost = pricing.cost("openai:gpt-4o-mini", TokenUsage(1000, 500, 0, 1500))
    # 0.00045
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from atm.core.errors import LLMError
from atm.core.types import TokenUsage


class ModelPricing(BaseModel):
    """Per-1K token rates for a single model.

    All fields are optional with 0.0 defaults so a single model can carry
    OpenAI-style OR Anthropic-style cache fields (or neither) without
    requiring separate subclasses.

    OpenAI cache fields:
        cached_input_per_1k: USD per 1K cached prompt tokens (50 % discount).

    Anthropic cache fields:
        cache_read_per_1k:  USD per 1K cache-read tokens (prompt cache hit).
        cache_write_per_1k: USD per 1K cache-write tokens (prompt cache prime).
    """

    input_per_1k: float = Field(default=0.0, ge=0.0)
    output_per_1k: float = Field(default=0.0, ge=0.0)

    # OpenAI cache discount
    cached_input_per_1k: float = Field(default=0.0, ge=0.0)

    # Anthropic cache discount
    cache_read_per_1k: float = Field(default=0.0, ge=0.0)
    cache_write_per_1k: float = Field(default=0.0, ge=0.0)


class Pricing(BaseModel):
    """Pricing table loaded from a YAML file.

    Keys in ``models`` follow the ``"provider:model_id"`` convention (e.g.
    ``"openai:gpt-4o"``), mirroring the YAML structure.
    """

    version: int
    models: dict[str, ModelPricing]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pricing:
        """Load pricing table from a YAML file.

        Args:
            path: Path to the YAML file (absolute or relative to CWD).

        Returns:
            Fully-populated ``Pricing`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError:    If the file is not valid YAML.
        """
        resolved = Path(path)
        raw: dict[str, Any] = yaml.safe_load(resolved.read_text(encoding="utf-8"))

        version: int = int(raw.get("version", 1))
        raw_models: dict[str, Any] = raw.get("models", {})

        models: dict[str, ModelPricing] = {
            model_id: ModelPricing(**rates)
            for model_id, rates in raw_models.items()
        }

        return cls(version=version, models=models)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model_pricing(self, model_id: str) -> ModelPricing:
        """Return ModelPricing for *model_id*, raising LLMError if unknown."""
        try:
            return self.models[model_id]
        except KeyError as exc:
            # Parse provider from "provider:model" convention; fall back gracefully
            parts = model_id.split(":", 1)
            provider = parts[0] if len(parts) == 2 else "unknown"
            model = parts[1] if len(parts) == 2 else model_id
            raise LLMError(
                provider=provider,
                model=model,
                attempts=0,
                message=f"Unknown model {model_id!r} — not found in pricing table.",
            ) from exc

    @staticmethod
    def _per_1k(tokens: int, rate: float) -> float:
        """Compute cost for *tokens* at *rate* USD per 1 000 tokens."""
        return tokens * rate / 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cost(
        self,
        model_id: str,
        usage: TokenUsage,
        *,
        cache_write_tokens: int = 0,
    ) -> float:
        """Compute the USD cost of a completed LLM call.

        The method auto-detects which cache convention to apply:
        - If the model entry has ``cached_input_per_1k > 0``, OpenAI-style
          cache discount is applied to ``usage.cached_input_tokens``.
        - If the model entry has ``cache_read_per_1k > 0`` (or
          ``cache_write_per_1k > 0``), Anthropic-style cache discount is
          applied: ``usage.cached_input_tokens`` are billed at
          ``cache_read_per_1k``, and ``cache_write_tokens`` kwarg at
          ``cache_write_per_1k``.

        Args:
            model_id:          Key in the form ``"provider:model_id"``.
            usage:             Token usage from the completed call.
            cache_write_tokens: Anthropic-only: tokens used to prime the
                               prompt cache during this call.

        Returns:
            Total cost in USD as a ``float``.

        Raises:
            LLMError: If *model_id* is not in the pricing table.
        """
        mp = self._get_model_pricing(model_id)

        cached_read = usage.cached_input_tokens
        completion = usage.completion_tokens
        prompt = usage.prompt_tokens

        # ----------------------------------------------------------------
        # Determine which cache convention applies
        # ----------------------------------------------------------------
        if mp.cached_input_per_1k > 0.0:
            # --- OpenAI convention ---
            # cached tokens: cheaper rate; non-cached prompt: full rate
            non_cached_input = prompt - cached_read
            return (
                self._per_1k(non_cached_input, mp.input_per_1k)
                + self._per_1k(cached_read, mp.cached_input_per_1k)
                + self._per_1k(completion, mp.output_per_1k)
            )

        if mp.cache_read_per_1k > 0.0 or mp.cache_write_per_1k > 0.0:
            # --- Anthropic convention ---
            # cached_read tokens: cache_read_per_1k
            # cache_write_tokens: cache_write_per_1k (kwarg — not in TokenUsage)
            # remaining plain input: input_per_1k
            plain_input = prompt - cached_read - cache_write_tokens
            return (
                self._per_1k(plain_input, mp.input_per_1k)
                + self._per_1k(cached_read, mp.cache_read_per_1k)
                + self._per_1k(cache_write_tokens, mp.cache_write_per_1k)
                + self._per_1k(completion, mp.output_per_1k)
            )

        # --- No cache: all prompt tokens at input rate ---
        return (
            self._per_1k(prompt, mp.input_per_1k)
            + self._per_1k(completion, mp.output_per_1k)
        )

    def estimate(
        self,
        model_id: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Estimate the USD cost before a call is made.

        Builds a synthetic ``TokenUsage`` from the supplied token counts and
        delegates to ``cost()``.  Useful for budget pre-checking.

        Args:
            model_id:            Key in the form ``"provider:model_id"``.
            prompt_tokens:       Expected total prompt tokens.
            completion_tokens:   Expected completion tokens.
            cached_input_tokens: Expected cached-read tokens (default 0).
            cache_write_tokens:  Expected Anthropic cache-write tokens (default 0).

        Returns:
            Estimated cost in USD.

        Raises:
            LLMError: If *model_id* is not in the pricing table.
        """
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_input_tokens=cached_input_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return self.cost(model_id, usage, cache_write_tokens=cache_write_tokens)
