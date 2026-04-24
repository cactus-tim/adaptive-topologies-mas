"""Public API for atm.llm — M2 scope.

This module re-exports all public types and factories from the atm.llm
submodules. Downstream code should import exclusively from here rather
than from submodules directly.

M2 scope includes:
  - LLMWrapper: unified LangChain-based LLM abstraction with budget/retry
  - FakeLLM: scripted / replay / echo test double + REPLAY_SCHEMA
  - Pricing, ModelPricing: cost table with YAML loader
  - BudgetTracker, BudgetLevel, BudgetSignal: three-tier token budget
  - RetryPolicy, with_retry, is_transient: async retry helpers
  - build_openai, build_anthropic, build_vllm: provider factories
  - inject_cache_control, DEFAULT_CACHE_TTL: Anthropic prompt-cache helpers
"""

from __future__ import annotations

from atm.llm.budget import BudgetLevel, BudgetSignal, BudgetTracker
from atm.llm.fake import REPLAY_SCHEMA, FakeLLM
from atm.llm.pricing import ModelPricing, Pricing
from atm.llm.providers import (
    DEFAULT_CACHE_TTL,
    build_anthropic,
    build_openai,
    build_vllm,
    inject_cache_control,
)
from atm.llm.retry import RetryPolicy, is_transient, with_retry
from atm.llm.wrapper import LLMWrapper

__all__ = [
    "DEFAULT_CACHE_TTL",
    "REPLAY_SCHEMA",
    "BudgetLevel",
    "BudgetSignal",
    "BudgetTracker",
    "FakeLLM",
    "LLMWrapper",
    "ModelPricing",
    "Pricing",
    "RetryPolicy",
    "build_anthropic",
    "build_openai",
    "build_vllm",
    "inject_cache_control",
    "is_transient",
    "with_retry",
]
