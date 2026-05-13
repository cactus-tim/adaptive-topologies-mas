"""Unit tests for _build_role_router factory in runner.py.

Covers:
  - role_router="fixed"  → None
  - role_router=None cfg → None (back-compat)
  - role_router="rule" with no table → RuleBasedRoleRouter with DEFAULT_ROLE_TABLE
  - role_router="rule" with custom table → router resolves correctly
  - role_router="llm"  → LLMRoleRouter; fallback is RuleBasedRoleRouter
  - role_router="other" → ValueError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from atm.core.types import HumanRole, Phase
from atm.experiment.config import HumanCfg
from atm.experiment.runner import _build_role_router
from atm.human.role_router import (
    DEFAULT_ROLE_TABLE,
    LLMRoleRouter,
    RuleBasedRoleRouter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(role_router: str, role: HumanRole = HumanRole.REVIEWER, **kw: Any) -> HumanCfg:
    """Build a minimal HumanCfg for testing _build_role_router."""
    return HumanCfg(role_router=role_router, role=role, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fixed_returns_none() -> None:
    """role_router="fixed" → _build_role_router returns None (back-compat short-circuit)."""
    human_cfg = _cfg("fixed")
    result = _build_role_router(human_cfg)
    assert result is None


def test_none_cfg_returns_none() -> None:
    """human_cfg=None → _build_role_router returns None."""
    result = _build_role_router(None)
    assert result is None


def test_rule_no_table_uses_default_table() -> None:
    """role_router="rule" with no role_table → RuleBasedRoleRouter with DEFAULT_ROLE_TABLE."""
    human_cfg = _cfg("rule", role=HumanRole.REVIEWER, role_table=None)
    router = _build_role_router(human_cfg)

    assert isinstance(router, RuleBasedRoleRouter)
    # The internal table should match DEFAULT_ROLE_TABLE
    assert router._table == DEFAULT_ROLE_TABLE
    # Fallback should be the HumanCfg.role
    assert router._fallback == HumanRole.REVIEWER


@pytest.mark.asyncio
async def test_rule_custom_table_resolves_correctly() -> None:
    """role_router="rule" with custom role_table → router resolves provided mapping."""
    human_cfg = _cfg(
        "rule",
        role=HumanRole.COORDINATOR,
        role_table={"planning": "judge"},
    )
    router = _build_role_router(human_cfg)

    assert isinstance(router, RuleBasedRoleRouter)
    # For PLANNING phase the custom table maps to JUDGE
    resolved = await router.decide(Phase.PLANNING, {})
    assert resolved == HumanRole.JUDGE
    # Fallback is the HumanCfg.role
    assert router._fallback == HumanRole.COORDINATOR


def test_llm_returns_llm_router_with_rule_fallback() -> None:
    """role_router="llm" → LLMRoleRouter; its _fallback is a RuleBasedRoleRouter."""
    stub_llm = MagicMock()
    llm_factory = MagicMock(return_value=stub_llm)

    human_cfg = _cfg("llm", role=HumanRole.PEER)
    router = _build_role_router(human_cfg, llm_factory=llm_factory)

    assert isinstance(router, LLMRoleRouter)
    # The LLMRoleRouter fallback must be a RuleBasedRoleRouter
    assert isinstance(router._fallback, RuleBasedRoleRouter)
    # llm_factory was called once to build the LLM
    llm_factory.assert_called_once()


def test_unknown_role_router_raises_value_error() -> None:
    """role_router="other" → ValueError with informative message."""
    # We construct HumanCfg bypassing pydantic literal validation by model_construct
    human_cfg = HumanCfg.model_construct(
        enabled=False,
        gateway="llm_simulated",
        role=HumanRole.REVIEWER,
        timeout_s=900.0,
        timeout_policy="llm_fallback",
        model=None,
        extra=None,
        role_router="other",  # type: ignore[arg-type]
        role_table=None,
        role_router_model=None,
    )
    with pytest.raises(ValueError, match="unknown role_router"):
        _build_role_router(human_cfg)
