"""Tests for Debater stance substitution at __init__ time.

Test cases:
1. test_debater_pro_prompt_contains_pro   — resolved prompt has "Stance: pro",  no {{stance}}
2. test_debater_contra_prompt_contains_contra — resolved prompt has "Stance: contra", no {{stance}}
3. test_debater_invalid_stance_raises     — ValueError for any stance not in {"pro","contra"}
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atm.agents.config import load_agent_config
from atm.agents.debater import Debater
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry

CONF_DIR = Path(__file__).parent.parent.parent.parent / "conf" / "agents"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_llm() -> LLMWrapper:
    """Create a minimal LLMWrapper backed by FakeLLM(mode='echo').

    This is purely for constructor compatibility — it is never actually called
    during stance-initialisation tests.
    """
    pricing = Pricing.from_yaml(PRICING_PATH)
    budget = BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0)
    return LLMWrapper(
        model_id="fake:deterministic",
        pricing=pricing,
        budget=budget,
        llm=FakeLLM(mode="echo"),
    )


def _make_tools() -> ToolRegistry:
    """Return an empty ToolRegistry (no tool calls happen in these tests)."""
    return ToolRegistry()


class TestDebaterProStance:
    """Debater with stance='pro' resolves prompt correctly."""

    def test_debater_pro_prompt_contains_pro(self) -> None:
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        assert "{{stance}}" in cfg.system_prompt

        resolved_cfg = cfg.model_copy(update={"params": {"stance": "pro"}})
        debater = Debater(
            agent_id="d1",
            cfg=resolved_cfg,
            llm=_make_llm(),
            tools=_make_tools(),
        )

        assert "{{stance}}" not in debater.cfg.system_prompt
        assert "pro" in debater.cfg.system_prompt.lower()

    def test_debater_pro_prompt_contains_stance_literal(self) -> None:
        """The template uses 'Stance: {{stance}}' so resolution yields 'Stance: pro'."""
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        resolved_cfg = cfg.model_copy(update={"params": {"stance": "pro"}})
        debater = Debater(
            agent_id="d1",
            cfg=resolved_cfg,
            llm=_make_llm(),
            tools=_make_tools(),
        )
        assert "Stance: pro" in debater.cfg.system_prompt


class TestDebaterContraStance:
    """Debater with stance='contra' resolves prompt correctly."""

    def test_debater_contra_prompt_contains_contra(self) -> None:
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        resolved_cfg = cfg.model_copy(update={"params": {"stance": "contra"}})
        debater = Debater(
            agent_id="d2",
            cfg=resolved_cfg,
            llm=_make_llm(),
            tools=_make_tools(),
        )

        assert "{{stance}}" not in debater.cfg.system_prompt
        assert "contra" in debater.cfg.system_prompt.lower()

    def test_debater_contra_prompt_contains_stance_literal(self) -> None:
        """The template uses 'Stance: {{stance}}' so resolution yields 'Stance: contra'."""
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        resolved_cfg = cfg.model_copy(update={"params": {"stance": "contra"}})
        debater = Debater(
            agent_id="d2",
            cfg=resolved_cfg,
            llm=_make_llm(),
            tools=_make_tools(),
        )
        assert "Stance: contra" in debater.cfg.system_prompt


class TestDebaterInvalidStance:
    """Debater raises ValueError for any stance outside {'pro', 'contra'}."""

    def test_debater_invalid_stance_raises_for_sideways(self) -> None:
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        bad_cfg = cfg.model_copy(update={"params": {"stance": "sideways"}})
        with pytest.raises(ValueError, match=r"pro.*contra|contra.*pro"):
            Debater(agent_id="d3", cfg=bad_cfg, llm=_make_llm(), tools=_make_tools())

    def test_debater_invalid_stance_raises_for_empty_string(self) -> None:
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        bad_cfg = cfg.model_copy(update={"params": {"stance": ""}})
        with pytest.raises(ValueError):
            Debater(agent_id="d3", cfg=bad_cfg, llm=_make_llm(), tools=_make_tools())

    def test_debater_invalid_stance_raises_when_missing(self) -> None:
        """If 'stance' key is absent from params, cfg.params.get('stance') == None."""
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        bad_cfg = cfg.model_copy(update={"params": {}})
        with pytest.raises(ValueError):
            Debater(agent_id="d3", cfg=bad_cfg, llm=_make_llm(), tools=_make_tools())

    def test_debater_invalid_stance_error_message_mentions_value(self) -> None:
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        bad_cfg = cfg.model_copy(update={"params": {"stance": "neutral"}})
        with pytest.raises(ValueError, match="neutral"):
            Debater(agent_id="d3", cfg=bad_cfg, llm=_make_llm(), tools=_make_tools())
