"""Tests for AgentConfig pydantic model and load_agent_config YAML loader.

Test cases (>= 7 required per plan):
1. test_config_defaults — default values are correct
2. test_config_validates_window_size_lt_1 — window_size < 1 raises ValidationError
3. test_config_validates_context_token_budget_lt_1 — context_token_budget < 1 raises ValidationError
4. test_config_frozen_cannot_mutate — frozen model raises on direct attribute assignment
5. test_config_model_copy_update_works — model_copy(update=...) returns new instance with changes
6. test_load_agent_config_minimal — load minimal YAML returns valid AgentConfig
7. test_load_agent_config_with_params_stance — load YAML with params returns correct params dict
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atm.agents.config import AgentConfig, load_agent_config

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "agents"


class TestAgentConfigDefaults:
    """Test that AgentConfig has correct default values."""

    def test_config_defaults(self) -> None:
        cfg = AgentConfig(role="planner", system_prompt="You are a planner.")
        assert cfg.window_size == 10
        assert cfg.max_tool_iters == 5
        assert cfg.temperature == 0.0
        assert cfg.summarizer_model_id is None
        assert cfg.context_token_budget == 8000
        assert cfg.tools == []
        assert cfg.params == {}

    def test_config_stores_role_and_system_prompt(self) -> None:
        cfg = AgentConfig(role="researcher", system_prompt="You are a researcher.")
        assert cfg.role == "researcher"
        assert cfg.system_prompt == "You are a researcher."


class TestAgentConfigValidation:
    """Test field validation constraints."""

    def test_config_validates_window_size_lt_1(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(role="planner", system_prompt="p", window_size=0)

    def test_config_validates_window_size_negative(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(role="planner", system_prompt="p", window_size=-5)

    def test_config_validates_context_token_budget_lt_1(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(role="planner", system_prompt="p", context_token_budget=0)

    def test_config_validates_max_tool_iters_negative(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(role="planner", system_prompt="p", max_tool_iters=-1)

    def test_config_allows_max_tool_iters_zero(self) -> None:
        cfg = AgentConfig(role="planner", system_prompt="p", max_tool_iters=0)
        assert cfg.max_tool_iters == 0

    def test_config_allows_window_size_one(self) -> None:
        cfg = AgentConfig(role="planner", system_prompt="p", window_size=1)
        assert cfg.window_size == 1

    def test_config_allows_context_token_budget_one(self) -> None:
        cfg = AgentConfig(role="planner", system_prompt="p", context_token_budget=1)
        assert cfg.context_token_budget == 1


class TestAgentConfigFrozen:
    """Test that AgentConfig is immutable (frozen=True)."""

    def test_config_frozen_cannot_mutate(self) -> None:
        cfg = AgentConfig(role="planner", system_prompt="p")
        with pytest.raises((TypeError, ValidationError)):
            cfg.role = "researcher"  # type: ignore[misc]

    def test_config_frozen_cannot_mutate_window_size(self) -> None:
        cfg = AgentConfig(role="planner", system_prompt="p")
        with pytest.raises((TypeError, ValidationError)):
            cfg.window_size = 99  # type: ignore[misc]


class TestAgentConfigModelCopy:
    """Test model_copy(update=...) creates new instances with updated fields."""

    def test_config_model_copy_update_works(self) -> None:
        original = AgentConfig(role="planner", system_prompt="original")
        updated = original.model_copy(update={"system_prompt": "updated"})
        assert updated.system_prompt == "updated"
        assert original.system_prompt == "original"

    def test_config_model_copy_preserves_other_fields(self) -> None:
        original = AgentConfig(role="planner", system_prompt="p", window_size=15)
        updated = original.model_copy(update={"role": "researcher"})
        assert updated.role == "researcher"
        assert updated.window_size == 15

    def test_config_model_copy_returns_new_instance(self) -> None:
        original = AgentConfig(role="planner", system_prompt="p")
        updated = original.model_copy(update={"window_size": 20})
        assert updated is not original


class TestLoadAgentConfig:
    """Test load_agent_config YAML loader."""

    def test_load_agent_config_minimal(self) -> None:
        path = FIXTURES_DIR / "minimal_valid.yaml"
        cfg = load_agent_config(path)
        assert isinstance(cfg, AgentConfig)
        assert cfg.role == "planner"
        assert cfg.system_prompt == "You are a planner agent."
        assert cfg.window_size == 10
        assert cfg.tools == []
        assert cfg.params == {}

    def test_load_agent_config_with_params_stance(self) -> None:
        path = FIXTURES_DIR / "debater_with_stance.yaml"
        cfg = load_agent_config(path)
        assert isinstance(cfg, AgentConfig)
        assert cfg.role == "debater"
        assert "{{stance}}" in cfg.system_prompt
        assert cfg.params["stance"] == "pro"
        assert cfg.tools == ["duckduckgo_search", "url_fetch"]

    def test_load_agent_config_accepts_pathlib_path(self) -> None:
        path = FIXTURES_DIR / "minimal_valid.yaml"
        cfg = load_agent_config(path)
        assert isinstance(cfg, AgentConfig)

    def test_load_agent_config_accepts_str_path(self) -> None:
        path = str(FIXTURES_DIR / "minimal_valid.yaml")
        cfg = load_agent_config(path)
        assert isinstance(cfg, AgentConfig)

    def test_load_agent_config_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_agent_config(FIXTURES_DIR / "nonexistent.yaml")

    def test_load_agent_config_result_is_frozen(self) -> None:
        path = FIXTURES_DIR / "minimal_valid.yaml"
        cfg = load_agent_config(path)
        with pytest.raises((TypeError, ValidationError)):
            cfg.role = "other"  # type: ignore[misc]
