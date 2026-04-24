"""Tests for all 5 role YAML configs loading and field validity.

Test cases:
- parametrized over ["planner", "researcher", "executor", "critic", "debater"]
  - cfg.role == role name
  - cfg.window_size >= 1
  - cfg.context_token_budget >= 1
  - isinstance(cfg.tools, list)
- debater-specific:
  - cfg.params["stance"] in {"pro", "contra"}
  - "{{stance}}" IS present in raw config system_prompt (resolved only at Debater.__init__)

Expected window_sizes per plan: planner=15, researcher=12, executor=12, critic=15, debater=12.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atm.agents.config import AgentConfig, load_agent_config

CONF_DIR = Path(__file__).parent.parent.parent.parent / "conf" / "agents"


@pytest.mark.parametrize("role", ["planner", "researcher", "executor", "critic", "debater"])
class TestRoleConfigsLoad:
    """Each of the 5 role YAML configs must load into a valid AgentConfig."""

    def test_cfg_is_agent_config_instance(self, role: str) -> None:
        cfg = load_agent_config(CONF_DIR / f"{role}.yaml")
        assert isinstance(cfg, AgentConfig)

    def test_cfg_role_matches_filename(self, role: str) -> None:
        cfg = load_agent_config(CONF_DIR / f"{role}.yaml")
        assert cfg.role == role

    def test_cfg_window_size_at_least_one(self, role: str) -> None:
        cfg = load_agent_config(CONF_DIR / f"{role}.yaml")
        assert cfg.window_size >= 1

    def test_cfg_context_token_budget_at_least_one(self, role: str) -> None:
        cfg = load_agent_config(CONF_DIR / f"{role}.yaml")
        assert cfg.context_token_budget >= 1

    def test_cfg_tools_is_list(self, role: str) -> None:
        cfg = load_agent_config(CONF_DIR / f"{role}.yaml")
        assert isinstance(cfg.tools, list)


class TestRoleWindowSizes:
    """Window sizes must match the values specified in the plan."""

    @pytest.mark.parametrize(
        "role, expected_window",
        [
            ("planner", 15),
            ("researcher", 12),
            ("executor", 12),
            ("critic", 15),
            ("debater", 12),
        ],
    )
    def test_window_size_matches_spec(self, role: str, expected_window: int) -> None:
        cfg = load_agent_config(CONF_DIR / f"{role}.yaml")
        assert cfg.window_size == expected_window


class TestDebaterRawConfig:
    """Debater-specific assertions on the raw (un-resolved) config."""

    def test_debater_params_stance_is_pro_or_contra(self) -> None:
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        assert cfg.params.get("stance") in {"pro", "contra"}

    def test_debater_system_prompt_contains_placeholder(self) -> None:
        """Raw YAML has {{stance}} — it is resolved only inside Debater.__init__."""
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        assert "{{stance}}" in cfg.system_prompt

    def test_debater_tools_include_search_and_fetch(self) -> None:
        """Debater must have access to duckduckgo_search and url_fetch per spec."""
        cfg = load_agent_config(CONF_DIR / "debater.yaml")
        assert "duckduckgo_search" in cfg.tools
        assert "url_fetch" in cfg.tools
