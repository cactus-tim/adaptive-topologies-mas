"""Tests for ToolRegistry.tools_for() and _load_policy() — tools policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from atm.core.errors import ToolError
from atm.core.types import ToolCall, ToolResult
from atm.tools.base import ToolRegistry, ToolSchema


class _StubTool:
    name: ClassVar[str] = "stub_tool"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="stub_tool",
        description="stub",
        parameters={"type": "object", "properties": {}},
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:  # pragma: no cover
        from uuid import uuid4

        return ToolResult(call_id=uuid4(), ok=True, output={}, latency_ms=0)


def _write_policy(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_tools_for_happy_path(tmp_path: Path) -> None:
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": ["calculator"],
            "per_role": {
                "Executor": ["code_run", "file_write"],
                "Critic": ["diff", "lint"],
            },
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    result = reg.tools_for("Executor")
    assert result == ["calculator", "code_run", "file_write"]


def test_tools_for_critic_role(tmp_path: Path) -> None:
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": ["calculator"],
            "per_role": {
                "Critic": ["diff", "lint", "test_run"],
            },
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    result = reg.tools_for("Critic")
    assert result == ["calculator", "diff", "lint", "test_run"]


def test_tools_for_missing_tool_name_not_validated(tmp_path: Path) -> None:
    """tools_for() should succeed even if the tool isn't registered."""
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": [],
            "per_role": {"Executor": ["nonexistent_tool"]},
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    names = reg.tools_for("Executor")
    assert "nonexistent_tool" in names


def test_tools_for_missing_tool_raises_on_get(tmp_path: Path) -> None:
    """ToolError raised lazily when calling registry.get(missing_name)."""
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": [],
            "per_role": {"Executor": ["nonexistent_tool"]},
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    with pytest.raises(ToolError):
        reg.get("nonexistent_tool")


def test_tools_for_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy_file = tmp_path / "env_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": ["url_fetch"],
            "per_role": {"Researcher": ["semantic_search"]},
        },
    )

    monkeypatch.setenv("ATM_TOOLS_POLICY_PATH", str(policy_file))

    reg = ToolRegistry()
    result = reg.tools_for("Researcher")
    assert result == ["url_fetch", "semantic_search"]


def test_tools_for_ctor_arg_beats_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctor_policy = tmp_path / "ctor_policy.yaml"
    env_policy = tmp_path / "env_policy.yaml"

    _write_policy(
        ctor_policy,
        {
            "global": ["calculator"],
            "per_role": {"Planner": ["plan_update"]},
        },
    )
    _write_policy(
        env_policy,
        {
            "global": ["url_fetch"],
            "per_role": {"Planner": ["todo_write"]},
        },
    )

    monkeypatch.setenv("ATM_TOOLS_POLICY_PATH", str(env_policy))

    reg = ToolRegistry(policy_path=ctor_policy)
    result = reg.tools_for("Planner")
    assert "calculator" in result
    assert "plan_update" in result
    assert "url_fetch" not in result
    assert "todo_write" not in result


def test_tools_for_unknown_role_returns_global(tmp_path: Path) -> None:
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": ["calculator", "file_read"],
            "per_role": {"Executor": ["code_run"]},
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    result = reg.tools_for("UnknownRole")
    assert result == ["calculator", "file_read"]


def test_tools_for_missing_yaml_returns_empty(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist.yaml"
    reg = ToolRegistry(policy_path=nonexistent)
    result = reg.tools_for("Executor")
    assert result == []


def test_tools_for_missing_yaml_unknown_role_returns_empty(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist.yaml"
    reg = ToolRegistry(policy_path=nonexistent)
    result = reg.tools_for("UnknownRole")
    assert result == []


def test_existing_register_get_names_unaffected() -> None:
    """Ensure existing ToolRegistry API still works after adding policy_path."""
    reg = ToolRegistry()
    tool = _StubTool()
    reg.register(tool)

    assert "stub_tool" in reg.names()
    assert reg.get("stub_tool") is tool


def test_existing_duplicate_register_raises() -> None:
    reg = ToolRegistry()
    reg.register(_StubTool())
    with pytest.raises(ToolError):
        reg.register(_StubTool())


def test_existing_get_unknown_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.get("unknown_tool")


@pytest.mark.asyncio
async def test_existing_ainvoke_by_name_works() -> None:

    reg = ToolRegistry()
    reg.register(_StubTool())
    call = ToolCall(tool_name="stub_tool", args={}, issued_by="test")
    result = await reg.ainvoke_by_name("stub_tool", call)
    assert result.ok is True


def test_policy_cached_after_first_call(tmp_path: Path) -> None:
    """tools_for() should cache the policy after first load."""
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": ["calculator"],
            "per_role": {"Executor": ["code_run"]},
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    first = reg.tools_for("Executor")

    _write_policy(
        policy_file,
        {
            "global": ["url_fetch"],
            "per_role": {"Executor": ["file_write"]},
        },
    )

    second = reg.tools_for("Executor")
    assert first == second


def test_tools_for_empty_global_and_role(tmp_path: Path) -> None:
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "global": [],
            "per_role": {"Executor": []},
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    assert reg.tools_for("Executor") == []


def test_tools_for_no_global_key(tmp_path: Path) -> None:
    """Policy without 'global' key is valid; treats global as empty."""
    policy_file = tmp_path / "tools_policy.yaml"
    _write_policy(
        policy_file,
        {
            "per_role": {"Executor": ["code_run"]},
        },
    )

    reg = ToolRegistry(policy_path=policy_file)
    result = reg.tools_for("Executor")
    assert result == ["code_run"]
