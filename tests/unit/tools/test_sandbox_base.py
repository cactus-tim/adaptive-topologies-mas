"""Tests for atm.tools.sandbox.base — ExecResult, SandboxConfig, CodeSandbox."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atm.tools.sandbox.base import ExecResult, SandboxConfig


# ---------------------------------------------------------------------------
# ExecResult tests
# ---------------------------------------------------------------------------


def test_exec_result_frozen_fields() -> None:
    result = ExecResult(
        stdout="hello\n",
        stderr="",
        exit_code=0,
        duration_ms=100,
        timed_out=False,
        oom_killed=False,
    )
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.oom_killed is False

    # Must be frozen — assignment should raise (Pydantic v2 raises ValidationError)
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        result.stdout = "other"  # type: ignore[misc]


def test_exec_result_all_fields_present() -> None:
    result = ExecResult(
        stdout="out",
        stderr="err",
        exit_code=1,
        duration_ms=500,
        timed_out=True,
        oom_killed=False,
    )
    assert result.stderr == "err"
    assert result.duration_ms == 500
    assert result.timed_out is True


# ---------------------------------------------------------------------------
# SandboxConfig tests
# ---------------------------------------------------------------------------


def test_sandbox_config_tmpfs_defaults_include_tmp() -> None:
    cfg = SandboxConfig()
    assert "/tmp" in cfg.tmpfs_mounts
    assert "/work" in cfg.tmpfs_mounts


def test_sandbox_config_tmpfs_work_options() -> None:
    cfg = SandboxConfig()
    work_opts = cfg.tmpfs_mounts["/work"]
    assert "noexec" in work_opts
    assert "nosuid" in work_opts
    assert "uid=1000" in work_opts


def test_sandbox_config_tmpfs_tmp_options() -> None:
    cfg = SandboxConfig()
    tmp_opts = cfg.tmpfs_mounts["/tmp"]
    assert "noexec" in tmp_opts
    assert "nosuid" in tmp_opts


def test_sandbox_config_custom_tmpfs() -> None:
    cfg = SandboxConfig(tmpfs_mounts={"/custom": "size=32m"})
    assert "/custom" in cfg.tmpfs_mounts
    assert "/tmp" not in cfg.tmpfs_mounts


def test_sandbox_config_default_values() -> None:
    cfg = SandboxConfig()
    assert cfg.timeout_s > 0
    assert cfg.mem_limit is not None
    assert cfg.pids_limit is not None
