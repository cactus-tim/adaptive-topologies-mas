"""CodeSandbox Protocol, ExecResult, and SandboxConfig.

Public API
----------
ExecResult    -- frozen Pydantic model for sandbox execution output
SandboxConfig -- configuration for sandbox resource limits and mounts
CodeSandbox   -- runtime-checkable Protocol for sandbox implementations
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ExecResult(BaseModel):
    """Frozen result of a sandbox code execution.

    Attributes:
        stdout:      Captured standard output.
        stderr:      Captured standard error.
        exit_code:   Process exit code (0 = success).
        duration_ms: Wall-clock execution time in milliseconds.
        timed_out:   True if the process was killed due to timeout.
        oom_killed:  True if the process was killed due to OOM.
    """

    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool
    oom_killed: bool


def _default_tmpfs_mounts() -> dict[str, str]:
    """Return the default tmpfs mount configuration.

    /work: writable workspace (uid=1000/gid=1000, noexec, nosuid)
    /tmp:  Python tempfile / .pyc cache support (noexec, nosuid)
    """
    return {
        "/work": "size=64m,noexec,nosuid,uid=1000,gid=1000",
        "/tmp": "size=64m,noexec,nosuid",
    }


class SandboxConfig(BaseModel):
    """Configuration for sandbox resource limits and filesystem mounts."""

    cpu_quota: int = 100_000
    mem_limit: str = "128m"
    pids_limit: int = 64
    timeout_s: float = 10.0
    workdir_size: str = "64m"
    tmpfs_mounts: dict[str, str] = Field(default_factory=_default_tmpfs_mounts)
    network: bool = False
    image_map: dict[str, str] = Field(default_factory=dict)
    seccomp_path: str | None = None
    rootless: bool = False


@runtime_checkable
class CodeSandbox(Protocol):
    """Protocol for sandbox implementations that execute arbitrary code."""

    IS_ISOLATED: ClassVar[bool]

    async def execute(
        self,
        lang: str,
        code: str,
        files: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute code in the sandbox."""
        ...


__all__ = ["CodeSandbox", "ExecResult", "SandboxConfig"]
