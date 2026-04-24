"""CodeSandbox Protocol, ExecResult, and SandboxConfig for the ATM framework.

Public API:
    ExecResult   -- frozen Pydantic model for sandbox execution output
    SandboxConfig -- configuration model for sandbox resource limits
    CodeSandbox  -- runtime-checkable Protocol for sandbox implementations
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ExecResult(BaseModel):
    """Result of a sandboxed code execution.

    Fields:
        stdout:      Captured standard output.
        stderr:      Captured standard error.
        exit_code:   Process exit code (0 = success).
        duration_ms: Wall-clock time in milliseconds.
        timed_out:   True if the process was killed due to timeout.
        oom_killed:  True if the process was killed due to OOM.
    """

    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool = False
    oom_killed: bool = False


class SandboxConfig(BaseModel):
    """Configuration for a code sandbox instance.

    Fields:
        cpu_quota:     CPU quota in microseconds per 100ms period (100000 = 1 CPU).
        mem_limit:     Memory limit string (e.g. "128m", "256m").
        pids_limit:    Maximum number of processes/threads.
        timeout_s:     Execution timeout in seconds.
        workdir_size:  Max size of the working directory (informational).
        tmpfs_mounts:  Dict of tmpfs mount paths to mount options.
                       Defaults include /work and /tmp with hardened options.
        network:       Whether network access is allowed.
        image_map:     Mapping of language identifiers to Docker image names.
        seccomp_path:  Path to seccomp profile JSON (None = default Docker profile).
        rootless:      Whether to run in rootless Docker mode.
    """

    model_config = ConfigDict(frozen=True)

    cpu_quota: int = 100_000
    mem_limit: str = "128m"
    pids_limit: int = 64
    timeout_s: float = 10.0
    workdir_size: str = "64m"
    tmpfs_mounts: dict[str, str] = Field(
        default_factory=lambda: {
            "/work": "size=64m,noexec,nosuid,uid=1000,gid=1000",
            "/tmp": "size=64m,noexec,nosuid",
        }
    )
    network: bool = False
    image_map: dict[str, str] = Field(
        default_factory=lambda: {
            "python": "python:3.11-slim",
            "node": "node:20-slim",
        }
    )
    seccomp_path: str | None = None
    rootless: bool = False


@runtime_checkable
class CodeSandbox(Protocol):
    """Protocol for sandbox implementations.

    Every sandbox must expose:
        IS_ISOLATED -- ClassVar[bool] indicating whether code runs in isolation
        execute(...)  -- async method to execute code and return ExecResult
    """

    IS_ISOLATED: ClassVar[bool]

    async def execute(
        self,
        lang: str,
        code: str,
        files: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute code in the sandbox.

        Args:
            lang:    Language identifier (e.g. "python", "node").
            code:    Source code to execute.
            files:   Optional additional files to include {filename: content}.
            timeout: Optional per-invocation timeout override (seconds).

        Returns:
            ExecResult with captured stdout/stderr and exit code.
        """
        ...


__all__ = ["ExecResult", "SandboxConfig", "CodeSandbox"]

# Satisfy unused import check — Any is used in Protocol type annotations
_: Any = None
