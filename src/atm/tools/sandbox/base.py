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
    """Configuration for sandbox resource limits and filesystem mounts.

    Attributes:
        cpu_quota:      CPU quota in microseconds per period (Docker: nano_cpus derived).
        mem_limit:      Memory limit string (e.g. "128m"). Docker accepts this directly.
        pids_limit:     Maximum number of PIDs (fork-bomb mitigation).
        timeout_s:      Wall-clock timeout in seconds before killing the process.
        workdir_size:   Maximum size of the working directory (informational).
        tmpfs_mounts:   Dict mapping mount path → options string. Defaults include
                        /work and /tmp with noexec,nosuid for security.
        network:        Whether to enable network access inside the sandbox.
        image_map:      Override the default Docker image per language.
        seccomp_path:   Path to a seccomp JSON profile (passed to Docker as a string).
        rootless:       Whether to use rootless Docker mode.
    """

    cpu_quota: int = 100_000  # 100ms per 100ms period = 1 CPU
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
    """Protocol for sandbox implementations that execute arbitrary code.

    Class variables
    ---------------
    IS_ISOLATED: ClassVar[bool]
        True for fully-isolated sandboxes (Docker, gVisor, etc.).
        False for dev-only sandboxes (SubprocessSandbox).

    Methods
    -------
    execute(lang, code, files, timeout) -> ExecResult
        Asynchronously execute *code* in *lang* inside the sandbox.
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

        Parameters
        ----------
        lang:    Language identifier (e.g. "python", "node").
        code:    Source code to execute.
        files:   Optional additional files to write alongside the main code.
                 Keys are filenames; values are file contents.
        timeout: Override the default ``SandboxConfig.timeout_s``.

        Returns
        -------
        ExecResult
        """
        ...


__all__ = ["CodeSandbox", "ExecResult", "SandboxConfig"]
