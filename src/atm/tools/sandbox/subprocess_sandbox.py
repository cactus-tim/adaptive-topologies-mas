"""Dev-only subprocess-based code sandbox.

Dev-only. NOT isolated from host. DO NOT use in production.
Use DockerSandbox for real isolation.

Public API
----------
SubprocessSandbox -- CodeSandbox implementation using asyncio.create_subprocess_exec
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import ClassVar

from atm.tools.sandbox.base import ExecResult


_LANG_FILENAME: dict[str, str] = {
    "python": "main.py",
    "node": "main.js",
}


class SubprocessSandbox:
    """Dev-only sandbox that executes code in a subprocess inside a temp directory.

    Dev-only. NOT isolated from host. DO NOT use in production.
    Use DockerSandbox for real isolation.

    Class Variables
    ---------------
    IS_ISOLATED : ClassVar[bool]
        Always False — this sandbox shares the host filesystem and process tree.

    Notes
    -----
    - Python code is executed with ``sys.executable`` (the same interpreter).
    - Node.js code is executed with ``shutil.which("node")``.
    - A fresh ``tempfile.TemporaryDirectory`` is created per ``execute`` call
      and removed in a ``finally`` block, even on error.
    - On timeout the process tree is killed and ``timed_out=True`` is returned.
    - ``oom_killed`` is always ``False`` — subprocess sandbox does not track OOM.
    """

    IS_ISOLATED: ClassVar[bool] = False

    async def execute(
        self,
        lang: str,
        code: str,
        files: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute *code* in a temporary directory using a subprocess.

        Parameters
        ----------
        lang:
            Language identifier: ``"python"`` or ``"node"``.
        code:
            Source code to execute.
        files:
            Optional mapping of filename → content for additional files
            written alongside the main entry-point before execution.
        timeout:
            Wall-clock timeout in seconds.  The process is killed (SIGKILL)
            on expiry.  ``None`` means no timeout.

        Returns
        -------
        ExecResult
            Always returned — never raises.  On timeout ``timed_out=True`` and
            ``exit_code=-1``.  ``oom_killed`` is always ``False``.

        Raises
        ------
        ValueError
            If *lang* is not ``"python"`` or ``"node"``.
        """
        if lang not in _LANG_FILENAME:
            raise ValueError(
                f"Unsupported language: {lang!r}. Supported: {sorted(_LANG_FILENAME)}"
            )

        tmpdir = tempfile.mkdtemp(prefix="atm_subprocess_")
        t0 = time.monotonic()

        try:
            work = Path(tmpdir)

            # Write the main entry-point file
            main_filename = _LANG_FILENAME[lang]
            (work / main_filename).write_text(code, encoding="utf-8")

            # Write any extra files
            if files:
                for filename, content in files.items():
                    (work / filename).write_text(content, encoding="utf-8")

            # Resolve the interpreter command
            if lang == "python":
                cmd = [sys.executable, main_filename]
            else:  # node
                node_bin = shutil.which("node")
                if node_bin is None:
                    raise RuntimeError("node is not installed or not on PATH")
                cmd = [node_bin, main_filename]

            # Minimal env: pass only PATH so the subprocess can find shared libs
            env = {"PATH": os.environ.get("PATH", "")}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work),
                env=env,
            )

            timed_out = False
            stdout_bytes = b""
            stderr_bytes = b""
            exit_code: int = -1

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                exit_code = proc.returncode if proc.returncode is not None else -1
            except asyncio.TimeoutError:
                timed_out = True
                # Kill the process and its entire process group
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    # Drain remaining output to avoid pipe buffer deadlocks
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=5.0,
                    )
                except (asyncio.TimeoutError, Exception):
                    stdout_bytes = b""
                    stderr_bytes = b""
                exit_code = -1

            duration_ms = int((time.monotonic() - t0) * 1000)

            return ExecResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                duration_ms=duration_ms,
                timed_out=timed_out,
                oom_killed=False,
            )

        finally:
            # Always remove the temp directory, even on error
            shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = ["SubprocessSandbox"]
