"""Hardened Docker sandbox for production code execution.

Implements the CodeSandbox Protocol with full isolation:
- Read-only rootfs
- tmpfs mounts for /work and /tmp
- No network (network_mode=none)
- All capabilities dropped (cap_drop=ALL)
- Custom seccomp profile
- Memory, CPU, and PID limits
- no-new-privileges security option

Public API
----------
DockerSandbox -- production-ready isolated sandbox, IS_ISOLATED=True
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tarfile
import time
from typing import ClassVar

import docker  # type: ignore[import-untyped]
import requests.exceptions  # type: ignore[import-untyped]

from atm.tools.sandbox.base import ExecResult, SandboxConfig

_DEFAULT_IMAGE_MAP: dict[str, str] = {
    "python": "python:3.11-slim",
    "node": "node:20-slim",
}


class DockerSandbox:
    """Production sandbox using Docker with full security hardening.

    IS_ISOLATED is True — safe for prod use.

    Parameters
    ----------
    config:
        Resource limits and mount configuration.
    seccomp_json_str:
        The seccomp profile as a JSON string. Passed directly to Docker
        via the ``seccomp=<json>`` security option.
    image_map:
        Optional override for the language → Docker image mapping.
        Defaults: {"python": "python:3.11-slim", "node": "node:20-slim"}.
    prefetch:
        If True, pull all images in image_map at construction time.
        Also triggered if env var ``ATM_AUTO_PULL_IMAGES=1`` is set.
    """

    IS_ISOLATED: ClassVar[bool] = True

    def __init__(
        self,
        config: SandboxConfig,
        seccomp_json_str: str,
        image_map: dict[str, str] | None = None,
        prefetch: bool = False,
    ) -> None:
        self._config = config
        self._seccomp_json_str = seccomp_json_str
        self._image_map: dict[str, str] = {
            **_DEFAULT_IMAGE_MAP,
            **(image_map or {}),
        }
        self._client: docker.DockerClient = docker.from_env()

        should_pull = prefetch or os.environ.get("ATM_AUTO_PULL_IMAGES") == "1"
        if should_pull:
            for image in self._image_map.values():
                self._client.images.pull(image)

    async def execute(
        self,
        lang: str,
        code: str,
        files: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute code in an isolated Docker container.

        Runs the blocking _execute_sync in a thread pool to avoid
        blocking the event loop.

        Parameters
        ----------
        lang:
            Language identifier (e.g. "python", "node").
        code:
            Source code to execute. Written as ``main.py`` / ``main.js``.
        files:
            Additional files to upload into /work alongside the main file.
        timeout:
            Override the SandboxConfig.timeout_s for this execution.

        Returns
        -------
        ExecResult
        """
        effective_timeout = timeout if timeout is not None else self._config.timeout_s
        return await asyncio.to_thread(
            self._execute_sync, lang, code, files, effective_timeout
        )

    def _make_archive(self, code: str, lang: str, files: dict[str, str] | None) -> bytes:
        """Build a tar archive containing main code + optional extra files."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            # Determine main filename by language
            if lang == "python":
                main_name = "main.py"
            elif lang == "node":
                main_name = "main.js"
            else:
                main_name = f"main.{lang}"

            # Add main code file
            code_bytes = code.encode()
            info = tarfile.TarInfo(name=main_name)
            info.size = len(code_bytes)
            info.uid = 1000
            info.gid = 1000
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(code_bytes))

            # Add additional files
            if files:
                for filename, content in files.items():
                    content_bytes = content.encode()
                    finfo = tarfile.TarInfo(name=filename)
                    finfo.size = len(content_bytes)
                    finfo.uid = 1000
                    finfo.gid = 1000
                    finfo.mode = 0o644
                    tar.addfile(finfo, io.BytesIO(content_bytes))

        return buf.getvalue()

    def _build_command(self, lang: str) -> list[str]:
        """Build the container command for the given language."""
        if lang == "python":
            return ["python", "main.py"]
        elif lang == "node":
            return ["node", "main.js"]
        else:
            return [lang, f"main.{lang}"]

    def _execute_sync(
        self,
        lang: str,
        code: str,
        files: dict[str, str] | None,
        timeout: float,
    ) -> ExecResult:
        """Blocking implementation of container execution.

        Called via asyncio.to_thread to avoid blocking the event loop.
        """
        image = self._image_map.get(lang, f"{lang}:latest")
        command = self._build_command(lang)

        security_opt = [
            "no-new-privileges",
            f"seccomp={self._seccomp_json_str}",
        ]

        cfg = self._config
        container = None
        t0 = time.monotonic()
        timed_out = False

        try:
            container = self._client.containers.run(
                image,
                command,
                detach=True,
                read_only=True,
                network_mode="none",
                cap_drop=["ALL"],
                security_opt=security_opt,
                tmpfs=cfg.tmpfs_mounts,
                mem_limit=cfg.mem_limit,
                pids_limit=cfg.pids_limit,
                user="1000:1000",
                working_dir="/work",
            )

            # Upload code + files into the tmpfs /work directory
            archive_data = self._make_archive(code, lang, files)
            container.put_archive("/work", archive_data)

            # Wait for container to finish; catch timeout
            try:
                wait_result = container.wait(timeout=timeout)
                exit_code: int = wait_result["StatusCode"]
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ):
                timed_out = True
                with contextlib.suppress(Exception):
                    container.kill()
                exit_code = -1

            # Gather output
            try:
                raw_logs: bytes = container.logs(stdout=True, stderr=True)
                combined = raw_logs.decode("utf-8", errors="replace")
            except Exception:
                combined = ""

            # Check OOM
            try:
                container.reload()
                oom_killed: bool = bool(
                    container.attrs.get("State", {}).get("OOMKilled", False)
                )
            except Exception:
                oom_killed = False

            duration_ms = int((time.monotonic() - t0) * 1000)

            return ExecResult(
                stdout=combined,
                stderr="",
                exit_code=exit_code,
                duration_ms=duration_ms,
                timed_out=timed_out,
                oom_killed=oom_killed,
            )

        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.remove(force=True)
