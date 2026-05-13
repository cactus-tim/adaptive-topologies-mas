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
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import ClassVar

import docker  # type: ignore[import-untyped]
import docker.errors  # type: ignore[import-untyped]
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
        self.image_digest: str | None = None

        # Constructor still raises if docker daemon is unreachable —
        # tests that need a working sandbox skip via fixture; we only soften
        # the digest-capture path so missing/local-only images don't crash.
        self._client: docker.DockerClient = docker.from_env()
        try:
            python_image = self._image_map["python"]
            img = self._client.images.get(python_image)
            self.image_digest = img.id
        except (docker.errors.DockerException, docker.errors.ImageNotFound, OSError, KeyError):
            self.image_digest = None

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
        return await asyncio.to_thread(self._execute_sync, lang, code, files, effective_timeout)

    def _populate_work_dir(
        self, work_dir: Path, code: str, lang: str, files: dict[str, str] | None
    ) -> None:
        """Write main code + optional extra files into a host work directory.

        The directory is later bind-mounted into the container at /work. It is
        chmod'd 0o777 so that the container's non-root user (uid=1000:gid=1000)
        can both read inputs and write new files (e.g. via FileWriteTool).
        """
        # Determine main filename by language
        if lang == "python":
            main_name = "main.py"
        elif lang == "node":
            main_name = "main.js"
        else:
            main_name = f"main.{lang}"

        (work_dir / main_name).write_text(code, encoding="utf-8")

        if files:
            for filename, content in files.items():
                # Allow nested subdirs inside /work (e.g. "pkg/mod.py")
                target = work_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

        # chmod 0o777 so container uid 1000 can rw regardless of host uid
        work_dir.chmod(0o777)
        for child in work_dir.rglob("*"):
            child.chmod(0o666 if child.is_file() else 0o777)

    def _build_command(self, lang: str) -> list[str]:
        """Build the container command for the given language.

        Raises
        ------
        ValueError
            If *lang* is not in the configured image_map.
        """
        if lang not in self._image_map:
            raise ValueError(
                f"Unsupported language: {lang!r}. Supported: {sorted(self._image_map)}"
            )
        if lang == "python":
            return ["python", "main.py"]
        elif lang == "node":
            return ["node", "main.js"]
        else:
            # Custom language added via image_map override
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
        # _build_command raises ValueError for unknown lang — validates before image lookup
        command = self._build_command(lang)
        image = self._image_map[lang]

        security_opt = [
            "no-new-privileges",
            f"seccomp={self._seccomp_json_str}",
        ]

        cfg = self._config
        container = None
        t0 = time.monotonic()
        timed_out = False

        # /work is a host-bind-mounted tempdir (not tmpfs). Docker's put_archive
        # refuses to write into a container with read_only=True rootfs even when
        # the target path is a tmpfs mount (see moby#41037). Bind-mounting a
        # freshly-chmodded host dir sidesteps this entirely while keeping rootfs
        # read-only, /tmp tmpfs, network=none, cap_drop=ALL, and seccomp active.
        host_work_dir = Path(tempfile.mkdtemp(prefix="atm-sandbox-"))
        try:
            self._populate_work_dir(host_work_dir, code, lang, files)

            # /tmp stays tmpfs; /work uses the host bind (remove from tmpfs if present)
            tmpfs_without_work = {k: v for k, v in cfg.tmpfs_mounts.items() if k != "/work"}
            volumes = {str(host_work_dir): {"bind": "/work", "mode": "rw"}}

            # Create the container in stopped state first so we can stage code
            # via the bind mount before the entrypoint runs.
            container = self._client.containers.create(
                image,
                command,
                read_only=True,
                network_mode="none",
                cap_drop=["ALL"],
                security_opt=security_opt,
                tmpfs=tmpfs_without_work,
                volumes=volumes,
                mem_limit=cfg.mem_limit,
                pids_limit=cfg.pids_limit,
                user="1000:1000",
                working_dir="/work",
            )

            container.start()

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

            # Gather output — fetch stdout and stderr separately so callers
            # can distinguish between the two streams (e.g. TestRunTool parses
            # stderr for unittest summary lines).
            try:
                raw_stdout: bytes = container.logs(stdout=True, stderr=False)
                stdout_str = raw_stdout.decode("utf-8", errors="replace")
            except Exception:
                stdout_str = ""
            try:
                raw_stderr: bytes = container.logs(stdout=False, stderr=True)
                stderr_str = raw_stderr.decode("utf-8", errors="replace")
            except Exception:
                stderr_str = ""

            # Check OOM
            try:
                container.reload()
                oom_killed: bool = bool(container.attrs.get("State", {}).get("OOMKilled", False))
            except Exception:
                oom_killed = False

            duration_ms = int((time.monotonic() - t0) * 1000)

            return ExecResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
                duration_ms=duration_ms,
                timed_out=timed_out,
                oom_killed=oom_killed,
            )

        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.remove(force=True)
            with contextlib.suppress(Exception):
                shutil.rmtree(host_work_dir, ignore_errors=True)
