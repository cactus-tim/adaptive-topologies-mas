"""Isolation tests for DockerSandbox.

Verifies that destructive operations inside the container cannot affect the host.

All tests are gated with @pytest.mark.docker.
"""

from __future__ import annotations

import contextlib
import hashlib
import pathlib
import textwrap
import time

import pytest

from atm.tools.sandbox.base import SandboxConfig
from atm.tools.sandbox.docker_sandbox import DockerSandbox

_SECCOMP_PATH = pathlib.Path(__file__).parents[3] / "conf" / "sandbox" / "seccomp.json"

# A stable host file whose hash we can compare before and after rm -rf /
_HOST_SENTINEL = pathlib.Path("/etc/hostname")


@pytest.fixture(scope="module")
def seccomp_str() -> str:
    return _SECCOMP_PATH.read_text()


@pytest.fixture(scope="module")
def sandbox(seccomp_str: str) -> DockerSandbox:
    cfg = SandboxConfig(
        mem_limit="256m",
        pids_limit=128,
        timeout_s=30.0,
        tmpfs_mounts={
            "/work": "size=64m,mode=700",
            "/tmp": "size=64m,noexec,nosuid",
        },
    )
    return DockerSandbox(config=cfg, seccomp_json_str=seccomp_str, prefetch=False)


# ---------------------------------------------------------------------------
# rm -rf / does not affect host
# ---------------------------------------------------------------------------


@pytest.mark.docker
async def test_rm_rf_root_does_not_affect_host(sandbox: DockerSandbox) -> None:
    """Running rm -rf / inside the container must not change host files.

    We compare the SHA-256 of /etc/hostname before and after the container runs.
    """
    if not _HOST_SENTINEL.exists():
        pytest.skip("/etc/hostname not available on this host")

    host_hash_before = hashlib.sha256(_HOST_SENTINEL.read_bytes()).hexdigest()

    code = textwrap.dedent("""\
        import subprocess
        import sys
        # Attempt to destroy the filesystem
        r = subprocess.run(
            ['sh', '-c', 'rm -rf / 2>/dev/null; echo done'],
            capture_output=True,
            text=True,
        )
        print("rm attempt finished:", r.returncode)
    """)

    # Even if this fails it's fine — we just want host to be untouched
    with contextlib.suppress(Exception):
        await sandbox.execute(lang="python", code=code)

    host_hash_after = hashlib.sha256(_HOST_SENTINEL.read_bytes()).hexdigest()
    assert host_hash_before == host_hash_after, (
        "Host /etc/hostname hash changed after rm -rf / inside container — isolation breach!"
    )


# ---------------------------------------------------------------------------
# Fork bomb — killed by pids_limit
# ---------------------------------------------------------------------------


@pytest.mark.docker
async def test_fork_bomb_killed_by_pids_limit(seccomp_str: str) -> None:
    """A fork bomb should be killed by pids_limit and not survive on the host.

    Acceptance (NIT-5): container.wait() returns non-zero OR wall-time < 30s.
    Key requirement: the process must not persist on the host.
    """
    cfg = SandboxConfig(
        mem_limit="256m",
        pids_limit=10,  # very tight PID limit
        timeout_s=25.0,
        tmpfs_mounts={
            "/work": "size=64m,mode=700",
            "/tmp": "size=64m,noexec,nosuid",
        },
    )
    sb = DockerSandbox(config=cfg, seccomp_json_str=seccomp_str, prefetch=False)

    t0 = time.monotonic()
    result = await sb.execute(
        lang="python",
        code=textwrap.dedent("""\
            import os
            while True:
                os.fork()
        """),
        timeout=20.0,
    )
    wall_time = time.monotonic() - t0

    # Either non-zero exit OR wall time well under 30s (meaning it was killed)
    assert result.exit_code != 0 or wall_time < 30.0, (
        f"Fork bomb should have been killed — exit={result.exit_code}, wall={wall_time:.1f}s"
    )
