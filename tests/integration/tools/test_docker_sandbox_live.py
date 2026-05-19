"""Live integration tests for DockerSandbox.

All tests are gated with @pytest.mark.docker — they only run when
ATM_ENABLE_DOCKER_TESTS=1 is set in the environment.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from atm.tools.sandbox.base import SandboxConfig
from atm.tools.sandbox.docker_sandbox import DockerSandbox

_SECCOMP_PATH = pathlib.Path(__file__).parents[3] / "conf" / "sandbox" / "seccomp.json"


@pytest.fixture(scope="module")
def seccomp_str() -> str:
    return _SECCOMP_PATH.read_text()


@pytest.fixture(scope="module")
def sandbox(seccomp_str: str) -> DockerSandbox:
    cfg = SandboxConfig(
        mem_limit="256m",
        pids_limit=64,
        timeout_s=15.0,
    )
    return DockerSandbox(
        config=cfg,
        seccomp_json_str=seccomp_str,
        prefetch=False,
    )


@pytest.mark.docker
async def test_hello_world(sandbox: DockerSandbox) -> None:
    """print(1+1) should produce stdout '2\n' with exit code 0."""
    result = await sandbox.execute(lang="python", code="print(1+1)")
    assert result.exit_code == 0
    assert "2" in result.stdout
    assert result.timed_out is False
    assert result.oom_killed is False


@pytest.mark.docker
async def test_exit_code_nonzero(sandbox: DockerSandbox) -> None:
    """Code that raises an exception should have a non-zero exit code."""
    result = await sandbox.execute(lang="python", code="raise ValueError('boom')")
    assert result.exit_code != 0


@pytest.mark.docker
async def test_readonly_rootfs_rejects_write(sandbox: DockerSandbox) -> None:
    """Writing to a non-tmpfs path on the read-only rootfs should fail."""
    code = textwrap.dedent("""\
        try:
            open('/etc/foo_atm_test', 'w').write('x')
            print('WRITE_OK')
        except (OSError, PermissionError) as e:
            print(f'WRITE_BLOCKED: {e}')
    """)
    result = await sandbox.execute(lang="python", code=code)
    assert "WRITE_BLOCKED" in result.stdout or result.exit_code != 0, (
        "Write to read-only rootfs should be blocked"
    )


@pytest.mark.docker
async def test_tmp_writable_via_tmpfs(sandbox: DockerSandbox) -> None:
    """Writing to /tmp should succeed (tmpfs mount allows writes)."""
    code = textwrap.dedent("""\
        with open('/tmp/atm_test_file.txt', 'w') as f:
            f.write('hello from tmpfs')
        with open('/tmp/atm_test_file.txt') as f:
            print(f.read())
    """)
    result = await sandbox.execute(lang="python", code=code)
    assert result.exit_code == 0, f"Writing to /tmp failed: {result.stderr}"
    assert "hello from tmpfs" in result.stdout


@pytest.mark.docker
async def test_work_dir_writable(sandbox: DockerSandbox) -> None:
    """Writing to /work (cwd) should succeed."""
    code = textwrap.dedent("""\
        import os
        with open('/work/test_output.txt', 'w') as f:
            f.write('in work dir')
        with open('/work/test_output.txt') as f:
            print(f.read())
    """)
    result = await sandbox.execute(lang="python", code=code)
    assert result.exit_code == 0, f"Writing to /work failed: {result.stderr}"
    assert "in work dir" in result.stdout


@pytest.mark.docker
async def test_multi_file_execution(sandbox: DockerSandbox) -> None:
    """Files dict should be uploaded and accessible from main code."""
    code = textwrap.dedent("""\
        from helper import greet
        print(greet('world'))
    """)
    files = {
        "helper.py": "def greet(name):\n    return f'Hello, {name}!'\n",
    }
    result = await sandbox.execute(lang="python", code=code, files=files)
    assert result.exit_code == 0, f"Multi-file execution failed: {result.stderr}"
    assert "Hello, world!" in result.stdout


@pytest.mark.docker
async def test_network_disabled(sandbox: DockerSandbox) -> None:
    """Network should be disabled inside the container."""
    code = textwrap.dedent("""\
        import socket
        try:
            s = socket.create_connection(('1.1.1.1', 53), timeout=2)
            s.close()
            print('CONNECTED')
        except OSError as e:
            print(f'BLOCKED: {e}')
    """)
    result = await sandbox.execute(lang="python", code=code)
    assert "BLOCKED" in result.stdout or result.exit_code != 0, (
        "Network access should be blocked in the container"
    )


@pytest.mark.docker
async def test_network_disabled_urllib(sandbox: DockerSandbox) -> None:
    """urllib.request should also fail when network is disabled."""
    code = textwrap.dedent("""\
        import urllib.request
        try:
            urllib.request.urlopen('http://8.8.8.8', timeout=2)
            print('CONNECTED')
        except Exception as e:
            print(f'BLOCKED: {e}')
    """)
    result = await sandbox.execute(lang="python", code=code)
    assert "BLOCKED" in result.stdout or result.exit_code != 0, (
        "urllib network access should be blocked"
    )


@pytest.mark.docker
async def test_timeout_infinite_loop(seccomp_str: str) -> None:
    """An infinite loop should be killed after timeout and timed_out=True."""
    cfg = SandboxConfig(
        mem_limit="128m",
        pids_limit=64,
        timeout_s=3.0,
    )
    sb = DockerSandbox(config=cfg, seccomp_json_str=seccomp_str, prefetch=False)
    result = await sb.execute(lang="python", code="while True: pass", timeout=2.0)
    assert result.timed_out is True, "timed_out must be True for infinite loop with short timeout"


@pytest.mark.docker
async def test_oom_killed(seccomp_str: str) -> None:
    """Allocating more memory than mem_limit should trigger OOM kill."""
    cfg = SandboxConfig(
        mem_limit="64m",
        pids_limit=64,
        timeout_s=30.0,
    )
    sb = DockerSandbox(config=cfg, seccomp_json_str=seccomp_str, prefetch=False)
    result = await sb.execute(
        lang="python",
        code="x = bytearray(512 * 1024 * 1024)",
    )
    assert result.oom_killed is True or result.exit_code != 0, (
        "OOM allocation should be killed or error"
    )
