"""Unit tests for DockerSandbox using a mocked docker client.

All tests in this file run without a live Docker daemon.
They verify that DockerSandbox constructs containers with the correct
hardening parameters and that lifecycle handling (remove, kill) is correct.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from atm.tools.sandbox.base import SandboxConfig
from atm.tools.sandbox.docker_sandbox import DockerSandbox

# ---------------------------------------------------------------------------
# Helpers — fake docker objects
# ---------------------------------------------------------------------------

_SECCOMP = '{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[]}'


def _make_fake_container(
    *,
    exit_code: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    oom_killed: bool = False,
    raise_wait: Exception | None = None,
    raise_logs: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that looks like a docker Container.

    Uses containers.create() lifecycle: create → put_archive → start → wait → logs.
    """
    container = MagicMock()

    # container.logs(stdout=True, stderr=False) / logs(stdout=False, stderr=True)
    if raise_logs:
        container.logs.side_effect = raise_logs
    else:
        def _logs_side_effect(stdout=True, stderr=True):  # type: ignore[override]
            if stdout and not stderr:
                return stdout if isinstance(stdout, bytes) else b""
            if stderr and not stdout:
                return stderr if isinstance(stderr, bytes) else b""
            return stdout if isinstance(stdout, bytes) else b""

        # Return stdout bytes for stdout=True,stderr=False; stderr bytes otherwise
        container.logs.side_effect = lambda **kw: (
            stdout if (kw.get("stdout") and not kw.get("stderr")) else
            (globals()["_make_fake_container"]  # never called — just for type hints
             if False else
             (stderr if (kw.get("stderr") and not kw.get("stdout")) else b""))
        )
        # Simpler: use a closure
        _stdout_bytes = stdout
        _stderr_bytes = stderr

        def _logs(**kw):
            if kw.get("stdout", True) and not kw.get("stderr", True):
                return _stdout_bytes
            if kw.get("stderr", True) and not kw.get("stdout", True):
                return _stderr_bytes
            return _stdout_bytes

        container.logs.side_effect = None
        container.logs.side_effect = _logs

    # container.wait() returns {"StatusCode": exit_code}
    if raise_wait:
        container.wait.side_effect = raise_wait
    else:
        container.wait.return_value = {"StatusCode": exit_code}

    # container.attrs for OOM check
    container.attrs = {
        "State": {
            "OOMKilled": oom_killed,
            "ExitCode": exit_code,
        }
    }

    # container.put_archive() — no-op
    container.put_archive.return_value = True

    return container


def _make_fake_docker_client(container: MagicMock) -> MagicMock:
    """Build a MagicMock docker.DockerClient with the given container.

    Uses containers.create() (not run()) to match the fixed DockerSandbox lifecycle.
    """
    client = MagicMock()
    client.containers.create.return_value = container
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seccomp_json() -> str:
    return _SECCOMP


@pytest.fixture()
def sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        mem_limit="128m",
        pids_limit=64,
        timeout_s=10.0,
    )


# ---------------------------------------------------------------------------
# IS_ISOLATED class variable
# ---------------------------------------------------------------------------


def test_is_isolated_true() -> None:
    """DockerSandbox must declare IS_ISOLATED = True."""
    assert DockerSandbox.IS_ISOLATED is True


# ---------------------------------------------------------------------------
# Constructor — prefetch
# ---------------------------------------------------------------------------


def test_prefetch_true_pulls_images(seccomp_json: str, sandbox_config: SandboxConfig) -> None:
    """When prefetch=True, client.images.pull should be called for each image."""
    fake_client = _make_fake_docker_client(_make_fake_container())

    with patch("docker.from_env", return_value=fake_client):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=True,
        )

    assert fake_client.images.pull.called, "images.pull should have been called when prefetch=True"
    _ = sandbox  # silence unused warning


def test_prefetch_false_no_pull(seccomp_json: str, sandbox_config: SandboxConfig) -> None:
    """When prefetch=False and ATM_AUTO_PULL_IMAGES != '1', no pull should occur."""
    fake_client = _make_fake_docker_client(_make_fake_container())

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )

    assert not fake_client.images.pull.called, (
        "images.pull should NOT be called when prefetch=False"
    )
    _ = sandbox


def test_auto_pull_env_triggers_pull(seccomp_json: str, sandbox_config: SandboxConfig) -> None:
    """When ATM_AUTO_PULL_IMAGES='1', images.pull should be called even without prefetch=True."""
    fake_client = _make_fake_docker_client(_make_fake_container())

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "1"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )

    assert fake_client.images.pull.called, (
        "images.pull should be called when ATM_AUTO_PULL_IMAGES=1"
    )
    _ = sandbox


# ---------------------------------------------------------------------------
# containers.create kwargs — hardening parameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_container_create_kwargs_hardening(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """containers.create must be called with the correct hardening kwargs."""
    fake_container = _make_fake_container(stdout=b"2\n")
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="print(1+1)")

    assert fake_client.containers.create.called, "containers.create must be called"
    call_kwargs = fake_client.containers.create.call_args[1]  # keyword args

    # Core hardening
    assert call_kwargs.get("read_only") is True, "read_only must be True"
    assert call_kwargs.get("cap_drop") == ["ALL"], "cap_drop must be ['ALL']"
    assert call_kwargs.get("network_mode") == "none", "network_mode must be 'none'"

    # security_opt must include no-new-privileges and the seccomp profile
    security_opt = call_kwargs.get("security_opt", [])
    assert "no-new-privileges" in security_opt, "security_opt must include no-new-privileges"
    seccomp_opt = next((o for o in security_opt if "seccomp=" in o), None)
    assert seccomp_opt is not None, "security_opt must include seccomp=<json>"
    assert _SECCOMP in seccomp_opt, "seccomp option must contain the seccomp JSON string"

    # tmpfs must include /work and /tmp
    tmpfs = call_kwargs.get("tmpfs", {})
    assert "/work" in tmpfs, "tmpfs must include /work"
    assert "/tmp" in tmpfs, "tmpfs must include /tmp"


@pytest.mark.asyncio
async def test_container_create_working_dir(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """working_dir must be set to /work."""
    fake_container = _make_fake_container(stdout=b"")
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="x = 1")

    call_kwargs = fake_client.containers.create.call_args[1]
    assert call_kwargs.get("working_dir") == "/work", "working_dir must be /work"


@pytest.mark.asyncio
async def test_container_create_mem_and_pids(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """mem_limit and pids_limit must be forwarded from SandboxConfig."""
    fake_container = _make_fake_container()
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="pass")

    call_kwargs = fake_client.containers.create.call_args[1]
    assert call_kwargs.get("mem_limit") == sandbox_config.mem_limit
    assert call_kwargs.get("pids_limit") == sandbox_config.pids_limit


@pytest.mark.asyncio
async def test_lifecycle_create_then_put_archive_then_start(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """Must call containers.create, then put_archive, then container.start in that order."""
    fake_container = _make_fake_container()
    fake_client = _make_fake_docker_client(fake_container)
    call_order: list[str] = []

    fake_client.containers.create.side_effect = lambda *a, **kw: (
        call_order.append("create") or fake_container
    )
    fake_container.put_archive.side_effect = lambda *a, **kw: call_order.append("put_archive")
    fake_container.start.side_effect = lambda *a, **kw: call_order.append("start")

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="pass")

    assert call_order[:3] == ["create", "put_archive", "start"], (
        f"Expected create→put_archive→start, got {call_order}"
    )


# ---------------------------------------------------------------------------
# container.remove — called even on exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_container_remove_called_on_success(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """container.remove(force=True) must be called after successful execution."""
    fake_container = _make_fake_container(stdout=b"ok\n")
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="print('ok')")

    fake_container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_container_remove_called_on_wait_exception(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """container.remove(force=True) must be called even when container.wait() raises."""
    fake_container = _make_fake_container(raise_wait=RuntimeError("wait failed"))
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        # Should not raise — exceptions during wait become error result or re-raised after remove
        with contextlib.suppress(Exception):
            await sandbox.execute(lang="python", code="pass")

    fake_container.remove.assert_called_with(force=True)


# ---------------------------------------------------------------------------
# Timeout path — kill + timed_out=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_kills_container_and_sets_timed_out(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """When container.wait() raises a timeout exception, container.kill() must be called
    and ExecResult.timed_out must be True."""
    import requests.exceptions

    fake_container = _make_fake_container(
        raise_wait=requests.exceptions.ReadTimeout("timed out"),
    )
    # After timeout kill, allow logs to return normally
    fake_container.logs.side_effect = None
    fake_container.logs.return_value = b""

    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        result = await sandbox.execute(lang="python", code="import time; time.sleep(999)")

    assert result.timed_out is True, "timed_out must be True after timeout"
    fake_container.kill.assert_called()
    fake_container.remove.assert_called_with(force=True)


# ---------------------------------------------------------------------------
# ExecResult shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_result_stdout_captured(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """stdout must be captured and returned in ExecResult."""
    fake_container = _make_fake_container(stdout=b"hello\n", exit_code=0)
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        result = await sandbox.execute(lang="python", code="print('hello')")

    assert "hello" in result.stdout
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_exec_result_oom_killed(seccomp_json: str, sandbox_config: SandboxConfig) -> None:
    """oom_killed must be True when OOMKilled is set in container state."""
    fake_container = _make_fake_container(exit_code=137, oom_killed=True)
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        result = await sandbox.execute(lang="python", code="x = [0] * 10**9")

    assert result.oom_killed is True


# ---------------------------------------------------------------------------
# image_map — default and custom
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_image_map_python(seccomp_json: str, sandbox_config: SandboxConfig) -> None:
    """Default image for 'python' lang must be python:3.11-slim."""
    fake_container = _make_fake_container()
    fake_client = _make_fake_docker_client(fake_container)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="pass")

    call_args = fake_client.containers.create.call_args
    # First positional arg is image
    image_used = call_args[0][0] if call_args[0] else call_args[1].get("image")
    assert image_used == "python:3.11-slim"


@pytest.mark.asyncio
async def test_custom_image_map_overrides(seccomp_json: str, sandbox_config: SandboxConfig) -> None:
    """Custom image_map should override the default for that language."""
    fake_container = _make_fake_container()
    fake_client = _make_fake_docker_client(fake_container)

    custom_image = "my-python:custom"

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            image_map={"python": custom_image},
            prefetch=False,
        )
        await sandbox.execute(lang="python", code="pass")

    call_args = fake_client.containers.create.call_args
    image_used = call_args[0][0] if call_args[0] else call_args[1].get("image")
    assert image_used == custom_image


# ---------------------------------------------------------------------------
# Unsupported language raises ValueError (finding #15)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_lang_raises_value_error(
    seccomp_json: str, sandbox_config: SandboxConfig
) -> None:
    """An unsupported language (not in image_map) must raise ValueError."""
    fake_client = _make_fake_docker_client(_make_fake_container())

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=seccomp_json,
            prefetch=False,
        )
        with pytest.raises(ValueError, match="Unsupported language"):
            await sandbox.execute(lang="pyhton", code="pass")  # intentional typo
