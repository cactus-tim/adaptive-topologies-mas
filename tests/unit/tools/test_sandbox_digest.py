"""Unit tests for sandbox image_digest capture.

Tests that:
- DockerSandbox.image_digest is populated from docker client after construction.
- DockerSandbox.image_digest is None when docker daemon is unavailable.
- SubprocessSandbox.image_digest is None (class-level attribute).
- Both sandbox types expose the image_digest attribute (Protocol compliance).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException

from atm.tools.sandbox.base import SandboxConfig
from atm.tools.sandbox.docker_sandbox import DockerSandbox
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox

_SECCOMP = '{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[]}'
_FAKE_DIGEST = "sha256:abc123"


def _make_docker_client_with_image(image_id: str = _FAKE_DIGEST) -> MagicMock:
    """Build a fake docker client whose images.get(name).id == image_id."""
    fake_image = MagicMock()
    fake_image.id = image_id

    client = MagicMock()
    client.images.get.return_value = fake_image
    return client


@pytest.fixture()
def sandbox_config() -> SandboxConfig:
    return SandboxConfig(mem_limit="128m", pids_limit=64, timeout_s=10.0)


def test_docker_sandbox_image_digest_populated(sandbox_config: SandboxConfig) -> None:
    """DockerSandbox.image_digest must equal the docker image id after construction."""
    fake_client = _make_docker_client_with_image(_FAKE_DIGEST)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    assert sandbox.image_digest == _FAKE_DIGEST


def test_docker_sandbox_image_digest_uses_python_image(sandbox_config: SandboxConfig) -> None:
    """DockerSandbox must look up the 'python' key in image_map for the digest."""
    fake_client = _make_docker_client_with_image(_FAKE_DIGEST)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    python_image = sandbox._image_map["python"]
    fake_client.images.get.assert_called_once_with(python_image)
    assert sandbox.image_digest == _FAKE_DIGEST


def test_docker_sandbox_custom_image_map_digest(sandbox_config: SandboxConfig) -> None:
    """DockerSandbox respects custom image_map when capturing digest."""
    custom_image = "my-python:custom"
    custom_digest = "sha256:deadbeef"
    fake_client = _make_docker_client_with_image(custom_digest)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            image_map={"python": custom_image},
            prefetch=False,
        )

    fake_client.images.get.assert_called_once_with(custom_image)
    assert sandbox.image_digest == custom_digest


def test_sandbox_digest_returns_none_when_docker_unavailable(
    sandbox_config: SandboxConfig,
) -> None:
    """When client.images.get() raises DockerException, image_digest is None.

    Note: docker.from_env() failures (no daemon at all) are intentionally NOT
    swallowed by the constructor — those propagate so that integration tests
    skip via fixture rather than silently degrading. This test exercises the
    digest-capture-only failure path (daemon up, image lookup fails).
    """
    fake_client = MagicMock()
    fake_client.images.get.side_effect = DockerException("daemon error")

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    assert sandbox.image_digest is None


def test_sandbox_digest_returns_none_when_os_error(sandbox_config: SandboxConfig) -> None:
    """When client.images.get() raises OSError, image_digest is None."""
    fake_client = MagicMock()
    fake_client.images.get.side_effect = OSError("socket error")

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    assert sandbox.image_digest is None


def test_sandbox_digest_returns_none_when_image_not_found(sandbox_config: SandboxConfig) -> None:
    """When client.images.get() raises, image_digest must be None (not raised)."""
    from docker.errors import ImageNotFound

    fake_client = MagicMock()
    fake_client.images.get.side_effect = ImageNotFound("not found")

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    assert sandbox.image_digest is None


def test_sandbox_digest_returns_none_when_key_error(sandbox_config: SandboxConfig) -> None:
    """When image_map has no 'python' key, image_digest must be None."""
    fake_client = _make_docker_client_with_image(_FAKE_DIGEST)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            image_map={"node": "node:20-slim"},
            prefetch=False,
        )
        sandbox._image_map = {"node": "node:20-slim"}

    with (
        patch("docker.from_env", return_value=fake_client),
        patch("atm.tools.sandbox.docker_sandbox._DEFAULT_IMAGE_MAP", {"node": "node:20-slim"}),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox2 = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            image_map={"node": "node:20-slim"},
            prefetch=False,
        )

    assert sandbox2.image_digest is None


def test_subprocess_sandbox_image_digest_is_none() -> None:
    """SubprocessSandbox.image_digest must be None at class level."""
    assert SubprocessSandbox.image_digest is None


def test_subprocess_sandbox_instance_image_digest_is_none() -> None:
    """SubprocessSandbox instance image_digest must be None."""
    sandbox = SubprocessSandbox()
    assert sandbox.image_digest is None


def test_docker_sandbox_has_image_digest_attr(sandbox_config: SandboxConfig) -> None:
    """getattr(docker_sandbox, 'image_digest', 'MISSING') must not be 'MISSING'."""
    fake_client = _make_docker_client_with_image(_FAKE_DIGEST)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    assert getattr(sandbox, "image_digest", "MISSING") != "MISSING"


def test_subprocess_sandbox_has_image_digest_attr() -> None:
    """getattr(subprocess_sandbox, 'image_digest', 'MISSING') must not be 'MISSING'."""
    sandbox = SubprocessSandbox()
    assert getattr(sandbox, "image_digest", "MISSING") != "MISSING"


def test_image_digest_type_is_str_or_none(sandbox_config: SandboxConfig) -> None:
    """image_digest must be str or None on both sandbox types."""
    fake_client = _make_docker_client_with_image(_FAKE_DIGEST)

    with (
        patch("docker.from_env", return_value=fake_client),
        patch.dict("os.environ", {"ATM_AUTO_PULL_IMAGES": "0"}, clear=False),
    ):
        docker_sandbox = DockerSandbox(
            config=sandbox_config,
            seccomp_json_str=_SECCOMP,
            prefetch=False,
        )

    subprocess_sandbox = SubprocessSandbox()

    assert docker_sandbox.image_digest is None or isinstance(docker_sandbox.image_digest, str)
    assert subprocess_sandbox.image_digest is None or isinstance(
        subprocess_sandbox.image_digest, str
    )
