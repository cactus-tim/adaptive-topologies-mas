"""Docker acceptance tests — gated by @pytest.mark.docker.

These tests require a running Docker daemon. Skip unless ATM_ENABLE_DOCKER_TESTS=1.

Tests
-----
- DockerSandbox.execute hello-world: stdout "2\\n", exit 0.
- TestRunTool via DockerSandbox: trivial unittest → passed=True.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.docker

_SECCOMP_PATH = pathlib.Path(__file__).parents[3] / "conf" / "sandbox" / "seccomp.json"


@pytest.fixture(scope="module")
def seccomp_str() -> str:
    return _SECCOMP_PATH.read_text()


@pytest.fixture(scope="module")
def cfg(seccomp_str: str):
    from atm.tools.sandbox.base import SandboxConfig

    return SandboxConfig(
        mem_limit="256m",
        pids_limit=64,
        timeout_s=30.0,
    )


@pytest.fixture(scope="module")
def docker_sandbox(cfg, seccomp_str: str):
    from atm.tools.sandbox.docker_sandbox import DockerSandbox

    return DockerSandbox(config=cfg, seccomp_json_str=seccomp_str, prefetch=False)


class TestDockerAcceptance:
    @pytest.mark.docker
    @pytest.mark.asyncio
    async def test_docker_sandbox_hello_world(self, docker_sandbox) -> None:
        """DockerSandbox.execute('python', 'print(1+1)') returns stdout='2\\n', exit=0."""
        result = await docker_sandbox.execute(lang="python", code="print(1+1)")

        assert "2" in result.stdout
        assert result.exit_code == 0
        assert result.timed_out is False

    @pytest.mark.docker
    @pytest.mark.asyncio
    async def test_docker_test_run_passing(self, docker_sandbox, tmp_path: Path) -> None:
        """TestRunTool backed by DockerSandbox: trivial passing test → passed=True."""
        from atm.core.types import ToolCall
        from atm.tools.local_.test_run import TestRunTool

        tool = TestRunTool(sandbox=docker_sandbox)

        test_code = (
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertEqual(1 + 1, 2)\n"
        )

        call = ToolCall(
            tool_name="test_run",
            issued_by="test_agent",
            args={"test_code": test_code},
        )
        result = await tool.ainvoke(call.args)
        assert result.ok is True
        assert result.output["passed"] is True
        assert result.output["summary"].startswith("OK")

    @pytest.mark.docker
    @pytest.mark.asyncio
    async def test_docker_registry_code_run(self, docker_sandbox, tmp_path: Path) -> None:
        """Registry with DockerSandbox can run code_run end-to-end."""
        from atm.core.types import ToolCall
        from atm.tools.defaults import build_default_registry

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "doc_1.txt").write_text("neural network test doc", encoding="utf-8")

        registry = build_default_registry(
            workspace=tmp_path / "workspace",
            corpus_dir=corpus_dir,
            sandbox=docker_sandbox,
            prod_mode=True,
        )
        (tmp_path / "workspace").mkdir(exist_ok=True)

        call = ToolCall(
            tool_name="code_run",
            issued_by="test_agent",
            args={"lang": "python", "code": "print('docker ok')"},
        )
        result = await registry.ainvoke_by_name("code_run", call)
        assert result.ok is True
        assert "docker ok" in result.output["stdout"]
