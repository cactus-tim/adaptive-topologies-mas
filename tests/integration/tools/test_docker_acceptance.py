"""Docker acceptance tests — gated by @pytest.mark.docker.

These tests require a running Docker daemon. Skip unless ATM_ENABLE_DOCKER_TESTS=1.

Tests
-----
- DockerSandbox.execute hello-world: stdout "2\\n", exit 0.
- TestRunTool via DockerSandbox: trivial unittest → passed=True.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip all tests in this module unless Docker env var is set
# (conftest.py also handles per-item skip; this module-level mark is backup)
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.docker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDockerAcceptance:
    @pytest.mark.docker
    def test_docker_sandbox_hello_world(self, tmp_path: Path) -> None:
        """DockerSandbox.execute('python', 'print(1+1)') returns stdout='2\\n', exit=0."""
        from atm.tools.sandbox.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox()
        result = run(sandbox.execute(lang="python", code="print(1+1)"))

        assert result.stdout == "2\n"
        assert result.exit_code == 0
        assert result.timed_out is False

    @pytest.mark.docker
    def test_docker_test_run_passing(self, tmp_path: Path) -> None:
        """TestRunTool backed by DockerSandbox: trivial passing test → passed=True."""
        from atm.core.types import ToolCall
        from atm.tools.local_.test_run import TestRunTool
        from atm.tools.sandbox.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox()
        tool = TestRunTool(sandbox=sandbox)

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
        result = run(tool.ainvoke(call.args))
        assert result.ok is True
        assert result.output["passed"] is True
        assert result.output["summary"].startswith("OK")

    @pytest.mark.docker
    def test_docker_registry_code_run(self, tmp_path: Path) -> None:
        """Registry with DockerSandbox can run code_run end-to-end."""
        from atm.core.types import ToolCall
        from atm.tools.defaults import build_default_registry
        from atm.tools.sandbox.docker_sandbox import DockerSandbox

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "doc_1.txt").write_text("neural network test doc", encoding="utf-8")

        sandbox = DockerSandbox()
        registry = build_default_registry(
            workspace=tmp_path / "workspace",
            corpus_dir=corpus_dir,
            sandbox=sandbox,
            prod_mode=True,  # DockerSandbox IS_ISOLATED=True
        )
        (tmp_path / "workspace").mkdir(exist_ok=True)

        call = ToolCall(
            tool_name="code_run",
            issued_by="test_agent",
            args={"lang": "python", "code": "print('docker ok')"},
        )
        result = run(registry.ainvoke_by_name("code_run", call))
        assert result.ok is True
        assert "docker ok" in result.output["stdout"]
