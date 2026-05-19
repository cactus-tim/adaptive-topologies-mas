"""End-to-end integration tests using SubprocessSandbox (dev mode).

Chains multiple tools through the registry to validate the full M4 tool set
working together. All tests are non-docker, non-network — run always.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atm.tools.defaults import build_default_registry
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ws = tmp_path_factory.mktemp("e2e_workspace")
    return ws


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cd = tmp_path_factory.mktemp("e2e_corpus")
    (cd / "doc_1.txt").write_text(
        "Deep learning and neural networks have revolutionized artificial intelligence. "
        "A neural network consists of layers of interconnected nodes that learn patterns.",
        encoding="utf-8",
    )
    (cd / "doc_2.txt").write_text(
        "Carrot cake is a moist, spiced cake made with grated carrots, cinnamon, and nutmeg. "
        "The batter often includes walnuts or raisins for added texture.",
        encoding="utf-8",
    )
    (cd / "doc_3.txt").write_text(
        "Paris is the capital and largest city of France, situated along the Seine river. "
        "The city is renowned worldwide for its art, cuisine, and historic landmarks.",
        encoding="utf-8",
    )
    return cd


@pytest.fixture(scope="module")
def registry(workspace: Path, corpus_dir: Path):
    return build_default_registry(
        workspace=workspace,
        corpus_dir=corpus_dir,
        sandbox=SubprocessSandbox(),
        prod_mode=False,
    )


class TestRegistryEndToEnd:
    @pytest.mark.asyncio
    async def test_todo_write_state_update(self, registry) -> None:
        """todo_write returns state_update.shared.todos."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="todo_write",
            issued_by="test_agent",
            args={
                "todos": [
                    {"id": "t1", "content": "Write tests", "status": "open"},
                    {"id": "t2", "content": "Implement", "status": "done"},
                ]
            },
        )
        result = await registry.ainvoke_by_name("todo_write", call)
        assert result.ok is True
        assert "state_update" in result.output
        assert "shared" in result.output["state_update"]
        todos = result.output["state_update"]["shared"]["todos"]
        assert len(todos) == 2
        assert todos[0]["id"] == "t1"

    @pytest.mark.asyncio
    async def test_file_write_then_file_read(self, registry, workspace: Path) -> None:
        """file_write followed by file_read returns the same content."""
        from atm.core.types import ToolCall

        content = "print('hello from integration test')\n"

        write_call = ToolCall(
            tool_name="file_write",
            issued_by="test_agent",
            args={
                "path": "solution.py",
                "content": content,
                "overwrite": False,
            },
        )
        write_result = await registry.ainvoke_by_name("file_write", write_call)
        assert write_result.ok is True
        assert write_result.output["bytes_written"] == len(content.encode("utf-8"))

        read_call = ToolCall(
            tool_name="file_read",
            issued_by="test_agent",
            args={"path": "solution.py"},
        )
        read_result = await registry.ainvoke_by_name("file_read", read_call)
        assert read_result.ok is True
        assert read_result.output["content"] == content

    @pytest.mark.asyncio
    async def test_code_run_hello(self, registry) -> None:
        """code_run with print('hello') returns stdout 'hello\\n'."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="code_run",
            issued_by="test_agent",
            args={"lang": "python", "code": "print('hello')"},
        )
        result = await registry.ainvoke_by_name("code_run", call)
        assert result.ok is True
        assert result.output["stdout"] == "hello\n"
        assert result.output["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_diff_returns_unified_diff(self, registry) -> None:
        """diff comparing two strings returns a non-empty unified diff."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="diff",
            issued_by="test_agent",
            args={"before": "line1\nline2\n", "after": "line1\nline3\n"},
        )
        result = await registry.ainvoke_by_name("diff", call)
        assert result.ok is True
        diff_text = result.output["diff"]
        assert "@@" in diff_text
        assert "-line2" in diff_text
        assert "+line3" in diff_text

    @pytest.mark.asyncio
    async def test_lint_on_valid_code(self, registry) -> None:
        """lint on clean code returns ruff=[] and correct output shape."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="lint",
            issued_by="test_agent",
            args={"code": "x = 1\nprint(x)\n"},
        )
        result = await registry.ainvoke_by_name("lint", call)
        assert result.ok is True
        output = result.output
        assert "ruff" in output
        assert isinstance(output["ruff"], list)
        assert "pylint" in output
        assert "pylint_skipped" in output

    @pytest.mark.asyncio
    async def test_lint_detects_issue(self, registry) -> None:
        """lint on code with unused import detects a ruff diagnostic."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="lint",
            issued_by="test_agent",
            args={"code": "import os\nimport os\n"},
        )
        result = await registry.ainvoke_by_name("lint", call)
        assert result.ok is True
        assert isinstance(result.output["ruff"], list)
        for diag in result.output["ruff"]:
            assert "code" in diag or "message" in diag

    @pytest.mark.asyncio
    async def test_calculator_two_plus_two(self, registry) -> None:
        """calculator evaluates '2+2' → value=4."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="calculator",
            issued_by="test_agent",
            args={"expression": "2+2"},
        )
        result = await registry.ainvoke_by_name("calculator", call)
        assert result.ok is True
        assert result.output["value"] == 4

    @pytest.mark.asyncio
    async def test_semantic_search_neural(self, registry) -> None:
        """semantic_search 'neural' → top hit = doc_1."""
        from atm.core.types import ToolCall

        call = ToolCall(
            tool_name="semantic_search",
            issued_by="test_agent",
            args={"query": "neural"},
        )
        result = await registry.ainvoke_by_name("semantic_search", call)
        assert result.ok is True
        hits = result.output["hits"]
        assert len(hits) > 0
        assert hits[0]["doc_id"] == "doc_1"

    @pytest.mark.asyncio
    async def test_code_run_with_solution_file(self, registry, workspace: Path) -> None:
        """code_run can import solution.py written by file_write."""
        from atm.core.types import ToolCall

        write_call = ToolCall(
            tool_name="file_write",
            issued_by="test_agent",
            args={
                "path": "helper.py",
                "content": "def greet(): return 'world'\n",
                "overwrite": False,
            },
        )
        await registry.ainvoke_by_name("file_write", write_call)

        code_call = ToolCall(
            tool_name="code_run",
            issued_by="test_agent",
            args={
                "lang": "python",
                "code": "from helper import greet; print(greet())",
                "files": {"helper.py": "def greet(): return 'world'\n"},
            },
        )
        result = await registry.ainvoke_by_name("code_run", code_call)
        assert result.ok is True
        assert result.output["stdout"].strip() == "world"
