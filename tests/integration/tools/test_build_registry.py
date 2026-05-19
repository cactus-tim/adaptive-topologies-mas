"""Integration tests for build_default_registry."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from atm.core.errors import ToolError
from atm.tools.defaults import build_default_registry
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture(autouse=True)
def _create_dirs(workspace: Path, tmp_path: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_1.txt").write_text("neural network deep learning", encoding="utf-8")


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    return tmp_path / "corpus"


class TestBuildDefaultRegistry:
    def test_prod_mode_rejects_subprocess(self, workspace: Path, corpus_dir: Path) -> None:
        """prod_mode=True with non-isolated sandbox must raise ToolError."""
        sandbox = SubprocessSandbox()
        assert sandbox.IS_ISOLATED is False

        with pytest.raises(ToolError) as exc_info:
            build_default_registry(
                workspace=workspace,
                corpus_dir=corpus_dir,
                sandbox=sandbox,
                prod_mode=True,
            )

        error_msg = str(exc_info.value)
        assert "isolated" in error_msg.lower()

    def test_prod_mode_accepts_docker(self, workspace: Path, corpus_dir: Path) -> None:
        """prod_mode=True with IS_ISOLATED=True sandbox must succeed."""

        class FakeDockerSandbox:
            IS_ISOLATED: ClassVar[bool] = True

            async def execute(self, lang, code, files=None, timeout=None):
                pass

        registry = build_default_registry(
            workspace=workspace,
            corpus_dir=corpus_dir,
            sandbox=FakeDockerSandbox(),  # type: ignore[arg-type]
            prod_mode=True,
        )
        assert registry is not None
        assert len(registry.names()) == 12

    def test_default_registry_has_all_tools(self, workspace: Path, corpus_dir: Path) -> None:
        """Registry built with SubprocessSandbox should have all 12 tool names."""
        registry = build_default_registry(
            workspace=workspace,
            corpus_dir=corpus_dir,
            sandbox=SubprocessSandbox(),
            prod_mode=False,
        )
        names = registry.names()
        assert len(names) == 12

        expected = {
            "calculator",
            "code_run",
            "diff",
            "duckduckgo_search",
            "file_read",
            "file_write",
            "lint",
            "plan_update",
            "semantic_search",
            "test_run",
            "todo_write",
            "url_fetch",
        }
        assert set(names) == expected

    def test_dev_mode_does_not_reject_subprocess(self, workspace: Path, corpus_dir: Path) -> None:
        """prod_mode=False (default) must accept non-isolated sandbox."""
        registry = build_default_registry(
            workspace=workspace,
            corpus_dir=corpus_dir,
            sandbox=SubprocessSandbox(),
        )
        assert registry is not None
