"""Tests for FileWriteTool — atomic, workspace-scoped file write."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from atm.tools.local_.file_write import FileWriteTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(workspace: Path, create_parents: bool = False) -> FileWriteTool:
    return FileWriteTool(workspace=workspace, create_parents=create_parents)


async def _invoke(tool: FileWriteTool, args: dict[str, Any]):
    return await tool.ainvoke(args)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_happy_path(tmp_path: Path) -> None:
    """Writing a new file returns ok=True with bytes_written and path."""
    tool = _tool(tmp_path)
    content = "Hello, World!\n"
    result = await _invoke(tool, {"path": "hello.txt", "content": content})

    assert result.ok is True
    assert result.error is None
    output = result.output
    assert isinstance(output, dict)

    expected_bytes = len(content.encode("utf-8"))
    assert output["bytes_written"] == expected_bytes
    assert output["path"] == "hello.txt"

    # Verify the file was actually written with correct content
    written = (tmp_path / "hello.txt").read_text(encoding="utf-8")
    assert written == content


# ---------------------------------------------------------------------------
# Reject absolute paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_rejects_absolute(tmp_path: Path) -> None:
    """Absolute path input (starting with /) must be rejected."""
    tool = _tool(tmp_path)
    result = await _invoke(tool, {"path": "/etc/shadow", "content": "evil"})

    assert result.ok is False
    assert result.error is not None
    assert "absolute" in result.error.lower()


# ---------------------------------------------------------------------------
# Reject path traversal (outside workspace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_rejects_traversal(tmp_path: Path) -> None:
    """Path traversal via ../ must be rejected with 'outside workspace' error."""
    tool = _tool(tmp_path)
    result = await _invoke(tool, {"path": "../outside.txt", "content": "evil"})

    assert result.ok is False
    assert result.error is not None
    assert "outside workspace" in result.error.lower()


# ---------------------------------------------------------------------------
# Reject overwrite when overwrite=False (default)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_rejects_exists_no_overwrite(tmp_path: Path) -> None:
    """Writing to an existing file without overwrite=True must be rejected."""
    target = tmp_path / "existing.txt"
    target.write_text("original content", encoding="utf-8")

    tool = _tool(tmp_path)
    result = await _invoke(tool, {"path": "existing.txt", "content": "new"})

    assert result.ok is False
    assert result.error is not None
    assert "exists" in result.error.lower()

    # Original file must be untouched
    assert target.read_text(encoding="utf-8") == "original content"


# ---------------------------------------------------------------------------
# Allow overwrite when overwrite=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_overwrite_true(tmp_path: Path) -> None:
    """overwrite=True allows replacing an existing file."""
    target = tmp_path / "existing.txt"
    target.write_text("old content", encoding="utf-8")

    tool = _tool(tmp_path)
    new_content = "new content"
    result = await _invoke(
        tool, {"path": "existing.txt", "content": new_content, "overwrite": True}
    )

    assert result.ok is True
    assert result.error is None
    assert result.output["bytes_written"] == len(new_content.encode("utf-8"))

    # Verify the content was replaced
    assert target.read_text(encoding="utf-8") == new_content


# ---------------------------------------------------------------------------
# Reject missing parent when create_parents=False (default)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_parent_missing_no_create_parents(tmp_path: Path) -> None:
    """Missing parent directory without create_parents=True must be rejected."""
    tool = _tool(tmp_path, create_parents=False)
    result = await _invoke(tool, {"path": "sub/deep/x.txt", "content": "data"})

    assert result.ok is False
    assert result.error is not None
    assert "parent" in result.error.lower()

    # Ensure no partial directories were created
    assert not (tmp_path / "sub").exists()


# ---------------------------------------------------------------------------
# Create parents when create_parents=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_parent_created_when_create_parents_true(tmp_path: Path) -> None:
    """create_parents=True causes missing parent directories to be created."""
    tool = _tool(tmp_path, create_parents=True)
    content = "deep content"
    result = await _invoke(tool, {"path": "sub/deep/x.txt", "content": content})

    assert result.ok is True
    assert result.error is None

    target = tmp_path / "sub" / "deep" / "x.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# Atomic write — no partial file on failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_atomic_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If os.replace raises, no tmp file should remain in the workspace."""
    tool = _tool(tmp_path)

    original_replace = os.replace

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    result = await _invoke(tool, {"path": "atomic.txt", "content": "data"})

    # The operation should fail (not ok)
    assert result.ok is False

    # No .tmp file should remain
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert tmp_files == [], f"Leftover tmp files found: {tmp_files}"

    # The target file must not exist either
    assert not (tmp_path / "atomic.txt").exists()

    monkeypatch.setattr(os, "replace", original_replace)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_file_write_name_and_schema(tmp_path: Path) -> None:
    """FileWriteTool must expose correct name and schema."""
    tool = _tool(tmp_path)
    assert tool.name == "file_write"
    assert "bytes_written" in tool.schema.returns
    assert "path" in tool.schema.returns
