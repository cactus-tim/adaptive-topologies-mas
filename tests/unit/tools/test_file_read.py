"""Tests for FileReadTool — workspace-scoped file read with path traversal protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from atm.tools.global_.file_read import FileReadTool


async def _invoke(tool: FileReadTool, path: str):
    """Thin wrapper to invoke the tool and return the ToolResult."""
    return await tool.ainvoke({"path": path})


@pytest.mark.asyncio
async def test_file_read_happy(tmp_path: Path) -> None:
    """Reading an existing file inside the workspace succeeds."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("hello")

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "a.txt")

    assert result.ok is True
    assert result.output is not None
    assert result.output["content"] == "hello"
    assert result.output["path"] == "a.txt"
    assert result.output["size"] == 5


@pytest.mark.asyncio
async def test_file_read_rejects_traversal(tmp_path: Path) -> None:
    """Path component ``../..`` that escapes the workspace is rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "../../../etc/passwd")

    assert result.ok is False
    assert result.error is not None
    assert "outside workspace" in result.error


@pytest.mark.asyncio
async def test_file_read_rejects_absolute(tmp_path: Path) -> None:
    """Absolute paths are rejected regardless of whether they are inside or outside workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "/etc/passwd")

    assert result.ok is False
    assert result.error is not None
    assert "absolute paths not allowed" in result.error


@pytest.mark.asyncio
async def test_file_read_rejects_oversize(tmp_path: Path) -> None:
    """Files exceeding max_bytes are rejected — NO silent truncation."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    big_file = workspace / "big.bin"
    big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

    tool = FileReadTool(workspace=workspace, max_bytes=1024 * 1024)
    result = await _invoke(tool, "big.bin")

    assert result.ok is False
    assert result.error is not None
    assert "too large" in result.error


@pytest.mark.asyncio
async def test_file_read_rejects_symlink_outside(tmp_path: Path) -> None:
    """A symlink inside the workspace that points to a file outside is rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    link = workspace / "secret_link"
    link.symlink_to("/etc/hostname")

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "secret_link")

    assert result.ok is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_file_read_output_shape(tmp_path: Path) -> None:
    """Successful read returns all three required keys: content, path, size."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "doc.txt").write_text("shape test")

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "doc.txt")

    assert result.ok is True
    assert result.output is not None
    keys = set(result.output.keys())
    assert "content" in keys
    assert "path" in keys
    assert "size" in keys


@pytest.mark.asyncio
async def test_file_read_nested_path(tmp_path: Path) -> None:
    """Files in subdirectories within the workspace can be read."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subdir = workspace / "sub" / "dir"
    subdir.mkdir(parents=True)
    (subdir / "data.txt").write_text("nested content")

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "sub/dir/data.txt")

    assert result.ok is True
    assert result.output["content"] == "nested content"
    assert result.output["path"] == "sub/dir/data.txt"


@pytest.mark.asyncio
async def test_file_read_file_not_found(tmp_path: Path) -> None:
    """Reading a non-existent file returns ok=False with an appropriate error."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tool = FileReadTool(workspace=workspace)
    result = await _invoke(tool, "nonexistent.txt")

    assert result.ok is False
    assert result.error is not None


def test_file_read_name_and_schema(tmp_path: Path) -> None:
    """FileReadTool has expected name and schema with returns field."""
    tool = FileReadTool(workspace=tmp_path)

    assert tool.name == "file_read"
    schema = tool.schema
    assert "content" in schema.returns
    assert "path" in schema.returns
    assert "size" in schema.returns
