"""Unit tests for DuckDuckGoSearchTool.

All tests use a FakeDDGS monkeypatch — no real network calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from atm.core.errors import ToolError
from atm.llm.retry import RetryPolicy
from atm.tools.global_.search import DuckDuckGoSearchTool


_CANNED_RESULTS: list[dict[str, Any]] = [
    {"title": "Result One", "href": "https://example.com/1", "body": "Snippet one."},
    {"title": "Result Two", "href": "https://example.com/2", "body": "Snippet two."},
    {"title": "Result Three", "href": "https://example.com/3", "body": "Snippet three."},
]


class FakeDDGS:
    """In-memory DDGS stub that returns canned results."""

    def __init__(self) -> None:
        pass

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return list(_CANNED_RESULTS)


class FakeEmptyDDGS:
    """In-memory DDGS stub that returns no results."""

    def __init__(self) -> None:
        pass

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class FakeFailingDDGS:
    """DDGS stub that raises on first call, then succeeds."""

    _call_count: int = 0

    def __init__(self) -> None:
        pass

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        FakeFailingDDGS._call_count += 1
        if FakeFailingDDGS._call_count == 1:
            raise OSError("network error")
        return list(_CANNED_RESULTS)


@pytest.mark.asyncio
async def test_happy_path_returns_results() -> None:
    """Tool returns correctly mapped results on success."""
    tool = DuckDuckGoSearchTool(max_results=10, retry_policy=None)

    with patch("atm.tools.global_.search.DDGS", FakeDDGS):
        result = await tool.ainvoke({"query": "python programming"})

    assert result.ok is True
    assert result.error is None
    assert isinstance(result.output, dict)
    results = result.output["results"]
    assert len(results) == 3
    first = results[0]
    assert first["title"] == "Result One"
    assert first["url"] == "https://example.com/1"
    assert first["snippet"] == "Snippet one."


@pytest.mark.asyncio
async def test_empty_results_returns_empty_list() -> None:
    """Tool returns ok=True with empty results list when DDGS returns nothing."""
    tool = DuckDuckGoSearchTool(max_results=10, retry_policy=None)

    with patch("atm.tools.global_.search.DDGS", FakeEmptyDDGS):
        result = await tool.ainvoke({"query": "xyzzy nonexistent"})

    assert result.ok is True
    assert result.output["results"] == []


@pytest.mark.asyncio
async def test_empty_query_returns_error() -> None:
    """Tool returns ok=False when query is empty string."""
    tool = DuckDuckGoSearchTool(max_results=10, retry_policy=None)

    result = await tool.ainvoke({"query": ""})

    assert result.ok is False
    assert result.error is not None
    assert "query" in result.error


@pytest.mark.asyncio
async def test_whitespace_query_returns_error() -> None:
    """Tool returns ok=False when query is all whitespace."""
    tool = DuckDuckGoSearchTool(max_results=10, retry_policy=None)

    result = await tool.ainvoke({"query": "   "})

    assert result.ok is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_schema_has_required_fields() -> None:
    """ToolSchema has correct name and returns field."""
    tool = DuckDuckGoSearchTool()

    assert tool.name == "duckduckgo_search"
    assert DuckDuckGoSearchTool.schema.name == "duckduckgo_search"
    assert "results" in DuckDuckGoSearchTool.schema.returns


@pytest.mark.asyncio
async def test_retry_on_first_failure_then_success() -> None:
    """Tool retries on transient error and returns results on second attempt."""
    FakeFailingDDGS._call_count = 0

    policy = RetryPolicy(
        max_retries=2,
        base_delay_s=0.0,
        max_delay_s=0.0,
        jitter=False,
        retry_on=(OSError,),
    )
    tool = DuckDuckGoSearchTool(max_results=10, retry_policy=policy)

    with patch("atm.tools.global_.search.DDGS", FakeFailingDDGS):
        result = await tool.ainvoke({"query": "retry test"})

    assert result.ok is True
    results = result.output["results"]
    assert len(results) == 3
    assert FakeFailingDDGS._call_count == 2
