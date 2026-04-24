"""Integration test for DuckDuckGoSearchTool — hits the live DDGS API.

Gated by @pytest.mark.network; skipped unless ATM_ENABLE_NETWORK_TESTS=1.
"""

from __future__ import annotations

import pytest

from atm.tools.global_.search import DuckDuckGoSearchTool


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_search_returns_results() -> None:
    """Live DDGS search returns non-empty results for a common query."""
    tool = DuckDuckGoSearchTool(max_results=5, retry_policy=None)

    result = await tool.ainvoke({"query": "Python programming language"})

    assert result.ok is True, f"Expected ok=True but got error: {result.error}"
    results = result.output["results"]
    assert len(results) > 0, "Expected at least one result from live DDGS search"
    # Verify the shape of each result
    for item in results:
        assert "title" in item
        assert "url" in item
        assert "snippet" in item
