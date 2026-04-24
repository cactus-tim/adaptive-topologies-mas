"""Integration test for UrlFetchTool — requires live network access.

Skipped unless ATM_ENABLE_NETWORK_TESTS=1 is set in the environment.
"""

from __future__ import annotations

import os

import pytest

from atm.tools.global_.url_fetch import UrlFetchTool

_NETWORK_ENABLED = os.environ.get("ATM_ENABLE_NETWORK_TESTS") == "1"


@pytest.mark.network
@pytest.mark.skipif(
    not _NETWORK_ENABLED, reason="Network tests disabled (set ATM_ENABLE_NETWORK_TESTS=1)"
)
@pytest.mark.asyncio
async def test_fetch_example_com() -> None:
    """Fetch https://example.com and assert status=200."""
    tool = UrlFetchTool()
    result = await tool.ainvoke({"url": "https://example.com"})
    assert result.ok, f"Expected ok=True, got error: {result.error}"
    assert result.output is not None
    assert result.output["status"] == 200
    assert len(result.output["body"]) > 0
