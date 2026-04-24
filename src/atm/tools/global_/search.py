"""DuckDuckGoSearchTool — web search via the ddgs package.

Uses asyncio.to_thread to call the synchronous DDGS().text() in a thread pool,
so the event loop is never blocked.

Retry behaviour is controlled by the injected RetryPolicy. When policy is None,
no retry is applied.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, ClassVar

from ddgs import DDGS

from atm.core.errors import LLMError, ToolError
from atm.core.types import ToolResult
from atm.llm.retry import RetryPolicy, with_retry
from atm.tools.base import ToolSchema


class DuckDuckGoSearchTool:
    """Global tool that performs web searches via DuckDuckGo.

    Each result dict has three keys:
        title   -- page title
        url     -- canonical URL (mapped from ddgs "href" field)
        snippet -- text excerpt (mapped from ddgs "body" field)

    Args:
        max_results:  Maximum number of results to return (default 10).
        retry_policy: RetryPolicy for transient failures (None = no retry).
    """

    name: ClassVar[str] = "duckduckgo_search"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="duckduckgo_search",
        description="Search the web using DuckDuckGo and return a list of results.",
        parameters={
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
        },
        returns={
            "results": "array of {title: string, url: string, snippet: string}",
        },
    )

    def __init__(
        self,
        max_results: int = 10,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._max_results = max_results
        self._retry_policy = retry_policy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_search(self, query: str) -> list[dict[str, Any]]:
        """Synchronous DDGS search — runs inside asyncio.to_thread."""
        raw: list[dict[str, Any]] = DDGS().text(query, max_results=self._max_results)
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            }
            for item in (raw or [])
        ]

    async def _search(self, query: str) -> list[dict[str, Any]]:
        """Run the search, with optional retry wrapping."""
        if self._retry_policy is None:
            return await asyncio.to_thread(self._run_search, query)

        # Inline retry: call _run_search via with_retry
        try:
            return await with_retry(
                lambda: asyncio.to_thread(self._run_search, query),
                policy=self._retry_policy,
                provider="tool",
                model=self.name,
            )
        except LLMError as exc:
            raise ToolError(tool_name=self.name, message=str(exc)) from exc

    # ------------------------------------------------------------------
    # Tool Protocol implementation
    # ------------------------------------------------------------------

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke DuckDuckGo search.

        Args:
            args: Must contain key "query" (str).

        Returns:
            ToolResult with ok=True and output={"results": [...]} on success,
            or ok=False with error message on failure.
        """
        query: str = args.get("query", "")
        if not query or not query.strip():
            return ToolResult(
                call_id=uuid.uuid4(),
                ok=False,
                output=None,
                error="missing or empty 'query' argument",
                latency_ms=0,
            )

        try:
            results = await self._search(query)
            return ToolResult(
                call_id=uuid.uuid4(),
                ok=True,
                output={"results": results},
                error=None,
                latency_ms=0,
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(tool_name=self.name, message=str(exc)) from exc
