"""SSRF-safe HTTP URL fetch tool.

Public API
----------
UrlFetchTool -- fetches a URL via httpx with SSRF protection, size limit, and retry
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

import httpx

from atm.core.errors import ToolError
from atm.core.types import ToolResult
from atm.llm.retry import RetryPolicy
from atm.tools._retry import with_tool_retry
from atm.tools._safety import resolve_and_validate_url
from atm.tools.base import ToolSchema

# Default retry policy: 2 retries, fast backoff, no jitter for predictability in tests
_DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    base_delay_s=0.1,
    max_delay_s=1.0,
    jitter=False,
    retry_on=(httpx.TimeoutException,),
)

_CALL_ID = uuid4  # callable alias for test clarity


class UrlFetchTool:
    """SSRF-safe HTTP GET tool backed by httpx.

    Features
    --------
    - Calls ``resolve_and_validate_url`` before each request to block SSRF.
    - ``follow_redirects=False`` by default — 3xx responses yield ok=False.
    - Body streamed via ``aiter_bytes``; aborts if ``max_bytes`` exceeded.
    - Body decoded as UTF-8 with ``errors='replace'``.
    - Retried via ``with_tool_retry`` on transient errors (e.g. timeout).

    Documented alternative for redirect handling (NOT implemented by default):
        Use ``httpx.AsyncClient(event_hooks={"response": [_revalidate_redirect]})``
        to revalidate each ``Location`` header through ``resolve_and_validate_url``
        before following it.  This is NOT enabled because it adds complexity and
        most callers should resolve the final URL explicitly.

    Parameters
    ----------
    allow_private: Allow private/loopback addresses (dev mode only).
    max_bytes:     Maximum body size in bytes; abort with error if exceeded.
    timeout_s:     Request timeout in seconds.
    retry_policy:  Retry policy for transient failures; None uses default.
    """

    name: ClassVar[str] = "url_fetch"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="url_fetch",
        description=(
            "Fetch a public URL via HTTP GET. "
            "Blocks SSRF (private IPs, loopback, link-local). "
            "Does not follow redirects."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
        returns={
            "status": "integer",
            "url": "string",
            "content_type": "string",
            "body": "string (utf-8 decoded)",
            "bytes_read": "integer",
        },
    )

    def __init__(
        self,
        allow_private: bool = False,
        max_bytes: int = 1_000_000,
        timeout_s: float = 30.0,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.allow_private = allow_private
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self._policy = retry_policy if retry_policy is not None else _DEFAULT_RETRY_POLICY

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Fetch the URL specified in *args["url"]*.

        Returns
        -------
        ToolResult
            Always returns a result; never raises.
        """
        url: str = args.get("url", "")
        call_id = _CALL_ID()

        @with_tool_retry(self._policy, self.name)
        async def _fetch() -> ToolResult:
            return await self._do_fetch(url, call_id)

        try:
            return await _fetch()
        except ToolError as exc:
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=str(exc),
                latency_ms=0,
            )

    async def _do_fetch(self, url: str, call_id: Any) -> ToolResult:
        """Core fetch logic — validates URL, sends GET, streams body."""
        # Step 1: SSRF validation
        try:
            resolve_and_validate_url(url, allow_private=self.allow_private)
        except ToolError as exc:
            msg = str(exc)
            # Extract the message portion after "Tool 'url_fetch' failed: "
            prefix = "Tool 'url_fetch' failed: "
            if msg.startswith(prefix):
                msg = msg[len(prefix) :]
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=f"url rejected: {msg}",
                latency_ms=0,
            )

        # Step 2: HTTP GET (no redirects)
        async with httpx.AsyncClient(
            timeout=self.timeout_s,
            follow_redirects=False,
        ) as client:
            try:
                response = await client.send(
                    client.build_request("GET", url),
                    stream=True,
                )
            except httpx.TimeoutException as exc:
                raise exc  # let with_tool_retry handle it

            # Step 3: Reject redirects
            if response.is_redirect or (300 <= response.status_code < 400):
                await response.aclose()
                return ToolResult(
                    call_id=call_id,
                    ok=False,
                    output=None,
                    error="redirects not followed",
                    latency_ms=0,
                )

            # Step 4: Stream body up to max_bytes
            chunks: list[bytes] = []
            bytes_read = 0
            too_large = False

            async for chunk in response.aiter_bytes():
                bytes_read += len(chunk)
                if bytes_read > self.max_bytes:
                    too_large = True
                    await response.aclose()
                    break
                chunks.append(chunk)

            if too_large:
                return ToolResult(
                    call_id=call_id,
                    ok=False,
                    output=None,
                    error="response too large",
                    latency_ms=0,
                )

            # Step 5: Decode body
            raw_body = b"".join(chunks)
            body = raw_body.decode("utf-8", errors="replace")
            content_type = response.headers.get("content-type", "")

            return ToolResult(
                call_id=call_id,
                ok=True,
                output={
                    "status": response.status_code,
                    "url": str(response.url),
                    "content_type": content_type,
                    "body": body,
                    "bytes_read": len(raw_body),
                },
                error=None,
                latency_ms=0,
            )
