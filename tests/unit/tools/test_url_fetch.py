"""Unit tests for UrlFetchTool (SSRF-safe HTTP fetch).

All network calls are intercepted via httpx.MockTransport.
All DNS lookups are monkeypatched via socket.getaddrinfo.
"""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

from atm.llm.retry import RetryPolicy
from atm.tools.global_.url_fetch import UrlFetchTool


def _fake_getaddrinfo_for(ip: str):
    """Return a fake getaddrinfo that resolves any host to *ip*."""

    def fake_getaddrinfo(
        host: str,
        port: Any,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[Any]:
        return [(2, 1, 6, "", (ip, port or 0))]

    return fake_getaddrinfo


def _make_tool(allow_private: bool = False, max_bytes: int = 1_000_000) -> UrlFetchTool:
    """Return a UrlFetchTool with fast, no-sleep retry for tests."""
    policy = RetryPolicy(max_retries=0, base_delay_s=0.0, max_delay_s=0.0, jitter=False)
    return UrlFetchTool(
        allow_private=allow_private,
        max_bytes=max_bytes,
        timeout_s=5.0,
        retry_policy=policy,
    )


def _mock_transport_200(
    body: bytes = b"Hello, world!", content_type: str = "text/plain"
) -> httpx.MockTransport:
    """Return a MockTransport that responds 200 OK with the given body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": content_type},
        )

    return httpx.MockTransport(handler)


def _mock_transport_302(location: str = "http://example.com/other") -> httpx.MockTransport:
    """Return a MockTransport that responds with a 302 redirect."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": location},
            content=b"",
        )

    return httpx.MockTransport(handler)


def _mock_transport_timeout() -> httpx.MockTransport:
    """Return a MockTransport that always raises TimeoutException."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_rejects_metadata_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """AWS metadata endpoint (169.254.169.254) must be blocked."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("169.254.169.254"))
    tool = _make_tool()
    result = await tool.ainvoke({"url": "http://169.254.169.254/latest/meta-data"})
    assert not result.ok
    assert result.error is not None
    assert "url rejected" in result.error


@pytest.mark.asyncio
async def test_rejects_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loopback address 127.0.0.1 must be blocked."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("127.0.0.1"))
    tool = _make_tool()
    result = await tool.ainvoke({"url": "http://127.0.0.1:8080"})
    assert not result.ok
    assert result.error is not None
    assert "url rejected" in result.error


@pytest.mark.asyncio
async def test_rejects_ipv6_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """IPv6 loopback [::1] must be blocked."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("::1"))
    tool = _make_tool()
    result = await tool.ainvoke({"url": "http://[::1]"})
    assert not result.ok
    assert result.error is not None
    assert "url rejected" in result.error


@pytest.mark.asyncio
async def test_rejects_rfc1918(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC 1918 address 10.0.0.1 must be blocked."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("10.0.0.1"))
    tool = _make_tool()
    result = await tool.ainvoke({"url": "http://10.0.0.1"})
    assert not result.ok
    assert result.error is not None
    assert "url rejected" in result.error


@pytest.mark.asyncio
async def test_rejects_ftp_scheme() -> None:
    """ftp:// scheme must be rejected without DNS lookup."""
    tool = _make_tool()
    result = await tool.ainvoke({"url": "ftp://example.com"})
    assert not result.ok
    assert result.error is not None
    assert "url rejected" in result.error


@pytest.mark.asyncio
async def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public IP (93.184.216.34 = example.com) with 200 OK must succeed."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("93.184.216.34"))
    tool = _make_tool()

    transport = _mock_transport_200(b"<html>Hello</html>", "text/html")
    original_init = httpx.AsyncClient.__init__

    def patched_init(self_client: httpx.AsyncClient, **kwargs: Any) -> None:
        kwargs["transport"] = transport
        original_init(self_client, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    result = await tool.ainvoke({"url": "http://example.com"})
    assert result.ok, f"Expected ok=True but got error: {result.error}"
    assert result.output is not None
    assert result.output["status"] == 200
    assert "Hello" in result.output["content"]
    assert result.output["size"] > 0


@pytest.mark.asyncio
async def test_allow_private_true_permits_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """With allow_private=True, localhost (127.0.0.1) must be permitted."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("127.0.0.1"))
    tool = _make_tool(allow_private=True)

    transport = _mock_transport_200(b"OK")

    class PatchedClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    monkeypatch.setattr("atm.tools.global_.url_fetch.httpx.AsyncClient", PatchedClient)

    result = await tool.ainvoke({"url": "http://127.0.0.1:8080/health"})
    assert result.ok, f"Expected ok=True but got error: {result.error}"
    assert result.output is not None
    assert result.output["status"] == 200


@pytest.mark.asyncio
async def test_size_limit_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Body exceeding max_bytes must yield ok=False with 'too large' error."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("93.184.216.34"))

    big_body = b"X" * 100
    transport = _mock_transport_200(big_body)

    class PatchedClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    monkeypatch.setattr("atm.tools.global_.url_fetch.httpx.AsyncClient", PatchedClient)

    tool = UrlFetchTool(
        allow_private=False,
        max_bytes=50,
        timeout_s=5.0,
        retry_policy=RetryPolicy(max_retries=0, base_delay_s=0.0, max_delay_s=0.0, jitter=False),
    )
    result = await tool.ainvoke({"url": "http://example.com"})
    assert not result.ok
    assert result.error is not None
    assert "too large" in result.error


@pytest.mark.asyncio
async def test_follow_redirects_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """302 redirect must yield ok=False with 'redirects' error."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("93.184.216.34"))

    transport = _mock_transport_302()

    class PatchedClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    monkeypatch.setattr("atm.tools.global_.url_fetch.httpx.AsyncClient", PatchedClient)

    tool = _make_tool()
    result = await tool.ainvoke({"url": "http://example.com"})
    assert not result.ok
    assert result.error is not None
    assert "redirect" in result.error


@pytest.mark.asyncio
async def test_timeout_raises_retry_then_toolerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """MockTransport always raises TimeoutException → retry exhausted → ok=False."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_for("93.184.216.34"))

    transport = _mock_transport_timeout()

    class PatchedClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    monkeypatch.setattr("atm.tools.global_.url_fetch.httpx.AsyncClient", PatchedClient)

    policy = RetryPolicy(
        max_retries=1,
        base_delay_s=0.0,
        max_delay_s=0.0,
        jitter=False,
        retry_on=(httpx.TimeoutException,),
    )
    tool = UrlFetchTool(
        allow_private=False,
        max_bytes=1_000_000,
        timeout_s=5.0,
        retry_policy=policy,
    )
    result = await tool.ainvoke({"url": "http://example.com"})
    assert not result.ok
    assert result.error is not None
