"""Tests for atm.tools._safety — resolve_and_validate_url."""

from __future__ import annotations

import socket

import pytest

from atm.core.errors import ToolError
from atm.tools._safety import resolve_and_validate_url


def _make_fake_getaddrinfo(ip_address: str):
    """Return a monkeypatched getaddrinfo that always resolves to ip_address."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip_address, 80))]

    return fake_getaddrinfo


def _make_fake_getaddrinfo_ipv6(ip_address: str):
    """Return a monkeypatched getaddrinfo that resolves to an IPv6 address."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip_address, 80, 0, 0))]

    return fake_getaddrinfo


def test_safety_rejects_ftp_scheme() -> None:
    with pytest.raises(ToolError) as exc_info:
        resolve_and_validate_url("ftp://example.com/file.txt")
    assert "scheme" in str(exc_info.value).lower() or exc_info.value.tool_name == "url_fetch"


def test_safety_rejects_file_scheme() -> None:
    with pytest.raises(ToolError):
        resolve_and_validate_url("file:///etc/passwd")


def test_safety_rejects_loopback(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo("127.0.0.1"))
    with pytest.raises(ToolError):
        resolve_and_validate_url("http://localhost/")


def test_safety_rejects_metadata_ip(monkeypatch) -> None:
    """169.254.169.254 is link-local (AWS metadata endpoint)."""
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo("169.254.169.254"))
    with pytest.raises(ToolError):
        resolve_and_validate_url("http://169.254.169.254/latest/meta-data/")


def test_safety_rejects_ipv6_localhost(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo_ipv6("::1"))
    with pytest.raises(ToolError):
        resolve_and_validate_url("http://[::1]/")


def test_safety_rejects_rfc1918(monkeypatch) -> None:
    """10.0.0.1 is a private address."""
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo("10.0.0.1"))
    with pytest.raises(ToolError):
        resolve_and_validate_url("http://10.0.0.1/")


def test_safety_rejects_private_192(monkeypatch) -> None:
    """192.168.1.1 is a private address."""
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo("192.168.1.1"))
    with pytest.raises(ToolError):
        resolve_and_validate_url("http://192.168.1.1/")


def test_safety_happy_path_public(monkeypatch) -> None:
    """A public IP (1.1.1.1) should pass."""
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo("1.1.1.1"))
    url = resolve_and_validate_url("https://example.com/page")
    assert url == "https://example.com/page"


def test_safety_allow_private_true_permits_localhost(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _make_fake_getaddrinfo("127.0.0.1"))
    url = resolve_and_validate_url("http://localhost/", allow_private=True)
    assert url == "http://localhost/"


def test_safety_missing_host_raises() -> None:
    with pytest.raises(ToolError):
        resolve_and_validate_url("http:///path")
