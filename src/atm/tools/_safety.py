"""SSRF defense utilities for the ATM tools layer.

Public API
----------
resolve_and_validate_url -- validates URL scheme and resolves host to reject private IPs
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse

from atm.core.errors import ToolError

_ALLOWED_SCHEMES = {"http", "https"}


def resolve_and_validate_url(url: str, allow_private: bool = False) -> str:
    """Validate *url* against SSRF threats and return it if safe.

    Validation steps
    ----------------
    1. Scheme must be in ``{"http", "https"}``; otherwise raise ToolError.
    2. Host must be present; otherwise raise ToolError.
    3. Resolve host via ``socket.getaddrinfo``; for each resolved IP:
       - Reject if ``is_private | is_loopback | is_link_local | is_reserved | is_multicast``
         *unless* ``allow_private=True``.

    Parameters
    ----------
    url:           The URL to validate.
    allow_private: When True, private/loopback addresses are permitted (dev use only).

    Returns
    -------
    str
        The original *url* string (unchanged) if validation passes.

    Raises
    ------
    ToolError(tool_name="url_fetch", ...)
        On any validation failure.
    """
    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ToolError(
            tool_name="url_fetch",
            message=f"scheme not allowed: {parsed.scheme!r} (allowed: http, https)",
        )

    host = parsed.hostname
    if not host:
        raise ToolError(
            tool_name="url_fetch",
            message="missing host in URL",
        )

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ToolError(
            tool_name="url_fetch",
            message=f"DNS resolution failed for {host!r}: {exc}",
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            # Malformed address — reject to be safe
            raise ToolError(
                tool_name="url_fetch",
                message=f"could not parse resolved address: {ip_str!r}",
            ) from exc

        if not allow_private and (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
        ):
            raise ToolError(
                tool_name="url_fetch",
                message=f"private/reserved address blocked: {ip_str}",
            )

    return url
