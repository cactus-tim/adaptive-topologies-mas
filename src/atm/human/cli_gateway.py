"""CLIGateway — stdin/stdout HITL gateway for debugging (M9 Step 2.2).

Renders the HumanContext to stdout and reads a decision from stdin via
``asyncio.to_thread(input, ...)`` so the event loop is never blocked.

Input format (one line):
    <action>
    <action> <comment text>
    {"action": "...", "comment": "...", "payload": {...}}

Unknown action → falls back to ``allowed_actions[0]`` with a comment
recording the invalid input.

Idempotency:
    Same ``(run_id, request_id)`` pair → same cached HumanResponse
    without re-prompting the user.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any
from uuid import UUID

from atm.core.types import HumanContext, HumanResponse

__all__ = ["CLIGateway"]

_SEPARATOR = "-" * 60


class CLIGateway:
    """Interactive CLI gateway — reads human decisions from stdin.

    This gateway is intended for local debugging only.  In CI and experiment
    runs use ``LLMSimulatedGateway`` instead.

    Idempotency contract: same ``(run_id, request_id)`` returns cached
    response without re-prompting, honouring the HumanGateway Protocol.

    Note on cancellation:
        ``asyncio.to_thread(input, ...)`` spawns a thread-pool task.  If the
        outer coroutine is cancelled while the user has not yet pressed Enter,
        the thread-pool task **cannot** be interrupted (the blocking ``input``
        call in the thread will finish regardless).  This is a known limitation
        of the CLI gateway and is acceptable because it is only used for
        interactive debugging.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[UUID, str], HumanResponse] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # HumanGateway Protocol
    # ------------------------------------------------------------------

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        """Render context to stdout and read a decision from stdin.

        Parameters
        ----------
        ctx:
            The HumanContext describing the decision point.
        request_id:
            Stable caller-assigned identifier; used for idempotency.

        Returns
        -------
        HumanResponse
            With ``source='human'``, ``timed_out=False``.
        """
        cache_key = (ctx.run_id, request_id)

        async with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Render to stdout
        self._render(ctx, request_id)

        prompt_str = "Your response: "
        raw: str = await asyncio.to_thread(input, prompt_str)
        raw = raw.strip()

        response = self._parse(raw, ctx)

        async with self._lock:
            # Double-checked: another coroutine might have filled the cache
            # while we were waiting for input.
            if cache_key not in self._cache:
                self._cache[cache_key] = response

        return self._cache[cache_key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render(self, ctx: HumanContext, request_id: str) -> None:
        """Print a formatted decision prompt to stdout."""
        lines: list[str] = [
            "",
            _SEPARATOR,
            f"  HUMAN REVIEW REQUEST  (request_id={request_id})",
            _SEPARATOR,
            f"  Run   : {ctx.run_id}",
            f"  Role  : {ctx.role.value}",
            "",
            "  Question:",
        ]
        for line in textwrap.wrap(ctx.question, width=56):
            lines.append(f"    {line}")
        lines.append("")

        if ctx.recent_messages:
            lines.append("  Recent messages:")
            for msg in ctx.recent_messages:
                lines.append(f"    [{msg.kind.value}] {msg.sender}: {msg.content}")
            lines.append("")

        actions_str = " | ".join(ctx.allowed_actions)
        lines.append(f"  Allowed actions: {actions_str}")
        lines.append("")
        lines.append("  Enter: <action> [optional comment]")
        lines.append('  Or JSON: {"action": "...", "comment": "...", "payload": {...}}')
        lines.append(_SEPARATOR)

        print("\n".join(lines))

    def _parse(self, raw: str, ctx: HumanContext) -> HumanResponse:
        """Parse a raw input line into a HumanResponse.

        Accepted formats:
        1. ``<action>``
        2. ``<action> <comment>``
        3. JSON object with ``action`` key (and optional ``comment``, ``payload``)

        Invalid / unknown action → ``allowed_actions[0]`` with a comment
        that records the invalid input verbatim.
        """
        action: str
        comment: str | None = None
        payload: dict[str, Any] = {}

        # Try JSON first
        if raw.startswith("{"):
            try:
                data = json.loads(raw)
                action = str(data.get("action", "")).strip()
                comment = data.get("comment") or None
                if isinstance(comment, str):
                    comment = comment.strip() or None
                payload = data.get("payload") or {}
                if not isinstance(payload, dict):
                    payload = {}
            except json.JSONDecodeError:
                action = ""
        else:
            parts = raw.split(maxsplit=1)
            action = parts[0] if parts else ""
            comment = parts[1].strip() if len(parts) > 1 else None

        # Validate action
        if action not in ctx.allowed_actions:
            fallback_action = ctx.allowed_actions[0] if ctx.allowed_actions else "abstain"
            comment = f"invalid_action_{raw!r}"
            action = fallback_action
            payload = {}

        return HumanResponse(
            action=action,
            comment=comment,
            payload=payload,
            timed_out=False,
            source="human",
        )
