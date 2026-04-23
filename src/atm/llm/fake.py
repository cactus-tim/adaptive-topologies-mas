"""FakeLLM — deterministic LLM stub for testing.

Three modes:
  scripted — plays back pre-recorded fixture YAML entries keyed by agent_id + step_idx.
  echo     — returns the content of the last Message in the input list.
  replay   — returns rows from a pyarrow Table that mirrors the llm_calls Parquet schema.

REPLAY_SCHEMA is exported so M3 and integration tests can build compatible tables without
duplicating the column definitions.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import pyarrow as pa
import yaml

from atm.core.types import LLMResponse, Message, MessageKind, TokenUsage, ToolCall

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# ---------------------------------------------------------------------------
# REPLAY_SCHEMA: mirrors arch.md §3.5 llm_calls Parquet schema.
# NOTE: call_id column maps to LLMResponse.id — M3 DB column convention uses call_id.
# ---------------------------------------------------------------------------

REPLAY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("call_id", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=True),
        pa.field("usage_input", pa.int64(), nullable=False),
        pa.field("usage_output", pa.int64(), nullable=False),
        pa.field("usage_total", pa.int64(), nullable=False),
        pa.field("usage_cached", pa.int64(), nullable=False),
        pa.field("cost_usd", pa.float64(), nullable=False),
        pa.field("latency_ms", pa.int64(), nullable=False),
        pa.field("finish_reason", pa.string(), nullable=False),
        pa.field("started_at", pa.string(), nullable=False),  # ISO 8601 string
        pa.field("tool_calls_json", pa.string(), nullable=True),  # JSON array string
    ]
)


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


class FakeLLM:
    """Deterministic LLM stub for tests.

    Modes
    -----
    scripted:
        Plays back entries from a YAML fixture file. Keyed by (agent_id, step_idx).
        Falls back to (role, step_idx) if agent_id is absent from an entry.
        Step index is per agent_id and increments atomically under an asyncio.Lock.

    echo:
        Returns the ``content`` of the last Message whose kind is REQUEST or BROADCAST
        (i.e. user-role messages). If no such message exists, returns the content of the
        last message regardless of kind. If the input is empty, returns an empty string.

    replay:
        Reads rows sequentially from a pyarrow Table built with REPLAY_SCHEMA.
        Each call advances an internal row counter and returns the corresponding row.
        The ``call_id`` column maps to ``LLMResponse.id``.

    Invariant: ``latency_ms=0`` is set on every response so that Pydantic ``__eq__``
    works correctly in determinism tests without needing field exclusion.
    """

    def __init__(
        self,
        *,
        mode: Literal["scripted", "replay", "echo"],
        fixture: str | Path | None = None,
        replay_table: pa.Table | None = None,
    ) -> None:
        self._mode = mode
        self._lock: asyncio.Lock = asyncio.Lock()

        # scripted mode state
        self._entries: dict[str, list[dict[str, Any]]] = {}  # agent_id -> ordered entries
        self._step: dict[str, int] = {}  # agent_id -> next step_idx

        # replay mode state
        self._replay_table: pa.Table | None = replay_table
        self._replay_row: int = 0

        if mode == "scripted":
            if fixture is None:
                raise ValueError("FakeLLM(mode='scripted') requires 'fixture' path")
            self._load_fixture(Path(fixture))

        if mode == "replay" and replay_table is None:
            raise ValueError("FakeLLM(mode='replay') requires 'replay_table'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fixture(self, path: Path) -> None:
        """Parse YAML fixture and index entries by agent_id."""
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        for entry in data.get("entries", []):
            agent_id: str = entry.get("agent_id", entry.get("role", "default"))
            self._entries.setdefault(agent_id, []).append(entry)
            # Also index by role as a fallback key if different from agent_id
            role: str | None = entry.get("role")
            if role and role != agent_id:
                self._entries.setdefault(role, []).append(entry)

    def _entry_for(self, agent_id: str, step_idx: int) -> dict[str, Any]:
        """Retrieve the fixture entry for (agent_id, step_idx)."""
        entries = self._entries.get(agent_id)
        if entries is None:
            raise LookupError(
                f"FakeLLM: no fixture entries for agent_id={agent_id!r}. "
                f"Available keys: {list(self._entries)}"
            )
        if step_idx >= len(entries):
            raise LookupError(
                f"FakeLLM: agent_id={agent_id!r} has {len(entries)} entries but "
                f"step_idx={step_idx} was requested"
            )
        return entries[step_idx]

    @staticmethod
    def _build_tool_calls(raw: list[dict[str, Any]], issued_by: str) -> tuple[ToolCall, ...]:
        """Convert raw fixture tool_call dicts to ToolCall domain objects."""
        result: list[ToolCall] = []
        for item in raw:
            result.append(
                ToolCall(
                    tool_name=item["name"],
                    args=item.get("args", {}),
                    issued_by=issued_by,
                )
            )
        return tuple(result)

    def _response_from_entry(
        self, entry: dict[str, Any], agent_id: str
    ) -> LLMResponse:
        """Build an LLMResponse from a scripted fixture entry."""
        usage_raw: dict[str, Any] = entry.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("input_tokens", 0),
            completion_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            cached_input_tokens=usage_raw.get("cached_tokens", 0),
        )
        raw_tool_calls: list[dict[str, Any]] = entry.get("tool_calls") or []
        tool_calls = self._build_tool_calls(raw_tool_calls, issued_by=agent_id)
        return LLMResponse(
            model=entry.get("model", "fake:deterministic"),
            text=entry.get("content") or None,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=0.0,
            latency_ms=0,
            finish_reason=entry.get("finish_reason", "stop"),
        )

    def _response_from_replay_row(self, row_idx: int) -> LLMResponse:
        """Build an LLMResponse from a replay table row."""
        if self._replay_table is None:
            raise RuntimeError("FakeLLM: replay_table is None")

        table = self._replay_table
        call_id_str: str = table.column("call_id")[row_idx].as_py()
        model: str = table.column("model")[row_idx].as_py()
        content_val = table.column("content")[row_idx].as_py()
        usage_input: int = table.column("usage_input")[row_idx].as_py()
        usage_output: int = table.column("usage_output")[row_idx].as_py()
        usage_total: int = table.column("usage_total")[row_idx].as_py()
        usage_cached: int = table.column("usage_cached")[row_idx].as_py()
        cost_usd: float = table.column("cost_usd")[row_idx].as_py()
        finish_reason_val: str = table.column("finish_reason")[row_idx].as_py()
        tool_calls_json_val = table.column("tool_calls_json")[row_idx].as_py()

        tool_calls: tuple[ToolCall, ...] = ()
        if tool_calls_json_val:
            raw_list: list[dict[str, Any]] = json.loads(tool_calls_json_val)
            tool_calls = self._build_tool_calls(raw_list, issued_by="replay")

        usage = TokenUsage(
            prompt_tokens=usage_input,
            completion_tokens=usage_output,
            total_tokens=usage_total,
            cached_input_tokens=usage_cached,
        )

        return LLMResponse(
            id=UUID(call_id_str),
            model=model,
            text=content_val or None,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost_usd,
            latency_ms=0,
            finish_reason=finish_reason_val,  # type: ignore[arg-type]
        )

    @staticmethod
    def _echo_content(
        messages: list[Message] | list[dict[str, Any]] | list[BaseMessage],
    ) -> str:
        """Return the content of the last user-kind message, or last message overall."""
        if not messages:
            return ""
        # Prefer last REQUEST or BROADCAST Message
        for msg in reversed(messages):
            if isinstance(msg, Message):
                if msg.kind in (MessageKind.REQUEST, MessageKind.BROADCAST):
                    return msg.content
            elif isinstance(msg, dict):
                kind = msg.get("kind", "")
                if kind in ("request", "broadcast"):
                    return str(msg.get("content", ""))
        # Fallback: last message regardless of kind
        last = messages[-1]
        if isinstance(last, Message):
            return last.content
        if isinstance(last, dict):
            return str(last.get("content", ""))
        # BaseMessage — both HumanMessage, AIMessage etc. have .content attribute
        return str(getattr(last, "content", ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        messages: list[Message] | list[dict[str, Any]] | list[BaseMessage],
        *,
        agent_id: str = "default",
        **opts: Any,
    ) -> LLMResponse:
        """Return the next scripted / echo / replay response."""
        if self._mode == "scripted":
            async with self._lock:
                step_idx = self._step.get(agent_id, 0)
                self._step[agent_id] = step_idx + 1

            entry = self._entry_for(agent_id, step_idx)
            return self._response_from_entry(entry, agent_id)

        if self._mode == "echo":
            content = self._echo_content(messages)
            return LLMResponse(
                model="fake:echo",
                text=content if content else None,
                usage=TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
                cost_usd=0.0,
                latency_ms=0,
                finish_reason="stop",
            )

        if self._mode == "replay":
            async with self._lock:
                row_idx = self._replay_row
                self._replay_row += 1

            return self._response_from_replay_row(row_idx)

        raise RuntimeError(f"FakeLLM: unknown mode {self._mode!r}")  # pragma: no cover

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Streaming is not supported by FakeLLM."""
        raise NotImplementedError("FakeLLM does not support streaming")
        yield  # make this an async generator
