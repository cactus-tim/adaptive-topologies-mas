"""Pure serializer functions: domain objects → dict rows matching PyArrow schemas.

Each function converts a pydantic/domain object into a flat dict whose keys exactly
match the field names of the corresponding schema in atm.storage.schemas.

Design rules:
- All functions are pure: no I/O, no side effects.
- datetime fields remain tz-aware (tzinfo is NOT stripped).
- UUIDs are converted via str(uuid_value).
- JSON fields use _dumps (deterministic: sort_keys=True, compact, unicode-safe,
  default=str for non-serialisable types such as UUID/datetime).
- For each serializer f: set(f(...).keys()) == set(SCHEMA.names).

Missing domain types (ScratchpadEntry, PhaseTransition.topology_used, etc.) are
defined here as minimal dataclasses/Protocols until they are promoted to
core/types.py in a later milestone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from atm.core.types import (
    LLMResponse,
    Message,
    PhaseTransition,
    ToolCall,
    ToolResult,
    TopologyTransition,
)

_JSON_KW: dict[str, Any] = {
    "default": str,
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
}


def _dumps(obj: object) -> str:
    """Serialise *obj* to a compact, deterministic JSON string.

    - ``sort_keys=True`` → deterministic field order for dicts.
    - ``separators=(",", ":")`` → no extra whitespace.
    - ``ensure_ascii=False`` → Cyrillic and other Unicode preserved literally.
    - ``default=str`` → UUIDs, datetimes, and other non-serialisable objects
      fall back to their str() representation.
    """
    return json.dumps(obj, **_JSON_KW)


@dataclass(frozen=True)
class ScratchpadEntry:
    """Minimal scratchpad entry type.

    The full type will be defined in core/types.py in a later milestone.
    Fields match SCRATCHPAD_SCHEMA column names.
    """

    at: datetime
    role: str
    content: str
    tool_calls: list[Any] = field(default_factory=list)


def llm_response_to_row(
    run_id: UUID,
    agent_id: str,
    model: str,
    at: datetime,
    response: LLMResponse,
) -> dict[str, Any]:
    """Convert an LLMResponse to a dict row matching LLM_CALL_SCHEMA.

    Schema fields: run_id, agent_id, model, at, input_tokens, output_tokens,
    cache_hit_tokens, cost_usd, latency_ms, cache_scope, fingerprint.

    Notes:
    - ``at`` is the wall-clock timestamp at invocation start (tz-aware).
    - ``fingerprint`` encodes the call identity (model + start time) for replay.
    - ``cache_scope`` defaults to "none" — set by the caller when known.
    - ``latency_ms`` cast to float to match pa.float64() schema field.
    """
    return {
        "run_id": str(run_id),
        "agent_id": agent_id,
        "model": model,
        "at": at,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cache_hit_tokens": response.usage.cached_input_tokens,
        "cost_usd": response.cost_usd,
        "latency_ms": float(response.latency_ms),
        "cache_scope": "none",
        "fingerprint": str(response.id),
    }


def message_to_row(
    run_id: UUID,
    msg: Message,
) -> dict[str, Any]:
    """Convert a Message to a dict row matching MESSAGE_SCHEMA.

    Schema fields: run_id, message_id, at, from_agent, to_agent, mtype,
    payload_json.

    Notes:
    - ``to_agent`` is set to the first recipient if any, else empty string
      (broadcast messages have no named recipient).
    - ``mtype`` maps from MessageKind string value.
    - ``at`` preserves tz-aware tzinfo.
    """
    to_agent = msg.recipients[0] if msg.recipients else ""
    return {
        "run_id": str(run_id),
        "message_id": str(msg.id),
        "at": msg.created_at,
        "from_agent": msg.sender,
        "to_agent": to_agent,
        "mtype": str(msg.kind),
        "payload_json": _dumps(msg.payload),
    }


def tool_call_to_row(
    run_id: UUID,
    agent_id: str,
    call: ToolCall,
    result: ToolResult,
) -> dict[str, Any]:
    """Convert a ToolCall + ToolResult pair to a dict row matching TOOL_CALL_SCHEMA.

    Schema fields: run_id, agent_id, tool_name, at, latency_ms, ok, args_json,
    result_json, error.

    Notes:
    - ``at`` is the tool invocation timestamp (call.issued_at, tz-aware).
    - ``latency_ms`` cast to float to match pa.float64() schema field.
    - ``result_json`` serialises result.output via _dumps.
    - ``error`` is "" (empty string) when result.error is None.
    """
    return {
        "run_id": str(run_id),
        "agent_id": agent_id,
        "tool_name": call.tool_name,
        "at": call.issued_at,
        "latency_ms": float(result.latency_ms),
        "ok": result.ok,
        "args_json": _dumps(call.args),
        "result_json": _dumps(result.output),
        "error": result.error if result.error is not None else "",
    }


def phase_transition_to_row(
    run_id: UUID,
    transition: PhaseTransition,
    *,
    topology_used: str = "",
    ended_at: datetime | None = None,
) -> dict[str, Any]:
    """Convert a PhaseTransition to a dict row matching PHASE_SCHEMA.

    Schema fields: run_id, phase_name, from_phase, started_at, ended_at,
    entry_reason, topology_used, decided_by.

    Notes:
    - ``phase_name`` maps from PhaseTransition.to_phase.
    - ``started_at`` maps from PhaseTransition.at (tz-aware).
    - ``ended_at`` may be None (phase still open); pyarrow timestamp is nullable.
    - ``topology_used`` is not on the domain model; caller must supply it.
    - ``from_phase`` is "" when transition.from_phase is None (initial transition).
    """
    return {
        "run_id": str(run_id),
        "phase_name": str(transition.to_phase),
        "from_phase": str(transition.from_phase) if transition.from_phase is not None else "",
        "started_at": transition.at,
        "ended_at": ended_at,
        "entry_reason": transition.entry_reason,
        "topology_used": topology_used,
        "decided_by": str(transition.decided_by),
    }


def topology_transition_to_row(
    run_id: UUID,
    transition: TopologyTransition,
) -> dict[str, Any]:
    """Convert a TopologyTransition to a dict row matching TOPOLOGY_TRANSITION_SCHEMA.

    Schema fields:
    run_id, from_topology, to_topology, phase_at_decision, iter_within_phase,
    iter_within_topology, decided_by, reason, considered_alternatives_json,
    guards_applied_json, signals_snapshot_json, router_cost_usd, at.

    Notes:
    - ``from_topology`` is "" when transition.from_topology is None (initial decision).
    - Tuple fields (considered_alternatives, guards_applied) are serialised to
      JSON arrays via _dumps for schema-stable storage.
    - ``signals_snapshot`` dict is serialised to compact JSON via _dumps.
    - ``at`` is tz-aware (UTC).
    """
    return {
        "run_id": str(run_id),
        "from_topology": transition.from_topology if transition.from_topology is not None else "",
        "to_topology": transition.to_topology,
        "phase_at_decision": str(transition.phase_at_decision),
        "iter_within_phase": transition.iter_within_phase,
        "iter_within_topology": transition.iter_within_topology,
        "decided_by": str(transition.decided_by),
        "reason": transition.reason,
        "considered_alternatives_json": _dumps(list(transition.considered_alternatives)),
        "guards_applied_json": _dumps(list(transition.guards_applied)),
        "signals_snapshot_json": _dumps(transition.signals_snapshot),
        "router_cost_usd": transition.router_cost_usd,
        "at": transition.at,
    }


def scratchpad_entry_to_row(
    run_id: UUID,
    agent_id: str,
    step_idx: int,
    entry: ScratchpadEntry,
) -> dict[str, Any]:
    """Convert a ScratchpadEntry to a dict row matching SCRATCHPAD_SCHEMA.

    Schema fields: run_id, agent_id, step_idx, at, role, content,
    tool_calls_json.

    Notes:
    - ``at`` is tz-aware (must be supplied as a tz-aware datetime on the entry).
    - ``tool_calls_json`` serialises entry.tool_calls list via _dumps.
    """
    return {
        "run_id": str(run_id),
        "agent_id": agent_id,
        "step_idx": step_idx,
        "at": entry.at,
        "role": entry.role,
        "content": entry.content,
        "tool_calls_json": _dumps(entry.tool_calls),
    }
