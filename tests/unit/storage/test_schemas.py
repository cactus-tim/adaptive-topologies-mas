"""Unit tests for atm.storage.schemas — 6 PyArrow parquet schemas.

Tests cover:
1. All 6 schemas are importable
2. All timestamp columns use UTC timezone (pa.timestamp("us", tz="UTC"))
3. Field names match expected lists (spot-checks)
"""

from __future__ import annotations

import pytest
import pyarrow as pa

# ---------------------------------------------------------------------------
# 1. Import test
# ---------------------------------------------------------------------------


def test_import_all_schemas() -> None:
    """All 6 schemas and _TS_UTC alias must be importable."""
    from atm.storage.schemas import (  # noqa: F401
        LLM_CALL_SCHEMA,
        MESSAGE_SCHEMA,
        TOOL_CALL_SCHEMA,
        PHASE_SCHEMA,
        TOPOLOGY_TRANSITION_SCHEMA,
        SCRATCHPAD_SCHEMA,
        _TS_UTC,
    )


# ---------------------------------------------------------------------------
# 2. UTC timezone on timestamp columns
# ---------------------------------------------------------------------------


def test_llm_call_schema_at_tz_utc() -> None:
    """LLM_CALL_SCHEMA 'at' field must be timestamp(us, UTC)."""
    from atm.storage.schemas import LLM_CALL_SCHEMA

    field = LLM_CALL_SCHEMA.field("at")
    assert pa.types.is_timestamp(field.type)
    assert field.type.tz == "UTC"


def test_message_schema_at_tz_utc() -> None:
    """MESSAGE_SCHEMA 'at' field must be timestamp(us, UTC)."""
    from atm.storage.schemas import MESSAGE_SCHEMA

    field = MESSAGE_SCHEMA.field("at")
    assert pa.types.is_timestamp(field.type)
    assert field.type.tz == "UTC"


def test_tool_call_schema_at_tz_utc() -> None:
    """TOOL_CALL_SCHEMA 'at' field must be timestamp(us, UTC)."""
    from atm.storage.schemas import TOOL_CALL_SCHEMA

    field = TOOL_CALL_SCHEMA.field("at")
    assert pa.types.is_timestamp(field.type)
    assert field.type.tz == "UTC"


def test_phase_schema_timestamps_tz_utc() -> None:
    """PHASE_SCHEMA 'started_at' and 'ended_at' fields must be timestamp(us, UTC)."""
    from atm.storage.schemas import PHASE_SCHEMA

    for col in ("started_at", "ended_at"):
        field = PHASE_SCHEMA.field(col)
        assert pa.types.is_timestamp(field.type), f"{col} not a timestamp"
        assert field.type.tz == "UTC", f"{col} missing UTC tz"


def test_topology_transition_schema_at_tz_utc() -> None:
    """TOPOLOGY_TRANSITION_SCHEMA 'at' field must be timestamp(us, UTC)."""
    from atm.storage.schemas import TOPOLOGY_TRANSITION_SCHEMA

    field = TOPOLOGY_TRANSITION_SCHEMA.field("at")
    assert pa.types.is_timestamp(field.type)
    assert field.type.tz == "UTC"


def test_scratchpad_schema_at_tz_utc() -> None:
    """SCRATCHPAD_SCHEMA 'at' field must be timestamp(us, UTC)."""
    from atm.storage.schemas import SCRATCHPAD_SCHEMA

    field = SCRATCHPAD_SCHEMA.field("at")
    assert pa.types.is_timestamp(field.type)
    assert field.type.tz == "UTC"


# ---------------------------------------------------------------------------
# 3. Field names spot-check
# ---------------------------------------------------------------------------


def test_schemas_field_names_match_expected() -> None:
    """Spot-check that each schema has the expected set of field names."""
    from atm.storage.schemas import (
        LLM_CALL_SCHEMA,
        MESSAGE_SCHEMA,
        TOOL_CALL_SCHEMA,
        PHASE_SCHEMA,
        TOPOLOGY_TRANSITION_SCHEMA,
        SCRATCHPAD_SCHEMA,
    )

    llm_names = set(LLM_CALL_SCHEMA.names)
    assert llm_names == {
        "run_id", "agent_id", "model", "at", "input_tokens", "output_tokens",
        "cache_hit_tokens", "cost_usd", "latency_ms", "cache_scope", "fingerprint",
    }, f"LLM_CALL_SCHEMA field names mismatch: {llm_names}"

    msg_names = set(MESSAGE_SCHEMA.names)
    assert msg_names == {
        "run_id", "message_id", "at", "from_agent", "to_agent", "mtype", "payload_json",
    }, f"MESSAGE_SCHEMA field names mismatch: {msg_names}"

    tool_names = set(TOOL_CALL_SCHEMA.names)
    assert tool_names == {
        "run_id", "agent_id", "tool_name", "at", "latency_ms", "ok", "args_json",
        "result_json", "error",
    }, f"TOOL_CALL_SCHEMA field names mismatch: {tool_names}"

    phase_names = set(PHASE_SCHEMA.names)
    assert phase_names == {
        "run_id", "phase_name", "from_phase", "started_at", "ended_at",
        "entry_reason", "topology_used", "decided_by",
    }, f"PHASE_SCHEMA field names mismatch: {phase_names}"

    topo_names = set(TOPOLOGY_TRANSITION_SCHEMA.names)
    assert topo_names == {
        "run_id", "at_iter", "at", "from_topology", "to_topology",
        "decided_by", "considered_alternatives_json", "rationale", "cost_usd", "guarded",
    }, f"TOPOLOGY_TRANSITION_SCHEMA field names mismatch: {topo_names}"

    scratch_names = set(SCRATCHPAD_SCHEMA.names)
    assert scratch_names == {
        "run_id", "agent_id", "step_idx", "at", "role", "content", "tool_calls_json",
    }, f"SCRATCHPAD_SCHEMA field names mismatch: {scratch_names}"


# ---------------------------------------------------------------------------
# 4. _TS_UTC alias is correct type
# ---------------------------------------------------------------------------


def test_ts_utc_alias_type() -> None:
    """_TS_UTC must equal pa.timestamp('us', tz='UTC')."""
    from atm.storage.schemas import _TS_UTC

    expected = pa.timestamp("us", tz="UTC")
    assert _TS_UTC == expected


# ---------------------------------------------------------------------------
# 5. Schema field types spot check
# ---------------------------------------------------------------------------


def test_llm_call_schema_numeric_types() -> None:
    """LLM_CALL_SCHEMA numeric fields: input_tokens/output_tokens/cache_hit_tokens=int32, cost_usd/latency_ms=float64."""
    from atm.storage.schemas import LLM_CALL_SCHEMA

    assert LLM_CALL_SCHEMA.field("input_tokens").type == pa.int32()
    assert LLM_CALL_SCHEMA.field("output_tokens").type == pa.int32()
    assert LLM_CALL_SCHEMA.field("cache_hit_tokens").type == pa.int32()
    assert LLM_CALL_SCHEMA.field("cost_usd").type == pa.float64()
    assert LLM_CALL_SCHEMA.field("latency_ms").type == pa.float64()


def test_topology_transition_schema_types() -> None:
    """TOPOLOGY_TRANSITION_SCHEMA: at_iter=int32, cost_usd=float64, guarded=bool."""
    from atm.storage.schemas import TOPOLOGY_TRANSITION_SCHEMA

    assert TOPOLOGY_TRANSITION_SCHEMA.field("at_iter").type == pa.int32()
    assert TOPOLOGY_TRANSITION_SCHEMA.field("cost_usd").type == pa.float64()
    assert TOPOLOGY_TRANSITION_SCHEMA.field("guarded").type == pa.bool_()


def test_scratchpad_schema_step_idx_type() -> None:
    """SCRATCHPAD_SCHEMA: step_idx must be int32."""
    from atm.storage.schemas import SCRATCHPAD_SCHEMA

    assert SCRATCHPAD_SCHEMA.field("step_idx").type == pa.int32()


def test_tool_call_schema_ok_bool() -> None:
    """TOOL_CALL_SCHEMA: ok must be bool."""
    from atm.storage.schemas import TOOL_CALL_SCHEMA

    assert TOOL_CALL_SCHEMA.field("ok").type == pa.bool_()
