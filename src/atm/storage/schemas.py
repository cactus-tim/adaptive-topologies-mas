"""PyArrow schemas for Parquet storage streams (6 schemas)."""

from __future__ import annotations

import pyarrow as pa

_TS_UTC: pa.DataType = pa.timestamp("us", tz="UTC")


LLM_CALL_SCHEMA: pa.Schema = pa.schema(
    [
        ("run_id", pa.string()),
        ("agent_id", pa.string()),
        ("model", pa.string()),
        ("at", _TS_UTC),
        ("input_tokens", pa.int32()),
        ("output_tokens", pa.int32()),
        ("cache_hit_tokens", pa.int32()),
        ("cost_usd", pa.float64()),
        ("latency_ms", pa.float64()),
        ("cache_scope", pa.string()),
        ("fingerprint", pa.string()),
    ]
)

MESSAGE_SCHEMA: pa.Schema = pa.schema(
    [
        ("run_id", pa.string()),
        ("message_id", pa.string()),
        ("at", _TS_UTC),
        ("from_agent", pa.string()),
        ("to_agent", pa.string()),
        ("mtype", pa.string()),
        ("payload_json", pa.string()),
    ]
)

TOOL_CALL_SCHEMA: pa.Schema = pa.schema(
    [
        ("run_id", pa.string()),
        ("agent_id", pa.string()),
        ("tool_name", pa.string()),
        ("at", _TS_UTC),
        ("latency_ms", pa.float64()),
        ("ok", pa.bool_()),
        ("args_json", pa.string()),
        ("result_json", pa.string()),
        ("error", pa.string()),
    ]
)

PHASE_SCHEMA: pa.Schema = pa.schema(
    [
        ("run_id", pa.string()),
        ("phase_name", pa.string()),
        ("from_phase", pa.string()),
        ("started_at", _TS_UTC),
        ("ended_at", _TS_UTC),
        ("entry_reason", pa.string()),
        ("topology_used", pa.string()),
        ("decided_by", pa.string()),
    ]
)

TOPOLOGY_TRANSITION_SCHEMA: pa.Schema = pa.schema(
    [
        ("run_id", pa.string()),
        ("from_topology", pa.string()),
        ("to_topology", pa.string()),
        ("phase_at_decision", pa.string()),
        ("iter_within_phase", pa.int32()),
        ("iter_within_topology", pa.int32()),
        ("decided_by", pa.string()),
        ("reason", pa.string()),
        ("considered_alternatives_json", pa.string()),
        ("guards_applied_json", pa.string()),
        ("signals_snapshot_json", pa.string()),
        ("router_cost_usd", pa.float64()),
        ("at", _TS_UTC),
    ]
)

SCRATCHPAD_SCHEMA: pa.Schema = pa.schema(
    [
        ("run_id", pa.string()),
        ("agent_id", pa.string()),
        ("step_idx", pa.int32()),
        ("at", _TS_UTC),
        ("role", pa.string()),
        ("content", pa.string()),
        ("tool_calls_json", pa.string()),
    ]
)
