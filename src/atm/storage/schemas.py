"""PyArrow schemas for Parquet storage streams.

MINOR CORRECTION vs arch.md §3.5: all timestamp columns use pa.timestamp('us', tz='UTC')
instead of naive pa.timestamp('us'). Preserves tzinfo round-trip; required for
pytest filterwarnings=['error'] compatibility with pydantic timezone-aware datetimes.

RECONCILIATION NOTE (step 4.1): TOPOLOGY_TRANSITION_SCHEMA field names updated from
draft names (at_iter, rationale, cost_usd, guarded) to arch.md §3.4 canonical names
(iter_within_phase, iter_within_topology, reason, router_cost_usd) matching the
TopologyTransition pydantic model in core/types.py. Lists serialized to JSON strings:
considered_alternatives_json, guards_applied_json, signals_snapshot_json.

Six schemas are defined:
- LLM_CALL_SCHEMA     — per-call LLM invocation data
- MESSAGE_SCHEMA      — inter-agent messages
- TOOL_CALL_SCHEMA    — tool invocation records
- PHASE_SCHEMA        — phase lifecycle records
- TOPOLOGY_TRANSITION_SCHEMA — topology switch events (arch.md §3.4 canonical fields)
- SCRATCHPAD_SCHEMA   — per-agent scratchpad entries
"""

from __future__ import annotations

import pyarrow as pa

# Canonical UTC-aware timestamp type used across all schemas.
# Using tz="UTC" instead of naive timestamp to:
#   1. Preserve tzinfo on round-trip (Python datetime.tzinfo not None)
#   2. Remain compatible with pytest filterwarnings=["error"] when combined
#      with pydantic v2 timezone-aware datetime fields.
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
        # Reconciled to arch.md §3.4 / TopologyTransition pydantic model field names.
        # Previous draft used at_iter/rationale/cost_usd/guarded — replaced in step 4.1
        # with the canonical names from the SQLAlchemy model and pydantic domain type.
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
