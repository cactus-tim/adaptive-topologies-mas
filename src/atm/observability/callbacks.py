"""ExperimentCallbackHandler — LangChain AsyncCallbackHandler for ATM experiment observability.

Writes all LLM calls, tool calls, messages, phase/topology transitions to Parquet streams
and updates PostgreSQL budget counters atomically.

4 flush invariants (arch.md §10.3, §17/#2):
(a) on_chain_end of root chain → parquet_writer.close() (flush + close all streams)
(b) phase_transition / topology_transition custom events → parquet_writer.flush()
    STRICTLY BEFORE PG INSERT of the transition row
(c) on_chain_error of root chain → parquet_writer.close()
(d) buffer overflow auto-flush — implemented at ParquetWriter layer (Step 2.3)
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atm.core.types import Message, PhaseTransition
from atm.core.types import TopologyTransition as TopologyTransitionDomain
from atm.observability.serializers import (
    _dumps,
    message_to_row,
    phase_transition_to_row,
    topology_transition_to_row,
)
from atm.storage.models import BudgetEvent, Experiment, HumanInteraction, Run
from atm.storage.models import Phase as PhaseModel
from atm.storage.models import TopologyTransition as TTModel
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import session_scope

# Constraint name for idempotent INSERT in human_interactions (migration 0002)
CONSTRAINT_NAME = "uq_human_interactions_run_request"


def _now_utc() -> datetime:
    """Return current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class ExperimentCallbackHandler(AsyncCallbackHandler):
    """AsyncCallbackHandler that persists experiment observability data to Parquet and PostgreSQL.

    Constructor parameters:
        run_id: UUID of the current run.
        exp_id: UUID of the current experiment.
        session_factory: SQLAlchemy async_sessionmaker for PostgreSQL sessions.
        parquet_writer: ParquetWriter instance for buffered Parquet writes.
        budget_warn_threshold: If set, emit a 'warn' BudgetEvent when run total reaches this value.
        budget_exceed_threshold: If set, emit an 'exceed' BudgetEvent when run total reaches this value.
        logger: Optional structlog BoundLogger; falls back to "atm.observability".
    """

    def __init__(
        self,
        run_id: UUID,
        exp_id: UUID,
        session_factory: async_sessionmaker[AsyncSession],
        parquet_writer: ParquetWriter,
        *,
        budget_warn_threshold: Decimal | None = None,
        budget_exceed_threshold: Decimal | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        super().__init__()
        self._run_id = run_id
        self._exp_id = exp_id
        self._session_factory = session_factory
        self._parquet_writer = parquet_writer
        self._budget_warn_threshold = budget_warn_threshold
        self._budget_exceed_threshold = budget_exceed_threshold
        self._log = (logger or structlog.get_logger("atm.observability")).bind(
            run_id=str(run_id), exp_id=str(exp_id)
        )
        self._root_run_id: UUID | None = None  # set on first root on_chain_start
        self._warn_emitted: bool = False
        self._exceed_emitted: bool = False
        # Maps tool run_id → start info dict for latency calculation and metadata
        self._tool_starts: dict[UUID, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Chain hooks
    # ------------------------------------------------------------------

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Detect the root run on first call."""
        if self._root_run_id is None and parent_run_id is None:
            # Prefer explicit metadata flag; fall back to first-seen
            if metadata is not None and metadata.get("is_root_run") is True:
                self._root_run_id = run_id
            else:
                # Fallback: set root on first top-level chain start
                self._root_run_id = run_id

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Invariant (a): flush + close parquet on root chain end."""
        if run_id != self._root_run_id:
            return
        try:
            await self._parquet_writer.close()
        except Exception:
            self._log.critical("on_chain_end: parquet close failed", exc_info=True)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Invariant (c): flush + close parquet on root chain error."""
        if run_id != self._root_run_id:
            return
        try:
            await self._parquet_writer.close()
        except Exception:
            self._log.critical("on_chain_error: parquet close failed", exc_info=True)

    # ------------------------------------------------------------------
    # LLM hook
    # ------------------------------------------------------------------

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Write LLM call row to Parquet and atomically update budget in PG (arch.md §4.2, §18/#4)."""
        try:
            llm_output: dict[str, Any] = response.llm_output or {}
            cost_usd: float = float(llm_output.get("cost_usd", 0.0))

            agent_id: str = str(llm_output.get("agent_id", ""))
            model: str = str(llm_output.get("model", ""))
            input_tokens: int = int(llm_output.get("input_tokens", 0))
            output_tokens: int = int(llm_output.get("output_tokens", 0))
            cache_hit_tokens: int = int(llm_output.get("cache_hit_tokens", 0))
            latency_ms: float = float(llm_output.get("latency_ms", 0))
            cache_scope: str = str(llm_output.get("cache_scope", "none"))
            fingerprint: str = str(llm_output.get("fingerprint", ""))

            row = {
                "run_id": str(self._run_id),
                "agent_id": agent_id,
                "model": model,
                "at": _now_utc(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_hit_tokens": cache_hit_tokens,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "cache_scope": cache_scope,
                "fingerprint": fingerprint,
            }

            await self._parquet_writer.write_llm_call(row)
        except Exception:
            self._log.critical("on_llm_end: parquet write failed", exc_info=True)
            # Still attempt budget update

        # Atomic budget update (arch.md §4.2, §18/#4)
        try:
            cost_delta_decimal = Decimal(
                str(float((response.llm_output or {}).get("cost_usd", 0.0)))
            )
            await self._update_budget(cost_delta_decimal)
        except Exception:
            self._log.critical("on_llm_end: budget update failed", exc_info=True)

    async def _update_budget(self, cost_delta: Decimal) -> None:
        """Atomically update budget_spent_usd on Run and total_cost_usd on Experiment.

        Fires BudgetEvent warn/exceed if thresholds are crossed (once each).
        All writes are in a single session_scope transaction (arch.md §11.3).
        """
        async with session_scope(self._session_factory) as session:
            # Update Run.budget_spent_usd
            result = await session.execute(
                update(Run)
                .where(Run.id == self._run_id)
                .values(budget_spent_usd=Run.budget_spent_usd + cost_delta)
                .returning(Run.budget_spent_usd)
            )
            new_run_total: Decimal = result.scalar_one()

            # Update Experiment.total_cost_usd
            result2 = await session.execute(
                update(Experiment)
                .where(Experiment.id == self._exp_id)
                .values(total_cost_usd=Experiment.total_cost_usd + cost_delta)
                .returning(Experiment.total_cost_usd)
            )
            result2.scalar_one()  # consume result

            now = _now_utc()

            # Budget warn threshold
            if (
                self._budget_warn_threshold is not None
                and not self._warn_emitted
                and new_run_total >= self._budget_warn_threshold
            ):
                self._warn_emitted = True
                session.add(
                    BudgetEvent(
                        run_id=self._run_id,
                        level="run",
                        event="warn",
                        limit_usd=self._budget_warn_threshold,
                        current_usd=new_run_total,
                        at=now,
                    )
                )

            # Budget exceed threshold
            if (
                self._budget_exceed_threshold is not None
                and not self._exceed_emitted
                and new_run_total >= self._budget_exceed_threshold
            ):
                self._exceed_emitted = True
                session.add(
                    BudgetEvent(
                        run_id=self._run_id,
                        level="run",
                        event="exceed",
                        limit_usd=self._budget_exceed_threshold,
                        current_usd=new_run_total,
                        at=now,
                    )
                )

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool start time, tool_name, agent_id and args for latency/row population."""
        self._tool_starts[run_id] = {
            "started_at": time.monotonic(),
            "tool_name": serialized.get("name", "") if serialized else "",
            "agent_id": (metadata or {}).get("agent_id", ""),
            "args_json": _dumps(inputs)
            if inputs is not None
            else (input_str if isinstance(input_str, str) else ""),
            "at": datetime.now(UTC),
        }

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Write tool call row to Parquet with ok=True."""
        try:
            start = self._tool_starts.pop(run_id, None)
            latency_ms = (
                (time.monotonic() - start["started_at"]) * 1000.0 if start is not None else 0.0
            )

            row = {
                "run_id": str(self._run_id),
                "agent_id": start["agent_id"] if start is not None else "",
                "tool_name": start["tool_name"] if start is not None else "",
                "at": start["at"] if start is not None else _now_utc(),
                "latency_ms": latency_ms,
                "ok": True,
                "args_json": start["args_json"] if start is not None else "{}",
                "result_json": _dumps(output),
                "error": "",
            }
            await self._parquet_writer.write_tool_call(row)
        except Exception:
            self._log.critical("on_tool_end: write failed", exc_info=True)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Write tool call row to Parquet with ok=False."""
        try:
            start = self._tool_starts.pop(run_id, None)
            latency_ms = (
                (time.monotonic() - start["started_at"]) * 1000.0 if start is not None else 0.0
            )

            row = {
                "run_id": str(self._run_id),
                "agent_id": start["agent_id"] if start is not None else "",
                "tool_name": start["tool_name"] if start is not None else "",
                "at": start["at"] if start is not None else _now_utc(),
                "latency_ms": latency_ms,
                "ok": False,
                "args_json": start["args_json"] if start is not None else "{}",
                "result_json": "",
                "error": str(error),
            }
            await self._parquet_writer.write_tool_call(row)
        except Exception:
            self._log.critical("on_tool_error: write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Custom event hook
    # ------------------------------------------------------------------

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch on event name: message_emit, phase_transition, topology_transition."""
        if name == "message_emit":
            await self._handle_message_emit(data)
        elif name == "phase_transition":
            await self._handle_phase_transition(data)
        elif name == "topology_transition":
            await self._handle_topology_transition(data)
        elif name == "human_request":
            await self._handle_human_request(data)
        elif name == "human_response":
            await self._handle_human_response(data)

    async def _handle_message_emit(self, data: Any) -> None:
        """Write a Message to the messages Parquet stream."""
        try:
            msg: Message = data
            row = message_to_row(self._run_id, msg)
            await self._parquet_writer.write_message(row)
        except Exception:
            self._log.critical("on_custom_event[message_emit]: write failed", exc_info=True)

    async def _handle_phase_transition(self, data: Any) -> None:
        """Invariant (b): flush Parquet FIRST, then INSERT phase into PG atomically."""
        # INVARIANT (b): flush BEFORE pg insert
        try:
            await self._parquet_writer.flush()
        except Exception:
            self._log.critical(
                "on_custom_event[phase_transition]: parquet flush failed", exc_info=True
            )

        try:
            transition: PhaseTransition = data
            now = _now_utc()

            async with session_scope(self._session_factory) as session:
                # SELECT the most recent open phase for this run
                result = await session.execute(
                    select(PhaseModel.id)
                    .where(PhaseModel.run_id == self._run_id, PhaseModel.ended_at.is_(None))
                    .order_by(PhaseModel.started_at.desc())
                    .limit(1)
                )
                prev_phase_id: UUID | None = result.scalar_one_or_none()

                # If there's a previous open phase, close it
                if prev_phase_id is not None:
                    await session.execute(
                        update(PhaseModel)
                        .where(PhaseModel.id == prev_phase_id)
                        .values(ended_at=now)
                    )

                # Insert new Phase row
                new_phase = PhaseModel(
                    id=uuid.uuid4(),
                    run_id=self._run_id,
                    phase_name=str(transition.to_phase),
                    from_phase=str(transition.from_phase)
                    if transition.from_phase is not None
                    else None,
                    started_at=transition.at,
                    ended_at=None,
                    entry_reason=transition.entry_reason,
                    topology_used="",
                    decided_by=str(transition.decided_by),
                )
                session.add(new_phase)

            # Write to Parquet after PG insert (flush already done above)
            row = phase_transition_to_row(self._run_id, transition)
            await self._parquet_writer.write_phase(row)
        except Exception:
            self._log.critical("on_custom_event[phase_transition]: pg insert failed", exc_info=True)

    async def _handle_topology_transition(self, data: Any) -> None:
        """Invariant (b): flush Parquet FIRST, then INSERT topology_transition into PG."""
        # INVARIANT (b): flush BEFORE pg insert
        try:
            await self._parquet_writer.flush()
        except Exception:
            self._log.critical(
                "on_custom_event[topology_transition]: parquet flush failed", exc_info=True
            )

        try:
            transition: TopologyTransitionDomain = data

            async with session_scope(self._session_factory) as session:
                tt = TTModel(
                    id=uuid.uuid4(),
                    run_id=self._run_id,
                    from_topology=transition.from_topology,
                    to_topology=transition.to_topology,
                    phase_at_decision=str(transition.phase_at_decision),
                    iter_within_phase=transition.iter_within_phase,
                    iter_within_topology=transition.iter_within_topology,
                    decided_by=str(transition.decided_by),
                    reason=transition.reason,
                    considered_alternatives=list(transition.considered_alternatives),
                    guards_applied=list(transition.guards_applied),
                    signals_snapshot=transition.signals_snapshot,
                    router_cost_usd=Decimal(str(transition.router_cost_usd)),
                    at=transition.at,
                )
                session.add(tt)

            # Write to Parquet after PG insert
            row = topology_transition_to_row(self._run_id, transition)
            await self._parquet_writer.write_topology_transition(row)
        except Exception:
            self._log.critical(
                "on_custom_event[topology_transition]: pg insert failed", exc_info=True
            )

    async def _handle_human_request(self, data: Any) -> None:
        """INSERT a new HumanInteraction row (idempotent via ON CONFLICT DO NOTHING).

        ``data`` is expected to be a dict with keys:
            run_id       (UUID)
            request_id   (str, <=64 chars)
            role         (str or HumanRole)
            context_json (dict — HumanContext.model_dump())
            requested_at (datetime, timezone-aware)

        Duplicate ``(run_id, request_id)`` pairs are silently ignored thanks to
        the UNIQUE constraint ``uq_human_interactions_run_request`` (migration 0002)
        and the ``ON CONFLICT DO NOTHING`` clause.
        """
        try:
            fields: dict[str, Any] = {
                "id": uuid.uuid4(),
                "run_id": data["run_id"],
                "request_id": data["request_id"],
                "role": str(data["role"]),
                "context_json": data.get("context_json", {}),
                "requested_at": data.get("requested_at", _now_utc()),
                "response_json": None,
            }
            stmt = pg_insert(HumanInteraction).values(**fields)
            stmt = stmt.on_conflict_do_nothing(constraint=CONSTRAINT_NAME)
            async with session_scope(self._session_factory) as session:
                await session.execute(stmt)
        except Exception:
            self._log.error("on_custom_event[human_request]: db insert failed", exc_info=True)

    async def _handle_human_response(self, data: Any) -> None:
        """UPDATE a HumanInteraction row with the human/gateway response.

        Matches on ``(run_id, request_id)`` AND ``response_json IS NULL`` so the
        UPDATE is idempotent — a re-executed node that re-dispatches this event
        will not overwrite an already-filled response.

        ``data`` is expected to be a dict with keys:
            run_id        (UUID)
            request_id    (str)
            answered_at   (datetime, timezone-aware)
            response_json (dict — HumanResponse.model_dump())
            source        (str)
            timed_out     (bool)
            latency_s     (float | None)
        """
        try:
            run_id: UUID = data["run_id"]
            request_id: str = data["request_id"]
            answered_at: datetime = data.get("answered_at", _now_utc())
            response_json: dict[str, Any] = data.get("response_json", {})
            source: str = data.get("source", "human")
            timed_out: bool = bool(data.get("timed_out", False))
            latency_s: float | None = data.get("latency_s")

            stmt = (
                update(HumanInteraction)
                .where(
                    HumanInteraction.run_id == run_id,
                    HumanInteraction.request_id == request_id,
                    HumanInteraction.response_json.is_(None),
                )
                .values(
                    answered_at=answered_at,
                    response_json=response_json,
                )
            )
            # Store source, timed_out, latency_s inside response_json if not already there,
            # or as separate computed values. Since the model has no dedicated columns for
            # these (they live in response_json as per HumanResponse.model_dump()), the
            # caller should include them in response_json. We log them for diagnostics.
            _ = source  # used by caller via response_json; kept for future column expansion
            _ = timed_out
            _ = latency_s

            async with session_scope(self._session_factory) as session:
                await session.execute(stmt)
        except Exception:
            self._log.error("on_custom_event[human_response]: db update failed", exc_info=True)
