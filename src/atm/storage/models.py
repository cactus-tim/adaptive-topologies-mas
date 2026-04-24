"""SQLAlchemy 2.x ORM models for the ATM storage layer.

6 business tables:
  - experiments    — root entity; one experiment = a grid/sweep run
  - runs           — one run within an experiment; reproducibility bundle §14.4
  - phases         — phase transitions within a run §3.4
  - human_interactions — HITL events with NASA-TLX §13.3
  - budget_events  — budget warn/exceed events §3.4
  - topology_transitions — TopologyRouter decisions §3.4 / RQ2

All datetime columns use TIMESTAMPTZ.
JSONB columns use server_default=sa.text("'{}'::jsonb") where arch.md requires it.
ARRAY(String) columns use server_default=sa.text("'{}'::text[]").
All relationships use lazy="raise".
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from enum import StrEnum
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# FinishReason — StrEnum with canonical values (arch.md §3.4)
# ---------------------------------------------------------------------------


class FinishReason(StrEnum):
    """Terminal reason for a run finishing."""

    SUCCESS = "success"
    MAX_ITER = "max_iter"
    TOPOLOGY_MAX = "topology_max"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"
    HUMAN_TIMEOUT = "human_timeout"


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Single declarative base for all ATM storage models."""


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class Experiment(Base):
    """Root entity — one experiment (a grid, sweep, or single run batch)."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    git_sha: Mapped[Optional[str]] = mapped_column(sa.String(40), nullable=True)
    started_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
        index=True,
    )
    finished_at: Mapped[Optional[sa.DateTime]] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    total_cost_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
        server_default=sa.text("0"),
    )
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False)

    # Relationships
    runs: Mapped[list[Run]] = relationship(
        "Run",
        back_populates="experiment",
        lazy="raise",
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run(Base):
    """One run within an experiment — atomic unit of execution.

    Contains the full reproducibility bundle (§14.4):
      - models_by_role_json: snapshot of ModelCfg.by_role at run start
      - model_version_snapshot: {model_id: version} for exact replay
      - sandbox_image_digest: sha256:… of the DockerSandbox image
    """

    __tablename__ = "runs"

    __table_args__ = (
        sa.Index("runs_exp_id_idx", "exp_id"),
        sa.Index("runs_topology_idx", "topology"),
        sa.Index("runs_task_id_idx", "task_id"),
        sa.Index("runs_status_idx", "status"),
        sa.Index("runs_started_idx", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    exp_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    topology: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    task_id: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    agent_set: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    human_role: Mapped[Optional[str]] = mapped_column(sa.String(32), nullable=True)
    seed: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    model: Mapped[str] = mapped_column(sa.String(64), nullable=False)

    # Reproducibility bundle (§14.4)
    models_by_role_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    model_version_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    sandbox_image_digest: Mapped[Optional[str]] = mapped_column(
        sa.String(80),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    finish_reason: Mapped[Optional[str]] = mapped_column(sa.String(32), nullable=True)
    budget_spent_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
        server_default=sa.text("0"),
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        sa.Double(),
        nullable=True,
    )
    wall_time_s: Mapped[Optional[float]] = mapped_column(
        sa.Double(),
        nullable=True,
    )
    iterations: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    started_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    finished_at: Mapped[Optional[sa.DateTime]] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    error: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # Relationships
    experiment: Mapped[Experiment] = relationship(
        "Experiment",
        back_populates="runs",
        lazy="raise",
    )
    phases: Mapped[list[Phase]] = relationship(
        "Phase",
        back_populates="run",
        lazy="raise",
    )
    human_interactions: Mapped[list[HumanInteraction]] = relationship(
        "HumanInteraction",
        back_populates="run",
        lazy="raise",
    )
    budget_events: Mapped[list[BudgetEvent]] = relationship(
        "BudgetEvent",
        back_populates="run",
        lazy="raise",
    )
    topology_transitions: Mapped[list[TopologyTransition]] = relationship(
        "TopologyTransition",
        back_populates="run",
        lazy="raise",
    )


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


class Phase(Base):
    """Phase transition within a run (planning → execution → verification → done)."""

    __tablename__ = "phases"

    __table_args__ = (sa.Index("phases_run_id_idx", "run_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    phase_name: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    from_phase: Mapped[Optional[str]] = mapped_column(
        sa.String(32),
        nullable=True,  # §3.4: None on initial init
    )
    started_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    ended_at: Mapped[Optional[sa.DateTime]] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,  # §3.4: nullable — phase may still be ongoing
    )
    entry_reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    topology_used: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    decided_by: Mapped[str] = mapped_column(sa.String(16), nullable=False)

    # Relationships
    run: Mapped[Run] = relationship(
        "Run",
        back_populates="phases",
        lazy="raise",
    )


# ---------------------------------------------------------------------------
# HumanInteraction
# ---------------------------------------------------------------------------


class HumanInteraction(Base):
    """One HITL event — a human was asked for input and (optionally) responded.

    NASA-TLX data (§13.3):
      - tlx_scores: raw 6-scale JSONB (NasaTLX.model_dump())
      - raw_tlx_score: aggregated float for fast filter queries
    Idempotency:
      - request_id: VARCHAR(64) — idempotency key paired with run_id
    """

    __tablename__ = "human_interactions"

    __table_args__ = (sa.Index("human_interactions_run_id_idx", "run_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    requested_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    answered_at: Mapped[Optional[sa.DateTime]] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    context_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,  # §3.4: HumanContext.model_dump()
    )
    response_json: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,  # §3.4: HumanResponse.model_dump(); null until answered
    )
    tlx_scores: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,  # §3.4: nullable
    )
    raw_tlx_score: Mapped[Optional[float]] = mapped_column(
        sa.Double(),
        nullable=True,  # §13.3: aggregated float for fast filter; null before TLX filled
    )
    request_id: Mapped[Optional[str]] = mapped_column(
        sa.String(64),
        nullable=True,  # idempotency key (run_id, request_id) pair
    )

    # Relationships
    run: Mapped[Run] = relationship(
        "Run",
        back_populates="human_interactions",
        lazy="raise",
    )


# ---------------------------------------------------------------------------
# BudgetEvent
# ---------------------------------------------------------------------------


class BudgetEvent(Base):
    """A budget warning or exceed event fired during a run."""

    __tablename__ = "budget_events"

    __table_args__ = (sa.Index("budget_events_run_id_idx", "run_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    level: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    event: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    limit_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
    )
    current_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
    )
    at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )

    # Relationships
    run: Mapped[Run] = relationship(
        "Run",
        back_populates="budget_events",
        lazy="raise",
    )


# ---------------------------------------------------------------------------
# TopologyTransition
# ---------------------------------------------------------------------------


class TopologyTransition(Base):
    """TopologyRouter decision — recorded for every call, switch or no-change.

    Source of truth for RQ2 metrics:
      topology_switch_count, guard_override_rate, router_cost_share, oracle_gap.
    """

    __tablename__ = "topology_transitions"

    __table_args__ = (
        sa.Index("topology_transitions_run_id_idx", "run_id"),
        sa.Index("topology_transitions_decided_by_idx", "decided_by"),
        sa.Index("topology_transitions_at_idx", "at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    from_topology: Mapped[Optional[str]] = mapped_column(
        sa.String(32),
        nullable=True,  # §3.4: null only for initial
    )
    to_topology: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,  # §3.4: == from_topology if no-change
    )
    phase_at_decision: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    iter_within_phase: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    iter_within_topology: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    decided_by: Mapped[str] = mapped_column(
        sa.String(24),
        nullable=False,  # enum: rule|llm_router|oracle|guard_override|initial
    )
    reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    considered_alternatives: Mapped[list[str]] = mapped_column(
        ARRAY(sa.String),
        nullable=False,
        server_default=sa.text("'{}'::text[]"),
    )
    guards_applied: Mapped[list[str]] = mapped_column(
        ARRAY(sa.String),
        nullable=False,
        server_default=sa.text("'{}'::text[]"),
    )
    signals_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    router_cost_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
        server_default=sa.text("0"),
    )
    at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )

    # Relationships
    run: Mapped[Run] = relationship(
        "Run",
        back_populates="topology_transitions",
        lazy="raise",
    )
