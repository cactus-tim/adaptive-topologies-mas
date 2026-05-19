"""SQLAlchemy 2.x ORM models for the ATM storage layer (6 business tables)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class FinishReason(StrEnum):
    """Terminal reason for a run finishing."""

    SUCCESS = "success"
    MAX_ITER = "max_iter"
    TOPOLOGY_MAX = "topology_max"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"
    HUMAN_TIMEOUT = "human_timeout"


class Base(DeclarativeBase):
    """Single declarative base for all ATM storage models."""


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
    git_sha: Mapped[str | None] = mapped_column(sa.String(40), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
        index=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    total_cost_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
        server_default=sa.text("0"),
    )
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False)

    runs: Mapped[list[Run]] = relationship(
        "Run",
        back_populates="experiment",
        lazy="raise",
    )


class Run(Base):
    """One run within an experiment — atomic unit of execution."""

    __tablename__ = "runs"

    __table_args__ = (
        sa.Index("runs_exp_id_idx", "exp_id"),
        sa.Index("runs_topology_idx", "topology"),
        sa.Index("runs_task_id_idx", "task_id"),
        sa.Index("runs_status_idx", "status"),
        sa.Index("runs_started_idx", "started_at"),
        sa.Index("runs_exp_status_idx", "exp_id", "status"),
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
    human_role: Mapped[str | None] = mapped_column(sa.String(32), nullable=True)
    seed: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    model: Mapped[str] = mapped_column(sa.String(64), nullable=False)

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
    sandbox_image_digest: Mapped[str | None] = mapped_column(
        sa.String(80),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    finish_reason: Mapped[str | None] = mapped_column(sa.String(32), nullable=True)
    budget_spent_usd: Mapped[Decimal] = mapped_column(
        sa.Numeric(10, 4),
        nullable=False,
        server_default=sa.text("0"),
    )
    quality_score: Mapped[float | None] = mapped_column(
        sa.Double(),
        nullable=True,
    )
    cognitive_load_proxy: Mapped[float | None] = mapped_column(
        sa.Double(),
        nullable=True,
        doc="NASA-TLX proxy from human_interactions (M9.2 RQ4 metric).",
    )
    wall_time_s: Mapped[float | None] = mapped_column(
        sa.Double(),
        nullable=True,
    )
    iterations: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)

    replay_of: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("runs.id", ondelete="SET NULL", name="fk_runs_replay_of_runs"),
        nullable=True,
    )
    host: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    process_pid: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)

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
    from_phase: Mapped[str | None] = mapped_column(
        sa.String(32),
        nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    entry_reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    topology_used: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    decided_by: Mapped[str] = mapped_column(sa.String(16), nullable=False)

    run: Mapped[Run] = relationship(
        "Run",
        back_populates="phases",
        lazy="raise",
    )


class HumanInteraction(Base):
    """One HITL event — a human was asked for input and (optionally) responded."""

    __tablename__ = "human_interactions"

    __table_args__ = (
        sa.Index("human_interactions_run_id_idx", "run_id"),
        sa.UniqueConstraint(
            "run_id",
            "request_id",
            name="uq_human_interactions_run_request",
        ),
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
    role: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )
    answered_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    context_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    response_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    tlx_scores: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    raw_tlx_score: Mapped[float | None] = mapped_column(
        sa.Double(),
        nullable=True,
    )
    request_id: Mapped[str | None] = mapped_column(
        sa.String(64),
        nullable=True,
    )

    run: Mapped[Run] = relationship(
        "Run",
        back_populates="human_interactions",
        lazy="raise",
    )


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
    at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )

    run: Mapped[Run] = relationship(
        "Run",
        back_populates="budget_events",
        lazy="raise",
    )


class TopologyTransition(Base):
    """TopologyRouter decision — recorded for every call, switch or no-change."""

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
    from_topology: Mapped[str | None] = mapped_column(
        sa.String(32),
        nullable=True,
    )
    to_topology: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
    )
    phase_at_decision: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    iter_within_phase: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    iter_within_topology: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    decided_by: Mapped[str] = mapped_column(
        sa.String(24),
        nullable=False,
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
    at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )

    run: Mapped[Run] = relationship(
        "Run",
        back_populates="topology_transitions",
        lazy="raise",
    )
