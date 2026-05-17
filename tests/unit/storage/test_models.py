"""Unit tests for SQLAlchemy ORM models in src/atm/storage/models.py.

These tests verify the structural properties of the 8 business tables:
experiments, runs, phases, human_interactions, budget_events, topology_transitions,
study_sessions, human_request_queue.

Column names, types, and nullability are asserted to match arch.md §3.4 exactly.
No live database connection is required — all assertions are metadata-level.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from atm.storage.models import (
    Base,
    BudgetEvent,
    Experiment,
    FinishReason,
    HumanInteraction,
    HumanRequestQueue,
    Phase,
    Run,
    StudySession,
    TopologyTransition,
)

# ---------------------------------------------------------------------------
# FinishReason enum — arch.md §3.4 canonical values
# ---------------------------------------------------------------------------


class TestFinishReason:
    def test_is_str_enum(self) -> None:
        from enum import StrEnum

        assert issubclass(FinishReason, StrEnum)

    def test_success_value(self) -> None:
        assert FinishReason("success").value == "success"

    def test_max_iter_value(self) -> None:
        assert FinishReason("max_iter").value == "max_iter"

    def test_topology_max_value(self) -> None:
        assert FinishReason("topology_max").value == "topology_max"

    def test_budget_exceeded_value(self) -> None:
        assert FinishReason("budget_exceeded").value == "budget_exceeded"

    def test_error_value(self) -> None:
        assert FinishReason("error").value == "error"

    def test_human_timeout_value(self) -> None:
        assert FinishReason("human_timeout").value == "human_timeout"

    def test_all_expected_members(self) -> None:
        expected = {
            "success",
            "max_iter",
            "topology_max",
            "budget_exceeded",
            "error",
            "human_timeout",
        }
        assert {m.value for m in FinishReason} == expected


# ---------------------------------------------------------------------------
# Base and metadata — exactly 6 tables in FK-safe order
# ---------------------------------------------------------------------------


EXPECTED_TABLE_NAMES = frozenset(
    {
        "budget_events",
        "experiments",
        "human_interactions",
        "human_request_queue",
        "phases",
        "runs",
        "study_sessions",
        "topology_transitions",
    }
)


class TestBaseMetadata:
    def test_exactly_eight_tables(self) -> None:
        tables = {t.name for t in Base.metadata.sorted_tables}
        assert tables == EXPECTED_TABLE_NAMES

    def test_sorted_tables_fk_safe_order(self) -> None:
        """experiments and runs must come before their dependents."""
        names = [t.name for t in Base.metadata.sorted_tables]
        experiments_idx = names.index("experiments")
        runs_idx = names.index("runs")
        # experiments must precede runs (runs.exp_id → experiments.id)
        assert experiments_idx < runs_idx
        # runs must precede all FK dependents
        for dependent in (
            "budget_events",
            "human_interactions",
            "human_request_queue",
            "phases",
            "topology_transitions",
        ):
            assert runs_idx < names.index(dependent)
        # study_sessions must precede human_interactions (FK study_session_id)
        assert names.index("study_sessions") < names.index("human_interactions")

    def test_declarative_base_used(self) -> None:
        from sqlalchemy.orm import DeclarativeBase

        assert issubclass(Base, DeclarativeBase)


# ---------------------------------------------------------------------------
# Experiment table
# ---------------------------------------------------------------------------


class TestExperimentModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["experiments"]

    def test_table_name(self) -> None:
        assert Experiment.__tablename__ == "experiments"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_name_unique(self) -> None:
        col = self._table().c["name"]
        assert col.unique or any(
            len(uc.columns) == 1 and "name" in [c.name for c in uc.columns]
            for uc in self._table().constraints
            if isinstance(uc, sa.UniqueConstraint)
        )

    def test_config_snapshot_jsonb(self) -> None:
        col = self._table().c["config_snapshot"]
        assert isinstance(col.type, JSONB)

    def test_config_snapshot_server_default(self) -> None:
        col = self._table().c["config_snapshot"]
        assert col.server_default is not None

    def test_git_sha_column_exists(self) -> None:
        assert "git_sha" in self._table().c

    def test_started_at_timestamptz(self) -> None:
        col = self._table().c["started_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_finished_at_nullable(self) -> None:
        col = self._table().c["finished_at"]
        assert col.nullable

    def test_finished_at_timestamptz(self) -> None:
        col = self._table().c["finished_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_total_cost_usd_numeric(self) -> None:
        col = self._table().c["total_cost_usd"]
        assert isinstance(col.type, sa.Numeric)

    def test_status_column_exists(self) -> None:
        assert "status" in self._table().c

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(Experiment)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# Run table
# ---------------------------------------------------------------------------


class TestRunModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["runs"]

    def test_table_name(self) -> None:
        assert Run.__tablename__ == "runs"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_exp_id_fk(self) -> None:
        col = self._table().c["exp_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "experiments.id" in str(fks[0].target_fullname)

    def test_topology_column_exists(self) -> None:
        assert "topology" in self._table().c

    def test_task_id_column_exists(self) -> None:
        assert "task_id" in self._table().c

    def test_agent_set_column_exists(self) -> None:
        assert "agent_set" in self._table().c

    def test_human_role_nullable(self) -> None:
        col = self._table().c["human_role"]
        assert col.nullable

    def test_seed_column_exists(self) -> None:
        assert "seed" in self._table().c

    def test_model_column_exists(self) -> None:
        assert "model" in self._table().c

    # Reproducibility bundle fields (§14.4)
    def test_models_by_role_json_jsonb(self) -> None:
        col = self._table().c["models_by_role_json"]
        assert isinstance(col.type, JSONB)

    def test_models_by_role_json_server_default(self) -> None:
        col = self._table().c["models_by_role_json"]
        assert col.server_default is not None

    def test_model_version_snapshot_jsonb(self) -> None:
        col = self._table().c["model_version_snapshot"]
        assert isinstance(col.type, JSONB)

    def test_model_version_snapshot_server_default(self) -> None:
        col = self._table().c["model_version_snapshot"]
        assert col.server_default is not None

    def test_sandbox_image_digest_nullable(self) -> None:
        col = self._table().c["sandbox_image_digest"]
        assert col.nullable

    def test_finish_reason_max_length(self) -> None:
        col = self._table().c["finish_reason"]
        assert col.nullable  # nullable
        assert isinstance(col.type, sa.String)
        assert col.type.length == 32  # type: ignore[union-attr]

    def test_budget_spent_usd_numeric(self) -> None:
        col = self._table().c["budget_spent_usd"]
        assert isinstance(col.type, sa.Numeric)

    def test_quality_score_nullable(self) -> None:
        col = self._table().c["quality_score"]
        assert col.nullable

    def test_wall_time_s_nullable(self) -> None:
        col = self._table().c["wall_time_s"]
        assert col.nullable

    def test_iterations_nullable(self) -> None:
        col = self._table().c["iterations"]
        assert col.nullable

    def test_started_at_timestamptz(self) -> None:
        col = self._table().c["started_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(Run)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# Phase table
# ---------------------------------------------------------------------------


class TestPhaseModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["phases"]

    def test_table_name(self) -> None:
        assert Phase.__tablename__ == "phases"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_run_id_fk(self) -> None:
        col = self._table().c["run_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "runs.id" in str(fks[0].target_fullname)

    def test_phase_name_column_exists(self) -> None:
        assert "phase_name" in self._table().c

    def test_from_phase_nullable(self) -> None:
        """§3.4: from_phase nullable (None on initial init)."""
        col = self._table().c["from_phase"]
        assert col.nullable

    def test_started_at_timestamptz(self) -> None:
        col = self._table().c["started_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_ended_at_nullable(self) -> None:
        """§3.4: ended_at nullable (phase may still be ongoing)."""
        col = self._table().c["ended_at"]
        assert col.nullable

    def test_ended_at_timestamptz(self) -> None:
        col = self._table().c["ended_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_entry_reason_column_exists(self) -> None:
        assert "entry_reason" in self._table().c

    def test_topology_used_column_exists(self) -> None:
        assert "topology_used" in self._table().c

    def test_decided_by_column_exists(self) -> None:
        assert "decided_by" in self._table().c

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(Phase)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# HumanInteraction table — arch.md §3.4 + §13.3
# ---------------------------------------------------------------------------


class TestHumanInteractionModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["human_interactions"]

    def test_table_name(self) -> None:
        assert HumanInteraction.__tablename__ == "human_interactions"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_run_id_fk(self) -> None:
        col = self._table().c["run_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "runs.id" in str(fks[0].target_fullname)

    def test_role_column_exists(self) -> None:
        assert "role" in self._table().c

    def test_requested_at_timestamptz(self) -> None:
        col = self._table().c["requested_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_answered_at_nullable(self) -> None:
        col = self._table().c["answered_at"]
        assert col.nullable

    def test_answered_at_timestamptz(self) -> None:
        col = self._table().c["answered_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_context_json_jsonb(self) -> None:
        """§3.4: context_json JSONB NOT NULL — HumanContext.model_dump()."""
        col = self._table().c["context_json"]
        assert isinstance(col.type, JSONB)
        assert not col.nullable

    def test_response_json_jsonb_nullable(self) -> None:
        """§3.4: response_json JSONB nullable — null until answered."""
        col = self._table().c["response_json"]
        assert isinstance(col.type, JSONB)
        assert col.nullable

    def test_tlx_scores_jsonb_nullable(self) -> None:
        """§3.4: tlx_scores JSONB nullable."""
        col = self._table().c["tlx_scores"]
        assert isinstance(col.type, JSONB)
        assert col.nullable

    def test_raw_tlx_score_nullable_float(self) -> None:
        """§13.3: raw_tlx_score present as DOUBLE PRECISION NULLABLE for fast filter."""
        col = self._table().c["raw_tlx_score"]
        assert col.nullable

    def test_request_id_varchar64(self) -> None:
        """request_id VARCHAR(64) for idempotency key (run_id, request_id) pair."""
        col = self._table().c["request_id"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 64  # type: ignore[union-attr]

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(HumanInteraction)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# BudgetEvent table — arch.md §3.4
# ---------------------------------------------------------------------------


class TestBudgetEventModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["budget_events"]

    def test_table_name(self) -> None:
        assert BudgetEvent.__tablename__ == "budget_events"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_run_id_fk(self) -> None:
        col = self._table().c["run_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "runs.id" in str(fks[0].target_fullname)

    def test_level_column_exists(self) -> None:
        assert "level" in self._table().c

    def test_event_column_exists(self) -> None:
        """§3.4: column named 'event' (not 'event_type')."""
        assert "event" in self._table().c

    def test_event_column_length(self) -> None:
        col = self._table().c["event"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 16  # type: ignore[union-attr]

    def test_limit_usd_numeric(self) -> None:
        """§3.4: limit_usd NUMERIC(10,4) NOT NULL."""
        col = self._table().c["limit_usd"]
        assert isinstance(col.type, sa.Numeric)
        assert not col.nullable

    def test_current_usd_numeric(self) -> None:
        """§3.4: current_usd NUMERIC(10,4) NOT NULL."""
        col = self._table().c["current_usd"]
        assert isinstance(col.type, sa.Numeric)
        assert not col.nullable

    def test_at_timestamptz(self) -> None:
        col = self._table().c["at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(BudgetEvent)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# TopologyTransition table — arch.md §3.4
# ---------------------------------------------------------------------------


class TestTopologyTransitionModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["topology_transitions"]

    def test_table_name(self) -> None:
        assert TopologyTransition.__tablename__ == "topology_transitions"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_run_id_fk(self) -> None:
        col = self._table().c["run_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "runs.id" in str(fks[0].target_fullname)

    def test_from_topology_nullable(self) -> None:
        """§3.4: from_topology nullable (null only for initial)."""
        col = self._table().c["from_topology"]
        assert col.nullable

    def test_to_topology_column_exists(self) -> None:
        assert "to_topology" in self._table().c

    def test_phase_at_decision_column_exists(self) -> None:
        """§3.4: phase_at_decision VARCHAR(32) NOT NULL."""
        col = self._table().c["phase_at_decision"]
        assert isinstance(col.type, sa.String)
        assert not col.nullable

    def test_iter_within_phase_column_exists(self) -> None:
        """§3.4: iter_within_phase INTEGER NOT NULL."""
        col = self._table().c["iter_within_phase"]
        assert isinstance(col.type, sa.Integer)
        assert not col.nullable

    def test_iter_within_topology_column_exists(self) -> None:
        """§3.4: iter_within_topology INTEGER NOT NULL."""
        col = self._table().c["iter_within_topology"]
        assert isinstance(col.type, sa.Integer)
        assert not col.nullable

    def test_decided_by_column_exists(self) -> None:
        assert "decided_by" in self._table().c

    def test_reason_text(self) -> None:
        """§3.4: reason TEXT NOT NULL (replaces rationale)."""
        col = self._table().c["reason"]
        assert isinstance(col.type, sa.Text)
        assert not col.nullable

    def test_considered_alternatives_array(self) -> None:
        col = self._table().c["considered_alternatives"]
        assert isinstance(col.type, ARRAY)

    def test_considered_alternatives_server_default(self) -> None:
        col = self._table().c["considered_alternatives"]
        assert col.server_default is not None

    def test_guards_applied_array(self) -> None:
        """§3.4: guards_applied TEXT[] NOT NULL DEFAULT '{}'."""
        col = self._table().c["guards_applied"]
        assert isinstance(col.type, ARRAY)
        assert col.server_default is not None

    def test_signals_snapshot_jsonb(self) -> None:
        """§3.4: signals_snapshot JSONB NOT NULL DEFAULT '{}'."""
        col = self._table().c["signals_snapshot"]
        assert isinstance(col.type, JSONB)
        assert col.server_default is not None

    def test_router_cost_usd_numeric(self) -> None:
        """§3.4: router_cost_usd NUMERIC(10,4) NOT NULL DEFAULT 0."""
        col = self._table().c["router_cost_usd"]
        assert isinstance(col.type, sa.Numeric)
        assert not col.nullable

    def test_at_timestamptz(self) -> None:
        col = self._table().c["at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(TopologyTransition)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# Import check — public API exports from models module
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_base_exported(self) -> None:
        from atm.storage import models

        assert hasattr(models, "Base")

    def test_finish_reason_exported(self) -> None:
        from atm.storage import models

        assert hasattr(models, "FinishReason")

    def test_all_six_models_exported(self) -> None:
        from atm.storage import models

        for name in (
            "Experiment",
            "Run",
            "Phase",
            "HumanInteraction",
            "BudgetEvent",
            "TopologyTransition",
        ):
            assert hasattr(models, name), f"models.{name} not found"

    def test_new_m14_models_exported(self) -> None:
        from atm.storage import models

        for name in ("StudySession", "HumanRequestQueue"):
            assert hasattr(models, name), f"models.{name} not found"


# ---------------------------------------------------------------------------
# HumanInteraction.study_session_id extension — M14 migration 0005
# ---------------------------------------------------------------------------


class TestHumanInteractionStudySessionExtension:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["human_interactions"]

    def test_study_session_id_column_exists(self) -> None:
        """M14: human_interactions must have study_session_id column."""
        assert "study_session_id" in self._table().c

    def test_study_session_id_nullable(self) -> None:
        """M14: study_session_id is nullable (SET NULL on delete)."""
        col = self._table().c["study_session_id"]
        assert col.nullable

    def test_study_session_id_uuid_type(self) -> None:
        """M14: study_session_id is a UUID column."""
        col = self._table().c["study_session_id"]
        assert "uuid" in type(col.type).__name__.lower()

    def test_study_session_id_fk_to_study_sessions(self) -> None:
        """M14: study_session_id FK → study_sessions.id."""
        col = self._table().c["study_session_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "study_sessions.id" in str(fks[0].target_fullname)

    def test_study_session_relationship_lazy_raise(self) -> None:
        """M14: study_session relationship on HumanInteraction uses lazy='raise'."""
        mapper = sa.inspect(HumanInteraction)
        rel = mapper.relationships["study_session"]
        assert rel.lazy == "raise"

    def test_mapped_column_python_type(self) -> None:
        """M14: study_session_id is mapped on the ORM class (not just table)."""

        mapper = sa.inspect(HumanInteraction)
        attr = mapper.attrs["study_session_id"]
        # Mapped column attribute must exist
        assert attr is not None


# ---------------------------------------------------------------------------
# StudySession table — M14 §HITL
# ---------------------------------------------------------------------------


class TestStudySessionModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["study_sessions"]

    def test_table_name(self) -> None:
        assert StudySession.__tablename__ == "study_sessions"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_participant_id_not_nullable(self) -> None:
        col = self._table().c["participant_id"]
        assert isinstance(col.type, sa.String)
        assert not col.nullable

    def test_participant_id_max_length(self) -> None:
        col = self._table().c["participant_id"]
        assert col.type.length == 64  # type: ignore[union-attr]

    def test_proctor_notes_nullable_text(self) -> None:
        col = self._table().c["proctor_notes"]
        assert isinstance(col.type, sa.Text)
        assert col.nullable

    def test_consent_given_boolean_not_nullable(self) -> None:
        col = self._table().c["consent_given"]
        assert isinstance(col.type, sa.Boolean)
        assert not col.nullable

    def test_consent_given_server_default_false(self) -> None:
        col = self._table().c["consent_given"]
        assert col.server_default is not None

    def test_started_at_timestamptz_not_nullable(self) -> None:
        col = self._table().c["started_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]
        assert not col.nullable

    def test_started_at_server_default(self) -> None:
        col = self._table().c["started_at"]
        assert col.server_default is not None

    def test_ended_at_timestamptz_nullable(self) -> None:
        col = self._table().c["ended_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]
        assert col.nullable

    def test_status_string16_not_nullable(self) -> None:
        col = self._table().c["status"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 16  # type: ignore[union-attr]
        assert not col.nullable

    def test_status_server_default_active(self) -> None:
        col = self._table().c["status"]
        assert col.server_default is not None

    def test_meta_json_jsonb_not_nullable(self) -> None:
        col = self._table().c["meta_json"]
        assert isinstance(col.type, JSONB)
        assert not col.nullable

    def test_meta_json_server_default_empty_object(self) -> None:
        col = self._table().c["meta_json"]
        assert col.server_default is not None

    def test_participant_id_index_exists(self) -> None:
        index_names = {idx.name for idx in self._table().indexes}
        assert "study_sessions_participant_id_idx" in index_names

    def test_status_index_exists(self) -> None:
        index_names = {idx.name for idx in self._table().indexes}
        assert "study_sessions_status_idx" in index_names

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(StudySession)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# HumanRequestQueue table — M14 §HITL
# ---------------------------------------------------------------------------


class TestHumanRequestQueueModel:
    def _table(self) -> sa.Table:
        return Base.metadata.tables["human_request_queue"]

    def test_table_name(self) -> None:
        assert HumanRequestQueue.__tablename__ == "human_request_queue"  # type: ignore[attr-defined]

    def test_pk_is_uuid(self) -> None:
        col = self._table().c["id"]
        assert col.primary_key
        assert "uuid" in type(col.type).__name__.lower()

    def test_run_id_fk_to_runs(self) -> None:
        col = self._table().c["run_id"]
        fks = list(col.foreign_keys)
        assert len(fks) == 1
        assert "runs.id" in str(fks[0].target_fullname)

    def test_run_id_not_nullable(self) -> None:
        col = self._table().c["run_id"]
        assert not col.nullable

    def test_request_id_varchar64_not_nullable(self) -> None:
        col = self._table().c["request_id"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 64  # type: ignore[union-attr]
        assert not col.nullable

    def test_context_json_jsonb_not_nullable(self) -> None:
        col = self._table().c["context_json"]
        assert isinstance(col.type, JSONB)
        assert not col.nullable

    def test_context_json_server_default(self) -> None:
        col = self._table().c["context_json"]
        assert col.server_default is not None

    def test_response_json_jsonb_nullable(self) -> None:
        """response_json is null until the human submits a response."""
        col = self._table().c["response_json"]
        assert isinstance(col.type, JSONB)
        assert col.nullable

    def test_status_string16_not_nullable(self) -> None:
        col = self._table().c["status"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 16  # type: ignore[union-attr]
        assert not col.nullable

    def test_status_server_default_pending(self) -> None:
        col = self._table().c["status"]
        assert col.server_default is not None

    def test_created_at_timestamptz_not_nullable(self) -> None:
        col = self._table().c["created_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]
        assert not col.nullable

    def test_created_at_server_default(self) -> None:
        col = self._table().c["created_at"]
        assert col.server_default is not None

    def test_claimed_at_timestamptz_nullable(self) -> None:
        col = self._table().c["claimed_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]
        assert col.nullable

    def test_responded_at_timestamptz_nullable(self) -> None:
        col = self._table().c["responded_at"]
        assert col.type.timezone is True  # type: ignore[union-attr]
        assert col.nullable

    def test_claimed_by_varchar64_nullable(self) -> None:
        col = self._table().c["claimed_by"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 64  # type: ignore[union-attr]
        assert col.nullable

    def test_unique_constraint_run_request(self) -> None:
        """UNIQUE(run_id, request_id) must exist as uq_human_request_queue_run_request."""
        uc_names = {c.name for c in self._table().constraints if isinstance(c, sa.UniqueConstraint)}
        assert "uq_human_request_queue_run_request" in uc_names

    def test_run_id_index_exists(self) -> None:
        index_names = {idx.name for idx in self._table().indexes}
        assert "human_request_queue_run_id_idx" in index_names

    def test_status_index_exists(self) -> None:
        index_names = {idx.name for idx in self._table().indexes}
        assert "human_request_queue_status_idx" in index_names

    def test_created_at_index_exists(self) -> None:
        index_names = {idx.name for idx in self._table().indexes}
        assert "human_request_queue_created_at_idx" in index_names

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(HumanRequestQueue)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"
