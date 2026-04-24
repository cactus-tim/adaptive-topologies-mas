"""Unit tests for SQLAlchemy ORM models in src/atm/storage/models.py.

These tests verify the structural properties of the 6 business tables:
experiments, runs, phases, human_interactions, budget_events, topology_transitions.

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
    Phase,
    Run,
    TopologyTransition,
)


# ---------------------------------------------------------------------------
# FinishReason enum
# ---------------------------------------------------------------------------


class TestFinishReason:
    def test_is_str_enum(self) -> None:
        from enum import StrEnum

        assert issubclass(FinishReason, StrEnum)

    def test_success_value(self) -> None:
        assert FinishReason("success").value == "success"

    def test_max_iter_value(self) -> None:
        assert FinishReason("max_iter").value == "max_iter"

    def test_budget_exceeded_value(self) -> None:
        assert FinishReason("budget_exceeded").value == "budget_exceeded"

    def test_error_value(self) -> None:
        assert FinishReason("error").value == "error"

    def test_timeout_value(self) -> None:
        assert FinishReason("timeout").value == "timeout"

    def test_human_abort_value(self) -> None:
        assert FinishReason("human_abort").value == "human_abort"

    def test_all_expected_members(self) -> None:
        expected = {"success", "max_iter", "budget_exceeded", "error", "timeout", "human_abort"}
        assert {m.value for m in FinishReason} == expected


# ---------------------------------------------------------------------------
# Base and metadata — exactly 6 tables in FK-safe order
# ---------------------------------------------------------------------------


EXPECTED_TABLE_NAMES = frozenset(
    {"budget_events", "experiments", "human_interactions", "phases", "runs", "topology_transitions"}
)


class TestBaseMetadata:
    def test_exactly_six_tables(self) -> None:
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
        for dependent in ("budget_events", "human_interactions", "phases", "topology_transitions"):
            assert runs_idx < names.index(dependent)

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
        # type should be UUID or PG_UUID
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
# HumanInteraction table
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

    def test_context_ref_column_exists(self) -> None:
        assert "context_ref" in self._table().c

    def test_answer_ref_column_exists(self) -> None:
        assert "answer_ref" in self._table().c

    def test_tlx_scores_jsonb(self) -> None:
        col = self._table().c["tlx_scores"]
        assert isinstance(col.type, JSONB)

    def test_tlx_scores_server_default(self) -> None:
        col = self._table().c["tlx_scores"]
        assert col.server_default is not None

    def test_raw_tlx_score_nullable_float(self) -> None:
        """§13.3: raw_tlx_score present as Float NULLABLE for fast filter."""
        col = self._table().c["raw_tlx_score"]
        assert col.nullable
        assert isinstance(col.type, sa.Float)

    def test_request_id_varchar64(self) -> None:
        """request_id VARCHAR(64) for idempotency."""
        col = self._table().c["request_id"]
        assert isinstance(col.type, sa.String)
        assert col.type.length == 64  # type: ignore[union-attr]

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(HumanInteraction)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# BudgetEvent table
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

    def test_event_type_column_exists(self) -> None:
        assert "event_type" in self._table().c

    def test_value_usd_numeric(self) -> None:
        col = self._table().c["value_usd"]
        assert isinstance(col.type, sa.Numeric)

    def test_at_timestamptz(self) -> None:
        col = self._table().c["at"]
        assert col.type.timezone is True  # type: ignore[union-attr]

    def test_all_relationships_lazy_raise(self) -> None:
        mapper = sa.inspect(BudgetEvent)
        for rel in mapper.relationships:
            assert rel.lazy == "raise", f"Relationship {rel.key!r} must have lazy='raise'"


# ---------------------------------------------------------------------------
# TopologyTransition table
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

    def test_at_iter_column_exists(self) -> None:
        assert "at_iter" in self._table().c

    def test_from_topology_column_exists(self) -> None:
        assert "from_topology" in self._table().c

    def test_to_topology_column_exists(self) -> None:
        assert "to_topology" in self._table().c

    def test_decided_by_column_exists(self) -> None:
        assert "decided_by" in self._table().c

    def test_considered_alternatives_array(self) -> None:
        col = self._table().c["considered_alternatives"]
        assert isinstance(col.type, ARRAY)

    def test_considered_alternatives_server_default(self) -> None:
        col = self._table().c["considered_alternatives"]
        assert col.server_default is not None

    def test_rationale_text(self) -> None:
        col = self._table().c["rationale"]
        assert isinstance(col.type, sa.Text)

    def test_cost_usd_numeric(self) -> None:
        col = self._table().c["cost_usd"]
        assert isinstance(col.type, sa.Numeric)

    def test_guarded_bool(self) -> None:
        col = self._table().c["guarded"]
        assert isinstance(col.type, sa.Boolean)

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
