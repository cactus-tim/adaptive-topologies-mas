"""Initial business schema — arch.md §3.4 DDL as of 2026-04-24.

Creates 6 business tables: experiments, runs, phases, human_interactions,
budget_events, topology_transitions. Checkpoint tables (checkpoints,
checkpoint_blobs, checkpoint_migrations, checkpoint_writes) are created
separately by langgraph-checkpoint-postgres AsyncPostgresSaver.setup()
and are NOT part of this migration (see arch.md §11.3, §17/#3).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = "bc5f66dd0897"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create 6 business tables in FK-safe order."""

    op.create_table(
        "experiments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
        sa.Column(
            "config_snapshot",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "total_cost_usd",
            sa.Numeric(10, 4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("status", sa.String(16), nullable=False),
    )
    op.create_index("experiments_started_at_idx", "experiments", ["started_at"])

    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("exp_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topology", sa.String(32), nullable=False),
        sa.Column("task_id", sa.String(128), nullable=False),
        sa.Column("agent_set", sa.String(64), nullable=False),
        sa.Column("human_role", sa.String(32), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column(
            "models_by_role_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "model_version_snapshot",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("sandbox_image_digest", sa.String(80), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("finish_reason", sa.String(32), nullable=True),
        sa.Column(
            "budget_spent_usd",
            sa.Numeric(10, 4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("quality_score", sa.Double(), nullable=True),
        sa.Column("wall_time_s", sa.Double(), nullable=True),
        sa.Column("iterations", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["exp_id"], ["experiments.id"], ondelete="CASCADE"),
    )
    op.create_index("runs_exp_id_idx", "runs", ["exp_id"])
    op.create_index("runs_topology_idx", "runs", ["topology"])
    op.create_index("runs_task_id_idx", "runs", ["task_id"])
    op.create_index("runs_status_idx", "runs", ["status"])
    op.create_index("runs_started_idx", "runs", ["started_at"])

    op.create_table(
        "phases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("phase_name", sa.String(32), nullable=False),
        sa.Column("from_phase", sa.String(32), nullable=True),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("entry_reason", sa.Text(), nullable=False),
        sa.Column("topology_used", sa.String(32), nullable=False),
        sa.Column("decided_by", sa.String(16), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
    )
    op.create_index("phases_run_id_idx", "phases", ["run_id"])

    op.create_table(
        "human_interactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("requested_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("answered_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("context_json", postgresql.JSONB(), nullable=False),
        sa.Column("response_json", postgresql.JSONB(), nullable=True),
        sa.Column(
            "tlx_scores",
            postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column("raw_tlx_score", sa.Double(), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
    )
    op.create_index("human_interactions_run_id_idx", "human_interactions", ["run_id"])

    op.create_table(
        "budget_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("event", sa.String(16), nullable=False),
        sa.Column("limit_usd", sa.Numeric(10, 4), nullable=False),
        sa.Column("current_usd", sa.Numeric(10, 4), nullable=False),
        sa.Column("at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
    )
    op.create_index("budget_events_run_id_idx", "budget_events", ["run_id"])

    op.create_table(
        "topology_transitions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("from_topology", sa.String(32), nullable=True),
        sa.Column("to_topology", sa.String(32), nullable=False),
        sa.Column("phase_at_decision", sa.String(32), nullable=False),
        sa.Column("iter_within_phase", sa.Integer(), nullable=False),
        sa.Column("iter_within_topology", sa.Integer(), nullable=False),
        sa.Column("decided_by", sa.String(24), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "considered_alternatives",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "guards_applied",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "signals_snapshot",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "router_cost_usd",
            sa.Numeric(10, 4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "topology_transitions_run_id_idx", "topology_transitions", ["run_id"]
    )
    op.create_index(
        "topology_transitions_decided_by_idx", "topology_transitions", ["decided_by"]
    )
    op.create_index("topology_transitions_at_idx", "topology_transitions", ["at"])


def downgrade() -> None:
    """Drop 6 business tables in reverse FK order."""
    op.drop_table("topology_transitions")
    op.drop_table("budget_events")
    op.drop_table("human_interactions")
    op.drop_table("phases")
    op.drop_table("runs")
    op.drop_table("experiments")
