"""Add replay_of (self-FK), host, process_pid to runs; composite index on (exp_id, status).

replay_of  — UUID self-referential FK; identifies the original run being replayed (§14.4).
host       — VARCHAR(64); hostname of the worker that executed the run.
process_pid — INTEGER; PID of the worker process (aids crash-recovery dedup).
runs_exp_status_idx — composite index for fast per-experiment status filter queries.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-14
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add replay_of, host, process_pid columns and supporting constraints/index."""
    op.add_column(
        "runs",
        sa.Column("replay_of", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("host", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "runs",
        sa.Column("process_pid", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_runs_replay_of_runs",
        "runs",
        "runs",
        ["replay_of"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("runs_exp_status_idx", "runs", ["exp_id", "status"], unique=False)


def downgrade() -> None:
    """Drop index, FK, and columns added by this migration (reverse order)."""
    op.drop_index("runs_exp_status_idx", table_name="runs")
    op.drop_constraint("fk_runs_replay_of_runs", "runs", type_="foreignkey")
    op.drop_column("runs", "process_pid")
    op.drop_column("runs", "host")
    op.drop_column("runs", "replay_of")
