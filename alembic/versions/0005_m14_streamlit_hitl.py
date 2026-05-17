"""Add study_sessions, human_request_queue tables; extend human_interactions with study_session_id.

study_sessions — tracks proctored user-study sessions (one per participant per experiment).
human_request_queue — async PG queue between the runner process and the Streamlit UI.
human_interactions.study_session_id — nullable UUID FK → study_sessions.id.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-17
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create study_sessions and human_request_queue; add study_session_id FK to human_interactions."""

    # 1. study_sessions (no FK dependencies on other new tables)
    op.create_table(
        "study_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("participant_id", sa.String(64), nullable=False),
        sa.Column("proctor_notes", sa.Text(), nullable=True),
        sa.Column("consent_given", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default=sa.text("'active'")),
        sa.Column(
            "meta_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("study_sessions_participant_id_idx", "study_sessions", ["participant_id"])
    op.create_index("study_sessions_status_idx", "study_sessions", ["status"])

    # 2. human_request_queue (FK → runs.id; ON DELETE CASCADE)
    op.create_table(
        "human_request_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column(
            "context_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "response_json",
            postgresql.JSONB(),
            nullable=True,  # null until the human submits a response
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default=sa.text("'pending'")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("claimed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("responded_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("claimed_by", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name="fk_human_request_queue_run_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "run_id",
            "request_id",
            name="uq_human_request_queue_run_request",
        ),
    )
    op.create_index("human_request_queue_run_id_idx", "human_request_queue", ["run_id"])
    op.create_index("human_request_queue_status_idx", "human_request_queue", ["status"])
    op.create_index("human_request_queue_created_at_idx", "human_request_queue", ["created_at"])

    # 3. Extend human_interactions with study_session_id (nullable FK → study_sessions.id)
    op.add_column(
        "human_interactions",
        sa.Column("study_session_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_human_interactions_study_session_id",
        "human_interactions",
        "study_sessions",
        ["study_session_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "human_interactions_study_session_id_idx",
        "human_interactions",
        ["study_session_id"],
    )


def downgrade() -> None:
    """Drop study_session_id FK/column from human_interactions; drop new tables (reverse order)."""

    # 3. Undo human_interactions extension
    op.drop_index("human_interactions_study_session_id_idx", table_name="human_interactions")
    op.drop_constraint(
        "fk_human_interactions_study_session_id", "human_interactions", type_="foreignkey"
    )
    op.drop_column("human_interactions", "study_session_id")

    # 2. Drop human_request_queue
    op.drop_index("human_request_queue_created_at_idx", table_name="human_request_queue")
    op.drop_index("human_request_queue_status_idx", table_name="human_request_queue")
    op.drop_index("human_request_queue_run_id_idx", table_name="human_request_queue")
    op.drop_table("human_request_queue")

    # 1. Drop study_sessions
    op.drop_index("study_sessions_status_idx", table_name="study_sessions")
    op.drop_index("study_sessions_participant_id_idx", table_name="study_sessions")
    op.drop_table("study_sessions")
