"""Add UNIQUE constraint on human_interactions(run_id, request_id) for idempotency.

Enables INSERT ... ON CONFLICT ON CONSTRAINT uq_human_interactions_run_request
in the HumanGateway callback layer (M9 Step 1.5).

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-12
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add unique constraint on (run_id, request_id) for idempotent HITL inserts."""
    op.create_unique_constraint(
        "uq_human_interactions_run_request",
        "human_interactions",
        ["run_id", "request_id"],
    )


def downgrade() -> None:
    """Drop the unique constraint."""
    op.drop_constraint(
        "uq_human_interactions_run_request",
        "human_interactions",
        type_="unique",
    )
