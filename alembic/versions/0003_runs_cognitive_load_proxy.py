"""Add cognitive_load_proxy FLOAT column to runs table.

Stores a lightweight cognitive-load proxy metric per run, derived from
NASA-TLX signals (§13.3). Nullable — populated post-run by the metrics
aggregation pipeline.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-12
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add cognitive_load_proxy FLOAT NULLABLE to runs."""
    op.add_column(
        "runs",
        sa.Column("cognitive_load_proxy", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Drop cognitive_load_proxy from runs."""
    op.drop_column("runs", "cognitive_load_proxy")
