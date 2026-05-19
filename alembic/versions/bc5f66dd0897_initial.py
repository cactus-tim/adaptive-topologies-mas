"""initial

Revision ID: bc5f66dd0897
Revises: 
Create Date: 2026-04-23 14:03:09.929134

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'bc5f66dd0897'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
