"""NASA-TLX model, aggregation helper, and TLX persistence.

Reference: arch.md §13.3
Storage: human_interactions.tlx_scores (JSONB) + raw_tlx_score (DOUBLE PRECISION).
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession


class NasaTLX(BaseModel):
    """6-scale NASA Task Load Index, each scale in [0, 100].

    performance is inverted in raw_score: lower performance rating = higher load.
    """

    mental_demand: int = Field(ge=0, le=100)
    physical_demand: int = Field(ge=0, le=100)
    temporal_demand: int = Field(ge=0, le=100)
    performance: int = Field(ge=0, le=100)
    effort: int = Field(ge=0, le=100)
    frustration: int = Field(ge=0, le=100)

    @property
    def raw_score(self) -> float:
        """Unweighted mean across 6 dimensions (arch.md §13.3).

        performance is inverted: (100 - performance) so that higher values
        always mean higher workload.
        """
        return (
            self.mental_demand
            + self.physical_demand
            + self.temporal_demand
            + (100 - self.performance)
            + self.effort
            + self.frustration
        ) / 6


def aggregate_tlx(ratings: list[NasaTLX]) -> float:
    """Return the mean raw_score across a list of NasaTLX ratings."""
    return sum(r.raw_score for r in ratings) / len(ratings)


async def persist_tlx(
    session: AsyncSession,
    interaction_id: uuid.UUID,
    raw_score: float,
) -> None:
    """Idempotent UPDATE: write raw_tlx_score to human_interactions row.

    Mirrors the style of evaluation/aggregator.py::persist_quality.
    If the row does not exist the UPDATE affects 0 rows and the call is a
    no-op (caller decides whether to treat that as an error).

    Args:
        session:        An open AsyncSession (caller owns transaction scope).
        interaction_id: UUID of the human_interactions row to update.
        raw_score:      The aggregated TLX score to persist.
    """
    table = sa.table(
        "human_interactions",
        sa.column("id"),
        sa.column("raw_tlx_score"),
    )
    await session.execute(
        sa.update(table).where(table.c.id == interaction_id).values(raw_tlx_score=raw_score)
    )
    await session.commit()
