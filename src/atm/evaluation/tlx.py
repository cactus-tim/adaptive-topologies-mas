"""NASA-TLX model and aggregation helper.

Reference: arch.md §13.3
Storage: human_interactions.tlx_scores (JSONB) + raw_tlx_score (DOUBLE PRECISION).
M9 owns the write path; this module provides the model and aggregate helper only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
