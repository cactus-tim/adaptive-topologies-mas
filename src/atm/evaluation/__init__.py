"""ATM evaluation package — public API.

M11 evaluation framework: five modules providing post-hoc quality scoring,
NASA-TLX human-load modelling, and RQ1-RQ4 metric functions.

Modules:
  - ``tlx``         — NasaTLX Pydantic model + aggregate_tlx helper
  - ``ground_truth`` — score_ground_truth facade over M10 EVALUATORS
  - ``judges``      — RubricJudge, PairwiseJudge, SelfConsistentJudge (LLM-as-judge)
  - ``metrics``     — pure metric functions (quality / efficiency / time / human / RQ2)
  - ``aggregator``  — compute_quality + persist_quality + aggregate_run (async, PG-backed)

All public symbols are listed in ``__all__`` (alphabetical).
"""

from __future__ import annotations

from atm.evaluation.aggregator import aggregate_run, compute_quality, persist_quality
from atm.evaluation.ground_truth import score_ground_truth
from atm.evaluation.judges import PairwiseJudge, PairwiseResult, RubricJudge, SelfConsistentJudge
from atm.evaluation.metrics import (
    aggregate_human_load,
    aggregate_quality,
    cost_per_quality,
    human_sim_cognitive_load_proxy,
    humaneval_pass_at_k,
    time_per_quality,
)
from atm.evaluation.tlx import NasaTLX, aggregate_tlx, persist_tlx

__all__ = [
    "NasaTLX",
    "PairwiseJudge",
    "PairwiseResult",
    "RubricJudge",
    "SelfConsistentJudge",
    "aggregate_human_load",
    "aggregate_quality",
    "aggregate_run",
    "aggregate_tlx",
    "compute_quality",
    "cost_per_quality",
    "human_sim_cognitive_load_proxy",
    "humaneval_pass_at_k",
    "persist_quality",
    "persist_tlx",
    "score_ground_truth",
    "time_per_quality",
]
