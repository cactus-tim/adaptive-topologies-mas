"""E2E integration test for the evaluation framework (M11).

Verifies the full contract:
    run_one(cfg) → runs.quality_score == 1.0

This test:
  - Builds an ExperimentConfig pointing at a Chain topology with scripted FakeLLM.
  - Mocks ``resolve_spec`` (in runner module) to return a pre-built MMLU TaskSpec
    with ``expected="B"`` and ``evaluator_key="mmlu_exact_match"``.
  - The scripted executor fixture emits "The answer is B." as final_answer.
  - MMLUEvaluator extracts "B" from the answer → score=1.0.
  - Asserts that ``runs.quality_score == 1.0`` is persisted in PostgreSQL.

Skipped unless ``ATM_INTEGRATION_PG=1`` is set.

Fixtures from ``tests/integration/conftest.py``:
  - ``pg_engine_fast`` (function scope) — Base.metadata.create_all DDL
  - ``session_factory_fast`` (function scope) — bound to pg_engine_fast
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.storage.models import Run
from atm.storage.session import create_session_factory, session_scope
from atm.tasks.base import TaskSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"

# Pre-built MMLU TaskSpec with expected="B" — no HuggingFace fetch needed.
_MMLU_SPEC = TaskSpec(
    id="mmlu/e2e/0",
    type="qa",
    input=(
        "Which letter comes after A in the alphabet?\n\n"
        "A. A\n"
        "B. B\n"
        "C. C\n"
        "D. D"
    ),
    expected="B",
    evaluator_key="mmlu_exact_match",
    metadata={
        "category": "general",
        "answer_index": 1,
        "options": ["A", "B", "C", "D"],
    },
)


def _make_cfg(pg_dsn: str, parquet_dir: str) -> object:
    """Build a minimal ExperimentConfig for the MMLU e2e test.

    Uses chain topology + scripted FakeLLM (no real LLM calls).
    The ``evaluation.judge_model`` is set to ``fake:echo`` so that the judge
    wrapper in the runner does not trigger real API calls.
    The ``task.name`` is set to ``"mmlu"`` (registered task); however, we
    mock ``resolve_spec`` in the runner to avoid loading the HuggingFace dataset.
    """
    from atm.experiment.config import (
        AgentSetCfg,
        BudgetCfg,
        EvaluationCfg,
        ExperimentConfig,
        ModelCfg,
        ObservabilityCfg,
        TaskCfg,
        TopologyCfg,
    )

    planner_fixture = str(_FIXTURES_DIR / "m6_chain_planner.yaml")
    executor_fixture = str(_FIXTURES_DIR / "m11_mmlu_e2e_executor.yaml")
    critic_fixture = str(_FIXTURES_DIR / "m6_chain_critic.yaml")

    for path in [planner_fixture, executor_fixture, critic_fixture]:
        if not Path(path).exists():
            pytest.skip(f"Fixture not found: {path}")

    return ExperimentConfig(
        name="m11_e2e_mmlu_quality",
        seed=42,
        budget=BudgetCfg(
            per_call_usd=1.0,
            per_run_usd=10.0,
            per_experiment_usd=100.0,
        ),
        model=ModelCfg(
            default="fake:scripted",
            by_role={
                "planner": "fake:scripted",
                "executor": "fake:scripted",
                "critic": "fake:scripted",
                "researcher": "fake:scripted",
            },
            fake_fixtures={
                "planner": planner_fixture,
                "executor": executor_fixture,
                "critic": critic_fixture,
                "researcher": planner_fixture,
            },
            judge="fake:echo",
        ),
        agents=AgentSetCfg(set="canonical_4"),
        topology=TopologyCfg(name="chain", max_iterations=5),
        task=TaskCfg(name="mmlu", input="", split="test", shuffle_seed=0),
        observability=ObservabilityCfg(
            pg_dsn=pg_dsn,
            parquet_dir=parquet_dir,
            callback_sync=True,
        ),
        evaluation=EvaluationCfg(
            judge_model="fake:echo",
            judge_self_consistency_n=1,
        ),
    )


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_mmlu_run_quality_score_is_one(
    pg_engine_fast: AsyncEngine,
    tmp_path: object,
) -> None:
    """Full e2e: run_one with MMLU + scripted answer 'B' → runs.quality_score == 1.0.

    Pipeline:
      1. Build ExperimentConfig (chain topology, scripted FakeLLM).
      2. Mock ``atm.experiment.runner.resolve_spec`` to return _MMLU_SPEC
         (evaluator_key='mmlu_exact_match', expected='B') — avoids HF download.
      3. Call ``run_one(cfg)`` — topology produces final_answer "The answer is B."
      4. MMLUEvaluator extracts 'B' → score=1.0 → persisted to runs.quality_score.
      5. Assert ``runs.quality_score == 1.0`` via SQLAlchemy read.
    """
    pg_dsn: str = pg_engine_fast.url.render_as_string(hide_password=False)
    # asyncpg DSN must include the driver suffix
    if "+asyncpg" not in pg_dsn:
        pg_dsn = pg_dsn.replace("postgresql://", "postgresql+asyncpg://")

    cfg = _make_cfg(pg_dsn=pg_dsn, parquet_dir=str(tmp_path))

    from atm.experiment.runner import run_one

    with patch("atm.experiment.runner.resolve_spec", return_value=_MMLU_SPEC):
        result = await run_one(cfg)  # type: ignore[arg-type]

    # The run must complete successfully
    assert result.status == "completed", (
        f"Expected status='completed', got {result.status!r}"
    )

    # quality_score in RunResult metrics
    quality = result.metrics.get("quality_score")
    assert quality == pytest.approx(1.0), (
        f"Expected quality_score=1.0 in RunResult.metrics, got {quality!r}"
    )

    # Verify the value is persisted to PostgreSQL
    factory = create_session_factory(pg_engine_fast)
    async with session_scope(factory) as session:
        row_result = await session.execute(
            select(Run).where(Run.id == result.run_id)
        )
        run_row = row_result.scalar_one()

    assert run_row is not None, f"Run row not found for run_id={result.run_id}"
    assert run_row.quality_score is not None, "runs.quality_score should not be NULL"
    assert float(run_row.quality_score) == pytest.approx(1.0), (
        f"Expected runs.quality_score=1.0, got {run_row.quality_score!r}"
    )
