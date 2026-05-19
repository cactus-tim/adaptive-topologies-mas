"""E2E integration test for the evaluation framework (M11).

Verifies the full contract:
    run_one(cfg) → runs.quality_score == 1.0

This test:
  - Builds an ExperimentConfig pointing at a Chain topology with scripted FakeLLM.
  - Mocks ``resolve_spec`` (in runner module) to return a pre-built GSM8K TaskSpec
    with ``expected="42"`` and ``evaluator_key="gsm8k_numeric"``.
  - The scripted executor fixture emits "The answer is 42." as final_answer.
  - GSM8KMatcher extracts the last numeric token "42" → score=1.0.
  - Asserts that ``runs.quality_score == 1.0`` is persisted in PostgreSQL.

Skipped unless ``ATM_ENABLE_PG_TESTS=1`` is set.

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

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"

_GSM8K_SPEC = TaskSpec(
    id="gsm8k/e2e/0",
    type="reasoning",
    input="What is 6 * 7?",
    expected="42",
    evaluator_key="gsm8k_numeric",
)


def _make_cfg(pg_dsn: str, parquet_dir: str) -> object:
    """Build a minimal ExperimentConfig for the GSM8K e2e test.

    Uses chain topology + scripted FakeLLM (no real LLM calls).
    The ``evaluation.judge_model`` is set to ``fake:echo`` so that the judge
    wrapper in the runner does not trigger real API calls.
    The ``task.name`` is set to ``"gsm8k"`` (registered task); however, we
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
    executor_fixture = str(_FIXTURES_DIR / "m11_gsm8k_e2e_executor.yaml")
    critic_fixture = str(_FIXTURES_DIR / "m6_chain_critic.yaml")

    for path in [planner_fixture, executor_fixture, critic_fixture]:
        if not Path(path).exists():
            pytest.skip(f"Fixture not found: {path}")

    return ExperimentConfig(
        name="m11_e2e_gsm8k_quality",
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
        task=TaskCfg(name="gsm8k", input="", split="test", shuffle_seed=0),
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


@pytest.mark.integration
async def test_gsm8k_run_quality_score_is_one(
    pg_engine_fast: AsyncEngine,
    tmp_path: object,
) -> None:
    """Full e2e: run_one with GSM8K + scripted answer '42' → runs.quality_score == 1.0.

    Pipeline:
      1. Build ExperimentConfig (chain topology, scripted FakeLLM).
      2. Mock ``atm.experiment.runner.resolve_spec`` to return _GSM8K_SPEC
         (evaluator_key='gsm8k_numeric', expected='42') — avoids HF download.
      3. Call ``run_one(cfg)`` — topology produces final_answer "The answer is 42."
      4. GSM8KMatcher extracts last numeric token '42' → score=1.0 → persisted.
      5. Assert ``runs.quality_score == 1.0`` via SQLAlchemy read.
    """
    pg_dsn: str = pg_engine_fast.url.render_as_string(hide_password=False)
    if "+asyncpg" not in pg_dsn:
        pg_dsn = pg_dsn.replace("postgresql://", "postgresql+asyncpg://")

    cfg = _make_cfg(pg_dsn=pg_dsn, parquet_dir=str(tmp_path))

    from atm.experiment.runner import run_one

    with patch("atm.experiment.runner.resolve_spec", return_value=_GSM8K_SPEC):
        result = await run_one(cfg)  # type: ignore[arg-type]

    assert result.status == "completed", f"Expected status='completed', got {result.status!r}"

    quality = result.metrics.get("quality_score")
    assert quality == pytest.approx(1.0), (
        f"Expected quality_score=1.0 in RunResult.metrics, got {quality!r}"
    )

    factory = create_session_factory(pg_engine_fast)
    async with session_scope(factory) as session:
        row_result = await session.execute(select(Run).where(Run.id == result.run_id))
        run_row = row_result.scalar_one()

    assert run_row is not None, f"Run row not found for run_id={result.run_id}"
    assert run_row.quality_score is not None, "runs.quality_score should not be NULL"
    assert float(run_row.quality_score) == pytest.approx(1.0), (
        f"Expected runs.quality_score=1.0, got {run_row.quality_score!r}"
    )
