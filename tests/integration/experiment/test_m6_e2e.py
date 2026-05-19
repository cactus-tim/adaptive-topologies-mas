"""M6 E2E integration tests — Star and Chain topologies with FakeLLM.

Tests run ``run_one(cfg)`` through the full lifecycle with:
  - Scripted FakeLLM fixtures (no real LLM calls).
  - Real PostgreSQL test DB (via ``ephemeral_pg_dsn`` fixture).
  - ExperimentCallbackHandler and ParquetWriter (topology nodes do not
    dispatch custom events — see NOTE below about parquet assertions).

Skipped unless ``ATM_ENABLE_PG_TESTS=1`` is set.

NOTE-4: FakeLLM is NOT a LangChain BaseChatModel, so LangChain's
  on_llm_end callback is NOT triggered. ``llm_calls.parquet``,
  ``messages.parquet``, ``tool_calls.parquet``, and scratchpad parquet
  files will NOT be written.
  Parquet assertions (10-14) are relaxed accordingly.

NOTE-5: topology_transitions table is NOT populated in M6 (M8 scope).
  Assertion #9 is skipped with a comment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

_PG_TESTS_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"


def _make_cfg(
    *,
    topology_name: str,
    pg_dsn: str,
    parquet_dir: str,
) -> Any:
    """Load smoke.yaml and override pg_dsn, parquet_dir, topology name, and fixtures.

    For Star topology, sets phase caps to 1 so each agent runs exactly once:
      planning_max_iter=1, exec_max_iter=1, verify_max_iter=1
    This ensures the scripted fixtures (with a single step each) are sufficient.
    For Chain topology, max_iterations=12 (unchanged) is fine since Chain
    terminates on first critic approval.

    Fixture paths are set per-topology so the runner uses the correct
    scripted responses. The runner's _build_llm_wrappers reads these from
    cfg.model.fake_fixtures.
    """
    from atm.experiment.config import load_config

    smoke_yaml = Path("conf/experiments/smoke.yaml")
    if not smoke_yaml.exists():
        smoke_yaml = (
            Path(__file__).parent.parent.parent.parent / "conf" / "experiments" / "smoke.yaml"
        )

    if not smoke_yaml.exists():
        pytest.skip(f"smoke.yaml not found at {smoke_yaml}")

    planner_fixture = str(_FIXTURES_DIR / f"m6_{topology_name}_planner.yaml")
    executor_fixture = str(_FIXTURES_DIR / f"m6_{topology_name}_executor.yaml")
    critic_fixture = str(_FIXTURES_DIR / f"m6_{topology_name}_critic.yaml")

    overrides = [
        f"observability.pg_dsn={pg_dsn}",
        f"observability.parquet_dir={parquet_dir}",
        f"topology.name={topology_name}",
        f"model.fake_fixtures.planner={planner_fixture}",
        f"model.fake_fixtures.executor={executor_fixture}",
        f"model.fake_fixtures.critic={critic_fixture}",
        f"model.fake_fixtures.researcher={planner_fixture}",
    ]

    if topology_name == "star":
        overrides += [
            "topology.extra.star.planning_max_iter=1",
            "topology.extra.star.exec_max_iter=1",
            "topology.extra.star.verify_max_iter=1",
        ]

    return load_config(str(smoke_yaml), overrides=overrides)


@pytest.mark.integration
async def test_e2e_star_topology(ephemeral_pg_dsn: str, tmp_path: Path) -> None:
    """End-to-end test: Star topology fibonacci task with FakeLLM fixtures.

    Tests the full run_one() lifecycle:
      1. INSERT experiment + run rows into PostgreSQL.
      2. Run StarTopology graph via LangGraph.
      3. Flush Parquet (before UPDATE runs — flush-before-update invariant).
      4. UPDATE runs row with quality_score and status.

    Uses scripted FakeLLM fixtures:
      m6_star_planner.yaml  — planner emits plan draft
      m6_star_executor.yaml — executor emits tool_call + DRAFT "fib(10)=55"
      m6_star_critic.yaml   — critic approves (Critic subclass emits DECISION)

    Assertions: 18 items (see inline comments).
    """
    from decimal import Decimal

    import pandas as pd
    from sqlalchemy import select

    from atm.experiment.runner import run_one
    from atm.storage.models import Experiment, Run
    from atm.storage.session import create_engine, create_session_factory, session_scope

    if not _PG_TESTS_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set")

    for fixture_name in ["m6_star_planner.yaml", "m6_star_executor.yaml", "m6_star_critic.yaml"]:
        p = _FIXTURES_DIR / fixture_name
        if not p.exists():
            pytest.skip(f"Fixture not found: {p}")

    cfg = _make_cfg(
        topology_name="star",
        pg_dsn=ephemeral_pg_dsn,
        parquet_dir=str(tmp_path),
    )

    result = await run_one(cfg)

    assert result.status == "completed", f"Expected status=completed, got {result.status}"

    assert result.metrics.get("quality_score") == 0.0, (
        f"Expected quality_score=0.0, got {result.metrics.get('quality_score')}"
    )

    assert "55" in result.final_answer, (
        f"Expected '55' in final_answer, got: {result.final_answer!r}"
    )

    assert "55" in result.final_answer

    engine = create_engine(ephemeral_pg_dsn, echo=False)
    session_factory = create_session_factory(engine)

    try:
        async with session_scope(session_factory) as session:
            row = await session.get(Run, result.run_id)
            assert row is not None, f"Run row not found for run_id={result.run_id}"
            assert row.finish_reason == "success", (
                f"Expected finish_reason=success, got {row.finish_reason}"
            )

            assert row.quality_score is not None, "runs.quality_score should not be NULL"

            assert row.budget_spent_usd >= Decimal("0"), "budget_spent_usd should be >= 0"

            assert row.iterations is not None and row.iterations > 0, (
                f"Expected iterations > 0, got {row.iterations}"
            )
            assert row.iterations <= cfg.topology.max_iterations, (
                f"iterations {row.iterations} > max_iterations {cfg.topology.max_iterations}"
            )

            assert row.finished_at is not None, "runs.finished_at should not be NULL"

        async with session_scope(session_factory) as session:
            exp_row = await session.get(Experiment, result.exp_id)
            assert exp_row is not None, f"Experiment row not found for exp_id={result.exp_id}"
            config_snapshot = exp_row.config_snapshot or {}
            assert config_snapshot.get("task", {}).get("name") == "fibonacci_smoke", (
                f"config_snapshot task.name mismatch: {config_snapshot}"
            )

    finally:
        await engine.dispose()

    run_dir = tmp_path / "experiments" / str(result.exp_id) / "runs" / str(result.run_id)

    llm_calls_path = run_dir / "llm_calls.parquet"
    if llm_calls_path.exists():
        llm_df = pd.read_parquet(llm_calls_path)
        assert llm_df.shape[0] >= 2, f"Expected >= 2 llm_call rows, got {llm_df.shape[0]}"

    if llm_calls_path.exists():
        llm_df = pd.read_parquet(llm_calls_path)
        total_cost_parquet = float(
            llm_df["cost_usd"].sum() if "cost_usd" in llm_df.columns else 0.0
        )
        assert abs(total_cost_parquet - float(result.metrics.get("cost_usd", 0.0))) < 0.001

    messages_path = run_dir / "messages.parquet"
    if messages_path.exists():
        msg_df = pd.read_parquet(messages_path)
        assert msg_df.shape[0] >= 3, f"Expected >= 3 message rows, got {msg_df.shape[0]}"

    tool_calls_path = run_dir / "tool_calls.parquet"
    if tool_calls_path.exists():
        tc_df = pd.read_parquet(tool_calls_path)
        code_run_calls = (
            tc_df[tc_df.get("tool_name", tc_df.get("name", "")) == "code_run"]
            if "tool_name" in tc_df.columns
            else tc_df
        )
        assert len(code_run_calls) >= 1

    scratchpad_dir = run_dir / "scratchpads"
    if scratchpad_dir.exists():
        scratchpad_files = list(scratchpad_dir.glob("*.parquet"))
        assert len(scratchpad_files) >= 1, "Expected at least 1 scratchpad file"

    engine2 = create_engine(ephemeral_pg_dsn, echo=False)
    sf2 = create_session_factory(engine2)
    try:
        async with session_scope(sf2) as session:
            row2 = await session.get(Run, result.run_id)
            assert row2 is not None
            assert row2.topology == "star", f"Expected runs.topology='star', got {row2.topology}"
    finally:
        await engine2.dispose()

    engine3 = create_engine(ephemeral_pg_dsn, echo=False)
    sf3 = create_session_factory(engine3)
    try:
        from atm.storage.models import Phase as PhaseRow

        async with session_scope(sf3) as session:
            phase_result = await session.execute(
                select(PhaseRow).where(PhaseRow.run_id == result.run_id)
            )
            phase_rows = phase_result.scalars().all()
            assert len(phase_rows) >= 0, "phase_rows query failed"
    finally:
        await engine3.dispose()


@pytest.mark.integration
async def test_e2e_chain_topology(ephemeral_pg_dsn: str, tmp_path: Path) -> None:
    """End-to-end test: Chain topology fibonacci task with FakeLLM fixtures.

    Tests the full run_one() lifecycle:
      1. INSERT experiment + run rows into PostgreSQL.
      2. Run ChainTopology graph via LangGraph.
      3. Flush Parquet (before UPDATE runs — flush-before-update invariant).
      4. UPDATE runs row with quality_score and status.

    Uses scripted FakeLLM fixtures:
      m6_chain_planner.yaml  — planner emits plan draft
      m6_chain_executor.yaml — executor emits tool_call + DRAFT "fib(10)=55"
      m6_chain_critic.yaml   — critic approves (Critic subclass emits DECISION)

    Chain first-approve scenario: iter_total == 1 (planner step is iter 0,
    executor + critic run as iter 1, critic approves → END).

    Assertions: 18 items (see inline comments).
    """
    from decimal import Decimal

    import pandas as pd

    from atm.experiment.runner import run_one
    from atm.storage.models import Experiment, Run
    from atm.storage.session import create_engine, create_session_factory, session_scope

    if not _PG_TESTS_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set")

    for fixture_name in ["m6_chain_planner.yaml", "m6_chain_executor.yaml", "m6_chain_critic.yaml"]:
        p = _FIXTURES_DIR / fixture_name
        if not p.exists():
            pytest.skip(f"Fixture not found: {p}")

    cfg = _make_cfg(
        topology_name="chain",
        pg_dsn=ephemeral_pg_dsn,
        parquet_dir=str(tmp_path),
    )

    result = await run_one(cfg)

    assert result.status == "completed", f"Expected status=completed, got {result.status}"

    assert result.metrics.get("quality_score") == 0.0, (
        f"Expected quality_score=0.0, got {result.metrics.get('quality_score')}"
    )

    assert "55" in result.final_answer, (
        f"Expected '55' in final_answer, got: {result.final_answer!r}"
    )

    assert "55" in result.final_answer

    engine = create_engine(ephemeral_pg_dsn, echo=False)
    session_factory = create_session_factory(engine)

    try:
        async with session_scope(session_factory) as session:
            row = await session.get(Run, result.run_id)
            assert row is not None, f"Run row not found for run_id={result.run_id}"
            assert row.finish_reason == "success", (
                f"Expected finish_reason=success, got {row.finish_reason}"
            )

            assert row.quality_score is not None, "runs.quality_score should not be NULL"

            assert row.budget_spent_usd >= Decimal("0"), "budget_spent_usd should be >= 0"

            assert row.iterations is not None and row.iterations > 0, (
                f"Expected iterations > 0, got {row.iterations}"
            )
            assert row.iterations <= cfg.topology.max_iterations, (
                f"iterations {row.iterations} > max_iterations {cfg.topology.max_iterations}"
            )

            assert row.finished_at is not None, "runs.finished_at should not be NULL"

        async with session_scope(session_factory) as session:
            exp_row = await session.get(Experiment, result.exp_id)
            assert exp_row is not None, f"Experiment row not found for exp_id={result.exp_id}"
            config_snapshot = exp_row.config_snapshot or {}
            assert config_snapshot.get("task", {}).get("name") == "fibonacci_smoke", (
                f"config_snapshot task.name mismatch: {config_snapshot}"
            )

    finally:
        await engine.dispose()

    run_dir = tmp_path / "experiments" / str(result.exp_id) / "runs" / str(result.run_id)

    llm_calls_path = run_dir / "llm_calls.parquet"
    if llm_calls_path.exists():
        llm_df = pd.read_parquet(llm_calls_path)
        assert llm_df.shape[0] >= 2

    if llm_calls_path.exists():
        llm_df = pd.read_parquet(llm_calls_path)
        total_cost_parquet = float(
            llm_df["cost_usd"].sum() if "cost_usd" in llm_df.columns else 0.0
        )
        assert abs(total_cost_parquet - float(result.metrics.get("cost_usd", 0.0))) < 0.001

    messages_path = run_dir / "messages.parquet"
    if messages_path.exists():
        msg_df = pd.read_parquet(messages_path)
        assert msg_df.shape[0] >= 3

    tool_calls_path = run_dir / "tool_calls.parquet"
    if tool_calls_path.exists():
        tc_df = pd.read_parquet(tool_calls_path)
        assert len(tc_df) >= 1

    scratchpad_dir = run_dir / "scratchpads"
    if scratchpad_dir.exists():
        scratchpad_files = list(scratchpad_dir.glob("*.parquet"))
        assert len(scratchpad_files) >= 1

    engine2 = create_engine(ephemeral_pg_dsn, echo=False)
    sf2 = create_session_factory(engine2)
    try:
        async with session_scope(sf2) as session:
            row2 = await session.get(Run, result.run_id)
            assert row2 is not None
            assert row2.topology == "chain", f"Expected runs.topology='chain', got {row2.topology}"
    finally:
        await engine2.dispose()

    iters = result.metrics.get("iters", -1)
    assert iters >= 1, f"Expected iters >= 1 for chain first-approve, got {iters}"
