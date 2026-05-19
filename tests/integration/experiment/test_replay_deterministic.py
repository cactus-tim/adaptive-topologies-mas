"""Integration test for `replay_one(mode='deterministic')` (m12-resume-replay G5).

Runs a fresh `run_one` with FakeLLM scripted fixtures, then replays the
recorded llm_calls.parquet through `replay_one`. Asserts that:

  - the new run row has ``replay_of = <original_run_id>``;
  - the new run lives in the SAME experiment;
  - final_answer is bit-identical;
  - iterations are identical.

Latency, started_at, and per-row UUIDs are excluded from equality (per dec).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

pytestmark = pytest.mark.requires_postgres

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_SMOKE_YAML = Path(__file__).parent.parent.parent.parent / "conf" / "experiments" / "smoke.yaml"


def _make_scripted_chain_cfg(*, pg_dsn: str, parquet_dir: str) -> Any:
    from atm.experiment.config import load_config

    if not _SMOKE_YAML.exists():
        pytest.skip(f"smoke.yaml not found at {_SMOKE_YAML}")

    overrides = [
        f"observability.pg_dsn={pg_dsn}",
        f"observability.parquet_dir={parquet_dir}",
        "topology.name=chain",
        "topology.max_iterations=4",
        "model.default=fake:scripted",
        f"model.fake_fixtures.planner={_FIXTURES_DIR / 'm6_chain_planner.yaml'}",
        f"model.fake_fixtures.executor={_FIXTURES_DIR / 'm6_chain_executor.yaml'}",
        f"model.fake_fixtures.critic={_FIXTURES_DIR / 'm6_chain_critic.yaml'}",
        "human.enabled=false",
    ]
    return load_config(_SMOKE_YAML, overrides=overrides)


@pytest.mark.asyncio
async def test_replay_deterministic_round_trip(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """Run once, replay deterministically, verify replay_of + answer parity."""
    import sqlalchemy as sa

    from atm.experiment.runner import replay_one, run_one
    from atm.storage.session import create_engine

    parquet_dir = tmp_path / "parquet"
    cfg = _make_scripted_chain_cfg(
        pg_dsn=ephemeral_pg_dsn,
        parquet_dir=str(parquet_dir),
    )

    original = await run_one(cfg)
    assert original.status == "completed", f"original did not complete: {original}"

    original_parquet = (
        parquet_dir
        / "experiments"
        / str(original.exp_id)
        / "runs"
        / str(original.run_id)
        / "llm_calls.parquet"
    )
    assert original_parquet.exists(), f"expected llm_calls.parquet at {original_parquet}"

    replayed = await replay_one(original.run_id, mode="deterministic")
    assert replayed.status == "completed", f"replay did not complete: {replayed}"

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    sa.text(
                        "SELECT replay_of, exp_id, iterations, host, process_pid "
                        "FROM runs WHERE id = :rid"
                    ).bindparams(rid=replayed.run_id)
                )
            ).fetchone()
    finally:
        await engine.dispose()

    assert row is not None
    replay_of, exp_id, iterations, host, pid = row
    assert replay_of == original.run_id
    assert exp_id == original.exp_id
    assert host is not None
    assert pid is not None

    assert replayed.final_answer == original.final_answer
    assert replayed.metrics.get("iters") == original.metrics.get("iters")
    if iterations is not None and original.metrics.get("iters") is not None:
        assert iterations == original.metrics["iters"]
