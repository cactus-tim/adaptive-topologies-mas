"""m10-merge-m8 Wave 8 — full-contract acceptance e2e for run_one().

Verifies that a single end-to-end run with HITL enabled populates ALL FOUR
post-run Run fields:

  - quality_score        (M11 — compute_quality)
  - cognitive_load_proxy (M9.2 — human_sim_cognitive_load_proxy)
  - model_version_snapshot (M11/G2 — provider fingerprint)
  - sandbox_image_digest (M11/G2 — DockerSandbox digest; None for SubprocessSandbox is acceptable)

This test will fail to RED until Wave 9 brings m8 topology HITL nodes. Marked
``requires_postgres``; gated on ATM_ENABLE_PG_TESTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

_PG_TESTS_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_SMOKE_YAML = Path(__file__).parent.parent.parent.parent / "conf" / "experiments" / "smoke.yaml"


pytestmark = pytest.mark.requires_postgres


def _make_hitl_chain_cfg(*, pg_dsn: str, parquet_dir: str) -> Any:
    """Load smoke.yaml and override for HITL Chain topology with fixed Reviewer role.

    Chain is the M9 reference HITL topology; FakeLLM fixtures from M9 e2e suite
    are reused via fake:scripted mode.
    """
    from atm.experiment.config import load_config

    if not _SMOKE_YAML.exists():
        pytest.skip(f"smoke.yaml not found at {_SMOKE_YAML}")

    planner_fixture = str(_FIXTURES_DIR / "m6_chain_planner.yaml")
    executor_fixture = str(_FIXTURES_DIR / "m6_chain_executor.yaml")
    critic_fixture = str(_FIXTURES_DIR / "m6_chain_critic.yaml")
    human_fixture = str(_FIXTURES_DIR / "m9_human_reviewer_approve.yaml")

    overrides = [
        f"observability.pg_dsn={pg_dsn}",
        f"observability.parquet_dir={parquet_dir}",
        "topology.name=chain",
        "topology.max_iterations=6",
        "model.default=fake:scripted",
        f"model.fake_fixtures.planner={planner_fixture}",
        f"model.fake_fixtures.executor={executor_fixture}",
        f"model.fake_fixtures.critic={critic_fixture}",
        f"model.fake_fixtures.human={human_fixture}",
        "human.enabled=true",
        "human.role=reviewer",
        "human.gateway=llm_simulated",
        "human.role_router=fixed",
        "human.timeout_policy=llm_fallback",
    ]
    return load_config(_SMOKE_YAML, overrides=overrides)


@pytest.mark.asyncio
async def test_run_one_populates_all_four_post_run_fields(
    ephemeral_pg_dsn: str,
    tmp_path: Path,
) -> None:
    """E2E acceptance — Chain + HITL + Reviewer fixed router → 4 Run fields filled."""
    import sqlalchemy as sa

    from atm.experiment.runner import run_one
    from atm.storage.session import create_engine

    cfg = _make_hitl_chain_cfg(
        pg_dsn=ephemeral_pg_dsn,
        parquet_dir=str(tmp_path / "parquet"),
    )
    result = await run_one(cfg)
    assert result.status == "completed", f"run did not complete: {result}"

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    sa.text(
                        "SELECT quality_score, cognitive_load_proxy,"
                        " model_version_snapshot, sandbox_image_digest, human_role"
                        " FROM runs WHERE id = :rid"
                    ).bindparams(rid=result.run_id)
                )
            ).fetchone()
    finally:
        await engine.dispose()

    assert row is not None, "Run row not found in PG"

    qs, cog, mvs, sdigest, hrole = row

    assert qs is not None and isinstance(qs, float)

    assert cog is not None and isinstance(cog, float)
    assert cog >= 0.0

    assert mvs is not None

    assert sdigest is None or isinstance(sdigest, str)

    assert hrole is None or isinstance(hrole, str)
