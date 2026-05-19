"""Unit tests for atm.analysis.oracle — Step 1.3 (RED phase).

Tests for:
  - OracleTable dataclass serialization roundtrip
  - build_loo_from_rows: basic leave-one-out aggregation
  - build_loo_from_rows: exclusion flips the winner (non-tautology test)
  - Edge cases: empty rows, single task, missing task_id, default fallback
  - Cross-contract: OracleTopologyRouter can consume the built table

ALL tests must FAIL with ImportError/ModuleNotFoundError until
src/atm/analysis/oracle.py is implemented (Step 2.1).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def rows_basic() -> list[dict]:
    """Three tasks of the same type with unambiguous winner."""
    return [
        {
            "task_id": "HumanEval/10",
            "task_type": "programming",
            "topology": "mesh",
            "quality_score": 0.90,
        },
        {
            "task_id": "HumanEval/10",
            "task_type": "programming",
            "topology": "linear",
            "quality_score": 0.40,
        },
        {
            "task_id": "HumanEval/11",
            "task_type": "programming",
            "topology": "mesh",
            "quality_score": 0.85,
        },
        {
            "task_id": "HumanEval/11",
            "task_type": "programming",
            "topology": "linear",
            "quality_score": 0.45,
        },
        {
            "task_id": "HumanEval/12",
            "task_type": "programming",
            "topology": "mesh",
            "quality_score": 0.88,
        },
        {
            "task_id": "HumanEval/12",
            "task_type": "programming",
            "topology": "linear",
            "quality_score": 0.42,
        },
    ]


@pytest.fixture
def rows_flip() -> list[dict]:
    """Fixture designed so that the LOO exclusion flips the winner for HumanEval/0.

    Without LOO (global aggregation over all 3 tasks):
      - mesh   mean across all = (5*0.95 + 1*0.40 + 1*0.40) / 7
                                 Note: only 1 mesh row each for /1 and /2.
      Simpler: per topology mean across all rows:
        mesh  rows: /0 x 5 = [0.95]*5, /1 x 1 = [0.40], /2 x 1 = [0.40]
                   total = 5*0.95 + 0.40 + 0.40 = 5.55 → mean = 5.55/7 ≈ 0.793
        linear rows: /0 x 1 = [0.50], /1 x 2 = [0.80, 0.80], /2 x 2 = [0.80, 0.80]
                   total = 0.50 + 0.80*4 = 3.70 → mean = 3.70/5 = 0.74
      → global winner: "mesh"

    WITH LOO for HumanEval/0 (exclude /0 from aggregation):
      Only /1 and /2 remain:
        mesh   rows: /1 x 1 = [0.40], /2 x 1 = [0.40] → mean = 0.40
        linear rows: /1 x 2 = [0.80, 0.80], /2 x 2 = [0.80, 0.80] → mean = 0.80
      → LOO winner for HumanEval/0: "linear"  ← flipped!

    Note: all task_ids use "HumanEval/N" (capital H, capital E).
    """
    rows: list[dict] = []
    for _ in range(5):
        rows.append(
            {
                "task_id": "HumanEval/0",
                "task_type": "programming",
                "topology": "mesh",
                "quality_score": 0.95,
            }
        )
    rows.append(
        {
            "task_id": "HumanEval/0",
            "task_type": "programming",
            "topology": "linear",
            "quality_score": 0.50,
        }
    )
    for _ in range(2):
        rows.append(
            {
                "task_id": "HumanEval/1",
                "task_type": "programming",
                "topology": "linear",
                "quality_score": 0.80,
            }
        )
    rows.append(
        {
            "task_id": "HumanEval/1",
            "task_type": "programming",
            "topology": "mesh",
            "quality_score": 0.40,
        }
    )
    for _ in range(2):
        rows.append(
            {
                "task_id": "HumanEval/2",
                "task_type": "programming",
                "topology": "linear",
                "quality_score": 0.80,
            }
        )
    rows.append(
        {
            "task_id": "HumanEval/2",
            "task_type": "programming",
            "topology": "mesh",
            "quality_score": 0.40,
        }
    )
    return rows


class TestOracleTableSerializationRoundtrip:
    """OracleTable can be serialized to a dict and deserialized back identically."""

    def test_oracle_table_serialization_roundtrip(self) -> None:
        from atm.analysis.oracle import OracleTable

        original = OracleTable(
            by_task_type={
                "programming": "mesh",
                "reasoning": "linear",
            },
            by_task_id={
                "HumanEval/0": "linear",
                "HumanEval/1": "mesh",
            },
            default_topology="linear",
        )

        serialized: dict = original.to_dict()
        restored = OracleTable.from_dict(serialized)

        assert restored.by_task_type == original.by_task_type
        assert restored.by_task_id == original.by_task_id
        assert restored.default_topology == original.default_topology


class TestBuildLooFromRowsBasic:
    """build_loo_from_rows returns the correct topology for a clear winner."""

    def test_build_loo_from_rows_basic(self, rows_basic: list[dict]) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        table: OracleTable = build_loo_from_rows(rows_basic)

        assert table.by_task_id["HumanEval/10"] == "mesh"
        assert table.by_task_id["HumanEval/11"] == "mesh"
        assert table.by_task_id["HumanEval/12"] == "mesh"

        assert table.by_task_type["programming"] == "mesh"


class TestBuildOracleTop1PerTaskIsIndependent:
    """Per-task TOP-1 oracle: each task's winner depends ONLY on its own rows.

    HumanEval/0 has 5x mesh@0.95 + 1x linear@0.50 → mesh wins for /0.
    HumanEval/1 has 2x linear@0.80 + 1x mesh@0.40 → linear wins for /1.
    HumanEval/2 has 2x linear@0.80 + 1x mesh@0.40 → linear wins for /2.

    Under the old LOO algorithm /0 would have flipped to "linear" (because
    excluding /0 leaves /1 and /2 which prefer linear). The new top-1
    semantics deliberately do NOT exclude — each task is judged on its own
    empirical evidence, giving the true upper-bound ceiling for RQ2.
    """

    def test_build_oracle_top1_per_task_is_independent(self, rows_flip: list[dict]) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        table: OracleTable = build_loo_from_rows(rows_flip)

        assert table.by_task_id["HumanEval/0"] == "mesh", (
            "Top-1 must pick the best topology FOR THIS task — mesh wins HumanEval/0"
        )

        assert table.by_task_id["HumanEval/1"] == "linear"
        assert table.by_task_id["HumanEval/2"] == "linear"


class TestBuildLooEdgeCaseEmptyRows:
    """build_loo_from_rows with no rows returns an empty table with a default."""

    def test_build_loo_empty_rows_returns_default(self) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        table: OracleTable = build_loo_from_rows([])

        assert isinstance(table, OracleTable)
        assert table.by_task_id == {}
        assert table.by_task_type == {}
        assert isinstance(table.default_topology, str)
        assert len(table.default_topology) > 0


class TestBuildLooEdgeCaseSingleTask:
    """A single task uses its own rows for top-1 (no LOO exclusion)."""

    def test_build_loo_single_task_fallback(self) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        rows = [
            {
                "task_id": "HumanEval/99",
                "task_type": "reasoning",
                "topology": "mesh",
                "quality_score": 0.90,
            },
            {
                "task_id": "HumanEval/99",
                "task_type": "reasoning",
                "topology": "linear",
                "quality_score": 0.50,
            },
        ]

        table: OracleTable = build_loo_from_rows(rows)

        assert isinstance(table, OracleTable)
        assert table.by_task_id["HumanEval/99"] == "mesh"


class TestOracleTableDefaultFallback:
    """OracleTable.lookup returns default_topology for an unknown task_id."""

    def test_oracle_table_unknown_task_id_uses_default(self) -> None:
        from atm.analysis.oracle import OracleTable

        table = OracleTable(
            by_task_type={"programming": "mesh"},
            by_task_id={"HumanEval/0": "linear"},
            default_topology="linear",
        )

        result = table.lookup(task_id="unknown_task", task_type="unknown_type")
        assert result == "linear"


class TestOracleTableByTaskIdPriority:
    """by_task_id lookup has higher priority than by_task_type."""

    def test_oracle_table_by_task_id_overrides_task_type(self) -> None:
        from atm.analysis.oracle import OracleTable

        table = OracleTable(
            by_task_type={"programming": "mesh"},
            by_task_id={"HumanEval/0": "debate"},
            default_topology="linear",
        )

        result = table.lookup(task_id="HumanEval/0", task_type="programming")
        assert result == "debate"


class TestOracleTableByTaskTypeFallback:
    """by_task_type is consulted when task_id is not in by_task_id."""

    def test_oracle_table_by_task_type_fallback(self) -> None:
        from atm.analysis.oracle import OracleTable

        table = OracleTable(
            by_task_type={"programming": "mesh"},
            by_task_id={},
            default_topology="linear",
        )

        result = table.lookup(task_id="HumanEval/999", task_type="programming")
        assert result == "mesh"


class TestBuildLooOracleTopologyRouterContract:
    """OracleTable produced by build_loo_from_rows is consumable by OracleTopologyRouter.

    State["task_type"] must be populated for the by_task_type fallback path,
    because OracleTopologyRouter (line 392 stub) does not auto-infer task_type.
    """

    def test_build_loo_oracle_topology_router_contract(self, rows_flip: list[dict]) -> None:
        import asyncio

        from atm.analysis.oracle import OracleTable, build_loo_from_rows
        from atm.phases.topology_router import OracleTopologyRouter

        table: OracleTable = build_loo_from_rows(rows_flip)

        router_dict = table.to_router_dict()
        router = OracleTopologyRouter(oracle_table=router_dict)

        state: dict = {
            "task_id": "HumanEval/0",
            "task_type": "programming",
            "phase": "planning",
        }
        decision = asyncio.get_event_loop().run_until_complete(router.decide(state))  # type: ignore[arg-type]
        assert decision.decided_by == "oracle"
        assert decision.topology == "mesh"
