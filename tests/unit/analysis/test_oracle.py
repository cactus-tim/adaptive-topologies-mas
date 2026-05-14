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

# ---------------------------------------------------------------------------
# Row structure used by build_loo_from_rows:
#   {"task_id": str, "task_type": str, "topology": str, "quality_score": float}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rows_basic() -> list[dict]:
    """Three tasks of the same type with unambiguous winner."""
    return [
        # task_A: mesh wins clearly
        {"task_id": "HumanEval/10", "task_type": "programming", "topology": "mesh",   "quality_score": 0.90},
        {"task_id": "HumanEval/10", "task_type": "programming", "topology": "linear", "quality_score": 0.40},
        # task_B: mesh wins clearly
        {"task_id": "HumanEval/11", "task_type": "programming", "topology": "mesh",   "quality_score": 0.85},
        {"task_id": "HumanEval/11", "task_type": "programming", "topology": "linear", "quality_score": 0.45},
        # task_C: mesh wins clearly
        {"task_id": "HumanEval/12", "task_type": "programming", "topology": "mesh",   "quality_score": 0.88},
        {"task_id": "HumanEval/12", "task_type": "programming", "topology": "linear", "quality_score": 0.42},
    ]


@pytest.fixture
def rows_flip() -> list[dict]:
    """Fixture designed so that the LOO exclusion flips the winner for HumanEval/0.

    Without LOO (global aggregation over all 3 tasks):
      - mesh   mean across all = (5*0.95 + 1*0.40 + 1*0.40) / 7
                                 Note: only 1 mesh row each for /1 and /2.
      Simpler: per topology mean across all rows:
        mesh  rows: /0 × 5 = [0.95]*5, /1 × 1 = [0.40], /2 × 1 = [0.40]
                   total = 5*0.95 + 0.40 + 0.40 = 5.55 → mean = 5.55/7 ≈ 0.793
        linear rows: /0 × 1 = [0.50], /1 × 2 = [0.80, 0.80], /2 × 2 = [0.80, 0.80]
                   total = 0.50 + 0.80*4 = 3.70 → mean = 3.70/5 = 0.74
      → global winner: "mesh"

    WITH LOO for HumanEval/0 (exclude /0 from aggregation):
      Only /1 and /2 remain:
        mesh   rows: /1 × 1 = [0.40], /2 × 1 = [0.40] → mean = 0.40
        linear rows: /1 × 2 = [0.80, 0.80], /2 × 2 = [0.80, 0.80] → mean = 0.80
      → LOO winner for HumanEval/0: "linear"  ← flipped!

    Note: all task_ids use "HumanEval/N" (capital H, capital E).
    """
    rows: list[dict] = []
    # HumanEval/0: 5 mesh runs at 0.95, 1 linear run at 0.50
    for _ in range(5):
        rows.append(
            {"task_id": "HumanEval/0", "task_type": "programming", "topology": "mesh", "quality_score": 0.95}
        )
    rows.append(
        {"task_id": "HumanEval/0", "task_type": "programming", "topology": "linear", "quality_score": 0.50}
    )
    # HumanEval/1: 2 linear runs at 0.80, 1 mesh run at 0.40
    for _ in range(2):
        rows.append(
            {"task_id": "HumanEval/1", "task_type": "programming", "topology": "linear", "quality_score": 0.80}
        )
    rows.append(
        {"task_id": "HumanEval/1", "task_type": "programming", "topology": "mesh", "quality_score": 0.40}
    )
    # HumanEval/2: 2 linear runs at 0.80, 1 mesh run at 0.40
    for _ in range(2):
        rows.append(
            {"task_id": "HumanEval/2", "task_type": "programming", "topology": "linear", "quality_score": 0.80}
        )
    rows.append(
        {"task_id": "HumanEval/2", "task_type": "programming", "topology": "mesh", "quality_score": 0.40}
    )
    return rows


# ---------------------------------------------------------------------------
# 1. OracleTable serialization roundtrip
# ---------------------------------------------------------------------------


class TestOracleTableSerializationRoundtrip:
    """OracleTable can be serialized to a dict and deserialized back identically."""

    def test_oracle_table_serialization_roundtrip(self) -> None:
        from atm.analysis.oracle import OracleTable

        original = OracleTable(
            by_task_type={
                "programming": "mesh",
                "reasoning":   "linear",
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
        assert restored.by_task_id   == original.by_task_id
        assert restored.default_topology == original.default_topology


# ---------------------------------------------------------------------------
# 2. build_loo_from_rows — basic case
# ---------------------------------------------------------------------------


class TestBuildLooFromRowsBasic:
    """build_loo_from_rows returns the correct topology for a clear winner."""

    def test_build_loo_from_rows_basic(self, rows_basic: list[dict]) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        table: OracleTable = build_loo_from_rows(rows_basic)

        # All three tasks are "programming" type; mesh wins in every LOO fold
        # because the other two tasks also clearly prefer mesh.
        assert table.by_task_id["HumanEval/10"] == "mesh"
        assert table.by_task_id["HumanEval/11"] == "mesh"
        assert table.by_task_id["HumanEval/12"] == "mesh"

        # by_task_type should also reflect mesh as best for "programming"
        assert table.by_task_type["programming"] == "mesh"


# ---------------------------------------------------------------------------
# 3. build_loo_from_rows — exclusion flips winner (non-tautology test)
# ---------------------------------------------------------------------------


class TestBuildLooOracleExclusionFlipsWinner:
    """The LOO exclusion changes the recommended topology for HumanEval/0.

    Key assertion: WITHOUT LOO this would be "mesh" (global winner);
    WITH LOO it MUST be "linear".
    """

    def test_build_loo_oracle_exclusion_flips_winner(self, rows_flip: list[dict]) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        table: OracleTable = build_loo_from_rows(rows_flip)

        # Sanity: verify the global (non-LOO) winner would be mesh by confirming
        # that the other tasks (not excluded) prefer linear in LOO.
        # HumanEval/1 and HumanEval/2: when excluding them from their own LOO fold,
        # HumanEval/0's mesh rows dominate → they also flip toward mesh.
        # But for HumanEval/0 itself: excluding it leaves only /1 and /2 which
        # both clearly prefer linear.

        # WITHOUT LOO this would be "mesh" (global winner); WITH LOO it MUST be "linear"
        assert table.by_task_id["HumanEval/0"] == "linear", (
            "LOO exclusion of HumanEval/0 must flip winner from mesh (global) to linear"
        )

        # HumanEval/1 and HumanEval/2: when excluded, remaining set = /0 (5×mesh@0.95, 1×linear@0.50)
        # mesh mean among remaining = 5*0.95/5 = 0.95 (only /0 rows contribute mesh)
        # linear mean among remaining = 0.50/1 = 0.50
        # → LOO winner for /1 and /2 is "mesh"
        assert table.by_task_id["HumanEval/1"] == "mesh"
        assert table.by_task_id["HumanEval/2"] == "mesh"


# ---------------------------------------------------------------------------
# 4. Edge case: empty rows
# ---------------------------------------------------------------------------


class TestBuildLooEdgeCaseEmptyRows:
    """build_loo_from_rows with no rows returns an empty table with a default."""

    def test_build_loo_empty_rows_returns_default(self) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        table: OracleTable = build_loo_from_rows([])

        assert isinstance(table, OracleTable)
        assert table.by_task_id == {}
        assert table.by_task_type == {}
        # default_topology must be a non-empty string (typically "linear")
        assert isinstance(table.default_topology, str)
        assert len(table.default_topology) > 0


# ---------------------------------------------------------------------------
# 5. Edge case: single task (LOO leaves empty fold → fallback)
# ---------------------------------------------------------------------------


class TestBuildLooEdgeCaseSingleTask:
    """A task type with only one task_id falls back gracefully (empty LOO fold)."""

    def test_build_loo_single_task_fallback(self) -> None:
        from atm.analysis.oracle import OracleTable, build_loo_from_rows

        rows = [
            {"task_id": "HumanEval/99", "task_type": "reasoning", "topology": "mesh",   "quality_score": 0.90},
            {"task_id": "HumanEval/99", "task_type": "reasoning", "topology": "linear", "quality_score": 0.50},
        ]

        table: OracleTable = build_loo_from_rows(rows)

        # With only one task, LOO fold is empty — result must be the default topology
        # (not crash, not use the excluded task's own data).
        assert isinstance(table, OracleTable)
        assert "HumanEval/99" in table.by_task_id
        # The value must equal default_topology (since no other tasks in group)
        assert table.by_task_id["HumanEval/99"] == table.default_topology


# ---------------------------------------------------------------------------
# 6. Edge case: OracleTable default fallback for unknown task_id
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 7. OracleTable by_task_id lookup takes priority over by_task_type
# ---------------------------------------------------------------------------


class TestOracleTableByTaskIdPriority:
    """by_task_id lookup has higher priority than by_task_type."""

    def test_oracle_table_by_task_id_overrides_task_type(self) -> None:
        from atm.analysis.oracle import OracleTable

        table = OracleTable(
            by_task_type={"programming": "mesh"},
            by_task_id={"HumanEval/0": "debate"},
            default_topology="linear",
        )

        # HumanEval/0 is in by_task_id → "debate", not "mesh" from by_task_type
        result = table.lookup(task_id="HumanEval/0", task_type="programming")
        assert result == "debate"


# ---------------------------------------------------------------------------
# 8. OracleTable by_task_type fallback (task_id not found, task_type found)
# ---------------------------------------------------------------------------


class TestOracleTableByTaskTypeFallback:
    """by_task_type is consulted when task_id is not in by_task_id."""

    def test_oracle_table_by_task_type_fallback(self) -> None:
        from atm.analysis.oracle import OracleTable

        table = OracleTable(
            by_task_type={"programming": "mesh"},
            by_task_id={},
            default_topology="linear",
        )

        # Not in by_task_id → falls back to by_task_type["programming"] = "mesh"
        result = table.lookup(task_id="HumanEval/999", task_type="programming")
        assert result == "mesh"


# ---------------------------------------------------------------------------
# 9. Cross-contract: OracleTopologyRouter can consume a built OracleTable
# ---------------------------------------------------------------------------


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

        # OracleTopologyRouter accepts a dict; convert OracleTable to the expected format.
        # The router expects {"by_task_id": {...}, "by_task_type": {...}, "_default": "..."}.
        router_dict = table.to_router_dict()
        router = OracleTopologyRouter(oracle_table=router_dict)

        # State with task_id present → resolved via by_task_id
        state: dict = {
            "task_id": "HumanEval/0",
            "task_type": "programming",  # state["task_type"] must be populated
            "phase": "planning",
        }
        decision = asyncio.get_event_loop().run_until_complete(router.decide(state))  # type: ignore[arg-type]
        assert decision.decided_by == "oracle"
        # LOO says HumanEval/0 → "linear"
        assert decision.topology == "linear"
