"""Unit tests for atm.experiment.estimator — heuristic path (no DB).

Covers:
  1. Empty configs list returns an empty GridEstimate.
  2. Single config + session_factory=None → heuristic path; cost matches
     pricing.estimate(prompt, completion).
  3. use_historical=False with a non-None session_factory still uses the
     heuristic path (DB is skipped entirely).
  4. Per-topology aggregation sums cell costs per topology key.
  5. source label is "heuristic" when no historical data is present.
  6. Unknown model in pricing table falls back to 0.0 (graceful).
  7. Token counts follow the documented formula
     (calls_per_iter * heuristic_tokens_per_call * max_iterations).
  8. Multiple cells: total_cost_usd == sum of cell costs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atm.experiment.config import EstimateCfg, ExperimentConfig
from atm.experiment.estimator import CellEstimate, GridEstimate, estimate_grid
from atm.experiment.loader import load_config
from atm.llm.pricing import ModelPricing, Pricing

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "experiment"
VALID_MINIMAL = FIXTURES_DIR / "valid_minimal.yaml"


def _make_pricing() -> Pricing:
    return Pricing(
        version=1,
        models={
            "fake:scripted": ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
            "openai:gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
        },
    )


def _load_cfg(**overrides: str) -> ExperimentConfig:
    """Load the minimal-valid fixture with optional dotlist overrides."""
    return load_config(str(VALID_MINIMAL), overrides=list(overrides.values()) or None)


def test_empty_configs_returns_empty_estimate() -> None:
    est_cfg = EstimateCfg()
    pricing = _make_pricing()
    result = asyncio.run(
        estimate_grid(configs=[], session_factory=None, pricing=pricing, cfg=est_cfg)
    )
    assert isinstance(result, GridEstimate)
    assert result.cells == []
    assert result.total_cost_usd == 0.0
    assert result.total_input_tokens == 0
    assert result.total_output_tokens == 0
    assert result.per_topology == {}


def test_heuristic_cost_matches_pricing_estimate() -> None:
    cfg = _load_cfg()
    est_cfg = EstimateCfg(heuristic_tokens_per_call=1000, calls_per_iter=4)
    pricing = _make_pricing()

    expected_in = 1000 * 4 * cfg.topology.max_iterations
    expected_out = expected_in // 3
    expected_cost = pricing.estimate(
        "fake:scripted",
        prompt_tokens=expected_in,
        completion_tokens=expected_out,
    )

    result = asyncio.run(
        estimate_grid(
            configs=[cfg],
            session_factory=None,
            pricing=pricing,
            cfg=est_cfg,
        )
    )
    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell.source == "heuristic"
    assert cell.est_input_tokens == expected_in
    assert cell.est_output_tokens == expected_out
    assert cell.est_cost_usd == pytest.approx(expected_cost)
    assert result.total_cost_usd == pytest.approx(expected_cost)


def test_use_historical_false_skips_db() -> None:
    cfg = _load_cfg()
    est_cfg = EstimateCfg(use_historical=False)
    pricing = _make_pricing()

    class _SentinelFactory:
        def __call__(self) -> object:  # pragma: no cover — must not be called
            raise AssertionError("session_factory should NOT be invoked when use_historical=False")

    result = asyncio.run(
        estimate_grid(
            configs=[cfg],
            session_factory=_SentinelFactory(),  # type: ignore[arg-type]
            pricing=pricing,
            cfg=est_cfg,
        )
    )
    assert len(result.cells) == 1
    assert result.cells[0].source == "heuristic"


def test_per_topology_aggregation() -> None:
    cfg_chain = _load_cfg()
    cfg_star = cfg_chain.model_copy(
        update={"topology": cfg_chain.topology.model_copy(update={"name": "star"})}
    )
    cfg_star2 = cfg_star.model_copy(update={"seed": 99})
    est_cfg = EstimateCfg()
    pricing = _make_pricing()

    result = asyncio.run(
        estimate_grid(
            configs=[cfg_chain, cfg_star, cfg_star2],
            session_factory=None,
            pricing=pricing,
            cfg=est_cfg,
        )
    )
    assert len(result.cells) == 3
    assert set(result.per_topology.keys()) == {"chain", "star"}

    chain_cost = sum(c.est_cost_usd for c in result.cells if c.topology == "chain")
    star_cost = sum(c.est_cost_usd for c in result.cells if c.topology == "star")
    assert result.per_topology["chain"] == pytest.approx(chain_cost)
    assert result.per_topology["star"] == pytest.approx(star_cost)
    assert result.total_cost_usd == pytest.approx(chain_cost + star_cost)


def test_source_label_heuristic_when_no_history() -> None:
    cfg = _load_cfg()
    est_cfg = EstimateCfg()
    pricing = _make_pricing()

    result = asyncio.run(
        estimate_grid(
            configs=[cfg],
            session_factory=None,
            pricing=pricing,
            cfg=est_cfg,
        )
    )
    assert all(c.source == "heuristic" for c in result.cells)


def test_unknown_model_falls_back_to_zero_cost() -> None:
    cfg = _load_cfg()
    est_cfg = EstimateCfg()
    pricing = Pricing(version=1, models={})

    result = asyncio.run(
        estimate_grid(
            configs=[cfg],
            session_factory=None,
            pricing=pricing,
            cfg=est_cfg,
        )
    )
    assert len(result.cells) == 1
    assert result.cells[0].est_cost_usd == 0.0
    assert result.cells[0].source == "heuristic"


def test_token_formula_matches_documented_shape() -> None:
    cfg = _load_cfg()
    est_cfg = EstimateCfg(heuristic_tokens_per_call=2000, calls_per_iter=3)
    pricing = _make_pricing()

    result = asyncio.run(
        estimate_grid(
            configs=[cfg],
            session_factory=None,
            pricing=pricing,
            cfg=est_cfg,
        )
    )
    cell = result.cells[0]
    expected_in = 2000 * 3 * cfg.topology.max_iterations
    assert cell.est_input_tokens == expected_in
    assert cell.est_output_tokens == expected_in // 3


def test_cell_estimate_is_frozen() -> None:
    cell = CellEstimate(
        topology="star",
        task="humaneval",
        seed=42,
        est_input_tokens=1000,
        est_output_tokens=300,
        est_cost_usd=0.01,
        source="heuristic",
    )
    with pytest.raises((AttributeError, TypeError)):
        cell.seed = 1  # type: ignore[misc]
