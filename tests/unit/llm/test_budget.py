"""Unit tests for BudgetTracker — three-tier async budget gating.

Tests cover:
1. Per-call cutoff (check raises BudgetExceededError when cost > per_call_usd)
2. Per-run cutoff (accumulated small calls cross threshold, next check raises)
3. Per-experiment cutoff (accumulated across runs, next check raises)
4. record() accumulates correctly across all three levels
5. Concurrent record() via asyncio.gather — asyncio.Lock correctness
6. Budget event callback fired on 'warn' and 'exceed'
"""

from __future__ import annotations

import asyncio

import pytest

from atm.core.errors import BudgetExceededError
from atm.llm.budget import BudgetEvent, BudgetLevel, BudgetTracker

# ---------------------------------------------------------------------------
# Test 1: Per-call cutoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_call_cutoff_raises() -> None:
    """check(0.20, level='call') when per_call_usd=0.10 raises BudgetExceededError."""
    tracker = BudgetTracker(
        per_call_usd=0.10,
        per_run_usd=10.0,
        per_experiment_usd=100.0,
    )
    with pytest.raises(BudgetExceededError) as exc_info:
        await tracker.check(0.20, level=BudgetLevel.CALL)

    err = exc_info.value
    assert err.level == "call"
    assert err.limit_usd == pytest.approx(0.10)
    assert err.spent_usd == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Test 2: Per-run cutoff via accumulated calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_run_cutoff_accumulates_and_raises() -> None:
    """Multiple small calls accumulate; on crossing per_run_usd, next check raises."""
    tracker = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=0.25,
        per_experiment_usd=100.0,
    )
    # Record 3 calls of 0.10 each = 0.30 total run spend
    await tracker.record(0.10, level=BudgetLevel.RUN)
    await tracker.record(0.10, level=BudgetLevel.RUN)
    await tracker.record(0.10, level=BudgetLevel.RUN)

    # Now per_run total is 0.30 > 0.25, check must raise
    with pytest.raises(BudgetExceededError) as exc_info:
        await tracker.check(0.01, level=BudgetLevel.RUN)

    err = exc_info.value
    assert err.level == "run"
    assert err.limit_usd == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Test 3: Per-experiment cutoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_experiment_cutoff_raises() -> None:
    """Accumulated experiment spend crossing per_experiment_usd raises BudgetExceededError."""
    tracker = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=100.0,
        per_experiment_usd=0.50,
    )
    # Record costs that push experiment total over threshold
    await tracker.record(0.30, level=BudgetLevel.EXPERIMENT)
    await tracker.record(0.30, level=BudgetLevel.EXPERIMENT)

    # Experiment total is 0.60 > 0.50, next check at experiment level raises
    with pytest.raises(BudgetExceededError) as exc_info:
        await tracker.check(0.01, level=BudgetLevel.EXPERIMENT)

    err = exc_info.value
    assert err.level == "experiment"
    assert err.limit_usd == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Test 4: record() accumulates across all three levels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_accumulates_all_levels() -> None:
    """record(cost_usd, level) accumulates correctly and totals reports all three levels."""
    tracker = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=10.0,
        per_experiment_usd=100.0,
    )
    await tracker.record(0.05, level=BudgetLevel.CALL)
    await tracker.record(0.10, level=BudgetLevel.RUN)
    await tracker.record(0.20, level=BudgetLevel.EXPERIMENT)
    await tracker.record(0.30, level=BudgetLevel.RUN)

    totals = tracker.totals
    assert totals[BudgetLevel.CALL] == pytest.approx(0.05)
    assert totals[BudgetLevel.RUN] == pytest.approx(0.40)
    assert totals[BudgetLevel.EXPERIMENT] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Test 5: Concurrent record() — asyncio.Lock correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_record_sums_correctly() -> None:
    """100 concurrent record(0.001) calls via asyncio.gather sum to exactly 0.1."""
    tracker = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=1.0,
        per_experiment_usd=100.0,
    )
    await asyncio.gather(*[tracker.record(0.001, level=BudgetLevel.RUN) for _ in range(100)])

    assert tracker.totals[BudgetLevel.RUN] == pytest.approx(0.1, abs=1e-10)


# ---------------------------------------------------------------------------
# Test 6: Budget event callback fired on warn and exceed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_callback_fired_on_warn_and_exceed() -> None:
    """on_event callback is called with BudgetEvent for warn (>=80%) and exceed."""
    events: list[BudgetEvent] = []

    async def collector(event: BudgetEvent) -> None:
        events.append(event)

    tracker = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=1.0,
        per_experiment_usd=100.0,
        warn_fraction=0.8,
        on_event=collector,
    )

    # Record 85% of per_run_usd to trigger a warn event
    await tracker.record(0.85, level=BudgetLevel.RUN)

    # The warn event should have been fired
    warn_events = [e for e in events if e.kind == "warn"]
    assert len(warn_events) >= 1
    warn = warn_events[0]
    assert warn.level == BudgetLevel.RUN
    assert warn.limit_usd == pytest.approx(1.0)
    assert warn.value_usd == pytest.approx(0.85)

    # Now exceed the budget to trigger an exceed event
    with pytest.raises(BudgetExceededError):
        await tracker.check(0.50, level=BudgetLevel.RUN)

    exceed_events = [e for e in events if e.kind == "exceed"]
    assert len(exceed_events) >= 1
    exceed = exceed_events[0]
    assert exceed.level == BudgetLevel.RUN
    assert exceed.limit_usd == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Extra: sync callback also works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_callback_also_accepted() -> None:
    """on_event accepts a synchronous callback (Callable[[BudgetEvent], None])."""
    events: list[BudgetEvent] = []

    def sync_collector(event: BudgetEvent) -> None:
        events.append(event)

    tracker = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=0.50,
        per_experiment_usd=100.0,
        warn_fraction=0.8,
        on_event=sync_collector,
    )

    # Trigger warn by reaching 85% of 0.50 = 0.425
    await tracker.record(0.45, level=BudgetLevel.RUN)
    warn_events = [e for e in events if e.kind == "warn"]
    assert len(warn_events) >= 1
