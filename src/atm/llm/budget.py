"""Three-tier async budget gating for LLM calls.

BudgetTracker enforces per-call, per-run, and per-experiment USD ceilings using
asyncio.Lock for concurrency safety. A callback (sync or async) can be registered
to receive BudgetSignal notifications on warn (≥warn_fraction of limit) and exceed.

Architecture: §4.2 of arch.md.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from atm.core.errors import BudgetExceededError


class BudgetLevel(StrEnum):
    """The scope at which a budget ceiling is applied."""

    CALL = "call"
    RUN = "run"
    EXPERIMENT = "experiment"


class BudgetSignal(BaseModel):
    """A budget notification — either a warning (≥80% spent) or an exceed event."""

    model_config = ConfigDict(frozen=True)

    level: BudgetLevel
    kind: Literal["warn", "exceed"]
    value_usd: float
    limit_usd: float
    at: datetime


# Callback type alias: accepts sync or async callables
_OnEvent = Callable[[BudgetSignal], "Awaitable[None] | None"]


class BudgetTracker:
    """Three-tier budget tracker with asyncio.Lock concurrency safety.

    Maintains running totals for call, run, and experiment budget levels.
    On each ``check()`` call the current total plus the proposed cost is
    compared against the configured ceiling; if it would exceed, a
    ``BudgetExceededError`` is raised *before* the cost is recorded.

    On each ``record()`` call the cost is added to the running total and
    the warn / exceed thresholds are evaluated; the optional ``on_event``
    callback is invoked when a threshold is crossed.

    Args:
        per_call_usd:       Maximum USD allowed per individual LLM call.
        per_run_usd:        Maximum USD cumulative across calls in one run.
        per_experiment_usd: Maximum USD cumulative across all runs in an experiment.
        warn_fraction:      Fraction of limit at which a 'warn' event fires (default 0.8).
        on_event:           Optional callback invoked with a ``BudgetSignal`` on warn/exceed.
                            May be synchronous or asynchronous.
    """

    def __init__(
        self,
        per_call_usd: float,
        per_run_usd: float,
        per_experiment_usd: float,
        *,
        warn_fraction: float = 0.8,
        on_event: _OnEvent | None = None,
    ) -> None:
        self._limits: dict[BudgetLevel, float] = {
            BudgetLevel.CALL: per_call_usd,
            BudgetLevel.RUN: per_run_usd,
            BudgetLevel.EXPERIMENT: per_experiment_usd,
        }
        self._warn_fraction = warn_fraction
        self._on_event = on_event
        self._totals: dict[BudgetLevel, float] = {
            BudgetLevel.CALL: 0.0,
            BudgetLevel.RUN: 0.0,
            BudgetLevel.EXPERIMENT: 0.0,
        }
        # Track which levels have already fired a warn event to avoid re-firing.
        self._warned: set[BudgetLevel] = set()
        self._lock = asyncio.Lock()

    @property
    def totals(self) -> dict[BudgetLevel, float]:
        """Current accumulated totals keyed by BudgetLevel (read-only snapshot)."""
        return dict(self._totals)

    async def check(self, cost_usd: float, *, level: BudgetLevel) -> None:
        """Raise BudgetExceededError if adding cost_usd would exceed the level's limit.

        This method does NOT record the cost — call ``record()`` after a successful
        LLM call. The check considers the *projected* total (current + cost_usd).

        Also fires an 'exceed' event via on_event if a callback is registered.

        Args:
            cost_usd: The projected cost of the upcoming LLM call in USD.
            level:    The budget level to check against.

        Raises:
            BudgetExceededError: If current total + cost_usd > configured limit.
        """
        async with self._lock:
            limit = self._limits[level]
            projected = self._totals[level] + cost_usd
            if projected > limit:
                await self._fire_event(level, "exceed", projected, limit)
                raise BudgetExceededError(
                    level=str(level),
                    limit_usd=limit,
                    spent_usd=projected,
                )

    async def record(
        self,
        cost_usd: float,
        *,
        level: BudgetLevel = BudgetLevel.RUN,
    ) -> None:
        """Accumulate cost_usd into the given budget level's running total.

        Fires a 'warn' event if the new total crosses the warn threshold (warn_fraction
        of the limit) and a warn has not yet been fired for this level.

        Args:
            cost_usd: The actual cost of the completed LLM call in USD.
            level:    The budget level to record against (default: RUN).
        """
        async with self._lock:
            self._totals[level] += cost_usd
            new_total = self._totals[level]
            limit = self._limits[level]
            warn_threshold = limit * self._warn_fraction

            if new_total >= warn_threshold and level not in self._warned:
                self._warned.add(level)
                await self._fire_event(level, "warn", new_total, limit)

    async def _fire_event(
        self,
        level: BudgetLevel,
        kind: Literal["warn", "exceed"],
        value_usd: float,
        limit_usd: float,
    ) -> None:
        """Construct and dispatch a BudgetSignal to the registered callback.

        Handles both synchronous and asynchronous callbacks transparently.
        Must be called while holding self._lock (or be otherwise safe).
        """
        if self._on_event is None:
            return

        event = BudgetSignal(
            level=level,
            kind=kind,
            value_usd=value_usd,
            limit_usd=limit_usd,
            at=datetime.now(UTC),
        )
        result = self._on_event(event)
        if inspect.isawaitable(result):
            await result
