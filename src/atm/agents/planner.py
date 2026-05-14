"""Planner agent subclass for ATM multi-agent framework.

The Planner is responsible for decomposing the problem into explicit steps,
writing them via todo_write, and updating the plan via plan_update as new
information becomes available. The Planner does NOT write code directly.
All prompt details come from the YAML config (conf/agents/planner.yaml).

Signal emission (M8.5):
    After each ``step()`` completion (i.e. the planner has produced a plan
    or updated it), emits ``signals["ready_for_execution"] = True``.  This
    signal informs the TopologyRouter that the execution phase can begin.
"""

from __future__ import annotations

from typing import Any

from atm.agents.base import Agent
from atm.core.state import GraphState
from atm.phases.signals import READY_FOR_EXECUTION, emit_signal


class Planner(Agent):
    """Planner role: decomposes tasks into steps, manages the plan lifecycle.

    Uses tools: todo_write, plan_update.
    Configured via conf/agents/planner.yaml (window_size=15).

    Overrides ``step()`` to emit ``signals["ready_for_execution"] = True``
    after each planning step, signalling the routing layer that the plan is
    ready for the Executor.
    """

    async def step(self, state: GraphState) -> dict[str, Any]:
        """Execute one planner step and emit the ready_for_execution signal.

        Delegates to ``Agent.step()`` for the tool-calling loop, then emits
        ``ready_for_execution = True`` in the shared signals dict.

        Args:
            state: Current GraphState dict.

        Returns:
            Delta dict with an updated ``shared`` key containing
            ``signals["ready_for_execution"] = True``.
        """
        delta = await super().step(state)

        # Read current shared state, merge any update already in delta
        shared: dict[str, Any] = dict(state.get("shared") or {})
        if "shared" in delta:
            shared.update(delta["shared"])

        shared = emit_signal(shared, READY_FOR_EXECUTION, True)

        delta = dict(delta)
        delta["shared"] = shared
        return delta
