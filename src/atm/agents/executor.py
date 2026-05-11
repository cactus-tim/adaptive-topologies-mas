"""Executor agent subclass for ATM multi-agent framework.

The Executor is responsible for implementing the plan produced by the Planner:
running code, writing files, and reporting results. The Executor acts on
instructions and produces concrete outputs (code, files, stdout).
All prompt details come from the YAML config (conf/agents/executor.yaml).

Signal emission (M8.5):
    - After ``stuck_threshold`` (default 3) consecutive failed code_run tool
      calls, sets ``signals["stuck"] = True``.
    - After the first successful code_run (even if preceded by failures),
      sets ``signals["ready_for_verification"] = True``.
    Both signals are emitted additively — existing delta content is preserved.
"""

from __future__ import annotations

from typing import Any

from atm.agents.base import Agent
from atm.core.state import GraphState
from atm.phases.signals import (
    READY_FOR_VERIFICATION,
    STUCK,
    emit_signal,
)


class Executor(Agent):
    """Executor role: implements the plan by running code and writing files.

    Uses tools: code_run, file_write, file_read, calculator.
    Configured via conf/agents/executor.yaml (window_size=12).

    Tracks consecutive failed ``code_run`` calls across ``step()`` invocations
    via the agent scratchpad stored in SharedState.  The ``stuck_threshold``
    parameter controls how many consecutive failures trigger the ``stuck``
    signal (default: 3).

    Args:
        stuck_threshold: Number of consecutive failed code_run calls before
                         emitting ``signals["stuck"] = True``.  Defaults to 3.
        *args, **kwargs: Forwarded to ``Agent.__init__``.
    """

    def __init__(self, *args: Any, stuck_threshold: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stuck_threshold: int = stuck_threshold

    async def step(self, state: GraphState) -> dict[str, Any]:
        """Execute one executor step, then emit code_run signals as needed.

        Delegates to ``Agent.step()`` for the tool-calling loop.  After the
        loop completes, inspects ``delta["agents"][agent_id]["tool_results"]``
        to count consecutive failed code_run calls and detect success.

        Signal logic:
            - Counts *consecutive* failed code_run results (i.e. resets on
              any success).  The count is accumulated across ``step()`` calls
              by reading the previous count from
              ``state["agents"][agent_id]["scratchpad"]``
              via a dedicated "code_run_fail_streak" marker.
            - On reaching ``stuck_threshold`` consecutive failures: emits
              ``stuck = True``.
            - On any successful code_run: emits ``ready_for_verification = True``
              and resets the failure streak counter.

        Args:
            state: Current GraphState dict.

        Returns:
            Delta dict (same structure as ``Agent.step()``) with an updated
            ``shared`` key containing any newly emitted signals.
        """
        # Read the current consecutive-failure count from scratchpad markers
        prev_fail_streak: int = self._read_fail_streak(state)

        delta = await super().step(state)

        # Inspect tool results from this step only
        agents_delta: dict[str, Any] = dict(delta.get("agents") or {})
        self_delta: dict[str, Any] = dict(agents_delta.get(self.agent_id) or {})
        tool_results: list[Any] = list(self_delta.get("tool_results") or [])

        # Only examine code_run tool results; resolve call ids from tool_calls
        tool_calls: list[Any] = list(self_delta.get("tool_calls") or [])
        code_run_call_ids: set[str] = {
            str(tc.id)
            for tc in tool_calls
            if getattr(tc, "tool_name", "") == "code_run"
        }

        shared: dict[str, Any] = dict(state.get("shared") or {})
        # Merge any shared update already in delta (e.g. from future subclasses)
        if "shared" in delta:
            shared.update(delta["shared"])

        current_fail_streak = prev_fail_streak
        emitted_verification = False

        for tr in tool_results:
            call_id = str(getattr(tr, "call_id", ""))
            if call_id not in code_run_call_ids:
                # Not a code_run result — skip
                continue
            ok: bool = bool(getattr(tr, "ok", False))
            if ok:
                # Success: emit ready_for_verification and reset streak
                shared = emit_signal(shared, READY_FOR_VERIFICATION, True)
                current_fail_streak = 0
                emitted_verification = True
            else:
                # Failure: increment consecutive failure counter
                current_fail_streak += 1
                if current_fail_streak >= self._stuck_threshold:
                    shared = emit_signal(shared, STUCK, True)

        # Store updated fail streak as a scratchpad marker so it persists
        # across step() calls (SharedState.agents[agent_id].scratchpad is
        # append-only — we append a lightweight marker event).
        if not emitted_verification or current_fail_streak != prev_fail_streak:
            marker: dict[str, Any] = {
                "kind": "_code_run_fail_streak",
                "value": current_fail_streak,
            }
            # Append to delta scratchpad (agent delta has a list here)
            existing_scratchpad: list[dict[str, Any]] = list(
                self_delta.get("scratchpad") or []
            )
            existing_scratchpad.append(marker)
            self_delta = dict(self_delta)
            self_delta["scratchpad"] = existing_scratchpad
            agents_delta = dict(agents_delta)
            agents_delta[self.agent_id] = self_delta

        delta = dict(delta)
        delta["agents"] = agents_delta
        delta["shared"] = shared
        return delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_fail_streak(self, state: GraphState) -> int:
        """Read the last known consecutive-failure count from state scratchpad.

        Scans the *full* accumulated scratchpad for ``_code_run_fail_streak``
        markers, returning the value from the most recent one, or 0 if none.
        """
        agents_state: dict[str, Any] = dict(state.get("agents") or {})
        self_state: dict[str, Any] = dict(agents_state.get(self.agent_id) or {})
        scratchpad: list[dict[str, Any]] = list(self_state.get("scratchpad") or [])

        last_streak = 0
        for event in reversed(scratchpad):
            if event.get("kind") == "_code_run_fail_streak":
                try:
                    last_streak = int(event.get("value") or 0)
                except (TypeError, ValueError):
                    last_streak = 0
                break

        return last_streak
