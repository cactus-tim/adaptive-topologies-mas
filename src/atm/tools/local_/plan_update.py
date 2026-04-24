"""PlanUpdateTool — writes plan steps into shared.plan (NOT shared.signals).

Output contains 'state_update' dict; ToolNode in M5 merges it into GraphState.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class PlanStep(BaseModel):
    """A single plan step."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    status: Literal["open", "done", "in_progress", "blocked", "skipped"]


class PlanUpdateTool:
    """Replace the current plan in shared state.

    Args (ainvoke):
        steps: list of step dicts, each with ``id``, ``title``, ``status``.

    Output: ``{"state_update": {"shared": {"plan": <steps_list>}}}``

    IMPORTANT: emits to ``shared.plan``, NOT to ``shared.signals``.
    This is a plan-locked contract (M4 Step 9).
    M5 reducer uses replace semantics for shared.plan.
    """

    name: ClassVar[str] = "plan_update"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="plan_update",
        description=(
            "Replace the agent's plan in shared state. "
            "Emits to shared.plan, not shared.signals. "
            "M5 reducer uses replace (not append) semantics."
        ),
        parameters={
            "plan": (
                "list of {id: str, title: str, status: 'open'|'done'|'in_progress'|'blocked'|'skipped'}"
            ),
        },
        returns={"state_update": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Validate plan steps and produce the state_update payload.

        Returns ToolResult with ok=False if any step fails validation.
        """
        raw_steps: list[Any] = args.get("plan", [])
        validated: list[dict[str, Any]] = []

        for item in raw_steps:
            try:
                step = PlanStep.model_validate(item)
                validated.append(step.model_dump())
            except ValidationError as exc:
                return ToolResult(
                    call_id=uuid.uuid4(),
                    ok=False,
                    output=None,
                    error=f"Invalid plan step: {exc}",
                    latency_ms=0,
                )

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=True,
            output={"state_update": {"shared": {"plan": validated}}},
            error=None,
            latency_ms=0,
        )
