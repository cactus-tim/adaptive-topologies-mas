"""HumanRoleRouter Protocol and concrete implementations (Fixed, RuleBased, LLM)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel

from atm.core.types import HumanRole, Message, MessageKind, Phase

_log = logging.getLogger(__name__)


@runtime_checkable
class HumanRoleRouter(Protocol):
    """Duck-typing contract for all role router implementations."""

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the HumanRole for the current phase and state."""
        ...


class FixedRoleRouter:
    """Role router that always returns the same configured HumanRole."""

    def __init__(self, role: HumanRole) -> None:
        self.role = role

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the configured role regardless of phase/state."""
        return self.role


DEFAULT_ROLE_TABLE: dict[Phase, HumanRole] = {
    Phase.PLANNING: HumanRole.COORDINATOR,
    Phase.EXECUTION: HumanRole.PEER,
    Phase.VERIFICATION: HumanRole.REVIEWER,
    Phase.DONE: HumanRole.REVIEWER,
}


class RuleBasedRoleRouter:
    """Phase → HumanRole table-driven router; falls back to ``fallback`` for unknown phases."""

    def __init__(
        self,
        table: dict[Phase, HumanRole] | None = None,
        fallback: HumanRole = HumanRole.REVIEWER,
    ) -> None:
        self._table: dict[Phase, HumanRole] = (
            dict(table) if table is not None else dict(DEFAULT_ROLE_TABLE)
        )
        self._fallback = fallback

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the HumanRole for *phase*, or *fallback* if not in table."""
        return self._table.get(phase, self._fallback)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuleBasedRoleRouter:
        """Build from a YAML file mapping phase strings to role strings."""
        path = Path(path)
        raw: dict[str, str] = yaml.safe_load(path.read_text())

        table: dict[Phase, HumanRole] = {}
        for phase_str, role_str in raw.items():
            try:
                phase = Phase(phase_str)
            except ValueError:
                valid = [p.value for p in Phase]
                raise ValueError(
                    f"Unknown phase {phase_str!r} in {path}. Valid values: {valid}"
                ) from None
            try:
                role = HumanRole(role_str)
            except ValueError:
                valid = [r.value for r in HumanRole]
                raise ValueError(
                    f"Unknown role {role_str!r} for phase {phase_str!r} in {path}. Valid values: {valid}"
                ) from None
            table[phase] = role

        return cls(table=table)


class _RoleRouterDecision(BaseModel):
    """Pydantic model for validating the LLM's JSON response."""

    role: HumanRole
    reason: str = ""


class LLMRoleRouter:
    """LLM-based role router with JSON/Pydantic validation; falls back to rule router on error."""

    _DEFAULT_PROMPT = (
        "You are a role router for a multi-agent system. "
        "Current phase: {phase}. State summary: {state_summary}. "
        "Choose the most appropriate human role for this interaction. "
        'Respond with JSON only: {{"role": "<one of: coordinator,reviewer,judge,peer,monitor>", '
        '"reason": "<brief explanation>"}}.'
    )

    def __init__(self, llm: Any, fallback: HumanRoleRouter) -> None:
        self._llm = llm
        self._fallback = fallback

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Invoke LLM to select a HumanRole; fallback to rule router on any error."""
        state_summary = _build_state_summary(state)
        prompt = self._DEFAULT_PROMPT.format_map({"phase": phase, "state_summary": state_summary})

        messages = [
            Message(
                sender="role_router",
                kind=MessageKind.REQUEST,
                content=prompt,
            )
        ]

        try:
            response = await self._llm.ainvoke(messages, agent_id="role_router")
        except Exception as exc:
            _log.warning(
                "LLMRoleRouter: LLM call failed (%s), falling back to rule-based router.", exc
            )
            return await self._fallback.decide(phase, state)

        raw_text = response.text or ""

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            _log.warning(
                "LLMRoleRouter: malformed JSON from LLM (%s), "
                "raw_preview=%r (len=%d), falling back to rule-based router.",
                exc,
                raw_text[:80],
                len(raw_text),
            )
            return await self._fallback.decide(phase, state)

        try:
            decision = _RoleRouterDecision.model_validate(data)
        except Exception as exc:
            _log.warning(
                "LLMRoleRouter: Pydantic validation failed (%s), "
                "data=%r, falling back to rule-based router.",
                exc,
                data,
            )
            return await self._fallback.decide(phase, state)

        return decision.role


def _build_state_summary(state: dict[str, Any]) -> str:
    """Build a compact text summary of state fields relevant to role selection."""
    parts: list[str] = []
    iter_total = state.get("iter_total") or state.get("iteration")
    if iter_total is not None:
        parts.append(f"iter={iter_total}")
    signals = state.get("signals")
    if signals:
        parts.append(f"signals={list(signals.keys())}")
    return ", ".join(parts) if parts else "no state"
