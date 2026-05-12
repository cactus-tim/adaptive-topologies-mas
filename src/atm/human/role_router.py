"""HumanRoleRouter Protocol and concrete implementations.

Public API:
  - HumanRoleRouter      — @runtime_checkable Protocol; async decide(phase, state) -> HumanRole
  - FixedRoleRouter      — returns a constant HumanRole regardless of phase/state (back-compat default)
  - RuleBasedRoleRouter  — table-driven Phase → HumanRole lookup with YAML support
  - LLMRoleRouter        — LLM-based router with JSON/Pydantic validation and fallback to RuleBased

Pattern mirrors the Topology Protocol in atm.topology.base with async decide semantics.

Design notes (m9.2-plan.md §Approach):
  - _build_role_router returns None for role_router="fixed" (short-circuit, step 4.1).
    FixedRoleRouter is kept as an explicit class for test-doubles and explicit usage.
  - RuleBasedRoleRouter (step 2.1) and LLMRoleRouter (step 3.1) are added in later waves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel

from atm.core.types import HumanRole, Message, MessageKind, Phase

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HumanRoleRouter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HumanRoleRouter(Protocol):
    """Duck-typing contract for all role router implementations.

    Concrete classes do NOT inherit from this Protocol — structural
    subtyping (same pattern as Topology in atm.topology.base).

    Contract:
      async decide(phase, state) -> HumanRole

    Args:
        phase: Current execution phase of the run.
        state: SharedState-compatible dict (may be empty).

    Returns:
        The HumanRole the topology should use for this interaction.
    """

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the HumanRole for the current phase and state."""
        ...


# ---------------------------------------------------------------------------
# FixedRoleRouter — back-compat constant-role implementation
# ---------------------------------------------------------------------------


class FixedRoleRouter:
    """Role router that always returns the same configured HumanRole.

    This is the back-compat default (m9.2-plan.md §Architecture Decisions):
    when role_router="fixed", _build_role_router returns None (topology
    uses cfg.human.role directly). FixedRoleRouter is available for
    explicit use in tests and for ad-hoc wiring.

    Satisfies HumanRoleRouter Protocol via structural subtyping.

    Example::

        router = FixedRoleRouter(role=HumanRole.REVIEWER)
        active_role = await router.decide(Phase.EXECUTION, state)
        # active_role is always HumanRole.REVIEWER

    Args:
        role: The HumanRole returned on every decide() call.
    """

    def __init__(self, role: HumanRole) -> None:
        self.role = role

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the configured role, ignoring phase and state.

        Args:
            phase: Ignored — kept for protocol compliance.
            state: Ignored — kept for protocol compliance.

        Returns:
            The HumanRole passed at construction time.
        """
        return self.role


# ---------------------------------------------------------------------------
# RuleBasedRoleRouter — table-driven Phase → HumanRole implementation
# ---------------------------------------------------------------------------

#: Default mapping used when no custom table is provided.
DEFAULT_ROLE_TABLE: dict[Phase, HumanRole] = {
    Phase.PLANNING: HumanRole.COORDINATOR,
    Phase.EXECUTION: HumanRole.PEER,
    Phase.VERIFICATION: HumanRole.REVIEWER,
    Phase.DONE: HumanRole.REVIEWER,
}


class RuleBasedRoleRouter:
    """Role router driven by a static Phase → HumanRole lookup table.

    Uses DEFAULT_ROLE_TABLE when no custom table is provided. Unknown phases
    fall back to the configured ``fallback`` role.

    Satisfies HumanRoleRouter Protocol via structural subtyping.

    Example::

        router = RuleBasedRoleRouter()
        role = await router.decide(Phase.PLANNING, {})
        # role == HumanRole.COORDINATOR

        router = RuleBasedRoleRouter.from_yaml("conf/human/role_table.yaml")

    Args:
        table:    Optional override mapping. Values are normalised to
                  ``dict[Phase, HumanRole]`` at construction time.
        fallback: Role returned when *phase* is absent from *table*.
                  Defaults to ``HumanRole.REVIEWER``.
    """

    def __init__(
        self,
        table: dict[Phase, HumanRole] | None = None,
        fallback: HumanRole = HumanRole.REVIEWER,
    ) -> None:
        self._table: dict[Phase, HumanRole] = dict(table) if table is not None else dict(DEFAULT_ROLE_TABLE)
        self._fallback = fallback

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the HumanRole for *phase*, or *fallback* if not in table.

        Args:
            phase: Current execution phase of the run.
            state: SharedState-compatible dict (unused by this implementation).

        Returns:
            Mapped HumanRole or the fallback role.
        """
        return self._table.get(phase, self._fallback)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuleBasedRoleRouter:
        """Build a RuleBasedRoleRouter from a YAML mapping file.

        The YAML file must contain a flat mapping of phase strings to role
        strings, e.g.::

            planning: coordinator
            execution: peer
            verification: reviewer
            done: reviewer

        Args:
            path: Path to the YAML file.

        Returns:
            A new RuleBasedRoleRouter with the table read from the file.

        Raises:
            ValueError: If the file contains an unknown phase or role string.
        """
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


# ---------------------------------------------------------------------------
# LLMRoleRouter — LLM-based router with Pydantic validation and fallback
# ---------------------------------------------------------------------------


class _RoleRouterDecision(BaseModel):
    """Pydantic model for validating the LLM's JSON response."""

    role: HumanRole
    reason: str = ""


class LLMRoleRouter:
    """LLM-based role router with JSON parsing, Pydantic validation, and fallback.

    Decision flow:
    1. Build a prompt embedding current phase + minimal state digest.
    2. Call llm.ainvoke(messages, agent_id='role_router').
    3. Parse JSON from LLMResponse.text.
    4. Validate via _RoleRouterDecision (Pydantic): role must be a valid HumanRole.
    5. On any failure (JSON parse error, validation error, LLM exception):
       log WARNING and delegate to fallback.decide(phase, state).
    6. On success: return the validated HumanRole.

    Satisfies HumanRoleRouter Protocol via structural subtyping.

    Args:
        llm:      Any object exposing ``async ainvoke(messages, *, agent_id, ...) -> LLMResponse``
                  (e.g. LLMWrapper or FakeLLM).
        fallback: HumanRoleRouter used when LLM call or parsing fails.
    """

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
        """Invoke LLM to select a HumanRole; fallback to rule router on any error.

        Args:
            phase: Current execution phase of the run.
            state: SharedState-compatible dict; used to build a minimal digest.

        Returns:
            The HumanRole from the LLM response, or the fallback router's decision.
        """
        # Build a minimal state digest for the prompt
        state_summary = _build_state_summary(state)
        prompt = self._DEFAULT_PROMPT.format_map(
            {"phase": phase, "state_summary": state_summary}
        )

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

        # Parse JSON
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

        # Validate via Pydantic — catches unknown role strings and type errors
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_state_summary(state: dict[str, Any]) -> str:
    """Build a compact text summary of relevant state fields for the LLM prompt.

    Only includes fields that help role selection: iteration count and signals.
    """
    parts: list[str] = []
    iter_total = state.get("iter_total") or state.get("iteration")
    if iter_total is not None:
        parts.append(f"iter={iter_total}")
    signals = state.get("signals")
    if signals:
        parts.append(f"signals={list(signals.keys())}")
    return ", ".join(parts) if parts else "no state"
