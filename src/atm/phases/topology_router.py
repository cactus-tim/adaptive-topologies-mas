"""TopologyRouter — three implementations of the runtime topology selection protocol.

Implements arch.md §8bis.2:
  - TopologyRouter   — Protocol (decide method, async)
  - RuleBasedTopologyRouter — deterministic table (phase x signals) -> topology
  - LLMTopologyRouter       — LLM-based selection with JSON validation + fallback
  - OracleTopologyRouter    — reads oracle_table.json keyed by task_id / task_type

Topology names (5 registered, per arch.md §7.7):
  "linear"  | "supervisor" | "mesh" | "debate" | "hierarchical"

Rule priority (RuleBasedTopologyRouter, documented per §7.7 table rows):
  Within a phase, rules are evaluated in priority order (first match wins).
  For execution phase:
    1. stuck=True AND iter_within_topology <= 5 → mesh
    2. rejected_count >= 3                      → debate
  For verification phase:
    1. needs_revision=True (from Critic)        → linear
  All other cases → keep current topology (no switch).

Architecture notes:
  - TopologyRouter.decide() takes SharedState (not full GraphState) per m8-routing TDD contract.
    The PhaseRouter (manager.py) takes GraphState; TopologyRouter takes SharedState to keep
    the L2 layer simpler and avoid type-assertion boilerplate in callers.
  - LLM messages use Message(sender='topology_router', kind=MessageKind.REQUEST, content=prompt)
    per arch.md §8bis.1 pattern.
  - LLM logs are sanitized: preview <= 80 chars + len, dict logged as keys only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from atm.core.state import SharedState
from atm.core.types import Message, MessageKind, Phase, TopologyDecision

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registered topology names (arch.md §7.7)
# ---------------------------------------------------------------------------

_VALID_TOPOLOGIES: frozenset[str] = frozenset(
    {"linear", "supervisor", "mesh", "debate", "hierarchical"}
)

# Default topology if no rule fires
_DEFAULT_TOPOLOGY: str = "linear"


def _extract_topology_from_hint(hint: str) -> str | None:
    """Return a topology name mentioned in *hint*, or None if no valid name found.

    Used by RuleBasedTopologyRouter to consume signals['human_advisor_hint']
    in advisory HITL mode.  The hint is a free-form string left by the human
    reviewer.  We do a case-insensitive whole-word scan for any name in
    _VALID_TOPOLOGIES; if exactly one valid name appears, that wins.
    Multiple/ambiguous mentions return None (let normal rules apply).

    Special tokens such as "override_applied:mesh" or
    "override_blocked_by_guards:debate" — written by adaptive's
    human_advisor_node itself — are intentionally NOT consumed here because
    the override path already mutated the topology decision via
    _topo_dec_slot.  We skip strings that begin with "override_".
    """
    if not hint:
        return None
    if hint.lower().startswith("override_"):
        return None
    tokens = {tok.strip(",.:;!?\"'()[]{}").lower() for tok in hint.split()}
    matches = tokens & _VALID_TOPOLOGIES
    if len(matches) == 1:
        return next(iter(matches))
    return None


# ---------------------------------------------------------------------------
# TopologyRouter — Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TopologyRouter(Protocol):
    """Protocol for all TopologyRouter implementations.

    Contract (arch.md §8bis.1):
    - decide() is called exactly once per meta-graph tick, AFTER PhaseRouter.
    - Input state already contains the updated shared.phase (if phase advanced).
    - Returns TopologyDecision; produces 0 LLM calls for 'rule'/'oracle'.
    - Must NOT mutate state or produce side-effects (DB writes).
      Recording is done by TransitionGate via callback.
    """

    async def decide(self, state: SharedState) -> TopologyDecision:
        """Decide which topology to use on the next meta-graph tick.

        Args:
            state: SharedState after current subgraph run (phase already updated).

        Returns:
            TopologyDecision with topology name and decided_by field.
        """
        ...


# ---------------------------------------------------------------------------
# RuleBasedTopologyRouter
# ---------------------------------------------------------------------------


class RuleBasedTopologyRouter:
    """Deterministic rule-based topology router.

    Decision logic follows the canonical table in arch.md §7.7:

    Rule priority (first match wins within each phase):

    Phase PLANNING:
        - No topology-switching rules (phase FSM handles advancement).
        - Stay with current or default topology.

    Phase EXECUTION:
        1. stuck=True AND iter_within_topology <= 5 → mesh (brainstorm)
        2. rejected_count >= 3                      → debate
        3. (no rule fires)                          → keep current topology

    Phase VERIFICATION:
        1. needs_revision=True (from Critic)        → linear (rewrite within phase)
        2. (no rule fires)                          → keep current topology

    Phase DONE:
        - Terminal: always stay with current topology (no switches).

    If no topology is currently active (active_topology is None), uses _DEFAULT_TOPOLOGY.
    All decisions have decided_by='rule'.
    """

    _REJECT_THRESHOLD: int = 3
    _STUCK_DWELL_CAP: int = 5

    async def decide(self, state: SharedState) -> TopologyDecision:
        """Evaluate rule table and return a TopologyDecision.

        Args:
            state: SharedState containing phase, signals, active_topology, counters.

        Returns:
            TopologyDecision with decided_by='rule'.
        """
        current_phase: Phase = state.get("phase", Phase.PLANNING)
        current_topology: str = state.get("active_topology") or _DEFAULT_TOPOLOGY
        signals: dict[str, Any] = state.get("signals", {})
        iter_within_topology: int = state.get("iter_total", 0) - state.get(
            "topology_started_at_iter", 0
        )

        if current_phase == Phase.DONE:
            return TopologyDecision(
                topology=current_topology,
                reason="terminal phase — no topology switches in DONE",
                decided_by="rule",
            )

        # Rule 0 (advisory HITL): a human reviewer can leave a hint in
        # signals['human_advisor_hint'].  When the hint mentions a valid
        # topology name (one of _VALID_TOPOLOGIES), the rule router treats
        # it as a soft suggestion and proposes that topology.  SwitchGuards
        # still apply downstream — guards may veto, in which case the
        # router_cost is preserved but the topology stays.  An override
        # written by the adaptive's human_advisor_node (decided_by=
        # 'human_override') goes through a different path (_topo_dec_slot)
        # and bypasses this rule.  Hints from advisory mode that don't name
        # a valid topology fall through to the rule logic below.
        hint = signals.get("human_advisor_hint")
        if isinstance(hint, str) and hint.strip():
            proposed = _extract_topology_from_hint(hint)
            if proposed is not None:
                return TopologyDecision(
                    topology=proposed,
                    reason=(
                        f"human_advisor_hint={hint!r} mentions valid topology"
                        f" {proposed!r} — applying advisory suggestion"
                    ),
                    decided_by="rule",
                    considered_alternatives=(current_topology,)
                    if proposed != current_topology
                    else (),
                )

        if current_phase == Phase.EXECUTION:
            # Rule 1: stuck + early in topology → brainstorm with mesh
            stuck: bool = bool(signals.get("stuck", False))
            if stuck and iter_within_topology <= self._STUCK_DWELL_CAP:
                return TopologyDecision(
                    topology="mesh",
                    reason=(
                        f"stuck=True with iter_within_topology={iter_within_topology}"
                        f" <= {self._STUCK_DWELL_CAP} — switching to mesh (brainstorm)"
                    ),
                    decided_by="rule",
                )

            # Rule 2: repeated rejection → structured debate
            rejected_count: int = int(signals.get("rejected_count", 0))
            if rejected_count >= self._REJECT_THRESHOLD:
                return TopologyDecision(
                    topology="debate",
                    reason=(
                        f"rejected_count={rejected_count}"
                        f" >= {self._REJECT_THRESHOLD} — switching to debate"
                    ),
                    decided_by="rule",
                )

        if current_phase == Phase.VERIFICATION:
            # Rule 1: critic wants revision → linear rewrite within verification
            needs_revision: bool = bool(signals.get("needs_revision", False))
            if needs_revision:
                return TopologyDecision(
                    topology="linear",
                    reason="needs_revision=True — switching to linear for rewrite within verification",
                    decided_by="rule",
                )

        # Default: no rule fired — keep current topology
        return TopologyDecision(
            topology=current_topology,
            reason=(
                f"no rule fired for phase={current_phase}"
                f" — keeping current topology '{current_topology}'"
            ),
            decided_by="rule",
        )


# ---------------------------------------------------------------------------
# LLMTopologyRouter
# ---------------------------------------------------------------------------


class LLMTopologyRouter:
    """LLM-based topology router with JSON validation and fallback to rule.

    Decision flow (arch.md §8bis.2):
    1. Build prompt from template + state snapshot.
    2. Call llm.ainvoke([Message(...)], agent_id='topology_router').
    3. Parse JSON from LLMResponse.text.
    4. Validate: 'topology' key present, topology in _VALID_TOPOLOGIES.
    5. On any failure (JSON parse error, missing field, unknown topology):
       log WARNING (sanitized) and delegate to rule_fallback.decide(state).
    6. On success: return TopologyDecision(decided_by='llm_router', router_cost_usd=cost).
    7. On fallback: decided_by='rule' (from rule_fallback), router_cost_usd=0.0.

    Sanitized logging: raw_text previewed at max 80 chars + total len.
    Dict fields logged as keys only (no values leaked).

    Args:
        llm: Object with async ainvoke(messages, *, agent_id) -> LLMResponse.
        rule_fallback: RuleBasedTopologyRouter for fallback decisions.
        prompt_template: String template with {phase}, {active_topology},
                         {iter_within_topology}, {signals_keys}, {valid_topologies}
                         placeholders. Uses str.format_map().
    """

    _DEFAULT_PROMPT = (
        "You are a topology router for a multi-agent LLM system. "
        "Current phase: {phase}. "
        "Active topology: {active_topology}. "
        "Iterations in current topology: {iter_within_topology}. "
        "Active signal keys: {signals_keys}. "
        "Human advisor hint (may be empty): {human_advisor_hint}. "
        "Choose the best topology from: {valid_topologies}. "
        'Respond with JSON: {{"topology": "<name>", "reason": "<reason>"}}.'
    )

    def __init__(
        self,
        llm: Any,
        rule_fallback: RuleBasedTopologyRouter,
        prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = rule_fallback
        self._prompt_template = prompt_template or self._DEFAULT_PROMPT

    async def decide(self, state: SharedState) -> TopologyDecision:
        """Invoke LLM to decide topology; fallback to rule on any error.

        Returns:
            TopologyDecision with decided_by='llm_router' and router_cost_usd>0 on success,
            or decided_by='rule' (from fallback) on any parse/validation error.
        """
        current_phase: Phase = state.get("phase", Phase.PLANNING)
        current_topology: str = state.get("active_topology") or _DEFAULT_TOPOLOGY
        signals: dict[str, Any] = state.get("signals", {})
        iter_total: int = state.get("iter_total", 0)
        topology_started: int = state.get("topology_started_at_iter", 0)
        iter_within_topology: int = iter_total - topology_started

        hint_value = signals.get("human_advisor_hint")
        hint_str: str = str(hint_value) if isinstance(hint_value, str) else ""
        prompt = self._prompt_template.format_map(
            {
                "phase": current_phase,
                "active_topology": current_topology,
                "iter_within_topology": iter_within_topology,
                "signals_keys": list(signals.keys()),  # sanitized: keys only
                "human_advisor_hint": hint_str,
                "valid_topologies": sorted(_VALID_TOPOLOGIES),
            }
        )

        messages = [
            Message(
                sender="topology_router",
                kind=MessageKind.REQUEST,
                content=prompt,
            )
        ]

        try:
            response = await self._llm.ainvoke(messages, agent_id="topology_router")
        except Exception as exc:
            _log.warning("LLMTopologyRouter: LLM call failed (%s), falling back to rule.", exc)
            return await self._fallback.decide(state)

        raw_text = response.text or ""
        cost_usd: float = float(response.cost_usd)
        _log.debug("LLMTopologyRouter: LLM call cost=%.6f USD", cost_usd)

        # Parse JSON — sanitized log on error
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            _log.warning(
                "LLMTopologyRouter: malformed JSON from LLM (%s), "
                "raw_preview=%r (len=%d), falling back to rule.",
                exc,
                raw_text[:80],
                len(raw_text),
            )
            return await self._fallback.decide(state)

        # Validate 'topology' field present — log keys only (sanitized)
        if "topology" not in data:
            _log.warning(
                "LLMTopologyRouter: missing 'topology' in LLM response, "
                "keys=%s, falling back to rule.",
                list(data.keys()),
            )
            return await self._fallback.decide(state)

        # Validate topology is in the registered set
        raw_topo: str = str(data["topology"])
        if raw_topo not in _VALID_TOPOLOGIES:
            _log.warning(
                "LLMTopologyRouter: unknown topology %r from LLM "
                "(valid: %s), falling back to rule.",
                raw_topo,
                sorted(_VALID_TOPOLOGIES),
            )
            return await self._fallback.decide(state)

        reason: str = str(data.get("reason", "llm_router decision"))

        return TopologyDecision(
            topology=raw_topo,
            reason=reason,
            decided_by="llm_router",
            router_cost_usd=cost_usd,
        )


# ---------------------------------------------------------------------------
# OracleTopologyRouter
# ---------------------------------------------------------------------------


class OracleTopologyRouter:
    """Upper-bound oracle topology router for E3 ablation experiments.

    Reads a pre-built oracle table (task_id or task_type → best topology)
    without looking at runtime signals — serves as theoretical upper bound.

    Lookup order (per m8-routing.md contract):
    1. state['task_id'] + phase → oracle by_task_id[task_id][phase]
    2. infer task_type from task_id prefix or state['task_type'] → by_task_type[type][phase]
    3. '_default' key in table → fallback topology
    4. 'linear' → hardcoded fallback if table has no '_default'

    Oracle table format (tests/fixtures/oracle_table.json):
    {
        "by_task_type": { "<type>": { "<phase>": "<topology>", ... }, ... },
        "by_task_id":   { "<task_id>": { "<phase>": "<topology>", ... }, ... },
        "_default": "<topology>"
    }

    Args:
        oracle_table: Either a Path/str to a JSON file, or a dict with the table directly.
                      The dict form is preferred for testing.

    decided_by='oracle'; router_cost_usd=0.0 (no LLM calls).
    """

    def __init__(self, oracle_table: Path | str | dict[str, Any]) -> None:
        if isinstance(oracle_table, dict):
            self._table: dict[str, Any] = oracle_table
        else:
            table_path = Path(oracle_table)
            with table_path.open(encoding="utf-8") as fh:
                self._table = json.load(fh)

    async def decide(self, state: SharedState) -> TopologyDecision:
        """Look up best topology from oracle table; fallback to default if unknown.

        Args:
            state: SharedState (only task_id and phase are consulted).

        Returns:
            TopologyDecision with decided_by='oracle'.
        """
        task_id: str = state.get("task_id", "")
        current_phase: Phase = state.get("phase", Phase.PLANNING)
        phase_str: str = str(current_phase)

        # 1. Lookup by specific task_id
        by_task_id: dict[str, Any] = self._table.get("by_task_id", {})
        if task_id and task_id in by_task_id:
            phase_map: dict[str, Any] = by_task_id[task_id]
            if phase_str in phase_map:
                topology: str = str(phase_map[phase_str])
                return TopologyDecision(
                    topology=topology,
                    reason=f"oracle: task_id={task_id!r}, phase={phase_str} → {topology}",
                    decided_by="oracle",
                )

        # 2. Lookup by task_type (try to infer from task_id prefix or task_type key)
        # SharedState does not define task_type — access via raw dict to avoid typeddict-item.
        raw_state: dict[str, Any] = dict(state)
        task_type: str = cast(str, raw_state.get("task_type", ""))
        if not task_type and task_id:
            # Try to infer task_type from task_id (e.g. "humaneval/..." → "programming")
            # This is a best-effort heuristic; real mapping is done in M8.7
            pass

        by_task_type: dict[str, Any] = self._table.get("by_task_type", {})
        if task_type and task_type in by_task_type:
            phase_map_t: dict[str, Any] = by_task_type[task_type]
            if phase_str in phase_map_t:
                topology_t: str = str(phase_map_t[phase_str])
                return TopologyDecision(
                    topology=topology_t,
                    reason=(f"oracle: task_type={task_type!r}, phase={phase_str} → {topology_t}"),
                    decided_by="oracle",
                )

        # 3. Fallback to _default key
        default_topology: str = str(self._table.get("_default", _DEFAULT_TOPOLOGY))
        return TopologyDecision(
            topology=default_topology,
            reason=(
                f"oracle: no entry for task_id={task_id!r} or task_type={task_type!r}"
                f" — using default '{default_topology}'"
            ),
            decided_by="oracle",
        )
