"""DebateTopology — parallel Debater fan-out with Critic-as-judge loop.

Architecture (arch.md §7.5):
  Graph (judge mode "critic" — default, identical to M7):
    START → planner
    planner → debater_pro   (parallel fan-out — same super-step)
    planner → debater_contra
    debater_pro   → judge
    debater_contra → judge   (fan-in — both must complete before judge runs)
    judge → judge_postprocess
    judge_postprocess → conditional:
      approved OR max_rounds → END
      else                   → debate_round_start → debater_pro + debater_contra (loop)

  Graph (judge mode "human"):
    Like "critic", but node "judge" is replaced by "human_judge" — a HITL node
    that writes a synthesized DECISION message to agents[judge_id]["outbox"].
    Module-level _judge_postprocess is called unchanged afterward.

  Graph (judge mode "both" — sequential composite):
    debater_pro, debater_contra → "judge_combined" → conditional
    "judge_combined" = _both_judge_postprocess closure:
      1. LLM judge inline (judge_agent.step())
      2. Human gateway call
      3. Aggregate: human-override > critic when they disagree
    "judge" and "judge_postprocess" nodes are NOT added (distinct node name).

Parallel fan-out semantics:
  LangGraph automatically creates a concurrent super-step when multiple edges
  leave the same node. Because the debater count is fixed at 2, Send() API is
  not needed. Each debater writes only to its own agents[agent_id] outbox to
  avoid shared-write collisions.

Stopping precedence (arch.md §7.1):
  1. max_iter (global)   — iter_total >= cfg.max_iterations → END
  2. topology_success    — judge approved=True             → END
  3. topology_max        — debate_round >= max_rounds      → END
  4. continue            — loop back to debaters

TopologyConfig.extra defaults (under namespaced extras.debate):
  max_rounds: 4                      — maximum debate rounds before forced END
  debater_pro_id: "debater_pro"      — key in agents dict for pro debater
  debater_contra_id: "debater_contra" — key in agents dict for contra debater
  judge_id: "judge"                  — key in agents dict for judge

  Legacy flat keys (e.g. ``extra: {max_rounds: 4}``) are auto-remapped to
  ``extras.debate.max_rounds`` (and ``extras.hierarchical.max_rounds``) with a
  ``DeprecationWarning`` by the bw-compat validator in
  ``atm.experiment.config.TopologyCfg``.

HumanCfg.extra keys (Debate-specific):
  judge: "critic" | "human" | "both"  (default "critic")
    "critic" — LLM judge only (M7 back-compat)
    "human"  — HITL judge replaces LLM judge; _judge_postprocess untouched
    "both"   — sequential composite: LLM first, then HITL, human-override > critic

Nodes (mode "critic"):
  planner, debater_pro, debater_contra,
  judge, judge_postprocess, debate_round_start

Nodes (mode "human"):
  planner, debater_pro, debater_contra,
  human_judge, judge_postprocess, debate_round_start

Nodes (mode "both"):
  planner, debater_pro, debater_contra,
  judge_combined, debate_round_start

Invariants:
  - build() raises ValueError if debater_pro_id == debater_contra_id
  - judge_postprocess treats malformed DECISION as approved=False
  - Debaters write only to agents[their_id] outbox (not shared)
  - Message.id uniqueness guaranteed by uuid4 default_factory
  - Without human_cfg.enabled=True, graph is byte-for-byte identical to M7 Debate

Registration:
  @TopologyRegistry.register("debate") — side-effect on import

Notes:
  iter_total double-increment: In all judge modes, ``iter_total`` is incremented
  TWICE per debate round — once in ``planner_node`` (at the start of each round)
  and once inside ``_judge_postprocess`` (after judging).  This means that if
  ``cfg.max_iterations=N``, the effective maximum number of debate rounds before
  the global-max-iter guard fires is approximately N/2.  Size ``max_iterations``
  accordingly (e.g. ``max_iterations = max_rounds * 2 + 1``).  This matches the
  pattern established in chain.py where each router tick increments iter_total.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import END, START, StateGraph

from atm.core.state import GraphState
from atm.core.types import HumanContext, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop, get_topology_extras

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg
    from atm.human.gateway import HumanGateway

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — patchable in tests (matching Chain pattern)
# ---------------------------------------------------------------------------

LLMSimulatedGateway: Any
CLIGateway: Any
request_with_timeout: Any

try:
    from atm.human.llm_simulated import LLMSimulatedGateway as LLMSimulatedGateway
except ImportError:  # pragma: no cover
    LLMSimulatedGateway = None

try:
    from atm.human.cli_gateway import CLIGateway as CLIGateway
except ImportError:  # pragma: no cover
    CLIGateway = None

try:
    from atm.human._timeout import request_with_timeout as request_with_timeout
except ImportError:  # pragma: no cover
    request_with_timeout = None

HumanRoleRouter: Any
try:
    from atm.human.role_router import HumanRoleRouter as HumanRoleRouter
except ImportError:  # pragma: no cover
    HumanRoleRouter = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ROUNDS: int = 4

# Route sentinels
_ROUTE_LOOP = "debate_round_start"
_ROUTE_END = "__end__"


# ---------------------------------------------------------------------------
# judge_postprocess — adapter node
# ---------------------------------------------------------------------------


async def _judge_postprocess(
    state: GraphState,
    *,
    judge_id: str,
    debater_pro_id: str,
    debater_contra_id: str,
    cfg: TopologyConfig,
) -> dict[str, Any]:
    """Parse the last DECISION message from judge's outbox.

    Actions:
      1. Find last DECISION-kind message in agents[judge_id]["outbox"].
      2. Parse payload["approved"] (bool) and payload["winner"] ("pro"|"contra").
         Malformed → approved=False + WARNING.
      3. If approved=True:
           - Set shared.signals["judge_decided"] = True
           - Set shared.signals["debate_winner"] = winner ("pro" or "contra")
           - Extract final_answer from winning debater's DRAFT message.
      4. Increment shared["debate_round"] by 1.

    Args:
        state: GraphState dict.
        judge_id: Agent id of the judge.
        debater_pro_id: Agent id of pro debater.
        debater_contra_id: Agent id of contra debater.
        cfg: TopologyConfig (unused here, passed for consistency).

    Returns:
        State delta dict.
    """
    agents: dict[str, Any] = dict(state.get("agents") or {})
    judge_state: dict[str, Any] = dict(agents.get(judge_id) or {})
    outbox: list[Any] = list(judge_state.get("outbox") or [])

    # Find last DECISION message
    decision_msg = None
    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DECISION or str(kind) == "decision":
            decision_msg = msg
            break

    approved: bool = False
    winner: str = ""

    if decision_msg is None:
        logger.warning(
            "judge_postprocess: no DECISION message in judge outbox; treating as rejected"
        )
    else:
        payload: dict[str, Any] = getattr(decision_msg, "payload", {}) or {}
        if "approved" not in payload:
            logger.warning(
                "judge_postprocess: DECISION message missing 'approved' key; treating as rejected"
            )
            approved = False
        else:
            try:
                approved = bool(payload["approved"])
            except (TypeError, ValueError):
                logger.warning(
                    "judge_postprocess: could not parse payload['approved'] as bool; "
                    "treating as rejected"
                )
                approved = False

        if approved:
            raw_winner = payload.get("winner", "pro")
            winner = str(raw_winner) if raw_winner else "pro"

    # Build shared delta
    existing_shared: dict[str, Any] = dict(state.get("shared") or {})
    existing_signals: dict[str, Any] = dict(existing_shared.get("signals") or {})

    # Increment debate round counter (one logical round = one judge_postprocess call)
    debate_round: int = int(existing_shared.get("debate_round") or 0) + 1
    existing_shared["debate_round"] = debate_round

    # Increment global iter_total — mirrors chain.py/_route_from_critic pattern;
    # ensures max_iter (global) precedence fires inside the debate loop.
    # See planner_node — iter_total increments twice per round (planner-tick + judge-tick).
    iter_total: int = int(existing_shared.get("iter_total") or 0) + 1
    existing_shared["iter_total"] = iter_total

    # Task-aware extraction: forwarded into _extract_winner_artifact so the
    # solution.py-vs-DRAFT preference matches the task type (mirror chain
    # d585bbb).
    task_id: str = str(existing_shared.get("task_id") or "")

    if approved:
        existing_signals["judge_decided"] = True
        existing_signals["debate_winner"] = winner

        # Extract final_answer from winning debater. Prefer a successful
        # file_write artifact (e.g. solution.py) over the DRAFT text — the
        # DRAFT typically contains the debater's argument while the actual
        # solution is written via the file_write tool.
        winner_id = debater_pro_id if winner == "pro" else debater_contra_id
        final_answer = _extract_winner_artifact(agents, winner_id, task_id=task_id)
        existing_shared["final_answer"] = final_answer
    else:
        # Explicitly set judge_decided=False on rejection (star.py pattern)
        existing_signals["judge_decided"] = False

        # Safety net: when the judge fails to converge (REJECT every round,
        # max_rounds reached), still extract the best DRAFT artifact so
        # the run reports SOMETHING rather than quality_score=0 on an
        # otherwise-correct debate. Pick the side whose extracted artifact
        # has more content. This used to be masked by a runner-level
        # workspace fallback that read solution.py from disk; that
        # fallback no longer fires because final_answer is non-empty for
        # rejected rounds only when the judge eventually rejects.
        pro_answer = _extract_winner_artifact(agents, debater_pro_id, task_id=task_id)
        contra_answer = _extract_winner_artifact(agents, debater_contra_id, task_id=task_id)
        # Pick the longer non-incomplete answer; ties → pro (deterministic).
        pro_len = len(pro_answer) if pro_answer != "<incomplete>" else 0
        contra_len = len(contra_answer) if contra_answer != "<incomplete>" else 0
        if pro_len == 0 and contra_len == 0:
            fallback = "<incomplete>"
        elif contra_len > pro_len:
            fallback = contra_answer
            existing_signals["debate_winner"] = "contra"
        else:
            fallback = pro_answer
            existing_signals["debate_winner"] = "pro"
        # Only overwrite final_answer if it isn't already set from an
        # earlier approved round in this run.
        if not existing_shared.get("final_answer"):
            existing_shared["final_answer"] = fallback

    existing_shared["signals"] = existing_signals
    return {"shared": existing_shared}


# ---------------------------------------------------------------------------
# _route_from_judge — conditional edge router
# ---------------------------------------------------------------------------


def _route_from_judge(
    state: GraphState,
    cfg: TopologyConfig,
    max_rounds: int,
) -> str:
    """Route after judge_postprocess: END or loop back to debaters.

    Stopping precedence (arch.md §7.1):
      1. max_iter  — iter_total >= cfg.max_iterations → END
      2. approved  — judge_decided=True               → END  (topology_success)
      3. max_rounds — debate_round >= max_rounds       → END  (topology_max)
      4. continue  — loop

    Returns:
        _ROUTE_END or _ROUTE_LOOP.
    """
    shared: dict[str, Any] = dict(state.get("shared") or {})
    signals: dict[str, Any] = dict(shared.get("signals") or {})

    judge_decided: bool = bool(signals.get("judge_decided", False))
    debate_round: int = int(shared.get("debate_round") or 0)
    topology_max_reached: bool = debate_round >= max_rounds

    stop, _reason = _should_stop(
        dict(state),
        cfg,
        topology_success=judge_decided,
        topology_max_reached=topology_max_reached,
    )

    if stop:
        return _ROUTE_END

    return _ROUTE_LOOP


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_draft(agents: dict[str, Any], agent_id: str) -> str:
    """Extract last DRAFT message content from agent's outbox.

    Args:
        agents: agents dict from GraphState.
        agent_id: Agent id to look up.

    Returns:
        Content string or "<incomplete>".
    """
    agent_state: dict[str, Any] = dict(agents.get(agent_id) or {})
    outbox: list[Any] = list(agent_state.get("outbox") or [])

    for msg in reversed(outbox):
        kind = getattr(msg, "kind", None)
        if kind == MessageKind.DRAFT or str(kind) == "draft":
            content = getattr(msg, "content", None)
            if content:
                return str(content)

    return "<incomplete>"


_CODE_TASKS: set[str] = {"humaneval"}


def _extract_winner_artifact(
    agents: dict[str, Any],
    agent_id: str,
    task_id: str = "",
) -> str:
    """Extract winner debater's primary artifact (final answer for the topology).

    Task-aware preference (mirrors atm.topology.chain post-d585bbb):

    * CODE tasks (``task_id`` in ``_CODE_TASKS``, currently ``humaneval``):
      prefer a written ``solution.py`` / ``main.py`` over DRAFT — debaters
      using the ``file_write`` tool put the actual solution there while the
      DRAFT carries argument/rationale only.

    * NON-CODE tasks (``gsm8k`` / ``commongen`` / ``dabench``): prefer the
      DRAFT message. The debate prompt override in ``runner._build_agents``
      instructs debaters to begin DRAFTs with a ``###ANSWER###...###END###``
      marker for non-code; we extract the marker block when present.
      Falling back to a ``solution.py`` artifact for non-code returns
      Python intermediates rather than the human-readable answer
      (executor.yaml unconditionally tells the model to dump "programming
      tasks" to solution.py).

    Unknown/empty ``task_id`` is treated as non-code so the
    DRAFT/marker path dominates — mirrors chain's behavior on unset
    ``shared.task_id``. Callers that genuinely test the code-task path
    must pass ``task_id="humaneval"``.

    Strategy order, applied within the task-aware preference:

      1. Last successful ``file_write`` to ``solution.py`` / ``main.py`` /
         any ``.py`` (filtered by ``ToolResult.ok`` to skip rejected
         overwrites).
      2. DRAFT message (with ``###ANSWER###`` marker extraction when
         present).
      3. Any non-Python file artifact (last resort).
      4. Fallback: ``"<incomplete>"``.
    """
    agent_state: dict[str, Any] = dict(agents.get(agent_id) or {})
    is_code_task: bool = task_id.lower() in _CODE_TASKS

    # Collect file_write candidates.
    tool_calls: list[Any] = list(agent_state.get("tool_calls") or [])
    tool_results: list[Any] = list(agent_state.get("tool_results") or [])
    ok_call_ids: set[Any] = {
        getattr(r, "call_id", None) for r in tool_results if getattr(r, "ok", False)
    }
    py_solution: str | None = None
    py_any: str | None = None
    any_file: str | None = None
    for tc in reversed(tool_calls):
        if getattr(tc, "tool_name", None) != "file_write":
            continue
        # If tool_results is empty (older runs without result tracking) treat
        # absence as success to preserve back-compat. Otherwise require ok=True.
        if tool_results and getattr(tc, "id", None) not in ok_call_ids:
            continue
        args = getattr(tc, "args", None) or {}
        content = args.get("content") if isinstance(args, dict) else None
        path = args.get("path") if isinstance(args, dict) else None
        if not content:
            continue
        if any_file is None:
            any_file = str(content)
        if isinstance(path, str) and path.endswith(".py"):
            if py_any is None:
                py_any = str(content)
            if py_solution is None and (path.endswith("solution.py") or path.endswith("main.py")):
                py_solution = str(content)
                break
    file_artifact: str | None = py_solution or py_any

    # Collect DRAFT candidate. For non-code tasks the debater is instructed
    # (via the [DEBATE ROLE — HARD RULES] prompt override) to begin its DRAFT
    # with ``###ANSWER###\\n<answer>\\n###END###``; extract just the answer
    # when the marker is present.
    draft: str | None = None
    raw_draft = _extract_draft(agents, agent_id)
    if raw_draft and raw_draft != "<incomplete>":
        import re as _re

        m = _re.search(r"###ANSWER###\s*\n(.*?)\n\s*###END###", raw_draft, _re.DOTALL)
        if m:
            extracted = m.group(1).strip()
            draft = extracted or raw_draft
        else:
            draft = raw_draft

    # Task-aware preference.
    if is_code_task:
        primary, secondary = file_artifact, draft
    else:
        primary, secondary = draft, file_artifact

    return primary or secondary or any_file or "<incomplete>"


# ---------------------------------------------------------------------------
# _build_human_judge_node — HITL node factory for "human" judge mode
# ---------------------------------------------------------------------------


def _build_human_judge_node(
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    *,
    judge_id: str,
    debater_pro_id: str,
    debater_contra_id: str,
    role_router: Any = None,
) -> Any:
    """Build an async HITL node that synthesizes a DECISION message for the judge.

    The node:
      1. Extracts run_id and iter_total from state["shared"].
      2. Builds a question from the debaters' latest DRAFT messages.
      3. Dispatches 'human_request' custom event.
      4. Calls gateway.request (via request_with_timeout if configured).
      5. Dispatches 'human_response' custom event.
      6. Synthesizes a DECISION Message and writes it to agents[judge_id]["outbox"].
         This satisfies _judge_postprocess's expectation of a DECISION in judge's outbox.
         - approve  → payload={"approved": True, "winner": "pro"}  (default winner)
         - reject   → payload={"approved": False, "winner": ""}
         - abstain/timeout → payload={"approved": False, "winner": ""}

    The synthesized DECISION format is identical to what LLM judge produces,
    so module-level _judge_postprocess runs unchanged after this node.

    Args:
        human_cfg: HumanCfg with gateway, role, timeout settings.
        gateway:   Pre-constructed HumanGateway instance to call.
        judge_id:  Agent id of the judge (where DECISION is written).
        debater_pro_id:   Agent id of pro debater.
        debater_contra_id: Agent id of contra debater.
        role_router: Optional HumanRoleRouter; when not None, overrides human_cfg.role
            dynamically via ``await role_router.decide(phase, state)``.

    Returns:
        An async callable compatible with LangGraph node signature.
    """
    _role_router = role_router

    async def human_judge_node(state: GraphState) -> dict[str, Any]:
        """HITL judge node — calls gateway and writes synthesized DECISION to judge outbox."""
        import uuid as _uuid_mod

        from atm.core.types import Message

        shared: dict[str, Any] = dict(deepcopy(state.get("shared") or {}))
        agents: dict[str, Any] = dict(state.get("agents") or {})

        # --- Extract identifiers ---
        _raw_run_id = shared.get("run_id") or state.get("run_id")
        run_id: _uuid_mod.UUID = (
            _raw_run_id
            if isinstance(_raw_run_id, _uuid_mod.UUID)
            else _uuid_mod.UUID(str(_raw_run_id))
            if _raw_run_id
            else _uuid_mod.uuid4()
        )
        iter_total: int = int(shared.get("iter_total", 0))
        request_id: str = f"debate:{iter_total}:judge"

        # --- Build question from debaters' drafts ---
        pro_draft = _extract_draft(agents, debater_pro_id)
        contra_draft = _extract_draft(agents, debater_contra_id)
        question = (
            f"Judge this debate. Pro argument: {pro_draft[:200]}. "
            f"Contra argument: {contra_draft[:200]}. "
            "Do you approve? Action: 'approve' to approve pro side, 'reject' otherwise."
        )

        # --- Resolve active role (dynamic via role_router or static from cfg) ---
        from atm.core.types import Phase

        if _role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            active_role = await _role_router.decide(_phase, shared)
        else:
            active_role = human_cfg.role

        # --- Build HumanContext ---
        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=tuple(state.get("messages", [])[-5:]),
            allowed_actions=("approve", "reject"),
            deadline_s=int(human_cfg.timeout_s) if human_cfg.timeout_s is not None else None,
        )

        # --- Dispatch human_request BEFORE gateway call ---
        _requested_at = datetime.now(UTC)
        try:
            await adispatch_custom_event(
                "human_request",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "role": str(
                        active_role.value if hasattr(active_role, "value") else active_role
                    ),
                    "context_json": ctx.model_dump(mode="json"),
                    "requested_at": _requested_at,
                },
            )
        except Exception:
            logger.debug(
                "debate.human_judge_node: adispatch human_request skipped (no callback ctx)",
                exc_info=True,
            )

        # --- Call gateway ---
        _t0 = time.monotonic()
        timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
        policy: str = getattr(human_cfg, "timeout_policy", "skip")

        if request_with_timeout is not None and timeout_s_val is not None:
            _fallback_gateway: Any = None
            if policy == "llm_fallback" and LLMSimulatedGateway is not None:
                _fb_llm: Any = getattr(gateway, "_llm", None)
                _fallback_gateway = LLMSimulatedGateway(llm=_fb_llm)
            response = await request_with_timeout(
                gateway,
                ctx,
                request_id=request_id,
                timeout_s=timeout_s_val,
                policy=policy,
                llm_fallback_gateway=_fallback_gateway,
            )
        else:
            response = await gateway.request(ctx, request_id=request_id)

        _latency_s = time.monotonic() - _t0

        # --- Dispatch human_response ---
        try:
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": response.model_dump(mode="json"),
                    "source": getattr(response, "source", "human"),
                    "timed_out": getattr(response, "timed_out", False),
                    "latency_s": _latency_s,
                },
            )
        except Exception:
            logger.debug(
                "debate.human_judge_node: adispatch human_response skipped (no callback ctx)",
                exc_info=True,
            )

        # --- Synthesize DECISION message and write to judge's outbox ---
        action: str = getattr(response, "action", "") or ""
        comment: str = getattr(response, "comment", None) or ""

        # Determine approved + winner from human response
        if action == "approve":
            # Human approves; infer winner from response payload or default "pro"
            _payload = getattr(response, "payload", {}) or {}
            human_winner: str = str(_payload.get("winner", "pro"))
            approved: bool = True
        else:
            # reject / abstain / timeout → not approved
            human_winner = ""
            approved = False

        decision_msg = Message(
            sender="human_judge",
            kind=MessageKind.DECISION,
            content="APPROVE" if approved else "REJECT",
            payload={
                "approved": approved,
                "winner": human_winner,
                "comment": comment,
                "source": "human",
            },
        )

        # Write DECISION to judge_id's outbox (merging with any existing outbox)
        judge_agent_state: dict[str, Any] = dict(agents.get(judge_id) or {})
        judge_outbox: list[Any] = list(judge_agent_state.get("outbox") or [])
        judge_outbox.append(decision_msg)
        judge_agent_state["outbox"] = judge_outbox

        # Build delta with updated agents
        new_agents: dict[str, Any] = dict(agents)
        new_agents[judge_id] = judge_agent_state

        logger.debug(
            "debate.human_judge_node: action=%r approved=%r iter_total=%d",
            action,
            approved,
            iter_total,
        )

        return {"agents": new_agents}

    return human_judge_node


# ---------------------------------------------------------------------------
# DebateTopology
# ---------------------------------------------------------------------------


@TopologyRegistry.register("debate")
class DebateTopology:
    """Parallel Debater fan-out topology with Critic-as-judge loop.

    Graph structure (default "critic" mode — identical to M7):
      START → planner → [debater_pro, debater_contra] (parallel)
      debater_pro, debater_contra → judge → judge_postprocess
      judge_postprocess → conditional:
        approved / max_rounds → END
        else                  → debate_round_start → [debater_pro, debater_contra]

    Graph structure (mode "human"):
      Like "critic", but "judge" node replaced by "human_judge" HITL node.
      _judge_postprocess unchanged — reads from agents[judge_id]["outbox"].

    Graph structure (mode "both" — sequential composite):
      debater_pro, debater_contra → "judge_combined" → conditional
      "judge_combined" = _both_judge_postprocess closure:
        1. LLM judge inline
        2. HITL gateway call
        3. Aggregate: human-override > critic when they disagree

    HITL config (via HumanCfg.extra["judge"]):
      "critic" (default) — back-compat, no changes
      "human"            — HITL replaces LLM judge
      "both"             — sequential: LLM then HITL, human wins on disagreement

    Invariants:
      - debater_pro_id != debater_contra_id (ValueError otherwise)
      - judge_postprocess treats malformed DECISION as rejected
      - Debaters write only to their own agents outbox
      - Without human_cfg.enabled=True → graph identical to M7

    TopologyConfig.extra:
      max_rounds:       int  (default 4)
      debater_pro_id:   str  (required — key in agents dict)
      debater_contra_id: str (required — key in agents dict)
      judge_id:         str  (required — key in agents dict)
    """

    name = "debate"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a CompiledStateGraph.

        Args:
            agents: Dict mapping agent_id → Agent instance with async .step() method.
                    Must contain debater_pro_id, debater_contra_id, and judge_id keys.
            cfg:    TopologyConfig with max_iterations and extra.max_rounds.
            **kwargs: Optional; checkpointer=... forwarded to graph.compile().
                      human_cfg=... and gateway=... for HITL modes.

        Returns:
            CompiledStateGraph ready for ainvoke.

        Raises:
            ValueError: If debater_pro_id == debater_contra_id.
        """
        checkpointer = kwargs.get("checkpointer")
        human_cfg: HumanCfg | None = kwargs.get("human_cfg")
        role_router: Any = kwargs.get("role_router")

        # Build gateway from kwargs. D7 canonical key is "human_gateway"; legacy key
        # "gateway" is still honoured for back-compat with existing unit tests.
        gateway: HumanGateway | None = kwargs.get("human_gateway") or kwargs.get("gateway")
        if gateway is None and human_cfg is not None and getattr(human_cfg, "enabled", False):
            _gateway_llm: Any = kwargs.get("human_gateway_llm")
            if getattr(human_cfg, "gateway", "llm_simulated") == "cli":
                gateway = CLIGateway() if CLIGateway is not None else None
            else:
                gateway = (
                    LLMSimulatedGateway(llm=_gateway_llm)
                    if LLMSimulatedGateway is not None
                    else None
                )

        extras: dict[str, Any] = get_topology_extras(cfg, "debate")

        max_rounds: int = int(extras.get("max_rounds", _DEFAULT_MAX_ROUNDS))
        debater_pro_id: str = str(extras.get("debater_pro_id", "debater_pro"))
        debater_contra_id: str = str(extras.get("debater_contra_id", "debater_contra"))
        judge_id: str = str(extras.get("judge_id", "judge"))

        # Determine judge mode (only active when human_cfg.enabled is True)
        judge_mode: str = "critic"  # default — back-compat
        if human_cfg is not None and getattr(human_cfg, "enabled", False):
            human_extra: dict[str, Any] = getattr(human_cfg, "extra", None) or {}
            judge_mode = str(human_extra.get("judge", "critic"))

        # Invariant: pro and contra must be different agents
        if debater_pro_id == debater_contra_id:
            raise ValueError(
                f"debater_pro_id and debater_contra_id must be different; "
                f"both are {debater_pro_id!r}"
            )

        # Resolve agent instances
        planner_agent = agents.get("planner")
        debater_pro_agent = agents.get(debater_pro_id)
        debater_contra_agent = agents.get(debater_contra_id)
        judge_agent = agents.get(judge_id)

        # ----------------------------------------------------------------
        # Node: planner — opening statement
        # ----------------------------------------------------------------

        async def planner_node(state: GraphState) -> dict[str, Any]:
            """Planner node: produces opening statement for the debate."""
            if planner_agent is None:
                return {}
            result: dict[str, Any] = await planner_agent.step(state)
            # Increment iter_total on each planner call.
            # Note: iter_total also increments in _judge_postprocess. Effective iterations
            # per debate round = 2. Size cfg.max_iterations accordingly.
            shared: dict[str, Any] = dict((result.get("shared") or state.get("shared")) or {})
            shared["iter_total"] = int(shared.get("iter_total") or 0) + 1
            result["shared"] = shared
            return result

        # ----------------------------------------------------------------
        # Node: debater_pro — argues for "pro" stance
        # ----------------------------------------------------------------

        async def debater_pro_node(state: GraphState) -> dict[str, Any]:
            """Debater pro node: writes only to its own outbox."""
            if debater_pro_agent is None:
                return {}
            result: dict[str, Any] = await debater_pro_agent.step(state)
            return result

        # ----------------------------------------------------------------
        # Node: debater_contra — argues for "contra" stance
        # ----------------------------------------------------------------

        async def debater_contra_node(state: GraphState) -> dict[str, Any]:
            """Debater contra node: writes only to its own outbox."""
            if debater_contra_agent is None:
                return {}
            result: dict[str, Any] = await debater_contra_agent.step(state)
            return result

        # ----------------------------------------------------------------
        # Node: debate_round_start — no-op entry for loop continuation
        # ----------------------------------------------------------------

        async def debate_round_start_node(state: GraphState) -> dict[str, Any]:
            """No-op node that serves as loop re-entry point."""
            return {}

        # ----------------------------------------------------------------
        # Routing function
        # ----------------------------------------------------------------

        def _route(state: GraphState) -> str:
            return _route_from_judge(state, cfg, max_rounds)

        # ----------------------------------------------------------------
        # Assemble the graph
        # ----------------------------------------------------------------

        graph: StateGraph[GraphState] = StateGraph(GraphState)

        # Add common nodes
        graph.add_node("planner", planner_node)
        graph.add_node("debater_pro", debater_pro_node)
        graph.add_node("debater_contra", debater_contra_node)
        graph.add_node("debate_round_start", debate_round_start_node)

        # Entry point
        graph.add_edge(START, "planner")

        # Parallel fan-out: planner → both debaters (same super-step)
        graph.add_edge("planner", "debater_pro")
        graph.add_edge("planner", "debater_contra")

        # Loop: round_start → both debaters again
        graph.add_edge("debate_round_start", "debater_pro")
        graph.add_edge("debate_round_start", "debater_contra")

        # ----------------------------------------------------------------
        # Mode-specific: "critic" (default) — M7 back-compat
        # ----------------------------------------------------------------

        if judge_mode == "critic":
            # Node: judge — Critic-as-judge, evaluates debate
            async def judge_node(state: GraphState) -> dict[str, Any]:
                """Judge node: evaluates arguments from both debaters."""
                if judge_agent is None:
                    return {}
                result: dict[str, Any] = await judge_agent.step(state)
                return result

            # Node: judge_postprocess — parses judge decision, emits signals
            async def judge_postprocess_node(state: GraphState) -> dict[str, Any]:
                """Judge postprocess: parse DECISION, emit signals, set final_answer."""
                return await _judge_postprocess(
                    state,
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

            graph.add_node("judge", judge_node)
            graph.add_node("judge_postprocess", judge_postprocess_node)

            # Fan-in: both debaters → judge
            graph.add_edge("debater_pro", "judge")
            graph.add_edge("debater_contra", "judge")

            # Judge pipeline
            graph.add_edge("judge", "judge_postprocess")

            # Conditional routing from postprocess
            graph.add_conditional_edges(
                "judge_postprocess",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        # ----------------------------------------------------------------
        # Mode-specific: "human" — HITL judge replaces LLM judge
        # ----------------------------------------------------------------

        elif judge_mode == "human":
            assert human_cfg is not None and gateway is not None, (
                "judge mode 'human' requires human_cfg and gateway to be provided to build()"
            )

            # Build the HITL judge node (writes DECISION to judge_id outbox)
            human_judge = _build_human_judge_node(
                human_cfg,
                gateway,
                judge_id=judge_id,
                debater_pro_id=debater_pro_id,
                debater_contra_id=debater_contra_id,
                role_router=role_router,
            )

            # Node: judge_postprocess — reads from judge_id outbox (unchanged)
            async def judge_postprocess_node_human(state: GraphState) -> dict[str, Any]:
                """Judge postprocess: parse DECISION, emit signals, set final_answer."""
                return await _judge_postprocess(
                    state,
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

            graph.add_node("human_judge", human_judge)
            graph.add_node("judge_postprocess", judge_postprocess_node_human)

            # Fan-in: both debaters → human_judge
            graph.add_edge("debater_pro", "human_judge")
            graph.add_edge("debater_contra", "human_judge")

            # Human judge pipeline
            graph.add_edge("human_judge", "judge_postprocess")

            # Conditional routing from postprocess
            graph.add_conditional_edges(
                "judge_postprocess",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        # ----------------------------------------------------------------
        # Mode-specific: "both" — sequential composite (LLM then HITL)
        # ----------------------------------------------------------------

        elif judge_mode == "both":
            assert human_cfg is not None and gateway is not None, (
                "judge mode 'both' requires human_cfg and gateway to be provided to build()"
            )

            # ----- Build _both_judge_postprocess inline closure -----

            async def _both_judge_postprocess(state: GraphState) -> dict[str, Any]:
                """Sequential composite judge: LLM judge inline → HITL gateway → aggregate.

                Execution order:
                  1. Run LLM judge inline (judge_agent.step())
                  2. Parse LLM judge's DECISION (approved_by_critic, winner_by_critic)
                  3. Call human gateway → approved_by_human, winner_by_human
                  4. Aggregate: human-override > critic when they disagree
                     - Both approve      → approved=True
                     - Both reject       → approved=False
                     - Human approves, critic rejects → approved=True  (human wins)
                     - Human rejects, critic approves → approved=False (human wins)
                  5. Write final synthetic DECISION to agents[judge_id]["outbox"]
                  6. Run _judge_postprocess logic inline (increment counters, set signals)

                Returns:
                    State delta dict (shared + agents).
                """
                import uuid as _uuid_mod

                from atm.core.types import Message

                agents_state: dict[str, Any] = dict(state.get("agents") or {})
                shared_state: dict[str, Any] = dict(deepcopy(state.get("shared") or {}))

                # ---- Step 1: LLM judge inline ----
                llm_judge_result: dict[str, Any] = {}
                if judge_agent is not None:
                    llm_judge_result = await judge_agent.step(state)

                # Merge LLM judge result into agents (for DECISION parse below)
                if llm_judge_result.get("agents"):
                    agents_state = dict(agents_state)
                    agents_state.update(llm_judge_result["agents"])

                # ---- Step 2: Parse LLM judge DECISION ----
                judge_agent_state: dict[str, Any] = dict(agents_state.get(judge_id) or {})
                llm_outbox: list[Any] = list(judge_agent_state.get("outbox") or [])

                critic_approved: bool = False
                critic_winner: str = ""
                for msg in reversed(llm_outbox):
                    kind = getattr(msg, "kind", None)
                    if kind == MessageKind.DECISION or str(kind) == "decision":
                        _payload: dict[str, Any] = getattr(msg, "payload", {}) or {}
                        try:
                            critic_approved = bool(_payload.get("approved", False))
                        except (TypeError, ValueError):
                            critic_approved = False
                        if critic_approved:
                            critic_winner = str(_payload.get("winner", "pro"))
                        break

                # ---- Step 3: Call human gateway ----
                _raw_run_id = shared_state.get("run_id") or state.get("run_id")
                run_id: _uuid_mod.UUID = (
                    _raw_run_id
                    if isinstance(_raw_run_id, _uuid_mod.UUID)
                    else _uuid_mod.UUID(str(_raw_run_id))
                    if _raw_run_id
                    else _uuid_mod.uuid4()
                )
                iter_total_val: int = int(shared_state.get("iter_total", 0))
                h_request_id: str = f"debate:{iter_total_val}:judge_combined"

                pro_draft = _extract_draft(agents_state, debater_pro_id)
                contra_draft = _extract_draft(agents_state, debater_contra_id)
                critic_verdict = "APPROVE" if critic_approved else "REJECT"
                question = (
                    f"[Debate Judge Review] Critic verdict: {critic_verdict}. "
                    f"Pro: {pro_draft[:150]}. Contra: {contra_draft[:150]}. "
                    "Override? 'approve' to finalize, 'reject' to continue debate."
                )

                # --- Resolve active role (dynamic via role_router or static from cfg) ---
                from atm.core.types import Phase

                if role_router is not None:
                    _raw_phase = shared_state.get("phase", "execution")
                    _phase = Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
                    active_role_both = await role_router.decide(_phase, shared_state)
                else:
                    active_role_both = human_cfg.role

                ctx = HumanContext(
                    run_id=run_id,
                    role=active_role_both,
                    question=question,
                    recent_messages=tuple(state.get("messages", [])[-5:]),
                    allowed_actions=("approve", "reject"),
                    deadline_s=int(human_cfg.timeout_s)
                    if human_cfg.timeout_s is not None
                    else None,
                )

                _requested_at = datetime.now(UTC)
                try:
                    await adispatch_custom_event(
                        "human_request",
                        {
                            "run_id": run_id,
                            "request_id": h_request_id,
                            "role": str(
                                active_role_both.value
                                if hasattr(active_role_both, "value")
                                else active_role_both
                            ),
                            "context_json": ctx.model_dump(mode="json"),
                            "requested_at": _requested_at,
                        },
                    )
                except Exception:
                    logger.debug(
                        "debate._both_judge_postprocess: adispatch human_request skipped",
                        exc_info=True,
                    )

                _t0 = time.monotonic()
                timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
                h_policy: str = getattr(human_cfg, "timeout_policy", "skip")

                if request_with_timeout is not None and timeout_s_val is not None:
                    _h_fallback_gateway: Any = None
                    if h_policy == "llm_fallback" and LLMSimulatedGateway is not None:
                        _h_fb_llm: Any = getattr(gateway, "_llm", None)
                        _h_fallback_gateway = LLMSimulatedGateway(llm=_h_fb_llm)
                    h_response = await request_with_timeout(
                        gateway,
                        ctx,
                        request_id=h_request_id,
                        timeout_s=timeout_s_val,
                        policy=h_policy,
                        llm_fallback_gateway=_h_fallback_gateway,
                    )
                else:
                    h_response = await gateway.request(ctx, request_id=h_request_id)

                _latency_s = time.monotonic() - _t0

                try:
                    await adispatch_custom_event(
                        "human_response",
                        {
                            "run_id": run_id,
                            "request_id": h_request_id,
                            "answered_at": datetime.now(UTC),
                            "response_json": h_response.model_dump(mode="json"),
                            "source": getattr(h_response, "source", "human"),
                            "timed_out": getattr(h_response, "timed_out", False),
                            "latency_s": _latency_s,
                        },
                    )
                except Exception:
                    logger.debug(
                        "debate._both_judge_postprocess: adispatch human_response skipped",
                        exc_info=True,
                    )

                # ---- Step 4: Aggregate — human-override > critic ----
                h_action: str = getattr(h_response, "action", "") or ""
                human_approved: bool = h_action == "approve"
                _h_payload: dict[str, Any] = getattr(h_response, "payload", {}) or {}
                human_winner: str = str(_h_payload.get("winner", "pro")) if human_approved else ""

                # Human overrides critic: final decision comes from human
                final_approved: bool = human_approved
                final_winner: str = human_winner if human_approved else critic_winner

                # Log aggregation decision
                if human_approved != critic_approved:
                    logger.debug(
                        "debate._both_judge_postprocess: human overrides critic "
                        "(human=%r, critic=%r) → final=%r",
                        human_approved,
                        critic_approved,
                        final_approved,
                    )

                # ---- Step 5: Write synthetic DECISION to judge_id's outbox ----
                h_comment: str = getattr(h_response, "comment", None) or ""
                final_decision = Message(
                    sender="judge_combined",
                    kind=MessageKind.DECISION,
                    content="APPROVE" if final_approved else "REJECT",
                    payload={
                        "approved": final_approved,
                        "winner": final_winner,
                        "comment": h_comment,
                        "source": "both",
                        "critic_approved": critic_approved,
                        "human_approved": human_approved,
                    },
                )

                # Write to judge_id outbox (replaces/appends)
                final_judge_state: dict[str, Any] = dict(agents_state.get(judge_id) or {})
                final_judge_outbox: list[Any] = list(final_judge_state.get("outbox") or [])
                final_judge_outbox.append(final_decision)
                final_judge_state["outbox"] = final_judge_outbox
                agents_state[judge_id] = final_judge_state

                # ---- Step 6: Run _judge_postprocess logic inline ----
                # Build a synthetic state with the updated agents to re-use _judge_postprocess
                synthetic_state = dict(state)
                synthetic_state["agents"] = agents_state
                synthetic_state["shared"] = shared_state

                postprocess_delta = await _judge_postprocess(
                    synthetic_state,  # type: ignore[arg-type]
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

                # Merge: return agents + shared delta
                return {
                    "agents": agents_state,
                    "shared": postprocess_delta["shared"],
                }

            # Add the combined node — replaces both "judge" and "judge_postprocess"
            graph.add_node("judge_combined", _both_judge_postprocess)

            # Fan-in: both debaters → judge_combined
            graph.add_edge("debater_pro", "judge_combined")
            graph.add_edge("debater_contra", "judge_combined")

            # Conditional routing directly from judge_combined
            graph.add_conditional_edges(
                "judge_combined",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        else:
            logger.warning(
                "DebateTopology.build: unknown judge mode %r; falling back to 'critic'",
                judge_mode,
            )

            # Fall back to critic mode — add judge + judge_postprocess nodes
            async def judge_node_fallback(state: GraphState) -> dict[str, Any]:
                """Judge node (fallback): evaluates arguments from both debaters."""
                if judge_agent is None:
                    return {}
                result: dict[str, Any] = await judge_agent.step(state)
                return result

            async def judge_postprocess_node_fallback(state: GraphState) -> dict[str, Any]:
                """Judge postprocess (fallback): parse DECISION, emit signals."""
                return await _judge_postprocess(
                    state,
                    judge_id=judge_id,
                    debater_pro_id=debater_pro_id,
                    debater_contra_id=debater_contra_id,
                    cfg=cfg,
                )

            graph.add_node("judge", judge_node_fallback)
            graph.add_node("judge_postprocess", judge_postprocess_node_fallback)

            graph.add_edge("debater_pro", "judge")
            graph.add_edge("debater_contra", "judge")
            graph.add_edge("judge", "judge_postprocess")

            graph.add_conditional_edges(
                "judge_postprocess",
                _route,
                {
                    _ROUTE_END: END,
                    _ROUTE_LOOP: "debate_round_start",
                },
            )

        return graph.compile(checkpointer=checkpointer)
