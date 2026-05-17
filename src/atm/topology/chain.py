"""ChainTopology — linear Planner → Executor → Critic pipeline with retry-loop.

Architecture (arch.md §6):
  Graph:  START → planner → executor → critic → critic_postprocess
          → conditional edge (→ END | → executor)

With HITL enabled (human_cfg.enabled=True):
  Graph:  START → planner → executor → critic → critic_postprocess
          → human_reviewer → conditional edge (→ END | → executor)

Chain does NOT use phases like Star. In M6, shared.phase is pinned to
"execution" for the entire run.

Stopping precedence (arch.md §7.1):
  1. max_iter (global):  iter_total >= cfg.max_iterations → END
  2. topology_success:   critic_approved=True            → END
  3. topology_max:       (unused in Chain for M6)
  4. continue:           → executor (retry loop)

NOTE: Steps 2 and 3 (star/chain) must NOT edit topology/__init__.py.
Registration is done via @TopologyRegistry.register("chain") side-effect.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import adispatch_custom_event

from atm.core.types import HumanContext, Message, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry, _should_stop

if TYPE_CHECKING:
    from atm.experiment.config import HumanCfg
    from atm.human.gateway import HumanGateway
    from atm.human.role_router import HumanRoleRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# END sentinel — matches LangGraph's END constant value ("__end__").
# Using a module-level constant avoids a hard import of langgraph.graph
# in functions that are unit-tested independently.
# ---------------------------------------------------------------------------

CHAIN_END: str = "__end__"

# ---------------------------------------------------------------------------
# StateGraph — module-level reference for testability.
# Tests patch "atm.topology.chain.StateGraph" to inject a mock graph factory.
# At runtime, build() resolves the real StateGraph via _import_state_graph().
# ---------------------------------------------------------------------------
#
# We use a private import function to avoid a module-level try/except that
# confuses mypy. The module attribute is set to the resolved class at module
# load time (if langgraph.graph is available) or left as a placeholder for
# mocking. Since pyproject.toml doesn't include langgraph as a direct dep
# yet, the module-level import is guarded and the None case is handled in build().

_StateGraphCls: type | None  # forward declaration for type checker


def _import_state_graph() -> type | None:
    """Import StateGraph from langgraph.graph if available.

    Returns:
        The StateGraph class, or None if langgraph.graph is not installed.
    """
    try:
        from langgraph.graph import StateGraph as _StateGraph

        return _StateGraph
    except ImportError:  # pragma: no cover
        return None


_StateGraphCls = _import_state_graph()

# Module-level alias for test patching: patch("atm.topology.chain.StateGraph")
StateGraph: type | None = _StateGraphCls

# ---------------------------------------------------------------------------
# LLMSimulatedGateway — lazy import for test patching
# ---------------------------------------------------------------------------
# Imported at module level so tests can patch "atm.topology.chain.LLMSimulatedGateway".
try:
    from atm.human.llm_simulated import LLMSimulatedGateway
except ImportError:  # pragma: no cover
    LLMSimulatedGateway = None  # type: ignore[assignment,misc]

try:
    from atm.human.cli_gateway import CLIGateway
except ImportError:  # pragma: no cover
    CLIGateway = None  # type: ignore[assignment,misc]

try:
    from atm.human._timeout import request_with_timeout
except ImportError:  # pragma: no cover
    request_with_timeout = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# _build_human_reviewer_node — HITL node factory
# ---------------------------------------------------------------------------


def _build_human_reviewer_node(
    human_cfg: HumanCfg,
    gateway: HumanGateway,
    role_router: HumanRoleRouter | None = None,
    fallback_llm: Any = None,
) -> Any:
    """Build and return an async node function for the human_reviewer step.

    The node:
      1. Reads run_id from state["shared"] (raises RuntimeError if missing).
      2. Determines active role: if role_router is not None, calls
         role_router.decide(phase, shared); otherwise uses human_cfg.role.
      3. Builds a HumanContext from state using the active role.
      4. Computes request_id = f"chain:{iter_total}:reviewer".
      5. Dispatches 'human_request' custom event (BEFORE gateway call).
      6. Calls gateway.request (via request_with_timeout).
      7. Dispatches 'human_response' custom event.
      8. Updates state based on action:
           approve  → shared["human_approved"] = True
           reject   → append synthesized Message + shared["needs_rerun"] = True
           abstain/timeout → no-op

    Args:
        human_cfg:   HumanCfg with gateway config, role, timeout settings.
        gateway:     Pre-constructed HumanGateway instance to call.
        role_router: Optional HumanRoleRouter; when None (default), human_cfg.role
                     is used (back-compat).  When provided, decide() determines the
                     active role for each interaction.

    Returns:
        An async function compatible with LangGraph node signature.
    """

    async def human_reviewer(state: dict[str, Any]) -> dict[str, Any]:
        shared: dict[str, Any] = dict(deepcopy(state.get("shared", {})))

        # --- Step 1: Extract run_id (required) ---
        run_id = shared.get("run_id")
        if run_id is None:
            raise RuntimeError(
                "human_reviewer node requires state['shared']['run_id'] to be set. "
                "Ensure Runner._build_initial_state populates run_id (M9 Step 3.1)."
            )

        iter_total: int = int(shared.get("iter_total", 0))

        # --- Step 2: Build HumanContext ---
        # Derive the question from the critic's latest DECISION message
        agents: dict[str, Any] = state.get("agents", {})
        critic_outbox: list[Any] = list((agents.get("critic") or {}).get("outbox", []))
        question: str = "Please review the latest output and decide to approve, reject, or abstain."
        for msg in reversed(critic_outbox):
            if getattr(msg, "kind", None) == MessageKind.DECISION:
                question = getattr(msg, "content", question) or question
                break

        # recent_messages: top-level messages list
        recent_msgs_raw: list[Any] = state.get("messages", [])
        # Convert to tuple of Message objects (filter valid)
        recent_messages: tuple[Message, ...] = tuple(
            m for m in recent_msgs_raw if isinstance(m, Message)
        )

        import uuid as _uuid

        if not isinstance(run_id, _uuid.UUID):
            run_id = _uuid.UUID(str(run_id))

        # Determine active role: router takes precedence over human_cfg.role
        from atm.core.types import Phase as _Phase

        if role_router is not None:
            _raw_phase = shared.get("phase", "execution")
            _active_phase: _Phase = (
                _Phase(_raw_phase) if isinstance(_raw_phase, str) else _raw_phase
            )
            active_role = await role_router.decide(_active_phase, shared)
        else:
            active_role = human_cfg.role

        ctx = HumanContext(
            run_id=run_id,
            role=active_role,
            question=question,
            recent_messages=recent_messages,
            allowed_actions=("approve", "reject", "abstain"),
        )

        # --- Step 3: Compute deterministic request_id ---
        request_id: str = f"chain:{iter_total}:reviewer"

        # --- Step 4: Dispatch human_request BEFORE gateway call ---
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
                    "requested_at": datetime.now(UTC),
                },
            )
        except Exception:
            logger.debug("adispatch human_request skipped (no callback ctx)", exc_info=True)

        # --- Step 5: Call gateway (with timeout wrapper if timeout_s is set) ---
        timeout_s_val: float | None = getattr(human_cfg, "timeout_s", None)
        policy: str = getattr(human_cfg, "timeout_policy", "skip")

        # Measure latency for human_response payload (F2)
        _t0 = time.monotonic()

        if request_with_timeout is not None and timeout_s_val is not None:
            # F3: build a fallback gateway for llm_fallback policy
            _fallback_gateway: Any = None
            if policy == "llm_fallback" and LLMSimulatedGateway is not None:
                # Prefer fallback_llm param; fall back to gateway._llm for back-compat.
                # Guard: only build LLMSimulatedGateway when an LLM is available
                # (StreamlitHumanGateway has no ._llm → would silently produce None).
                _fb_llm: Any = (
                    fallback_llm
                    if fallback_llm is not None
                    else getattr(gateway, "_llm", None)
                )
                _fallback_gateway = (
                    LLMSimulatedGateway(llm=_fb_llm) if _fb_llm is not None else None
                )
            response = await request_with_timeout(
                gateway,
                ctx,
                request_id=request_id,
                timeout_s=timeout_s_val,
                policy=policy,  # type: ignore[arg-type]
                llm_fallback_gateway=_fallback_gateway,
            )
        else:
            response = await gateway.request(ctx, request_id=request_id)

        latency_s = time.monotonic() - _t0

        # --- Step 6: Dispatch human_response AFTER gateway returns ---
        try:
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": response.model_dump(mode="json"),
                    "source": response.source,
                    "timed_out": response.timed_out,
                    "latency_s": latency_s,
                },
            )
        except Exception:
            logger.debug("adispatch human_response skipped (no callback ctx)", exc_info=True)

        # --- Step 7: Update state based on action ---
        action: str = response.action
        delta: dict[str, Any] = {}

        if action == "approve":
            shared["human_approved"] = True
            delta["shared"] = shared

        elif action == "reject":
            # Synthesize a rejection message with the comment
            comment: str = response.comment or "Human reviewer rejected this iteration."
            rejection_msg = Message(
                sender="human_reviewer",
                kind=MessageKind.CRITIQUE,
                content=f"[Human Reviewer Rejection] {comment}",
                payload={"human_rejected": True, "comment": comment},
            )
            shared["needs_rerun"] = True
            delta["shared"] = shared
            delta["messages"] = [rejection_msg]

        # abstain / timeout → no-op (return empty delta)

        return delta

    return human_reviewer


# ---------------------------------------------------------------------------
# _critic_postprocess — adapter node
# ---------------------------------------------------------------------------


async def _critic_postprocess(state: dict[str, Any]) -> dict[str, Any]:
    """Parse the last DECISION message from critic.outbox and update shared state.

    Actions:
      1. Read last DECISION-kind message from state["agents"]["critic"]["outbox"].
      2. Parse payload["approved"] (bool). Malformed → approved=False + warning.
      3. Write shared.signals["critic_approved"].
      4. If approved=True:
           - Find last DRAFT message in state["agents"]["executor"]["outbox"].
           - Set shared["final_answer"] = msg.payload["draft"] or msg.content.
           - Fallback to "<incomplete>" if no DRAFT found.
      5. Pin shared["phase"] = "execution" (Chain M6 invariant).

    Returns:
        A state delta dict (only "shared" key) to be merged by LangGraph.
    """
    shared = dict(deepcopy(state.get("shared", {})))
    agents = state.get("agents", {})

    # Initialize signals dict if missing
    signals: dict[str, Any] = dict(shared.get("signals", {}))

    # --- Step 1 & 2: Find last DECISION message in critic outbox ---
    critic_state = agents.get("critic", {})
    critic_outbox: list[Any] = list(critic_state.get("outbox", []))

    approved = False
    for msg in reversed(critic_outbox):
        if getattr(msg, "kind", None) == MessageKind.DECISION:
            try:
                approved = bool(msg.payload["approved"])
            except (KeyError, TypeError, AttributeError):
                logger.warning(
                    "ChainTopology: malformed DECISION message — missing payload['approved']; "
                    "treating as approved=False. message=%s",
                    str(msg),
                )
                approved = False
            break

    # --- Step 3: Write signals ---
    signals["critic_approved"] = approved

    # --- Step 4: Populate final_answer ALWAYS (regardless of approval). ---
    # Setting final_answer only on approval was the original behaviour, but it
    # leaves shared.final_answer empty whenever Chain hits max_iterations
    # without the critic ever approving. The evaluator then receives an empty
    # string and scores 0 even when the executor produced correct code. We
    # always extract the executor's best artifact so the evaluator can score
    # whatever work was done; the critic's approval status is independently
    # surfaced via shared.signals["critic_approved"] for downstream policy.
    executor_state = agents.get("executor", {})

    # 4a. Most recent SUCCESSFUL file_write to a .py path.
    #     Filtering by ToolResult.ok avoids returning the payload of a
    #     rejected overwrite (file_write defaults to overwrite=False, so a
    #     duplicate write on the same path silently fails — and the stale
    #     file on disk still holds the first, correct, content).
    executor_tool_calls: list[Any] = list(executor_state.get("tool_calls", []))
    executor_tool_results: list[Any] = list(executor_state.get("tool_results", []))
    ok_call_ids: set[Any] = {
        getattr(r, "call_id", None) for r in executor_tool_results if getattr(r, "ok", False)
    }
    file_artifact: str | None = None
    for tc in reversed(executor_tool_calls):
        if getattr(tc, "tool_name", None) != "file_write":
            continue
        if executor_tool_results and getattr(tc, "id", None) not in ok_call_ids:
            continue
        args = getattr(tc, "args", None) or {}
        content = args.get("content") if isinstance(args, dict) else None
        path = args.get("path") if isinstance(args, dict) else None
        if not content:
            continue
        if isinstance(path, str) and path.endswith(".py"):
            file_artifact = str(content)
            break
        if file_artifact is None:
            file_artifact = str(content)  # last-resort non-py artifact

    # 4b. Fall back to DRAFT text (text-only executors, no tool use).
    executor_outbox: list[Any] = list(executor_state.get("outbox", []))
    draft_text: str | None = None
    for msg in reversed(executor_outbox):
        if getattr(msg, "kind", None) == MessageKind.DRAFT:
            raw_payload = getattr(msg, "payload", {}) or {}
            draft_text = raw_payload.get("draft") or getattr(msg, "content", None)
            break

    # 4c. Task-aware preference. For CODE tasks (executor told to write
    # solution.py), prefer the file artifact — the DRAFT is usually just
    # narrative ("Solution written to solution.py"). For NON-CODE tasks
    # (gsm8k / commongen / dabench), the executor STILL writes solution.py
    # under instruction (executor.yaml unconditionally tells it to dump
    # "programming tasks" to solution.py), but that file contains Python
    # full of intermediate variables, test cases, and restated problem
    # numbers — and GSM8KMatcher takes the LAST numeric token, which is
    # almost never the right answer. Use the DRAFT (where the executor
    # writes the human-readable answer) for these tasks.
    task_id = str(shared.get("task_id") or "").lower()
    _code_tasks = {"humaneval"}
    if task_id in _code_tasks:
        final_answer = file_artifact or draft_text or "<incomplete>"
    else:
        final_answer = draft_text or file_artifact or "<incomplete>"

    # --- Step 5: Pin phase to "execution" ---
    shared["phase"] = "execution"
    shared["signals"] = signals
    shared["final_answer"] = final_answer

    return {"shared": shared}


# ---------------------------------------------------------------------------
# _route_from_critic — conditional edge router
# ---------------------------------------------------------------------------


def _route_from_critic(state: dict[str, Any], cfg: TopologyConfig) -> str:
    """Decide whether to go to END or retry via executor.

    Side effects:
      - Increments state["shared"]["iter_total"] by 1.
      - Increments state["shared"]["iteration"] by 1.

    Routing logic (after counter increment):
      1. Run _should_stop:
           - If stop (max_iter or topology_success) → CHAIN_END.
      2. If critic_approved in signals → CHAIN_END.
      3. Otherwise → "executor" (retry loop; planner does NOT re-run).

    Returns:
        CHAIN_END ("__end__") or "executor".
    """
    shared = state["shared"]

    # Increment counters BEFORE evaluating stop conditions
    shared["iter_total"] = shared.get("iter_total", 0) + 1
    shared["iteration"] = shared.get("iteration", 0) + 1

    signals: dict[str, Any] = shared.get("signals", {})
    critic_approved: bool = bool(signals.get("critic_approved", False))

    # Evaluate stopping conditions (max_iter priority over topology_success)
    stop, _reason = _should_stop(
        state,
        cfg,
        topology_success=critic_approved,
        topology_max_reached=False,
    )

    if stop:
        return CHAIN_END

    return "executor"


# ---------------------------------------------------------------------------
# ChainTopology — main class
# ---------------------------------------------------------------------------


@TopologyRegistry.register("chain")
class ChainTopology:
    """Linear topology: Planner → Executor → Critic → (retry | END).

    Graph structure:
      START → planner → executor → critic → critic_postprocess
      critic_postprocess → conditional: approved/max → END | → executor

    M6 constraints:
      - shared.phase pinned to "execution" (no phase transitions).
      - max_iterations from TopologyConfig.max_iterations.
      - TopologyConfig.extra is empty for Chain in M6.
    """

    name: str = "chain"

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Build and compile the Chain LangGraph.

        Args:
            agents:      Dict mapping agent_id → Agent instance.
                         Expected keys: "planner", "executor", "critic".
            cfg:         TopologyConfig with max_iterations.
            **kwargs:    Passed to compile() (e.g. checkpointer=...).

        Returns:
            A compiled LangGraph CompiledStateGraph.

        Raises:
            ImportError: If langgraph.graph is not installed.
            KeyError: If "planner", "executor", or "critic" missing from agents.
        """
        # Resolve StateGraph — may be mocked in tests
        sg_cls = StateGraph
        if sg_cls is None:  # pragma: no cover
            raise ImportError(
                "langgraph.graph is required for ChainTopology.build(). Install: uv add langgraph"
            )

        planner = agents["planner"]
        executor = agents["executor"]
        critic = agents["critic"]

        # Build the graph — GraphState is passed as state schema
        from atm.core.state import GraphState

        graph: Any = sg_cls(GraphState)

        # --- Node: planner ---
        graph.add_node("planner", planner.step)

        # --- Node: executor ---
        graph.add_node("executor", executor.step)

        # --- Node: critic ---
        graph.add_node("critic", critic.step)

        # --- Node: critic_postprocess ---
        graph.add_node("critic_postprocess", _critic_postprocess)

        # --- Edges: linear chain ---
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "critic")
        graph.add_edge("critic", "critic_postprocess")

        # --- HITL: optionally insert human_reviewer between critic_postprocess and routing ---
        human_cfg: HumanCfg | None = kwargs.get("human_cfg")

        # Conditional edge: postprocess (or human_reviewer) → (END | executor)
        def _route(state: dict[str, Any]) -> str:
            return _route_from_critic(state, cfg)

        if human_cfg is not None and human_cfg.enabled:
            # D7: honour pre-built gateway from runner (canonical key: human_gateway).
            # gateway_llm captured unconditionally so it is available to
            # _build_human_reviewer_node's fallback_llm even when gateway came
            # pre-built from the runner (Streamlit path).
            gateway_llm: Any = kwargs.get("human_gateway_llm")
            human_gateway: Any = kwargs.get("human_gateway")
            if human_gateway is not None:
                gateway_instance: Any = human_gateway
            else:
                # Build gateway based on human_cfg.gateway setting.
                # The LLM wrapper for LLMSimulatedGateway may be passed via kwargs
                # (e.g. by Runner); if absent, it is left as None and the class is
                # expected to be patched in tests.
                if human_cfg.gateway == "cli":
                    gateway_instance = CLIGateway() if CLIGateway is not None else None
                else:
                    # default: llm_simulated
                    gateway_instance = (
                        LLMSimulatedGateway(llm=gateway_llm)
                        if LLMSimulatedGateway is not None
                        else None
                    )

            if gateway_instance is None:  # pragma: no cover
                raise ImportError(
                    f"Gateway class for '{human_cfg.gateway}' could not be imported. "
                    "Ensure atm.human is installed."
                )

            role_router = kwargs.get("role_router")
            node_fn = _build_human_reviewer_node(
                human_cfg,
                gateway_instance,
                role_router=role_router,
                fallback_llm=gateway_llm,
            )
            graph.add_node("human_reviewer", node_fn)

            # critic_postprocess → human_reviewer → conditional
            graph.add_edge("critic_postprocess", "human_reviewer")
            graph.add_conditional_edges(
                "human_reviewer",
                _route,
                {
                    "executor": "executor",
                    CHAIN_END: CHAIN_END,
                },
            )
        else:
            # Default path: critic_postprocess → conditional
            graph.add_conditional_edges(
                "critic_postprocess",
                _route,
                {
                    "executor": "executor",
                    CHAIN_END: CHAIN_END,
                },
            )

        # --- Entry point ---
        graph.set_entry_point("planner")

        # --- Compile ---
        checkpointer = kwargs.get("checkpointer")
        return graph.compile(checkpointer=checkpointer)
