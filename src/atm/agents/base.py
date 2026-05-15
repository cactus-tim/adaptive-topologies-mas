"""Agent base class for ATM multi-agent framework.

Implements:
  - ``_StepOutcome``: frozen dataclass aggregating one tool-loop execution result.
  - ``AgentView``: immutable snapshot of the slice of GraphState used during step().
  - ``Agent``: LangGraph-compatible node with scratchpad policy C, tool-calling loop,
    and optional summarizer integration.

Scratchpad policy C:
  - ``window_size`` is measured in **scratchpad events**, not logical steps.
  - One logical step may produce up to 3 events: reasoning, tool_call, observation.
  - Only the tail ``scratchpad[-window_size:]`` is included in the prompt window.
  - The full scratchpad is always returned in the delta (append-only, never truncated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import structlog

from atm.agents._tokens import estimate_prompt_tokens
from atm.agents.config import AgentConfig
from atm.core.errors import ToolError
from atm.core.state import AgentState, GraphState
from atm.core.types import (
    LLMResponse,
    Message,
    MessageKind,
    ToolCall,
    ToolResult,
)
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# _StepOutcome — result of one _run_tool_loop() execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StepOutcome:
    """Aggregated result of one tool-loop execution.

    Attributes:
        response:          Final LLMResponse from the last iteration.
        scratchpad_events: All events produced during the loop (reasoning /
                           tool_call / observation), in order.
        tool_calls:        All tool calls accumulated across ALL iterations
                           (critical C1 fix — not reset between iterations).
        tool_results:      All tool results accumulated across ALL iterations.
    """

    response: LLMResponse
    scratchpad_events: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]


# ---------------------------------------------------------------------------
# AgentView — immutable snapshot of state for one step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentView:
    """Immutable snapshot of the agent's slice of GraphState for one step.

    Purpose: decouples prompt building and summarizer invocation from the
    live GraphState, preventing accidental mutation of state.

    Attributes:
        agent_id:              The agent's identifier.
        self_state:            Agent's own AgentState (TypedDict).
        shared:                Shared global state (task_input, phase, etc.).
        inbox:                 Incoming messages addressed to this agent.
        scratchpad:            Full append-only scratchpad up to current step.
        summary_before_window: Summarizer output for events before the window.
    """

    agent_id: str
    self_state: AgentState
    shared: dict[str, Any]
    inbox: tuple[Message, ...]
    scratchpad: tuple[dict[str, Any], ...]
    summary_before_window: str | None
    # Most recent peer message per (sender, kind) from state["messages"], used
    # by chain/star/etc. topologies that don't (yet) populate inbox via an
    # explicit routing node. Empty tuple when no peers have spoken yet.
    peer_messages: tuple[Message, ...] = ()


# ---------------------------------------------------------------------------
# Agent — base class
# ---------------------------------------------------------------------------


class Agent:
    """Base LangGraph-compatible agent node with scratchpad policy C.

    Args:
        agent_id:       Unique identifier for this agent instance.
        cfg:            Frozen ``AgentConfig`` with all tuning parameters.
        llm:            Primary ``LLMWrapper`` for generation.
        tools:          ``ToolRegistry`` providing tool dispatch.
        summarizer_llm: Optional second ``LLMWrapper`` for scratchpad
                        summarization. Used when prompt exceeds
                        ``cfg.context_token_budget``.
    """

    def __init__(
        self,
        agent_id: str,
        cfg: AgentConfig,
        llm: LLMWrapper,
        tools: ToolRegistry,
        summarizer_llm: LLMWrapper | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.summarizer_llm = summarizer_llm

    # ------------------------------------------------------------------
    # Public API — LangGraph node entry point
    # ------------------------------------------------------------------

    async def step(self, state: GraphState) -> dict[str, Any]:
        """Execute one agent step and return a delta to merge into GraphState.

        Returns a *delta* only — never the full state.  The LangGraph reducer
        (``merge_agent_states``) handles merging:
          - list fields (inbox, outbox, scratchpad, tool_calls, tool_results) — concat.
          - ``step_count`` — max (so we return the ABSOLUTE new value).
          - ``summary_before_window`` — right wins if non-empty.

        Flow:
          1. Build ``AgentView`` snapshot from state.
          2. Optionally summarize pre-window events (_maybe_summarize).
          3. Build prompt via _build_prompt.
          4. Run tool-loop via _run_tool_loop.
          5. Produce DRAFT outbox message from final response.
          6. Assemble and return delta dict.
        """
        agents_state: dict[str, Any] = cast(dict[str, Any], state.get("agents") or {})
        self_state: AgentState = cast(AgentState, agents_state.get(self.agent_id) or {})
        shared: dict[str, Any] = cast(dict[str, Any], state.get("shared") or {})

        # Reconstruct inbox and scratchpad from current state (already accumulated)
        inbox_raw: list[Any] = list(self_state.get("inbox") or [])
        scratchpad_raw: list[dict[str, Any]] = list(self_state.get("scratchpad") or [])
        summary_before_window: str | None = self_state.get("summary_before_window") or None
        step_count: int = int(self_state.get("step_count") or 0) + 1

        # Collect the most recent peer message per (sender, kind) from the
        # global messages channel. Required for chain/star/mesh/debate where
        # no routing node copies outbox→inbox: without this, every agent runs
        # blind and never sees the previous agent's draft/decision.
        # Self-emitted messages are excluded; the agent already has its own
        # context via scratchpad. Order: chronological (oldest first).
        global_messages: list[Any] = list(state.get("messages") or [])
        latest_per_key: dict[tuple[str, Any], Message] = {}
        for _m in global_messages:
            _sender = getattr(_m, "sender", "")
            if not _sender or _sender == self.agent_id:
                continue
            _kind = getattr(_m, "kind", None)
            latest_per_key[(_sender, _kind)] = _m
        peer_messages_tuple = tuple(latest_per_key.values())

        view = AgentView(
            agent_id=self.agent_id,
            self_state=self_state,
            shared=shared,
            inbox=tuple(inbox_raw),
            scratchpad=tuple(scratchpad_raw),
            summary_before_window=summary_before_window,
            peer_messages=peer_messages_tuple,
        )

        # Optional pre-loop summarization (policy C)
        view = await self._maybe_summarize(view)

        # Build prompt for the tool loop
        prompt_messages = self._build_prompt(view)

        # Execute tool-calling loop
        outcome: _StepOutcome = await self._run_tool_loop(prompt_messages)

        # Build DRAFT outbox message from final response
        response_text = outcome.response.text or ""
        draft_msg = Message(
            sender=self.agent_id,
            kind=MessageKind.DRAFT,
            content=response_text,
        )

        # Compute cumulative token and cost stats
        usage = outcome.response.usage
        prev_tokens = int(self_state.get("tokens_spent") or 0)
        prev_cost = float(self_state.get("cost_spent_usd") or 0.0)
        new_tokens = prev_tokens + usage.total_tokens
        new_cost = prev_cost + outcome.response.cost_usd

        # Build per-agent state delta
        agent_delta: AgentState = {
            "agent_id": self.agent_id,
            "role": self.cfg.role,
            "inbox": [],  # delta — inbox consumed (empty delta = no new inbox)
            "outbox": [draft_msg],
            "scratchpad": outcome.scratchpad_events,
            "tool_calls": outcome.tool_calls,
            "tool_results": outcome.tool_results,
            "summary_before_window": view.summary_before_window or "",
            "step_count": step_count,
            "tokens_spent": new_tokens,
            "cost_spent_usd": new_cost,
        }

        return {
            "agents": {self.agent_id: agent_delta},
            "messages": [draft_msg],
            "llm_calls": [outcome.response],
        }

    # ------------------------------------------------------------------
    # Prompt building — scratchpad policy C
    # ------------------------------------------------------------------

    def _build_prompt(self, view: AgentView) -> list[Message]:
        """Assemble the LLM prompt using scratchpad policy C.

        Structure:
          1. System message (cfg.system_prompt).
          2. Task description from shared state (task_input).
          3. Summary of events before the window (if any).
          4. Inbox messages.
          5. Tail of scratchpad events: ``scratchpad[-window_size:]``.

        **Window semantics:** ``window_size`` is measured in *scratchpad events*,
        not logical steps. One logical step may produce up to 3 events
        (reasoning, tool_call, observation). This is intentional — a
        window_size of 15 for Planner means up to 5 logical steps of full
        context visibility.

        The original scratchpad list is NEVER mutated here.
        """
        messages: list[Message] = []

        # 1. System message
        messages.append(
            Message(
                sender="system",
                kind=MessageKind.REQUEST,
                content=self.cfg.system_prompt,
            )
        )

        # 2. Task description
        task_input: str = view.shared.get("task_input") or ""
        if task_input:
            messages.append(
                Message(
                    sender="system",
                    kind=MessageKind.REQUEST,
                    content=f"Task: {task_input}",
                )
            )

        # 3. Summary before window (if present)
        if view.summary_before_window:
            messages.append(
                Message(
                    sender="system",
                    kind=MessageKind.REQUEST,
                    content=f"[Summary of earlier context]\n{view.summary_before_window}",
                )
            )

        # 4. Inbox messages (explicitly routed by topology — empty for
        #    chain/star/mesh which rely on the peer_messages fallback below).
        for msg in view.inbox:
            messages.append(msg)

        # 4b. Peer fallback: most recent message from each peer agent. Provides
        #     critic↔executor↔planner visibility in chain/star where no
        #     routing node populates inbox. Skipped if inbox already contains
        #     an explicit delivery from that sender (avoids double-prompting).
        if view.peer_messages:
            seen_inbox_senders = {getattr(m, "sender", "") for m in view.inbox}
            for msg in view.peer_messages:
                if getattr(msg, "sender", "") in seen_inbox_senders:
                    continue
                messages.append(msg)

        # 5. Scratchpad window tail (read-only slice — does not mutate)
        window: tuple[dict[str, Any], ...] = view.scratchpad[-self.cfg.window_size :]
        for event in window:
            kind = event.get("kind", "reasoning")
            if kind == "reasoning":
                content = f"[Reasoning step {event.get('step', '?')}]: {event.get('content', '')}"
            elif kind == "tool_call":
                content = (
                    f"[Tool call step {event.get('step', '?')}]: "
                    f"{event.get('tool_name', '?')}({event.get('args', {})})"
                )
            elif kind == "observation":
                content = (
                    f"[Observation step {event.get('step', '?')}]: "
                    f"tool={event.get('tool_name', '?')} "
                    f"ok={event.get('ok', '?')} "
                    f"output={event.get('output_preview', '')}"
                )
            else:
                content = str(event)

            messages.append(
                Message(
                    sender=self.agent_id,
                    kind=MessageKind.REQUEST,
                    content=content,
                )
            )

        return messages

    # ------------------------------------------------------------------
    # Summarizer (policy C trigger)
    # ------------------------------------------------------------------

    async def _maybe_summarize(self, view: AgentView) -> AgentView:
        """Optionally summarize pre-window scratchpad events.

        Invoked ONCE per step, before the tool-loop. Returns a new
        ``AgentView`` (immutable — no mutation of original) with updated
        ``summary_before_window`` if summarization was triggered.

        Conditions for summarization:
          - ``summarizer_llm`` is not None.
          - ``estimate_prompt_tokens(provisional_prompt, llm.model_id) > cfg.context_token_budget``.

        When summarization fires:
          - Builds a summarizer prompt from the pre-window scratchpad events.
          - Calls ``summarizer_llm.ainvoke`` with ``agent_id=f"{agent_id}_summarizer"``.
          - Returns a new ``AgentView`` with the summarizer output as
            ``summary_before_window`` (pre-window events are dropped from the
            prompt-building view but the full scratchpad remains for the delta).

        When NOT triggered:
          - Returns ``view`` unchanged.
        """
        if self.summarizer_llm is None:
            return view

        # Build a provisional prompt and estimate its size
        provisional = self._build_prompt(view)
        estimated = estimate_prompt_tokens(provisional, self.llm.model_id)

        if estimated <= self.cfg.context_token_budget:
            return view

        # Summarization triggered — build a prompt for the summarizer
        pre_window_events = (
            view.scratchpad[: -self.cfg.window_size]
            if len(view.scratchpad) > self.cfg.window_size
            else ()
        )
        window_events = view.scratchpad[-self.cfg.window_size :]

        if not pre_window_events:
            # Nothing before the window to summarize
            return view

        # Build summarizer messages
        summary_content_parts = []
        for event in pre_window_events:
            summary_content_parts.append(str(event))
        events_text = "\n".join(summary_content_parts)

        summarizer_messages: list[Message] = [
            Message(
                sender="system",
                kind=MessageKind.REQUEST,
                content=(
                    "You are a concise summarizer. Summarize the following scratchpad "
                    "events into a compact narrative (1-3 sentences) preserving key facts "
                    "and outcomes:\n\n" + events_text
                ),
            )
        ]

        summarizer_response = await self.summarizer_llm.ainvoke(
            summarizer_messages,
            agent_id=f"{self.agent_id}_summarizer",
        )

        new_summary = summarizer_response.text or ""

        # Return a new view with updated summary and truncated scratchpad
        # (only window events remain in the prompt-building view)
        return AgentView(
            agent_id=view.agent_id,
            self_state=view.self_state,
            shared=view.shared,
            inbox=view.inbox,
            # Prompt-building view only has window events; full scratchpad returned in delta
            scratchpad=window_events,
            summary_before_window=new_summary,
            peer_messages=view.peer_messages,
        )

    # ------------------------------------------------------------------
    # Tool-calling loop
    # ------------------------------------------------------------------

    async def _run_tool_loop(self, messages: list[Message]) -> _StepOutcome:
        """Execute the tool-calling loop up to ``cfg.max_tool_iters`` iterations.

        Accumulation invariant (C1 fix):
          ``tool_calls_accum`` is NEVER reset between iterations.
          Both ``tool_calls_accum`` and ``tool_results`` accumulate across ALL
          loop iterations.

        Error handling:
          - ``ToolError`` is caught per-tool and converted to a failed
            ``ToolResult(ok=False)``.
          - ``BudgetExceededError`` is NEVER caught here — it propagates up
            through ``LLMWrapper.ainvoke`` as intended.

        TODO (M6/M7): Consider re-entrant summarization when tool outputs
        bloat the prompt. Currently mitigated by truncating observation
        preview to 2000 chars.
        """
        tool_calls_accum: list[ToolCall] = []  # CRITICAL: accumulated across ALL iterations
        tool_results: list[ToolResult] = []
        scratchpad_events: list[dict[str, Any]] = []
        current_messages = list(messages)  # working copy

        tools_schema = self._build_tools_schema()

        # Last response placeholder — will be set on first iteration at minimum
        response: LLMResponse | None = None

        for loop_iter in range(
            max(1, self.cfg.max_tool_iters) if self.cfg.max_tool_iters > 0 else 1
        ):
            response = await self.llm.ainvoke(
                current_messages,
                tools=tools_schema,
                agent_id=self.agent_id,
            )

            # Record reasoning event
            scratchpad_events.append(
                {
                    "kind": "reasoning",
                    "content": response.text or "",
                    "step": loop_iter,
                }
            )

            # Check if we should continue the loop
            if response.finish_reason != "tool_calls" or not response.tool_calls:
                break

            # Process each tool call
            for tool_call in response.tool_calls:
                # CRITICAL C1: accumulate across all iterations
                tool_calls_accum.append(tool_call)

                # Record tool_call event
                scratchpad_events.append(
                    {
                        "kind": "tool_call",
                        "tool_name": tool_call.tool_name,
                        "args": dict(tool_call.args),
                        "call_id": str(tool_call.id),
                        "step": loop_iter,
                    }
                )

                # Invoke the tool — only ToolError is caught; BudgetExceededError propagates
                try:
                    result = await self.tools.ainvoke_by_name(tool_call.tool_name, tool_call)
                except ToolError as exc:
                    result = ToolResult(
                        call_id=tool_call.id,
                        ok=False,
                        output=None,
                        error=str(exc),
                        latency_ms=0,
                    )

                tool_results.append(result)

                # Record observation event (truncated to 2000 chars)
                output_preview = str(result.output)[:2000]
                scratchpad_events.append(
                    {
                        "kind": "observation",
                        "tool_name": tool_call.tool_name,
                        "ok": result.ok,
                        "output_preview": output_preview,
                        "step": loop_iter,
                    }
                )

                # Append synthetic tool-result message to working conversation
                current_messages.append(
                    Message(
                        sender="tool",
                        kind=MessageKind.REQUEST,
                        content=f"Tool {tool_call.tool_name} result: {output_preview}",
                        refs=(tool_call.id,),
                    )
                )

            # If max_tool_iters reached on this iteration, stop
            if loop_iter >= self.cfg.max_tool_iters - 1:
                break

        # Guaranteed non-None after at least one ainvoke call above
        assert response is not None, "Tool loop ran 0 iterations — this should not happen"

        return _StepOutcome(
            response=response,
            scratchpad_events=scratchpad_events,
            tool_calls=tool_calls_accum,
            tool_results=tool_results,
        )

    # ------------------------------------------------------------------
    # Tool schema builder
    # ------------------------------------------------------------------

    def _build_tools_schema(self) -> list[dict[str, Any]] | None:
        """Build the tools schema list for LLMWrapper.ainvoke.

        Iterates ``cfg.tools``, looks up each name in the registry, and
        constructs an OpenAI-compatible tool-call schema dict.

        Unknown tool names are logged via structlog at WARNING level (NOT
        ``warnings.warn`` — pyproject.toml has ``filterwarnings = ["error"]``
        which would convert Python warnings into exceptions under pytest).

        Returns:
            List of schema dicts, or ``None`` if no valid tools found.
        """
        schemas: list[dict[str, Any]] = []

        for name in self.cfg.tools:
            try:
                tool = self.tools.get(name)
            except ToolError:
                logger.warning(
                    "tool name not in registry",
                    tool_name=name,
                    agent_id=self.agent_id,
                    role=self.cfg.role,
                )
                continue

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.schema.name,
                        "description": tool.schema.description,
                        "parameters": tool.schema.parameters,
                    },
                }
            )

        return schemas if schemas else None
