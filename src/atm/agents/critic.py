"""Critic agent subclass for ATM multi-agent framework.

The Critic is responsible for reviewing the Executor's output: running tests,
checking diffs, and linting code. The Critic emits an approval signal
(critic_approved) when the output meets quality standards, or requests
revision with structured feedback otherwise.
All prompt details come from the YAML config (conf/agents/critic.yaml).

BUG-3 fix:
    Agent.step() always emits ``MessageKind.DRAFT`` for outbox messages.
    But ``_critic_postprocess`` in Star and Chain topologies expects
    ``MessageKind.DECISION`` with ``payload={"approved": bool, "comment": str}``.
    This subclass overrides ``step()`` to convert the outbox DRAFT message into
    a DECISION message by parsing the LLM response text for "APPROVE" / "REJECT"
    keywords (case-insensitive). "APPROVE" in the response text → approved=True;
    otherwise → approved=False.
"""

from __future__ import annotations

from typing import Any

from atm.agents.base import Agent
from atm.core.state import GraphState
from atm.core.types import Message, MessageKind


class Critic(Agent):
    """Critic role: reviews output quality via tests, diffs, and linting.

    Uses tools: test_run, diff, lint.
    Configured via conf/agents/critic.yaml (window_size=15).

    Overrides ``step()`` to emit ``MessageKind.DECISION`` outbox messages
    (with ``payload={"approved": bool, "comment": str}``) instead of the
    base class ``MessageKind.DRAFT``. This is required by ``_critic_postprocess``
    in both StarTopology and ChainTopology.

    Approval heuristic: the LLM response text is scanned for "APPROVE"
    (case-insensitive). Presence → approved=True; absence → approved=False.
    """

    async def step(self, state: GraphState) -> dict[str, Any]:
        """Execute one critic step and emit a DECISION-kind outbox message.

        Delegates to the parent ``Agent.step()`` for the tool-loop and
        scratchpad management, then replaces any DRAFT messages in the
        returned delta with DECISION messages.

        The approved flag is True if the LLM response text contains "APPROVE"
        (case-insensitive). The original response text is preserved as the
        ``comment`` field in the payload.

        Args:
            state: GraphState dict.

        Returns:
            Delta dict with DECISION messages in agents[agent_id]["outbox"]
            and messages list.
        """
        delta = await super().step(state)

        # Convert DRAFT outbox messages to DECISION with approved payload
        agents_delta: dict[str, Any] = dict(delta.get("agents", {}))
        self_delta: dict[str, Any] = dict(agents_delta.get(self.agent_id, {}))
        outbox: list[Any] = list(self_delta.get("outbox", []))
        messages: list[Any] = list(delta.get("messages", []))

        new_outbox: list[Message] = []
        new_messages: list[Message] = []

        for msg in outbox:
            content: str = getattr(msg, "content", "") or ""
            approved: bool = "approve" in content.lower()
            decision_msg = Message(
                sender=self.agent_id,
                kind=MessageKind.DECISION,
                content=content,
                payload={"approved": approved, "comment": content},
            )
            new_outbox.append(decision_msg)

        for msg in messages:
            content = getattr(msg, "content", "") or ""
            approved = "approve" in content.lower()
            decision_msg = Message(
                sender=self.agent_id,
                kind=MessageKind.DECISION,
                content=content,
                payload={"approved": approved, "comment": content},
            )
            new_messages.append(decision_msg)

        # Rebuild delta with DECISION messages
        self_delta = dict(self_delta)
        self_delta["outbox"] = new_outbox
        agents_delta = dict(agents_delta)
        agents_delta[self.agent_id] = self_delta
        delta = dict(delta)
        delta["agents"] = agents_delta
        delta["messages"] = new_messages

        return delta
