"""Unit tests for Critic.step() — BUG-3 regression tests.

Verifies that the Critic subclass emits MessageKind.DECISION outbox messages
(not DRAFT) with payload["approved"] correctly parsed from LLM response text.

Test cases:
  1. test_critic_emits_decision_kind        — outbox kind is DECISION (not DRAFT)
  2. test_critic_approved_true_on_approve   — "APPROVE" in text → approved=True
  3. test_critic_approved_false_on_reject   — "REJECT" in text → approved=False
  4. test_critic_payload_has_comment        — payload["comment"] == response text
  5. test_critic_messages_list_is_decision  — delta["messages"] also DECISION
  6. test_base_agent_still_emits_draft      — base Agent is not affected
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.agents.critic import Critic
from atm.core.types import MessageKind
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_echo_llm(response_text: str) -> LLMWrapper:
    """Build an echo LLMWrapper that always returns response_text."""
    fake = FakeLLM(mode="echo")
    return LLMWrapper(
        model_id="fake:echo",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_scripted_llm(fixture_name: str) -> LLMWrapper:
    """Build a scripted FakeLLM from a fixture file."""
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_cfg(role: str = "critic") -> AgentConfig:
    return AgentConfig(
        role=role,
        system_prompt=f"You are a {role}.",
        tools=[],
        max_tool_iters=0,
    )


def _make_state() -> dict[str, Any]:
    return {
        "shared": {"task_input": "Review the executor output."},
        "agents": {},
        "messages": [],
        "llm_calls": [],
    }


class TestCriticEmitsDecision:
    """Critic.step() must emit DECISION kind messages."""

    async def test_critic_emits_decision_kind(self) -> None:
        """Outbox messages from Critic.step() must have kind=DECISION."""
        llm = _make_scripted_llm("m6_chain_critic.yaml")
        cfg = _make_cfg()
        critic = Critic(agent_id="critic", cfg=cfg, llm=llm, tools=ToolRegistry())

        state = _make_state()
        delta = await critic.step(state)

        outbox = delta["agents"]["critic"]["outbox"]
        assert len(outbox) >= 1
        for msg in outbox:
            assert msg.kind == MessageKind.DECISION, f"Expected DECISION, got {msg.kind}"

    async def test_critic_approved_true_on_approve(self) -> None:
        """APPROVE in response text → payload['approved'] == True."""
        llm = _make_scripted_llm("m6_chain_critic.yaml")
        cfg = _make_cfg()
        critic = Critic(agent_id="critic", cfg=cfg, llm=llm, tools=ToolRegistry())

        delta = await critic.step(_make_state())
        outbox = delta["agents"]["critic"]["outbox"]
        assert len(outbox) >= 1
        msg = outbox[0]
        assert msg.kind == MessageKind.DECISION
        assert msg.payload is not None
        assert msg.payload["approved"] is True, (
            f"Expected approved=True for 'APPROVE fib', got {msg.payload}"
        )

    async def test_critic_approved_false_on_non_approve(self) -> None:
        """Absence of 'approve' in response text → payload['approved'] == False."""
        llm = _make_echo_llm("REJECT: code is incorrect")
        cfg = _make_cfg()
        critic = Critic(agent_id="critic", cfg=cfg, llm=llm, tools=ToolRegistry())

        state = {
            "shared": {"task_input": "REJECT: output is wrong."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }
        delta = await critic.step(state)
        outbox = delta["agents"]["critic"]["outbox"]
        assert len(outbox) >= 1
        msg = outbox[0]
        assert msg.kind == MessageKind.DECISION
        assert msg.payload is not None
        assert msg.payload["approved"] is False, (
            f"Expected approved=False for non-approve text, got {msg.payload}"
        )

    async def test_critic_payload_has_comment(self) -> None:
        """payload['comment'] should equal the response content."""
        llm = _make_scripted_llm("m6_chain_critic.yaml")
        cfg = _make_cfg()
        critic = Critic(agent_id="critic", cfg=cfg, llm=llm, tools=ToolRegistry())

        delta = await critic.step(_make_state())
        outbox = delta["agents"]["critic"]["outbox"]
        msg = outbox[0]
        assert "comment" in (msg.payload or {}), "payload must have 'comment' key"
        assert msg.payload["comment"] == msg.content

    async def test_critic_messages_list_is_decision(self) -> None:
        """delta['messages'] list should also contain DECISION messages."""
        llm = _make_scripted_llm("m6_chain_critic.yaml")
        cfg = _make_cfg()
        critic = Critic(agent_id="critic", cfg=cfg, llm=llm, tools=ToolRegistry())

        delta = await critic.step(_make_state())
        messages = delta.get("messages", [])
        for msg in messages:
            assert msg.kind == MessageKind.DECISION, (
                f"Expected DECISION in messages list, got {msg.kind}"
            )


class TestBaseAgentUnaffected:
    """Base Agent.step() must still emit DRAFT (Critic subclass must not affect it)."""

    async def test_base_agent_still_emits_draft(self) -> None:
        """Agent.step() (non-critic) must emit DRAFT, not DECISION."""
        fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / "m5_agent_basic.yaml")
        llm = LLMWrapper(
            model_id="fake:scripted",
            pricing=_make_pricing(),
            budget=_make_budget(),
            llm=fake,
        )
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
        )
        agent = Agent(agent_id="planner", cfg=cfg, llm=llm, tools=ToolRegistry())
        state = {
            "shared": {"task_input": "Write a plan."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }
        delta = await agent.step(state)
        outbox = delta["agents"]["planner"]["outbox"]
        assert len(outbox) == 1
        assert outbox[0].kind == MessageKind.DRAFT, (
            f"Base Agent must emit DRAFT, got {outbox[0].kind}"
        )
