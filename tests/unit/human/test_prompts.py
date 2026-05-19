"""Unit tests for atm.human.prompts — build_role_prompt (M9 Step 1.1)."""

from __future__ import annotations

import uuid

import pytest

from atm.core.types import HumanContext, HumanRole, Message, MessageKind


def _make_ctx(
    role: HumanRole,
    question: str = "Should the agent proceed with the proposed plan?",
) -> HumanContext:
    msg = Message(
        sender="planner",
        kind=MessageKind.DRAFT,
        content="Draft proposal text",
    )
    return HumanContext(
        run_id=uuid.uuid4(),
        role=role,
        question=question,
        recent_messages=(msg,),
        allowed_actions=("approve", "reject", "revise"),
    )


ALL_ROLES = list(HumanRole)


@pytest.mark.parametrize("role", ALL_ROLES)
def test_build_role_prompt_returns_two_strings(role: HumanRole) -> None:
    from atm.human.prompts import build_role_prompt

    ctx = _make_ctx(role)
    result = build_role_prompt(role, ctx)
    assert isinstance(result, tuple), "build_role_prompt must return a tuple"
    assert len(result) == 2, "build_role_prompt must return exactly 2 elements"
    system_prompt, user_prompt = result
    assert isinstance(system_prompt, str), "system_prompt must be str"
    assert isinstance(user_prompt, str), "user_prompt must be str"
    assert len(system_prompt) > 0, "system_prompt must not be empty"
    assert len(user_prompt) > 0, "user_prompt must not be empty"


@pytest.mark.parametrize("role", ALL_ROLES)
def test_system_prompt_mentions_role(role: HumanRole) -> None:
    from atm.human.prompts import build_role_prompt

    ctx = _make_ctx(role)
    system_prompt, _ = build_role_prompt(role, ctx)
    assert role.value.lower() in system_prompt.lower(), (
        f"System prompt for role '{role.value}' must mention the role name; "
        f"got: {system_prompt[:120]!r}"
    )


@pytest.mark.parametrize("role", ALL_ROLES)
def test_user_prompt_embeds_question(role: HumanRole) -> None:
    from atm.human.prompts import build_role_prompt

    question = f"Unique question for {role.value}: proceed or not?"
    ctx = _make_ctx(role, question=question)
    _, user_prompt = build_role_prompt(role, ctx)
    assert question in user_prompt, (
        f"User prompt must embed ctx.question; "
        f"question={question!r}, user_prompt={user_prompt[:200]!r}"
    )


@pytest.mark.parametrize("role", ALL_ROLES)
def test_user_prompt_lists_allowed_actions(role: HumanRole) -> None:
    from atm.human.prompts import build_role_prompt

    ctx = _make_ctx(role)
    _, user_prompt = build_role_prompt(role, ctx)
    for action in ctx.allowed_actions:
        assert action in user_prompt, (
            f"User prompt must list allowed action '{action}'; user_prompt={user_prompt[:200]!r}"
        )


def test_different_roles_have_different_system_prompts() -> None:
    from atm.human.prompts import build_role_prompt

    prompts: dict[str, str] = {}
    for role in ALL_ROLES:
        ctx = _make_ctx(role)
        sys_p, _ = build_role_prompt(role, ctx)
        prompts[role.value] = sys_p

    unique_prompts = set(prompts.values())
    assert len(unique_prompts) == len(ALL_ROLES), "Each role must have a distinct system prompt"
