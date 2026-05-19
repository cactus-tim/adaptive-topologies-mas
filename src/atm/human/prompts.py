"""Per-role system prompts and user prompt builder for HITL interactions."""

from __future__ import annotations

from atm.core.types import HumanContext, HumanRole

__all__ = ["ROLE_SYSTEM_PROMPTS", "build_role_prompt"]

ROLE_SYSTEM_PROMPTS: dict[HumanRole, str] = {
    HumanRole.COORDINATOR: (
        "You are the coordinator overseeing this multi-agent run. "
        "Your role as coordinator is to assess whether the agents are on track, "
        "identify blocking issues, and make high-level steering decisions. "
        "Be decisive: the agents cannot proceed without your input."
    ),
    HumanRole.REVIEWER: (
        "You are a reviewer evaluating the agents' output for quality and correctness. "
        "Your role as reviewer is to check that the result meets the stated requirements, "
        "flag specific deficiencies, and request targeted revisions if needed. "
        "Provide actionable feedback."
    ),
    HumanRole.JUDGE: (
        "You are the judge adjudicating between competing agent proposals. "
        "Your role as judge is to evaluate each argument on its merits, apply the stated "
        "criteria consistently, and deliver a clear verdict. "
        "Explain your reasoning concisely."
    ),
    HumanRole.PEER: (
        "You are a peer collaborator providing domain expertise to the agents. "
        "Your role as peer is to validate technical assumptions, supply missing context, "
        "and suggest improvements grounded in your expertise. "
        "Be specific and constructive."
    ),
    HumanRole.MONITOR: (
        "You are a monitor tracking the run for safety, compliance, and resource usage. "
        "Your role as monitor is to flag policy violations, anomalies, or unexpected "
        "agent behaviour. Intervene only when necessary; otherwise confirm that the run "
        "may continue."
    ),
}


def build_role_prompt(role: HumanRole, ctx: HumanContext) -> tuple[str, str]:
    """Build ``(system_prompt, user_prompt)`` for a given role and HumanContext."""
    system_prompt = ROLE_SYSTEM_PROMPTS[role]

    actions_str = ", ".join(f'"{a}"' for a in ctx.allowed_actions)
    deadline_line = (
        f"\nDeadline: respond within {ctx.deadline_s} seconds."
        if ctx.deadline_s is not None
        else ""
    )

    user_prompt = (
        f"Question: {ctx.question}\n"
        f"\nAllowed actions: {actions_str}"
        f"{deadline_line}\n"
        f"\nRecent context ({len(ctx.recent_messages)} message(s)):\n"
        + _format_recent_messages(ctx)
        + "\n\nRespond with one of the allowed actions and an optional comment."
    )

    return system_prompt, user_prompt


def _format_recent_messages(ctx: HumanContext) -> str:
    """Render recent_messages as a readable block for inclusion in user_prompt."""
    if not ctx.recent_messages:
        return "  (none)"
    lines: list[str] = []
    for msg in ctx.recent_messages:
        lines.append(f"  [{msg.kind.value}] {msg.sender}: {msg.content}")
    return "\n".join(lines)
