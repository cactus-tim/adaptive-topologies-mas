"""Critic agent subclass for ATM multi-agent framework.

The Critic is responsible for reviewing the Executor's output: running tests,
checking diffs, and linting code. The Critic emits an approval signal
(critic_approved) when the output meets quality standards, or requests
revision with structured feedback otherwise.
All prompt details come from the YAML config (conf/agents/critic.yaml).
"""

from __future__ import annotations

from atm.agents.base import Agent


class Critic(Agent):
    """Critic role: reviews output quality via tests, diffs, and linting.

    Uses tools: test_run, diff, lint.
    Configured via conf/agents/critic.yaml (window_size=15).
    """

    pass
