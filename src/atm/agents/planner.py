"""Planner agent subclass for ATM multi-agent framework.

The Planner is responsible for decomposing the problem into explicit steps,
writing them via todo_write, and updating the plan via plan_update as new
information becomes available. The Planner does NOT write code directly.
All prompt details come from the YAML config (conf/agents/planner.yaml).
"""

from __future__ import annotations

from atm.agents.base import Agent


class Planner(Agent):
    """Planner role: decomposes tasks into steps, manages the plan lifecycle.

    Uses tools: todo_write, plan_update.
    Configured via conf/agents/planner.yaml (window_size=15).
    """

    pass
