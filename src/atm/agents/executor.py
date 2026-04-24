"""Executor agent subclass for ATM multi-agent framework.

The Executor is responsible for implementing the plan produced by the Planner:
running code, writing files, and reporting results. The Executor acts on
instructions and produces concrete outputs (code, files, stdout).
All prompt details come from the YAML config (conf/agents/executor.yaml).
"""

from __future__ import annotations

from atm.agents.base import Agent


class Executor(Agent):
    """Executor role: implements the plan by running code and writing files.

    Uses tools: code_run, file_write, file_read, calculator.
    Configured via conf/agents/executor.yaml (window_size=12).
    """

    pass
