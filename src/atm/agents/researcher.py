"""Researcher agent subclass for ATM multi-agent framework.

The Researcher is responsible for gathering information relevant to the task:
running semantic searches, web searches, and fetching URL content. The
Researcher synthesizes findings and passes them to the Executor or Planner.
All prompt details come from the YAML config (conf/agents/researcher.yaml).
"""

from __future__ import annotations

from atm.agents.base import Agent


class Researcher(Agent):
    """Researcher role: information gathering via search and web tools.

    Uses tools: semantic_search, duckduckgo_search, url_fetch, file_read.
    Configured via conf/agents/researcher.yaml (window_size=12).
    """

    pass
