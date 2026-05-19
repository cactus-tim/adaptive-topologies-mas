"""Debater agent subclass for ATM multi-agent framework.

The Debater participates in structured debate topologies. It defends a fixed
stance ('pro' or 'contra') passed via AgentConfig.params['stance'], which is
resolved into the system_prompt at construction time. The prompt template
must contain a '{{stance}}' placeholder that is replaced once — the resulting
prompt is immutable for the lifetime of the agent.
All other prompt details come from the YAML config (conf/agents/debater.yaml).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atm.agents.base import Agent

if TYPE_CHECKING:
    from atm.agents.config import AgentConfig
    from atm.llm.wrapper import LLMWrapper
    from atm.tools.base import ToolRegistry


class Debater(Agent):
    """Debater role: defends a fixed stance in structured debate topologies.

    Resolves '{{stance}}' in the system_prompt at __init__ time via
    cfg.model_copy(update={"system_prompt": resolved_prompt}).

    Uses tools: duckduckgo_search, url_fetch.
    Configured via conf/agents/debater.yaml (window_size=12).

    Raises:
        ValueError: If cfg.params['stance'] is not 'pro' or 'contra'.
    """

    def __init__(
        self,
        agent_id: str,
        cfg: AgentConfig,
        llm: LLMWrapper,
        tools: ToolRegistry,
        summarizer_llm: LLMWrapper | None = None,
    ) -> None:
        stance: Any = cfg.params.get("stance")
        if stance not in ("pro", "contra"):
            raise ValueError(f"Debater stance must be 'pro' or 'contra', got: {stance!r}")

        resolved_prompt = cfg.system_prompt.replace("{{stance}}", stance)
        new_cfg = cfg.model_copy(update={"system_prompt": resolved_prompt})

        super().__init__(
            agent_id,
            new_cfg,
            llm,
            tools,
            summarizer_llm=summarizer_llm,
        )

        assert "{{stance}}" not in self.cfg.system_prompt, (
            "Debater invariant violated: '{{stance}}' still present in system_prompt "
            "after resolution."
        )
