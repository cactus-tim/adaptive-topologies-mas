"""LLMWrapper — unified LLM abstraction for ATM.

Wraps a LangChain BaseChatModel (or FakeLLM) with:
- Per-call budget pre-check and post-call recording (three-tier).
- Automatic retry on transient errors (429 / 5xx) via RetryPolicy.
- Usage parsing for both OpenAI and Anthropic response shapes.
- Cost calculation via Pricing table.
- Message.to_lc conversion for list[Message] inputs.
- Prompt-cache support via inject_cache_control (Anthropic only).

Published contract: ``LLMWrapper.ainvoke(messages) -> LLMResponse``
``astream`` is stubbed — raises ``NotImplementedError`` until M4+.
"""

from __future__ import annotations

import datetime
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from atm.core.types import LLMResponse, Message, TokenUsage, ToolCall
from atm.llm.budget import BudgetLevel, BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.retry import RetryPolicy, with_retry

# ---------------------------------------------------------------------------
# Default retry policy used when caller passes retry_policy=None
# ---------------------------------------------------------------------------

_DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay_s=1.0,
    max_delay_s=30.0,
    jitter=True,
)


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _count_tokens_tiktoken(text: str, model_id: str) -> int:
    """Count tokens using tiktoken for OpenAI-family models.

    Strips the ``"provider:"`` prefix before calling ``encoding_for_model``.
    Falls back to heuristic on any error.
    """
    try:
        import tiktoken

        bare_model = model_id.split(":", 1)[-1]
        enc = tiktoken.encoding_for_model(bare_model)
        return len(enc.encode(text))
    except Exception:
        return _count_tokens_heuristic(text)


def _count_tokens_heuristic(text: str) -> int:
    """Heuristic token count: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _count_prompt_tokens(messages: list[BaseMessage], model_id: str) -> int:
    """Estimate total prompt token count for a list of LC messages."""
    provider = model_id.split(":", 1)[0] if ":" in model_id else "unknown"
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)

        if provider == "openai":
            total += _count_tokens_tiktoken(text, model_id)
        else:
            total += _count_tokens_heuristic(text)

    return max(1, total)


# ---------------------------------------------------------------------------
# Usage parsing helpers
# ---------------------------------------------------------------------------


def _parse_usage_openai(ai_msg: AIMessage) -> tuple[TokenUsage, int]:
    """Parse OpenAI-style usage_metadata. Returns (TokenUsage, cache_write_tokens=0)."""
    raw_meta = ai_msg.usage_metadata
    # usage_metadata can be UsageMetadata (TypedDict-like) or dict
    meta: dict[str, Any] = dict(raw_meta) if raw_meta else {}
    input_tokens: int = int(meta.get("input_tokens", 0))
    output_tokens: int = int(meta.get("output_tokens", 0))
    total_tokens: int = int(meta.get("total_tokens", input_tokens + output_tokens))

    raw_details = meta.get("input_token_details")
    details: dict[str, Any] = dict(raw_details) if raw_details else {}
    cache_read: int = int(details.get("cache_read", 0))

    usage = TokenUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        cached_input_tokens=cache_read,
        total_tokens=total_tokens,
    )
    return usage, 0  # no cache_write in OpenAI convention


def _parse_usage_anthropic(ai_msg: AIMessage) -> tuple[TokenUsage, int]:
    """Parse Anthropic-style response_metadata['usage']. Returns (TokenUsage, cache_write_tokens)."""
    resp_meta: dict[str, Any] = ai_msg.response_metadata or {}
    usage_raw_val = resp_meta.get("usage") or {}
    usage_raw: dict[str, Any] = dict(usage_raw_val) if usage_raw_val else {}

    input_tokens: int = int(usage_raw.get("input_tokens", 0))
    output_tokens: int = int(usage_raw.get("output_tokens", 0))
    cache_read: int = int(usage_raw.get("cache_read_input_tokens", 0))
    cache_write: int = int(usage_raw.get("cache_creation_input_tokens", 0))
    total_tokens: int = input_tokens + output_tokens

    usage = TokenUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        cached_input_tokens=cache_read,
        total_tokens=total_tokens,
    )
    return usage, cache_write


def _detect_and_parse_usage(
    ai_msg: AIMessage,
    provider: str,
) -> tuple[TokenUsage, int]:
    """Detect response shape and parse usage accordingly.

    Priority:
    1. If ``usage_metadata`` is present and non-empty → OpenAI shape.
    2. If ``response_metadata.usage`` is present → Anthropic shape.
    3. Fallback: zero usage.
    """
    if ai_msg.usage_metadata:
        return _parse_usage_openai(ai_msg)

    resp_meta: dict[str, Any] = ai_msg.response_metadata or {}
    if "usage" in resp_meta:
        return _parse_usage_anthropic(ai_msg)

    # Fallback: zero usage
    usage = TokenUsage(
        prompt_tokens=0,
        completion_tokens=0,
        cached_input_tokens=0,
        total_tokens=0,
    )
    return usage, 0


# ---------------------------------------------------------------------------
# Tool-call parsing helper
# ---------------------------------------------------------------------------


def _parse_tool_calls(ai_msg: AIMessage, issued_by: str) -> tuple[ToolCall, ...]:
    """Convert LC tool_calls list (list of dicts) → tuple[ToolCall, ...]."""
    raw = getattr(ai_msg, "tool_calls", None)
    if not raw:
        return ()
    result: list[ToolCall] = []
    for tc in raw:
        result.append(
            ToolCall(
                tool_name=tc.get("name", ""),
                args=tc.get("args", {}),
                issued_by=issued_by,
            )
        )
    return tuple(result)


# ---------------------------------------------------------------------------
# Model version extraction helper
# ---------------------------------------------------------------------------


def _extract_model_version(response_metadata: dict[str, Any]) -> str | None:
    """Extract the actual model version/fingerprint from LLM response_metadata.

    Priority order (matches both Anthropic and OpenAI response shapes):
      1. ``system_fingerprint`` — OpenAI uniquely identifies a model deployment
      2. ``model_name``         — OpenAI canonical model name
      3. ``model``              — Anthropic canonical model name (also OpenAI fallback)

    Returns None if no non-empty value is found.
    """
    for key in ("system_fingerprint", "model_name", "model"):
        value = response_metadata.get(key)
        if value and isinstance(value, str) and value.strip():
            return str(value)
    return None


# ---------------------------------------------------------------------------
# Finish reason helper
# ---------------------------------------------------------------------------

_VALID_FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "content_filter", "error"})


def _parse_finish_reason(ai_msg: AIMessage) -> str:
    """Extract finish_reason from response_metadata, normalising to known values."""
    resp_meta: dict[str, Any] = ai_msg.response_metadata or {}

    # OpenAI uses finish_reason; Anthropic uses stop_reason
    raw: str | None = resp_meta.get("finish_reason") or resp_meta.get("stop_reason")

    if raw is None:
        return "stop"

    # Normalise Anthropic "end_turn" → "stop"
    if raw == "end_turn":
        return "stop"
    if raw in _VALID_FINISH_REASONS:
        return raw
    return "stop"


# ---------------------------------------------------------------------------
# LLMWrapper
# ---------------------------------------------------------------------------


class LLMWrapper:
    """Unified LLM wrapper with budget gating, retry, and usage tracking.

    Accepts an injected ``llm`` for testing (``FakeLLM`` or an ``AsyncMock``),
    or builds a real ``BaseChatModel`` via ``init_chat_model`` when ``llm`` is None.

    Args:
        model_id:      Model identifier in ``"provider:model"`` format.
        pricing:       Loaded ``Pricing`` instance for cost calculation.
        budget:        ``BudgetTracker`` for three-tier budget enforcement.
        cfg:           Optional model config (reserved for future use).
        retry_policy:  ``RetryPolicy`` for transient-error retry.
                       Defaults to ``_DEFAULT_RETRY_POLICY`` when None.
        llm:           Injected LLM (testing). When None, a real model is built.
        callbacks:     Optional LangChain callback list.
    """

    def __init__(
        self,
        model_id: str,
        *,
        pricing: Pricing,
        budget: BudgetTracker,
        cfg: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
        llm: BaseChatModel | FakeLLM | Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self._model_id = model_id
        self._pricing = pricing
        self._budget = budget
        self._cfg = cfg or {}
        self._retry_policy: RetryPolicy = retry_policy or _DEFAULT_RETRY_POLICY
        self._callbacks = callbacks

        if llm is not None:
            self._llm: Any = llm
        else:
            # Build a real model via init_chat_model when no injection provided.
            from langchain.chat_models import init_chat_model

            self._llm = init_chat_model(model_id, **self._cfg)

        # Parse provider for cost/retry logic
        parts = model_id.split(":", 1)
        self._provider: str = parts[0] if len(parts) == 2 else "unknown"
        self._bare_model: str = parts[1] if len(parts) == 2 else model_id

        # Reproducibility: actual model version reported by provider after each call.
        # Updated in-place by ainvoke(); only set to a non-None value, never cleared.
        self.last_model_version: str | None = None

    @property
    def model_id(self) -> str:
        """The provider-qualified model identifier (e.g. ``'fake:deterministic'``)."""
        return self._model_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_lc_messages(
        self,
        messages: list[Any],
    ) -> list[BaseMessage]:
        """Convert input messages to list[BaseMessage]."""
        if not messages:
            return []

        first = messages[0]

        if isinstance(first, Message):
            # list[Message] → convert via to_lc
            return [m.to_lc() for m in messages]

        if isinstance(first, BaseMessage):
            # Already BaseMessage
            return list(messages)

        if isinstance(first, dict):
            # dict format — use LangChain conversion
            from langchain_core.messages import convert_to_messages

            return convert_to_messages(messages)

        return list(messages)

    async def _invoke_llm(
        self,
        lc_messages: list[BaseMessage],
        invoke_kwargs: dict[str, Any],
        *,
        agent_id: str = "default",
    ) -> Any:
        """Call the underlying LLM with retry, returning response (AIMessage or LLMResponse)."""

        async def _call() -> Any:
            # FakeLLM has its own ainvoke signature and returns LLMResponse directly
            if isinstance(self._llm, FakeLLM):
                return await self._llm.ainvoke(list(lc_messages), agent_id=agent_id)
            return await self._llm.ainvoke(lc_messages, **invoke_kwargs)

        return await with_retry(
            _call,
            policy=self._retry_policy,
            provider=self._provider,
            model=self._bare_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        messages: list[Message] | list[BaseMessage] | list[dict[str, Any]],
        *,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        agent_id: str = "default",
        cache_key: str | None = None,
        **opts: Any,
    ) -> LLMResponse:
        """Invoke the LLM with budget gating, retry, and usage tracking.

        Flow:
        1. Convert input messages to list[BaseMessage].
        2. Estimate prompt tokens → compute estimated cost.
        3. Pre-call budget check at CALL / RUN / EXPERIMENT levels.
        4. If Anthropic + cache_key: inject cache_control.
        5. Invoke underlying LLM with retry.
        6. Parse usage and compute actual cost.
        7. Record cost at RUN and EXPERIMENT levels.
        8. Build and return LLMResponse.
        """
        started_monotonic = time.monotonic()
        started_dt = datetime.datetime.now(datetime.UTC)

        # --- Step 1: convert messages ---
        lc_messages = self._to_lc_messages(list(messages))

        # --- Step 2: estimate prompt tokens and cost ---
        prompt_tokens = _count_prompt_tokens(lc_messages, self._model_id)
        # Estimate completion tokens as 10% of prompt tokens (conservative)
        estimated_completion = max(1, prompt_tokens // 10)
        estimated_cost = self._pricing.estimate(
            self._model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=estimated_completion,
        )

        # --- Step 3: pre-call budget checks ---
        await self._budget.check(estimated_cost, level=BudgetLevel.CALL)
        await self._budget.check(estimated_cost, level=BudgetLevel.RUN)
        await self._budget.check(estimated_cost, level=BudgetLevel.EXPERIMENT)

        # --- Step 4: inject cache control for Anthropic if cache_key is set ---
        if self._provider == "anthropic" and cache_key is not None:
            from atm.llm.providers.anthropic import inject_cache_control

            lc_messages = inject_cache_control(lc_messages, key=cache_key)

        # --- Step 5: build invoke kwargs and call LLM with retry ---
        invoke_kwargs: dict[str, Any] = {}
        if tools is not None:
            invoke_kwargs["tools"] = tools
        if tool_choice is not None:
            invoke_kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            invoke_kwargs["response_format"] = response_format
        invoke_kwargs.update(opts)

        ai_msg = await self._invoke_llm(lc_messages, invoke_kwargs, agent_id=agent_id)

        # --- Step 6: parse usage and cost ---
        # FakeLLM returns LLMResponse directly; bypass AIMessage parsing in that case.
        if isinstance(ai_msg, LLMResponse):
            fake_resp = ai_msg
            actual_cost = self._pricing.cost(
                self._model_id,
                fake_resp.usage,
                cache_write_tokens=0,
            )
            # --- Step 7: record cost ---
            await self._budget.record(actual_cost, level=BudgetLevel.RUN)
            await self._budget.record(actual_cost, level=BudgetLevel.EXPERIMENT)

            latency_ms = int((time.monotonic() - started_monotonic) * 1000)
            return LLMResponse(
                id=fake_resp.id,
                model=self._model_id,
                text=fake_resp.text,
                tool_calls=fake_resp.tool_calls,
                usage=fake_resp.usage,
                cost_usd=actual_cost,
                latency_ms=latency_ms,
                finish_reason=fake_resp.finish_reason,
                started_at=started_dt,
            )

        # --- Capture actual model version from provider response_metadata ---
        resp_meta_for_version: dict[str, Any] = dict(ai_msg.response_metadata or {})
        extracted_version = _extract_model_version(resp_meta_for_version)
        if extracted_version is not None:
            self.last_model_version = extracted_version

        usage, cache_write_tokens = _detect_and_parse_usage(ai_msg, self._provider)
        actual_cost = self._pricing.cost(
            self._model_id,
            usage,
            cache_write_tokens=cache_write_tokens,
        )

        # --- Step 7: record cost ---
        await self._budget.record(actual_cost, level=BudgetLevel.RUN)
        await self._budget.record(actual_cost, level=BudgetLevel.EXPERIMENT)

        # --- Step 8: build LLMResponse ---
        latency_ms = int((time.monotonic() - started_monotonic) * 1000)
        finish_reason = _parse_finish_reason(ai_msg)
        tool_calls = _parse_tool_calls(ai_msg, issued_by=agent_id)

        # text: None if content is empty string (no text response)
        text_raw = ai_msg.content
        if isinstance(text_raw, str):
            text: str | None = text_raw if text_raw else None
        else:
            # List of content blocks
            extracted = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text_raw
            ).strip()
            text = extracted or None

        return LLMResponse(
            id=uuid.uuid4(),
            model=self._model_id,
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=actual_cost,
            latency_ms=latency_ms,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            started_at=started_dt,
            raw=dict(ai_msg.response_metadata or {}),
        )

    async def astream(self, *args: Any, **kwargs: Any) -> None:
        """Streaming is not supported in M2."""
        raise NotImplementedError("streaming not supported in M2")
