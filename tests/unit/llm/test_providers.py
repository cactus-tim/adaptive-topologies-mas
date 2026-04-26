"""Unit tests for atm.llm.providers — factory functions + inject_cache_control."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from atm.llm.providers import (
    DEFAULT_CACHE_TTL,
    build_anthropic,
    build_openai,
    build_vllm,
    inject_cache_control,
)

# ---------------------------------------------------------------------------
# build_openai tests
# ---------------------------------------------------------------------------


class TestBuildOpenAI:
    def test_returns_base_chat_model(self) -> None:
        """build_openai returns a BaseChatModel subclass without making API call."""
        from langchain_core.language_models.chat_models import BaseChatModel

        model = build_openai("openai:gpt-4o-mini", {"temperature": 0.0})
        assert isinstance(model, BaseChatModel)

    def test_strips_openai_prefix(self) -> None:
        """build_openai strips 'openai:' prefix; resulting model name is bare."""
        from langchain_openai import ChatOpenAI

        model = build_openai("openai:gpt-4o-mini", {})
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"

    def test_forwards_temperature(self) -> None:
        """opts dict values are forwarded to the underlying model."""
        from langchain_openai import ChatOpenAI

        model = build_openai("openai:gpt-4o", {"temperature": 0.7})
        assert isinstance(model, ChatOpenAI)
        assert model.temperature == pytest.approx(0.7)

    def test_no_api_call_on_construction(self) -> None:
        """Constructing the model object does not trigger any network call."""
        # If this test hangs or raises a network error, the factory is wrong.
        model = build_openai("openai:gpt-4o-mini", {"temperature": 0.0})
        assert model is not None


# ---------------------------------------------------------------------------
# build_anthropic tests
# ---------------------------------------------------------------------------


class TestBuildAnthropic:
    def test_returns_base_chat_model(self) -> None:
        """build_anthropic returns a BaseChatModel subclass without making API call."""
        from langchain_core.language_models.chat_models import BaseChatModel

        model = build_anthropic("anthropic:claude-3-5-haiku-latest", {})
        assert isinstance(model, BaseChatModel)

    def test_returns_chat_anthropic_instance(self) -> None:
        """build_anthropic specifically returns a ChatAnthropic instance."""
        from langchain_anthropic import ChatAnthropic

        model = build_anthropic("anthropic:claude-3-5-haiku-latest", {})
        assert isinstance(model, ChatAnthropic)

    def test_strips_anthropic_prefix(self) -> None:
        """build_anthropic strips 'anthropic:' prefix; bare model name is passed."""
        from langchain_anthropic import ChatAnthropic

        model = build_anthropic("anthropic:claude-3-5-sonnet-latest", {})
        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-3-5-sonnet-latest"

    def test_forwards_temperature(self) -> None:
        """opts temperature is forwarded to the underlying model."""
        from langchain_anthropic import ChatAnthropic

        model = build_anthropic("anthropic:claude-3-5-haiku-latest", {"temperature": 0.3})
        assert isinstance(model, ChatAnthropic)
        assert model.temperature == pytest.approx(0.3)

    def test_no_api_call_on_construction(self) -> None:
        """Constructing the model object does not trigger any network call."""
        model = build_anthropic("anthropic:claude-3-5-haiku-latest", {})
        assert model is not None


# ---------------------------------------------------------------------------
# build_vllm tests
# ---------------------------------------------------------------------------


class TestBuildVLLM:
    def test_returns_base_chat_model(self) -> None:
        """build_vllm returns a BaseChatModel subclass."""
        from langchain_core.language_models.chat_models import BaseChatModel

        model = build_vllm("vllm:my-model", {"base_url": "http://localhost:8000/v1"})
        assert isinstance(model, BaseChatModel)

    def test_returns_chat_openai_instance(self) -> None:
        """build_vllm uses ChatOpenAI under the hood."""
        from langchain_openai import ChatOpenAI

        model = build_vllm("vllm:my-model", {"base_url": "http://localhost:8000/v1"})
        assert isinstance(model, ChatOpenAI)

    def test_sets_base_url(self) -> None:
        """base_url from opts is set on the model."""
        from langchain_openai import ChatOpenAI

        model = build_vllm("vllm:llama-3", {"base_url": "http://localhost:8000/v1"})
        assert isinstance(model, ChatOpenAI)
        assert "localhost:8000" in str(model.openai_api_base)

    def test_api_key_is_empty(self) -> None:
        """vLLM expects api_key='EMPTY' per LangChain+vLLM convention."""
        from langchain_openai import ChatOpenAI

        model = build_vllm("vllm:my-model", {"base_url": "http://localhost:8000/v1"})
        assert isinstance(model, ChatOpenAI)
        # openai_api_key is a SecretStr in langchain-openai
        secret = model.openai_api_key
        key_value = (
            secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
        )
        assert key_value == "EMPTY"

    def test_strips_vllm_prefix(self) -> None:
        """build_vllm strips 'vllm:' prefix; bare model name passed to ChatOpenAI."""
        from langchain_openai import ChatOpenAI

        model = build_vllm("vllm:llama-3-8b", {"base_url": "http://localhost:8000/v1"})
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "llama-3-8b"

    def test_no_api_call_on_construction(self) -> None:
        """Constructing the model does not trigger any network call."""
        model = build_vllm("vllm:my-model", {"base_url": "http://localhost:8000/v1"})
        assert model is not None


# ---------------------------------------------------------------------------
# inject_cache_control tests
# ---------------------------------------------------------------------------


class TestInjectCacheControl:
    # ---- immutability invariants ----

    def test_returns_new_list(self) -> None:
        """inject_cache_control returns a new list — input is not the returned object."""
        messages: list[BaseMessage] = [HumanMessage(content="hello")]
        result = inject_cache_control(messages, key="k1")
        assert result is not messages

    def test_input_list_unchanged(self) -> None:
        """Input list length and references are not modified."""
        original_msg = HumanMessage(content="hello")
        messages: list[BaseMessage] = [original_msg]
        _ = inject_cache_control(messages, key="k1")
        assert len(messages) == 1
        assert messages[0] is original_msg

    def test_first_messages_not_mutated(self) -> None:
        """Messages other than the last one are not mutated."""
        first = HumanMessage(content="first message")
        last = HumanMessage(content="last message")
        messages: list[BaseMessage] = [first, last]
        result = inject_cache_control(messages, key="k1")
        # The returned first message must be unchanged
        assert result[0].content == "first message"
        # Input first message object is unchanged
        assert first.content == "first message"

    def test_original_last_message_not_mutated(self) -> None:
        """The original last message object in the input list is NOT mutated."""
        original_last = HumanMessage(content="last")
        messages: list[BaseMessage] = [HumanMessage(content="first"), original_last]
        _ = inject_cache_control(messages, key="k2")
        # original_last.content must still be a plain string (or unchanged list)
        assert original_last.content == "last"

    # ---- ephemeral cache_control added to last text block ----

    def test_adds_cache_control_to_last_message_simple_string(self) -> None:
        """For a string-content last message, content becomes list-of-blocks with cache_control."""
        messages: list[BaseMessage] = [
            HumanMessage(content="hello"),
            HumanMessage(content="last plain text"),
        ]
        result = inject_cache_control(messages, key="some-key")
        last = result[-1]
        # content should be converted to list-of-blocks
        assert isinstance(last.content, list)
        # The last block must have cache_control
        last_block = last.content[-1]
        assert isinstance(last_block, dict)
        assert "cache_control" in last_block
        assert last_block["cache_control"]["type"] == "ephemeral"

    def test_cache_control_ttl_default(self) -> None:
        """Default TTL is DEFAULT_CACHE_TTL == '5m'."""
        assert DEFAULT_CACHE_TTL == "5m"
        messages: list[BaseMessage] = [HumanMessage(content="text")]
        result = inject_cache_control(messages, key="k")
        last_block = result[-1].content[-1]
        assert last_block["cache_control"]["ttl"] == "5m"

    def test_cache_control_custom_ttl(self) -> None:
        """Custom ttl is forwarded into the cache_control block."""
        messages: list[BaseMessage] = [HumanMessage(content="text")]
        result = inject_cache_control(messages, key="k", ttl="10m")
        last_block = result[-1].content[-1]
        assert last_block["cache_control"]["ttl"] == "10m"

    def test_preserves_other_messages(self) -> None:
        """All messages except the last are preserved in the new list."""
        msgs: list[BaseMessage] = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="user turn 1"),
            AIMessage(content="ai turn 1"),
            HumanMessage(content="user turn 2"),
        ]
        result = inject_cache_control(msgs, key="k")
        assert len(result) == len(msgs)
        # First 3 messages are unchanged
        for i in range(3):
            assert result[i].content == msgs[i].content

    # ---- multi-modal last message ----

    def test_multimodal_adds_cache_control_to_last_text_block(self) -> None:
        """For a last message with content=list[dict], cache_control goes on the last text-type block."""
        content_blocks: list[dict] = [
            {"type": "text", "text": "describe this image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "be concise"},
        ]
        messages: list[BaseMessage] = [HumanMessage(content=content_blocks)]  # type: ignore[arg-type]
        result = inject_cache_control(messages, key="img-key")
        last = result[-1]
        assert isinstance(last.content, list)
        # Find the last text block
        text_blocks = [b for b in last.content if isinstance(b, dict) and b.get("type") == "text"]
        assert text_blocks, "Expected at least one text block"
        last_text_block = text_blocks[-1]
        assert "cache_control" in last_text_block
        assert last_text_block["cache_control"]["type"] == "ephemeral"

    def test_multimodal_image_blocks_unchanged(self) -> None:
        """Image blocks in a multi-modal message do NOT get cache_control added."""
        content_blocks: list[dict] = [
            {"type": "text", "text": "look at this:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        messages: list[BaseMessage] = [HumanMessage(content=content_blocks)]  # type: ignore[arg-type]
        result = inject_cache_control(messages, key="img-key2")
        last = result[-1]
        img_blocks = [
            b for b in last.content if isinstance(b, dict) and b.get("type") == "image_url"
        ]
        for img_block in img_blocks:
            assert "cache_control" not in img_block

    def test_empty_messages_raises_or_returns_unchanged(self) -> None:
        """inject_cache_control with empty list returns empty list gracefully."""
        result = inject_cache_control([], key="k")
        assert result == []


# ---------------------------------------------------------------------------
# build_cerebras tests
# ---------------------------------------------------------------------------


class TestBuildCerebras:
    def test_returns_base_chat_model(self) -> None:
        """build_cerebras returns a BaseChatModel subclass without making API call."""
        from langchain_core.language_models.chat_models import BaseChatModel

        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras("cerebras:llama3.1-8b", {})
        assert isinstance(model, BaseChatModel)

    def test_returns_chat_cerebras_instance(self) -> None:
        """build_cerebras specifically returns a ChatCerebras instance."""
        from langchain_cerebras import ChatCerebras

        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras("cerebras:llama3.1-8b", {})
        assert isinstance(model, ChatCerebras)

    def test_strips_cerebras_prefix(self) -> None:
        """build_cerebras strips 'cerebras:' prefix; bare model name is passed through."""
        from langchain_cerebras import ChatCerebras

        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras("cerebras:gpt-oss-120b", {})
        assert isinstance(model, ChatCerebras)
        assert model.model_name == "gpt-oss-120b"

    def test_forwards_temperature(self) -> None:
        """opts temperature is forwarded to the underlying model."""
        from langchain_cerebras import ChatCerebras

        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras("cerebras:llama3.1-8b", {"temperature": 0.5})
        assert isinstance(model, ChatCerebras)
        assert model.temperature == pytest.approx(0.5)

    def test_no_api_call_on_construction(self) -> None:
        """Constructing the model object does not trigger any network call."""
        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras("cerebras:llama3.1-8b", {})
        assert model is not None

    def test_api_key_is_empty_by_default(self) -> None:
        """Default api_key is 'EMPTY' so construction works without CEREBRAS_API_KEY env var."""
        from langchain_cerebras import ChatCerebras

        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras("cerebras:llama3.1-8b", {})
        assert isinstance(model, ChatCerebras)
        secret = model.cerebras_api_key
        key_value = (
            secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
        )
        assert key_value == "EMPTY"

    @pytest.mark.parametrize("model_id", ["cerebras:llama3.1-8b", "cerebras:gpt-oss-120b"])
    def test_supports_active_models(self, model_id: str) -> None:
        """build_cerebras constructs successfully for each supported active model ID."""
        from langchain_cerebras import ChatCerebras

        from atm.llm.providers.cerebras import build_cerebras

        model = build_cerebras(model_id, {})
        assert isinstance(model, ChatCerebras)
        bare_id = model_id.split(":", 1)[1]
        assert model.model_name == bare_id
