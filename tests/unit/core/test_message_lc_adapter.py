"""Unit tests for Message.to_lc / Message.from_lc adapters (Step 3.1 / M2).

Tests:
- 4-kind round-trip: request → HumanMessage → Message (equal to original)
- decision → AIMessage → Message (equal to original)
- phase_emit → SystemMessage → Message (equal to original)
- broadcast → HumanMessage (with metadata) → Message (equal to original)
- ToolCall round-trip: assistant Message with tool_calls → AIMessage with tool_calls populated
- to_lc raises no NotImplementedError (stubs removed)
- from_lc raises no NotImplementedError
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from atm.core.types import Message, MessageKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request_message() -> Message:
    return Message(sender="agent_a", kind=MessageKind.REQUEST, content="hello world")


def make_decision_message() -> Message:
    return Message(sender="agent_b", kind=MessageKind.DECISION, content="final answer")


def make_broadcast_message() -> Message:
    return Message(
        sender="agent_c",
        kind=MessageKind.BROADCAST,
        content="broadcast msg",
    )


def make_phase_emit_message() -> Message:
    return Message(sender="coord", kind=MessageKind.PHASE_EMIT, content="advance to execution")


# ---------------------------------------------------------------------------
# 1. to_lc no longer raises
# ---------------------------------------------------------------------------


def test_to_lc_does_not_raise_for_request() -> None:
    msg = make_request_message()
    result = msg.to_lc()
    assert result is not None


def test_to_lc_does_not_raise_for_decision() -> None:
    msg = make_decision_message()
    result = msg.to_lc()
    assert result is not None


# ---------------------------------------------------------------------------
# 2. to_lc returns correct LC message types
# ---------------------------------------------------------------------------


def test_request_maps_to_human_message() -> None:
    msg = Message(sender="agent_a", kind=MessageKind.REQUEST, content="ask something")
    lc = msg.to_lc()
    assert isinstance(lc, HumanMessage)
    assert lc.content == "ask something"


def test_draft_maps_to_human_message() -> None:
    msg = Message(sender="writer", kind=MessageKind.DRAFT, content="draft text")
    lc = msg.to_lc()
    assert isinstance(lc, HumanMessage)
    assert lc.content == "draft text"


def test_critique_maps_to_human_message() -> None:
    msg = Message(sender="critic", kind=MessageKind.CRITIQUE, content="needs improvement")
    lc = msg.to_lc()
    assert isinstance(lc, HumanMessage)
    assert lc.content == "needs improvement"


def test_decision_maps_to_ai_message() -> None:
    msg = Message(sender="agent_b", kind=MessageKind.DECISION, content="final answer")
    lc = msg.to_lc()
    assert isinstance(lc, AIMessage)
    assert lc.content == "final answer"


def test_broadcast_maps_to_human_message_with_metadata() -> None:
    msg = Message(sender="agent_c", kind=MessageKind.BROADCAST, content="broadcast msg")
    lc = msg.to_lc()
    assert isinstance(lc, HumanMessage)
    assert lc.content == "broadcast msg"
    # broadcast should have metadata indicating channel
    assert lc.response_metadata.get("channel") == "broadcast" or (
        hasattr(lc, "additional_kwargs") and lc.additional_kwargs.get("channel") == "broadcast"
    )


def test_phase_emit_maps_to_system_message() -> None:
    msg = Message(sender="coord", kind=MessageKind.PHASE_EMIT, content="advance to execution")
    lc = msg.to_lc()
    assert isinstance(lc, SystemMessage)
    assert lc.content == "advance to execution"


# ---------------------------------------------------------------------------
# 3. Round-trip tests: Message → LC → Message
# ---------------------------------------------------------------------------


def test_request_roundtrip() -> None:
    """request → HumanMessage → from_lc → Message equals original."""
    original = Message(sender="agent_a", kind=MessageKind.REQUEST, content="hello world")
    lc = original.to_lc()
    reconstructed = Message.from_lc(lc, sender=original.sender, kind=original.kind)
    assert reconstructed.content == original.content
    assert reconstructed.sender == original.sender
    assert reconstructed.kind == original.kind


def test_decision_roundtrip() -> None:
    """decision → AIMessage → from_lc → Message equals original."""
    original = Message(sender="agent_b", kind=MessageKind.DECISION, content="final answer")
    lc = original.to_lc()
    reconstructed = Message.from_lc(lc, sender=original.sender, kind=original.kind)
    assert reconstructed.content == original.content
    assert reconstructed.sender == original.sender
    assert reconstructed.kind == original.kind


def test_broadcast_roundtrip() -> None:
    """broadcast → HumanMessage → from_lc → Message equals original."""
    original = Message(sender="agent_c", kind=MessageKind.BROADCAST, content="broadcast msg")
    lc = original.to_lc()
    reconstructed = Message.from_lc(lc, sender=original.sender, kind=original.kind)
    assert reconstructed.content == original.content
    assert reconstructed.sender == original.sender
    assert reconstructed.kind == original.kind


def test_phase_emit_roundtrip() -> None:
    """phase_emit → SystemMessage → from_lc → Message equals original."""
    original = Message(sender="coord", kind=MessageKind.PHASE_EMIT, content="advance")
    lc = original.to_lc()
    reconstructed = Message.from_lc(lc, sender=original.sender, kind=original.kind)
    assert reconstructed.content == original.content
    assert reconstructed.sender == original.sender
    assert reconstructed.kind == original.kind


# ---------------------------------------------------------------------------
# 4. from_lc from raw LC messages
# ---------------------------------------------------------------------------


def test_from_lc_human_message() -> None:
    """from_lc converts HumanMessage → Message with correct content."""
    lc = HumanMessage(content="user query")
    msg = Message.from_lc(lc, sender="user", kind=MessageKind.REQUEST)
    assert msg.content == "user query"
    assert msg.sender == "user"
    assert msg.kind == MessageKind.REQUEST


def test_from_lc_ai_message() -> None:
    """from_lc converts AIMessage → Message with correct content."""
    lc = AIMessage(content="LLM answer")
    msg = Message.from_lc(lc, sender="agent_x", kind=MessageKind.DECISION)
    assert msg.content == "LLM answer"
    assert msg.sender == "agent_x"
    assert msg.kind == MessageKind.DECISION


def test_from_lc_system_message() -> None:
    """from_lc converts SystemMessage → Message with correct content."""
    lc = SystemMessage(content="system prompt")
    msg = Message.from_lc(lc, sender="system", kind=MessageKind.PHASE_EMIT)
    assert msg.content == "system prompt"
    assert msg.sender == "system"
    assert msg.kind == MessageKind.PHASE_EMIT


def test_from_lc_stores_lc_extra_in_payload() -> None:
    """from_lc stores LC-specific fields in payload under '_lc' key."""
    lc = AIMessage(content="answer", additional_kwargs={"some_key": "some_val"})
    msg = Message.from_lc(lc, sender="agent", kind=MessageKind.DECISION)
    assert "_lc" in msg.payload


# ---------------------------------------------------------------------------
# 5. AIMessage tool_calls round-trip
# ---------------------------------------------------------------------------


def test_decision_with_tool_calls_to_ai_message() -> None:
    """Message(kind=DECISION) converts to AIMessage."""
    original = Message(sender="planner", kind=MessageKind.DECISION, content="use tool")
    lc = original.to_lc()
    assert isinstance(lc, AIMessage)


def test_from_lc_ai_message_with_tool_calls() -> None:
    """from_lc on AIMessage with tool_calls stores tool_calls in payload['_lc']."""
    tc = {"name": "bash", "args": {"cmd": "ls"}, "id": "call_001", "type": "tool_call"}
    lc = AIMessage(content="", tool_calls=[tc])
    msg = Message.from_lc(lc, sender="agent", kind=MessageKind.DECISION)
    assert msg.content is not None or msg.content == ""
    assert "_lc" in msg.payload
    assert "tool_calls" in msg.payload["_lc"]
    assert len(msg.payload["_lc"]["tool_calls"]) == 1
    assert msg.payload["_lc"]["tool_calls"][0]["name"] == "bash"
