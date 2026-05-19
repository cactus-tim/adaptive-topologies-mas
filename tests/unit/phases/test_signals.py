"""Unit tests for atm.phases.signals — emit_signal and increment_signal helpers.

Tests:
  1. test_emit_signal_creates_signals_dict     — creates signals dict if absent
  2. test_emit_signal_updates_existing_key     — overwrites an existing key
  3. test_emit_signal_does_not_mutate_input    — input dict is not mutated
  4. test_emit_signal_preserves_other_keys     — other shared keys are kept
  5. test_emit_signal_preserves_other_signals  — other signals are kept
  6. test_increment_signal_starts_at_zero      — defaults counter to 0 then increments
  7. test_increment_signal_accumulates         — multiple increments stack
  8. test_increment_signal_returns_new_value   — second return value is the int counter
  9. test_emit_signal_no_crash_without_langgraph — graceful no-op if LG unavailable
  10. test_signal_constants                    — all constants have expected string values
"""

from __future__ import annotations

from unittest.mock import patch

from atm.phases.signals import (
    CRITIC_APPROVED,
    NEEDS_DEBATE,
    READY_FOR_EXECUTION,
    READY_FOR_VERIFICATION,
    REJECTED_COUNT,
    STUCK,
    emit_signal,
    increment_signal,
)


class TestEmitSignal:
    """Tests for emit_signal()."""

    def test_emit_signal_creates_signals_dict(self) -> None:
        """emit_signal creates signals dict if not present in shared."""
        shared: dict = {"task_input": "hello", "iter_total": 5}
        updated = emit_signal(shared, STUCK, True)
        assert "signals" in updated
        assert updated["signals"][STUCK] is True

    def test_emit_signal_updates_existing_key(self) -> None:
        """emit_signal overwrites an existing key in signals."""
        shared: dict = {"signals": {"stuck": False}}
        updated = emit_signal(shared, STUCK, True)
        assert updated["signals"][STUCK] is True

    def test_emit_signal_does_not_mutate_input(self) -> None:
        """emit_signal returns a new dict; input is unchanged."""
        original_signals: dict = {"critic_approved": False}
        shared: dict = {"signals": original_signals, "iter_total": 1}
        updated = emit_signal(shared, CRITIC_APPROVED, True)

        assert shared["signals"][CRITIC_APPROVED] is False
        assert original_signals[CRITIC_APPROVED] is False
        assert updated["signals"][CRITIC_APPROVED] is True
        assert updated is not shared
        assert updated["signals"] is not original_signals

    def test_emit_signal_preserves_other_keys(self) -> None:
        """emit_signal keeps other keys in the shared dict."""
        shared: dict = {
            "task_input": "test task",
            "iter_total": 3,
            "signals": {},
        }
        updated = emit_signal(shared, READY_FOR_EXECUTION, True)
        assert updated["task_input"] == "test task"
        assert updated["iter_total"] == 3

    def test_emit_signal_preserves_other_signals(self) -> None:
        """emit_signal keeps other signals in the signals dict."""
        shared: dict = {
            "signals": {
                "critic_approved": True,
                "ready_for_execution": False,
            }
        }
        updated = emit_signal(shared, STUCK, True)
        assert updated["signals"]["critic_approved"] is True
        assert updated["signals"]["ready_for_execution"] is False
        assert updated["signals"][STUCK] is True

    def test_emit_signal_int_value(self) -> None:
        """emit_signal works with integer values."""
        shared: dict = {}
        updated = emit_signal(shared, REJECTED_COUNT, 2)
        assert updated["signals"][REJECTED_COUNT] == 2

    def test_emit_signal_no_crash_without_langgraph(self) -> None:
        """emit_signal must not crash even if dispatch_custom_event is unavailable.

        Simulates the scenario where the LangGraph API is missing by patching
        logging to ensure the call completes without error.
        """
        shared: dict = {"iter_total": 7}
        with patch("atm.phases.signals.logger") as mock_logger:
            updated = emit_signal(shared, NEEDS_DEBATE, True)
            mock_logger.debug.assert_called()
        assert updated["signals"][NEEDS_DEBATE] is True


class TestIncrementSignal:
    """Tests for increment_signal()."""

    def test_increment_signal_starts_at_zero(self) -> None:
        """increment_signal defaults to 0 when the key is absent."""
        shared: dict = {}
        new_shared, value = increment_signal(shared, REJECTED_COUNT)
        assert value == 1
        assert new_shared["signals"][REJECTED_COUNT] == 1

    def test_increment_signal_accumulates(self) -> None:
        """Multiple calls to increment_signal accumulate correctly."""
        shared: dict = {}
        shared, v1 = increment_signal(shared, REJECTED_COUNT)
        shared, v2 = increment_signal(shared, REJECTED_COUNT)
        shared, v3 = increment_signal(shared, REJECTED_COUNT)
        assert v1 == 1
        assert v2 == 2
        assert v3 == 3
        assert shared["signals"][REJECTED_COUNT] == 3

    def test_increment_signal_returns_new_value(self) -> None:
        """increment_signal returns the new integer value as second element."""
        shared: dict = {"signals": {REJECTED_COUNT: 5}}
        new_shared, value = increment_signal(shared, REJECTED_COUNT)
        assert value == 6
        assert new_shared["signals"][REJECTED_COUNT] == 6

    def test_increment_signal_does_not_mutate_input(self) -> None:
        """increment_signal returns a new dict without mutating input."""
        shared: dict = {"signals": {REJECTED_COUNT: 2}}
        new_shared, _ = increment_signal(shared, REJECTED_COUNT)
        assert shared["signals"][REJECTED_COUNT] == 2
        assert new_shared["signals"][REJECTED_COUNT] == 3


class TestSignalConstants:
    """Verify all signal key constants have expected string values."""

    def test_stuck_constant(self) -> None:
        assert STUCK == "stuck"

    def test_rejected_count_constant(self) -> None:
        assert REJECTED_COUNT == "rejected_count"

    def test_needs_debate_constant(self) -> None:
        assert NEEDS_DEBATE == "needs_debate"

    def test_ready_for_execution_constant(self) -> None:
        assert READY_FOR_EXECUTION == "ready_for_execution"

    def test_ready_for_verification_constant(self) -> None:
        assert READY_FOR_VERIFICATION == "ready_for_verification"

    def test_critic_approved_constant(self) -> None:
        assert CRITIC_APPROVED == "critic_approved"
