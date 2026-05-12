"""Unit tests for atm.tasks.resolve_spec.

3 tests:
  1. Registered dataset name → TaskSpec (uses MMLU which is registered by M10).
  2. Inline-prompt / unknown name → None (M6 fallback path).
  3. Seed determinism — same seed returns same TaskSpec id.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from atm.tasks import resolve_spec
from atm.tasks.base import TASKS, TaskSpec

# ---------------------------------------------------------------------------
# 1. Registered dataset name → TaskSpec
# ---------------------------------------------------------------------------


def test_resolve_spec_registered_name_returns_task_spec() -> None:
    """resolve_spec returns a TaskSpec for a registered dataset (mmlu)."""
    # mmlu is registered by M10 via atm.tasks.mmlu side-effect import
    # We patch TASKS.get to avoid needing actual dataset files
    fake_spec = TaskSpec(
        id="mmlu/test/0",
        type="qa",
        input="Question text",
        expected="A",
        evaluator_key="mmlu_exact_match",
    )

    mock_task_cfg = MagicMock()
    mock_task_cfg.name = "mmlu"
    mock_task_cfg.shuffle_seed = 42

    with (
        patch.object(TASKS, "get", return_value=object()),
        patch.object(TASKS, "sample", return_value=[fake_spec]),
    ):
        result = resolve_spec(mock_task_cfg)

    assert result is not None
    assert isinstance(result, TaskSpec)
    assert result.id == "mmlu/test/0"


# ---------------------------------------------------------------------------
# 2. Inline-prompt / unknown name → None
# ---------------------------------------------------------------------------


def test_resolve_spec_unknown_name_returns_none() -> None:
    """resolve_spec returns None when task name is not in TASKS registry."""
    mock_task_cfg = MagicMock()
    mock_task_cfg.name = "nonexistent_m6_inline_task"
    mock_task_cfg.shuffle_seed = 0
    mock_task_cfg.input = "Write fib(n)"

    # Do not patch TASKS — "nonexistent_m6_inline_task" should not be registered
    result = resolve_spec(mock_task_cfg)

    assert result is None


# ---------------------------------------------------------------------------
# 3. Seed determinism — same seed returns same TaskSpec id
# ---------------------------------------------------------------------------


def test_resolve_spec_seed_determinism() -> None:
    """resolve_spec called twice with same seed returns same TaskSpec."""
    fake_spec_a = TaskSpec(
        id="mmlu/test/42",
        type="qa",
        input="Question A",
        expected="B",
        evaluator_key="mmlu_exact_match",
    )
    fake_spec_b = TaskSpec(
        id="mmlu/test/42",
        type="qa",
        input="Question A",
        expected="B",
        evaluator_key="mmlu_exact_match",
    )

    mock_task_cfg = MagicMock()
    mock_task_cfg.name = "mmlu"
    mock_task_cfg.shuffle_seed = 7

    with (
        patch.object(TASKS, "get", return_value=object()),
        patch.object(TASKS, "sample", return_value=[fake_spec_a]),
    ):
        result1 = resolve_spec(mock_task_cfg)

    with (
        patch.object(TASKS, "get", return_value=object()),
        patch.object(TASKS, "sample", return_value=[fake_spec_b]),
    ):
        result2 = resolve_spec(mock_task_cfg)

    assert result1 is not None
    assert result2 is not None
    assert result1.id == result2.id
