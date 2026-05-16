"""Unit tests for the post-graph finalize hook (Path C).

Covers:
  * _is_non_code_task / _looks_like_code / _needs_finalize heuristics
  * maybe_finalize_answer dispatch — no-op cases vs LLM-call cases
  * error swallowing — LLM raises → original answer kept
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atm.core.types import TaskSpec, TokenUsage
from atm.experiment.finalize import (
    _DABENCH_TEMPLATE_RE,
    _is_non_code_task,
    _looks_like_code,
    _needs_finalize,
    maybe_finalize_answer,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk_spec(
    task_id: str,
    *,
    input_text: str = "Question?",
    metadata: dict[str, Any] | None = None,
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        type="decision",
        input=input_text,
        expected=None,
        evaluator_key="exact_match",
        metadata=metadata or {},
    )


def _mk_llm_response(text: str) -> Any:
    return SimpleNamespace(
        text=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


class TestTaskClassification:
    def test_humaneval_is_code(self) -> None:
        assert _is_non_code_task("humaneval/HumanEval/0") is False

    @pytest.mark.parametrize(
        "task_id",
        ["gsm8k/test/0", "commongen/test/1", "dabench/12"],
    )
    def test_known_non_code_tasks(self, task_id: str) -> None:
        assert _is_non_code_task(task_id) is True

    def test_unknown_prefix_treated_as_code(self) -> None:
        # Conservative default: anything we don't recognize is not finalized.
        assert _is_non_code_task("unknown/0") is False


class TestLooksLikeCode:
    def test_code_fence(self) -> None:
        assert _looks_like_code("Here is the answer: ```py\nprint(1)\n```") is True

    def test_multiple_markers(self) -> None:
        text = "import pandas as pd\ndef foo():\n    return 1\n"
        assert _looks_like_code(text) is True

    def test_one_marker_not_enough(self) -> None:
        # Single accidental keyword in prose shouldn't trigger finalize.
        assert _looks_like_code("The function should return the same value.") is False

    def test_plain_text(self) -> None:
        assert _looks_like_code("The mean is 34.65.") is False


class TestNeedsFinalize:
    def test_code_task_never_triggers(self) -> None:
        # Even with empty / code-shaped answers, humaneval is left alone:
        # its file-write artifact is the right answer.
        assert _needs_finalize(_mk_spec("humaneval/0"), "") is False
        assert _needs_finalize(_mk_spec("humaneval/0"), "import x\ndef foo(): pass") is False

    def test_non_code_empty_triggers(self) -> None:
        spec = _mk_spec("gsm8k/0")
        assert _needs_finalize(spec, "") is True
        assert _needs_finalize(spec, "   ") is True
        assert _needs_finalize(spec, "<incomplete>") is True

    def test_non_code_python_shaped_triggers(self) -> None:
        py = "import pandas as pd\ndf = pd.read_csv('x.csv')\nprint(df.mean())"
        assert _needs_finalize(_mk_spec("gsm8k/0"), py) is True

    def test_dabench_missing_template_triggers(self) -> None:
        # Non-empty, prose-shaped, but no @name[value] — DABench evaluator
        # would score 0; we should ask the model to re-format.
        assert _needs_finalize(_mk_spec("dabench/5"), "The mean fare is 34.65.") is True

    def test_dabench_with_template_passes(self) -> None:
        assert _needs_finalize(_mk_spec("dabench/5"), "@mean_fare[34.65]") is False

    def test_gsm8k_plain_numeric_passes(self) -> None:
        # GSM8K answer is a number — no template required.
        assert _needs_finalize(_mk_spec("gsm8k/0"), "72") is False

    def test_commongen_missing_concept_triggers(self) -> None:
        spec = _mk_spec(
            "commongen/0",
            metadata={"concepts": ["dog", "fence", "jump"], "references": []},
        )
        # "fence" missing → trigger.
        assert _needs_finalize(spec, "The dog jumps over the wall.") is True

    def test_commongen_all_concepts_present_passes(self) -> None:
        spec = _mk_spec(
            "commongen/0",
            metadata={"concepts": ["dog", "fence", "jump"], "references": []},
        )
        assert _needs_finalize(spec, "The dog jumps over the fence.") is False

    def test_commongen_meta_marker_triggers(self) -> None:
        spec = _mk_spec(
            "commongen/0",
            metadata={"concepts": ["dog", "fence", "jump"], "references": []},
        )
        # All concepts technically present, but answer is a critic complaint.
        meta_text = (
            "Issues identified: the executor output does not contain a story "
            "with the required concepts (dog, fence, jump)."
        )
        assert _needs_finalize(spec, meta_text) is True

    def test_dabench_template_regex(self) -> None:
        assert _DABENCH_TEMPLATE_RE.search("@a[1] @b[2]") is not None
        assert _DABENCH_TEMPLATE_RE.search("a[1] b[2]") is None


# ---------------------------------------------------------------------------
# maybe_finalize_answer
# ---------------------------------------------------------------------------


class TestMaybeFinalizeAnswer:
    @pytest.mark.asyncio
    async def test_no_spec_passthrough(self) -> None:
        llm = AsyncMock()
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=None,
            final_state={},
            current_answer="anything",
        )
        assert out == "anything"
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_task_passthrough(self) -> None:
        llm = AsyncMock()
        spec = _mk_spec("humaneval/0")
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=spec,
            final_state={},
            current_answer="",  # would trigger for non-code
        )
        assert out == ""
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_good_answer_passthrough(self) -> None:
        llm = AsyncMock()
        spec = _mk_spec("gsm8k/0")
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=spec,
            final_state={},
            current_answer="42",
        )
        assert out == "42"
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_for_empty_non_code(self) -> None:
        llm = AsyncMock()
        llm.ainvoke.return_value = _mk_llm_response("synthesized answer")
        spec = _mk_spec("gsm8k/0")
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=spec,
            final_state={"agents": {"executor": {}}},
            current_answer="",
        )
        assert out == "synthesized answer"
        llm.ainvoke.assert_called_once()
        # Tools must be disabled on the finalize call.
        kwargs = llm.ainvoke.call_args.kwargs
        assert kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_triggers_for_dabench_missing_template(self) -> None:
        llm = AsyncMock()
        llm.ainvoke.return_value = _mk_llm_response("@mean_fare[34.65]")
        spec = _mk_spec(
            "dabench/5",
            input_text="Find the mean fare.",
            metadata={"format": "@mean_fare[value]", "constraints": "Round to 2dp"},
        )
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=spec,
            final_state={
                "agents": {
                    "executor": {
                        "tool_calls": [
                            SimpleNamespace(
                                id="c1", tool_name="code_run", args={}, issued_by="executor"
                            )
                        ],
                        "tool_results": [
                            SimpleNamespace(
                                call_id="c1", ok=True, output="mean=34.65", error=None
                            )
                        ],
                    }
                }
            },
            current_answer="The mean fare is 34.65.",  # missing @name[value]
        )
        assert out == "@mean_fare[34.65]"
        llm.ainvoke.assert_called_once()
        # Prompt should mention the required format from metadata.
        messages = llm.ainvoke.call_args.args[0]
        joined = "\n".join(getattr(m, "content", "") for m in messages)
        assert "@mean_fare[value]" in joined
        # And the tool history should be referenced.
        assert "mean=34.65" in joined

    @pytest.mark.asyncio
    async def test_llm_error_keeps_original_answer(self) -> None:
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("boom")
        spec = _mk_spec("gsm8k/0")
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=spec,
            final_state={},
            current_answer="",  # triggers, but llm raises
        )
        assert out == ""
        llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_llm_response_keeps_original(self) -> None:
        llm = AsyncMock()
        llm.ainvoke.return_value = _mk_llm_response("")  # empty text
        spec = _mk_spec("gsm8k/0")
        out = await maybe_finalize_answer(
            llm=llm,
            task_spec=spec,
            final_state={},
            current_answer="",
        )
        assert out == ""
