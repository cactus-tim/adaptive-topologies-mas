"""Unit tests for atm.tasks.creative — CreativeLoader and CreativeJudgeEvaluator.

Tests (TDD red phase — step 6):
1. CreativeLoader.load() returns list[TaskSpec] with correct structure.
2. CreativeLoader.load() raises ValueError if fewer than 5 prompts in YAML.
3. CreativeJudgeEvaluator with valid FakeLLM (scripted) returns score=0.8, passed=True.
4. CreativeJudgeEvaluator with broken-JSON FakeLLM returns score=0.0, error is not None.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atm.core.types import TaskSpec
from atm.llm.fake import FakeLLM
from atm.tasks.creative import CreativeJudgeEvaluator, CreativeLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures"
_LLM_FIXTURE = _FIXTURE_DIR / "llm" / "m10_judge_creative.yaml"
_PROMPTS_YAML = Path(__file__).parent.parent.parent.parent / "conf" / "tasks" / "creative_prompts.yaml"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_minimal_yaml(tmp_path: Path, n: int) -> Path:
    """Write a minimal creative_prompts.yaml with ``n`` entries and return the path."""
    entries = [
        {"id": f"c{i}", "prompt": f"Prompt {i}", "rubric": f"Rubric {i}"}
        for i in range(1, n + 1)
    ]
    data = {"version": 1, "prompts": entries}
    p = tmp_path / "creative_prompts.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1: loader returns correct TaskSpec list
# ---------------------------------------------------------------------------


def test_creative_loader_returns_task_specs() -> None:
    """CreativeLoader.load() returns a non-empty list[TaskSpec] with expected fields."""
    loader = CreativeLoader(yaml_path=_PROMPTS_YAML)
    specs = loader.load()

    assert len(specs) >= 5, "Expected at least 5 TaskSpecs from creative_prompts.yaml"

    for spec in specs:
        assert isinstance(spec, TaskSpec)
        assert spec.type == "creative"
        assert spec.evaluator_key == "creative_judge"
        assert spec.expected is None
        assert "rubric" in spec.metadata
        assert spec.input  # non-empty prompt
        assert spec.id.startswith("c"), f"Expected id starting with 'c', got {spec.id!r}"


# ---------------------------------------------------------------------------
# Test 2: loader raises ValueError when fewer than 5 prompts
# ---------------------------------------------------------------------------


def test_creative_loader_raises_if_fewer_than_5(tmp_path: Path) -> None:
    """CreativeLoader.load() raises ValueError if the YAML has fewer than 5 prompts."""
    yaml_path = _make_minimal_yaml(tmp_path, n=4)
    loader = CreativeLoader(yaml_path=yaml_path)

    with pytest.raises(ValueError, match="creative: need at least 5 prompts"):
        loader.load()


# ---------------------------------------------------------------------------
# Test 3: evaluator with valid FakeLLM returns score=0.8, passed=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creative_judge_evaluator_valid_response() -> None:
    """CreativeJudgeEvaluator with a scripted FakeLLM returning score=8 yields score=0.8, passed=True."""
    fake_llm = FakeLLM(mode="scripted", fixture=_LLM_FIXTURE)
    evaluator = CreativeJudgeEvaluator(judge_llm=fake_llm)

    spec = TaskSpec(
        id="c1",
        type="creative",
        input="Write a short story about a lighthouse keeper.",
        expected=None,
        evaluator_key="creative_judge",
        metadata={"rubric": "Evaluate for vivid imagery and narrative tension."},
    )

    result = await evaluator.evaluate(spec, "Once upon a time, a lighthouse keeper watched the storm roll in...")

    assert result.score == pytest.approx(0.8)
    assert result.passed is True
    assert result.error is None
    assert "reasoning" in result.details


# ---------------------------------------------------------------------------
# Test 4: evaluator with broken-JSON FakeLLM returns score=0.0, error not None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creative_judge_evaluator_broken_json() -> None:
    """CreativeJudgeEvaluator with FakeLLM returning broken JSON yields score=0.0 and error set."""
    # Write an inline broken-JSON fixture
    broken_fixture_data = {
        "version": 1,
        "mode": "scripted",
        "entries": [
            {
                "agent_id": "creative_judge",
                "step_idx": 0,
                "content": "this is not valid json {{{",
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "model": "fake:scripted",
            }
        ],
    }
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(broken_fixture_data, f)
        broken_fixture_path = Path(f.name)

    try:
        fake_llm = FakeLLM(mode="scripted", fixture=broken_fixture_path)
        evaluator = CreativeJudgeEvaluator(judge_llm=fake_llm)

        spec = TaskSpec(
            id="c2",
            type="creative",
            input="Write a poem about autumn.",
            expected=None,
            evaluator_key="creative_judge",
            metadata={"rubric": "Evaluate for poetic imagery."},
        )

        result = await evaluator.evaluate(spec, "Leaves fall softly...")

        assert result.score == pytest.approx(0.0)
        assert result.error is not None
        assert isinstance(result.error, str)
    finally:
        broken_fixture_path.unlink(missing_ok=True)
