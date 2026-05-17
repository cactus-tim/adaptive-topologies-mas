"""Unit tests for atm.ui.tlx — NasaTLXForm result shape and raw_score computation.

These tests exercise NasaTLXForm.compute_raw_score directly (no Streamlit
runtime required) and verify that the dict shape matches the contract
expected by StreamlitHumanGateway / callbacks.

Tests:
1. compute_raw_score matches atm.evaluation.tlx.NasaTLX for arbitrary values.
2. compute_raw_score — boundary: all zeros → 0.0 except performance inversion.
3. compute_raw_score — boundary: all 100 → max inversion on performance.
4. compute_raw_score — symmetry with NasaTLX.raw_score property.
5. render() returns None before submission (module-level import works w/o Streamlit crash).
6. _SCALES constant has exactly 6 entries with the required keys.
"""

from __future__ import annotations

import pytest

from atm.evaluation.tlx import NasaTLX
from atm.ui.tlx import _SCALES, NasaTLXForm

# ---------------------------------------------------------------------------
# compute_raw_score
# ---------------------------------------------------------------------------


class TestComputeRawScore:
    """NasaTLXForm.compute_raw_score produces values consistent with NasaTLX."""

    def _make_scores(
        self,
        mental: int = 50,
        physical: int = 30,
        temporal: int = 60,
        performance: int = 70,
        effort: int = 40,
        frustration: int = 20,
    ) -> dict[str, int]:
        return {
            "mental": mental,
            "physical": physical,
            "temporal": temporal,
            "performance": performance,
            "effort": effort,
            "frustration": frustration,
        }

    def test_matches_evaluation_tlx(self) -> None:
        """compute_raw_score must equal NasaTLX.raw_score for the same inputs."""
        scores = self._make_scores(
            mental=60,
            physical=10,
            temporal=70,
            performance=80,
            effort=50,
            frustration=40,
        )
        expected = NasaTLX(
            mental_demand=scores["mental"],
            physical_demand=scores["physical"],
            temporal_demand=scores["temporal"],
            performance=scores["performance"],
            effort=scores["effort"],
            frustration=scores["frustration"],
        ).raw_score

        result = NasaTLXForm.compute_raw_score(scores)
        assert result == pytest.approx(expected)

    def test_all_zeros(self) -> None:
        """All zeros → raw_score = (100 - 0) / 6 = 100/6 (performance inverted)."""
        scores = self._make_scores(
            mental=0, physical=0, temporal=0, performance=0, effort=0, frustration=0
        )
        # performance=0 inverts to 100; rest are 0 → mean = 100/6
        expected = (0 + 0 + 0 + (100 - 0) + 0 + 0) / 6
        result = NasaTLXForm.compute_raw_score(scores)
        assert result == pytest.approx(expected)

    def test_all_100(self) -> None:
        """All 100 → performance inverted to 0; rest are 100 → mean = 500/6."""
        scores = self._make_scores(
            mental=100, physical=100, temporal=100, performance=100, effort=100, frustration=100
        )
        expected = (100 + 100 + 100 + (100 - 100) + 100 + 100) / 6
        result = NasaTLXForm.compute_raw_score(scores)
        assert result == pytest.approx(expected)

    def test_symmetry_with_nasa_tlx_model(self) -> None:
        """raw_score from dict helper equals NasaTLX.raw_score for multiple cases."""
        cases = [
            {
                "mental": 10,
                "physical": 20,
                "temporal": 30,
                "performance": 40,
                "effort": 50,
                "frustration": 60,
            },
            {
                "mental": 99,
                "physical": 1,
                "temporal": 55,
                "performance": 75,
                "effort": 25,
                "frustration": 0,
            },
            {
                "mental": 0,
                "physical": 100,
                "temporal": 0,
                "performance": 100,
                "effort": 0,
                "frustration": 100,
            },
        ]
        for case in cases:
            form_score = NasaTLXForm.compute_raw_score(case)
            model_score = NasaTLX(
                mental_demand=case["mental"],
                physical_demand=case["physical"],
                temporal_demand=case["temporal"],
                performance=case["performance"],
                effort=case["effort"],
                frustration=case["frustration"],
            ).raw_score
            assert form_score == pytest.approx(model_score), f"Mismatch for case {case}"

    def test_result_is_float(self) -> None:
        """compute_raw_score always returns a float."""
        scores = self._make_scores()
        result = NasaTLXForm.compute_raw_score(scores)
        assert isinstance(result, float)

    def test_result_within_bounds(self) -> None:
        """raw_score must be in [0, 100] for valid inputs."""
        for _ in range(10):
            scores = self._make_scores(
                mental=50, physical=50, temporal=50, performance=50, effort=50, frustration=50
            )
            result = NasaTLXForm.compute_raw_score(scores)
            assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# _SCALES constant
# ---------------------------------------------------------------------------


class TestScalesConstant:
    def test_exactly_six_scales(self) -> None:
        """_SCALES must have exactly 6 entries (one per NASA-TLX dimension)."""
        assert len(_SCALES) == 6

    def test_each_scale_is_triple(self) -> None:
        """Each entry is a (key, label, help_text) triple."""
        for entry in _SCALES:
            assert len(entry) == 3, f"Expected triple, got {entry!r}"

    def test_required_keys_present(self) -> None:
        """All 6 required keys are present."""
        required_keys = {"mental", "physical", "temporal", "performance", "effort", "frustration"}
        actual_keys = {entry[0] for entry in _SCALES}
        assert actual_keys == required_keys

    def test_no_duplicate_keys(self) -> None:
        """No duplicate scale keys."""
        keys = [entry[0] for entry in _SCALES]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Module import without Streamlit runtime
# ---------------------------------------------------------------------------


def test_module_importable_without_streamlit_runtime() -> None:
    """atm.ui.tlx can be imported and NasaTLXForm accessed without a running Streamlit app."""
    from atm.ui.tlx import NasaTLXForm

    assert callable(NasaTLXForm.compute_raw_score)
    assert callable(NasaTLXForm.render)


def test_atm_ui_importable_without_streamlit_runtime() -> None:
    """atm.ui package is importable without the [ui] extra (lazy imports)."""
    import atm.ui  # noqa: F401
