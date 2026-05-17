"""NASA-TLX 6-scale form for the Streamlit HITL UI (M14 §UI).

All Streamlit calls are inside the ``render`` classmethod so this module is
importable without the ``[ui]`` extra installed.

The six scales follow the NASA-TLX specification:
  Mental Demand, Physical Demand, Temporal Demand,
  Performance, Effort, Frustration — each in [0, 100].

``Performance`` is inverted in ``raw_score`` (lower self-rated performance =
higher workload), consistent with ``atm.evaluation.tlx.NasaTLX``.

Usage (inside a Streamlit page)::

    from atm.ui.tlx import NasaTLXForm

    result = NasaTLXForm.render()
    if result is not None:
        # dict: {"mental": int, "physical": int, "temporal": int,
        #        "performance": int, "effort": int, "frustration": int,
        #        "raw_score": float}
        st.json(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# NasaTLXForm
# ---------------------------------------------------------------------------

_SCALES: list[tuple[str, str, str]] = [
    # (key, label, help_text)
    (
        "mental",
        "Mental Demand",
        "How much mental and perceptual activity was required?",
    ),
    (
        "physical",
        "Physical Demand",
        "How much physical activity was required?",
    ),
    (
        "temporal",
        "Temporal Demand",
        "How much time pressure did you feel?",
    ),
    (
        "performance",
        "Performance",
        "How successful were you at accomplishing the task? "
        "(higher = better performance = lower load).",
    ),
    (
        "effort",
        "Effort",
        "How hard did you have to work?",
    ),
    (
        "frustration",
        "Frustration",
        "How irritated, stressed, or annoyed were you?",
    ),
]


@dataclass
class NasaTLXForm:
    """Helper that renders a NASA-TLX 6-slider form inside a Streamlit page.

    Call ``NasaTLXForm.render()`` to display the sliders and a submit button.
    Returns a dict with keys ``mental``, ``physical``, ``temporal``,
    ``performance``, ``effort``, ``frustration``, and ``raw_score`` when
    submitted, or ``None`` when the form has not been submitted yet.

    The ``raw_score`` mirrors :class:`atm.evaluation.tlx.NasaTLX`:
    unweighted mean of 6 scales with ``performance`` inverted.
    """

    @classmethod
    def render(
        cls,
        *,
        key_prefix: str = "tlx",
    ) -> dict[str, Any] | None:
        """Render the NASA-TLX form and return scores when submitted.

        Parameters
        ----------
        key_prefix:
            Prefix for Streamlit widget keys; change when embedding multiple
            TLX forms on the same page.

        Returns
        -------
        dict[str, Any] | None
            ``{"mental": int, ..., "raw_score": float}`` on submit;
            ``None`` when not yet submitted.
        """
        try:
            import streamlit as st
        except ImportError as exc:
            raise RuntimeError(
                "Streamlit is not installed. Install the [ui] extra: uv pip install 'atm[ui]'"
            ) from exc

        st.subheader("NASA Task Load Index (TLX)")
        st.caption(
            "Please rate the workload you experienced during this interaction. "
            "All scales range from 0 (very low) to 100 (very high)."
        )

        scores: dict[str, int] = {}
        with st.form(key=f"{key_prefix}_form"):
            for key, label, help_text in _SCALES:
                scores[key] = st.slider(
                    label=label,
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help=help_text,
                    key=f"{key_prefix}_{key}",
                )

            submitted = st.form_submit_button("Submit TLX Ratings")

        if submitted:
            raw_score = cls.compute_raw_score(scores)
            return {**scores, "raw_score": raw_score}

        return None

    @staticmethod
    def compute_raw_score(scores: dict[str, int]) -> float:
        """Compute unweighted NASA-TLX raw score from the six-scale dict.

        ``performance`` is inverted: ``(100 - performance)`` so that higher
        values always indicate higher workload, consistent with NasaTLX in
        ``atm.evaluation.tlx``.

        Parameters
        ----------
        scores:
            Dict with keys ``mental``, ``physical``, ``temporal``,
            ``performance``, ``effort``, ``frustration``, each in ``[0, 100]``.

        Returns
        -------
        float
            Unweighted mean across 6 dimensions (0.0-100.0).
        """
        mental = int(scores["mental"])
        physical = int(scores["physical"])
        temporal = int(scores["temporal"])
        performance = int(scores["performance"])
        effort = int(scores["effort"])
        frustration = int(scores["frustration"])

        return (mental + physical + temporal + (100 - performance) + effort + frustration) / 6
