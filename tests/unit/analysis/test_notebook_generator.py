"""Tests for scripts/gen_analysis_notebook.py.

Four tests:
  1. test_notebook_generates       — running generate_notebook() returns a valid nbformat notebook
  2. test_notebook_validates        — nbformat.validate(nb) passes (no ValidationError)
  3. test_committed_notebook_matches_generator — committed notebooks/analysis_template.ipynb
                                      has the same cell sources as a freshly generated one
  4. test_rq2_section_covers_all_metrics — all 7 metric symbols appear in RQ2 cells
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Locate the generator script (importable as module)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
GENERATOR_PATH = SCRIPTS_DIR / "gen_analysis_notebook.py"
NOTEBOOKS_DIR = Path(__file__).resolve().parents[3] / "notebooks"
COMMITTED_NB_PATH = NOTEBOOKS_DIR / "analysis_template.ipynb"

# 7 RQ2 metric symbols that must appear somewhere in RQ2 cells
_RQ2_REQUIRED_SYMBOLS = [
    "compute_hurt_rate",
    "compute_oracle_gap_manual",
    "compute_oracle_gap_loo",
    "compute_topology_switch_counts",
    "compute_guard_override_rate",
    "compute_router_cost_share",
    "compute_time_per_topology",
]


def _import_generator() -> object:
    """Import gen_analysis_notebook as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location("gen_analysis_notebook", GENERATOR_PATH)
    assert spec is not None, f"Could not build spec for {GENERATOR_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Stash in sys.modules so relative imports (if any) resolve correctly
    sys.modules["gen_analysis_notebook"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell_sources(nb: nbformat.NotebookNode) -> list[str]:
    """Return list of source strings for every cell in the notebook."""
    return [cell["source"] for cell in nb.cells]


def _rq2_sources(nb: nbformat.NotebookNode) -> str:
    """Return a single concatenated string of all RQ2-related cell sources.

    We detect RQ2 cells by looking for the string "RQ2" (case-insensitive) in
    the cell source.  This is intentionally broad so that a markdown header
    like ``## RQ2`` and subsequent code cells are both matched.
    """
    rq2_start = False
    rq3_start = False
    collected: list[str] = []

    for cell in nb.cells:
        src: str = cell["source"]
        src_upper = src.upper()
        # Start collecting when we see an RQ2 section marker
        if "## RQ2" in src_upper or "# RQ2" in src_upper:
            rq2_start = True
            rq3_start = False
        # Stop collecting when RQ3 starts
        if rq2_start and ("## RQ3" in src_upper or "# RQ3" in src_upper):
            rq3_start = True
        if rq2_start and not rq3_start:
            collected.append(src)

    return "\n".join(collected)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNotebookGenerator:
    """Tests for scripts/gen_analysis_notebook.py."""

    def test_generator_script_exists(self) -> None:
        """The generator script must be present at scripts/gen_analysis_notebook.py."""
        assert GENERATOR_PATH.exists(), f"Script not found: {GENERATOR_PATH}"

    def test_notebook_generates(self) -> None:
        """generate_notebook() returns an nbformat NotebookNode with cells."""
        mod = _import_generator()
        assert hasattr(mod, "generate_notebook"), (
            "gen_analysis_notebook must expose a generate_notebook() function"
        )
        nb = mod.generate_notebook()  # type: ignore[union-attr]
        assert isinstance(nb, nbformat.NotebookNode)
        assert len(nb.cells) > 0, "Notebook must have at least one cell"

    def test_notebook_validates(self) -> None:
        """nbformat.validate() must pass without raising ValidationError."""
        mod = _import_generator()
        nb = mod.generate_notebook()  # type: ignore[union-attr]
        # raises nbformat.ValidationError on failure
        nbformat.validate(nb)

    def test_committed_notebook_matches_generator(self) -> None:
        """The committed notebooks/analysis_template.ipynb cell sources must match
        what generate_notebook() produces.

        Cell *sources* are compared (not raw bytes) so that kernel metadata
        differences between machines do not cause false failures.
        """
        assert COMMITTED_NB_PATH.exists(), (
            f"Committed notebook not found: {COMMITTED_NB_PATH}. "
            "Run `python scripts/gen_analysis_notebook.py` to generate it."
        )
        mod = _import_generator()
        generated_nb = mod.generate_notebook()  # type: ignore[union-attr]
        committed_nb = nbformat.read(str(COMMITTED_NB_PATH), as_version=4)

        generated_sources = _cell_sources(generated_nb)
        committed_sources = _cell_sources(committed_nb)

        assert generated_sources == committed_sources, (
            "Committed notebook cell sources differ from generator output.\n"
            "Re-run `python scripts/gen_analysis_notebook.py` to update the committed file."
        )

    def test_rq2_section_covers_all_metrics(self) -> None:
        """All 7 RQ2 metric symbols must appear somewhere in the RQ2 section cells."""
        mod = _import_generator()
        nb = mod.generate_notebook()  # type: ignore[union-attr]
        rq2_text = _rq2_sources(nb)
        assert rq2_text, "No RQ2 section found in the generated notebook"

        missing = [sym for sym in _RQ2_REQUIRED_SYMBOLS if sym not in rq2_text]
        assert not missing, (
            f"The following metric symbols are missing from RQ2 cells: {missing}\n"
            f"RQ2 cell content (first 500 chars): {rq2_text[:500]}"
        )
