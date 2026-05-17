"""Streamlit HITL UI package for the ATM framework (M14 §UI).

All Streamlit imports are lazy (inside functions or guarded with try/except
ImportError) so that ``import atm.ui`` works without the ``[ui]`` optional
extra installed.  This lets non-UI unit tests import helpers from this package
without requiring Streamlit to be present.

Entry point::

    streamlit run src/atm/ui/app.py

Public helpers (importable without Streamlit)::

    from atm.ui.tlx import NasaTLXForm
"""

__all__ = ["NasaTLXForm"]


def __getattr__(name: str) -> object:
    if name == "NasaTLXForm":
        from atm.ui.tlx import NasaTLXForm

        return NasaTLXForm
    raise AttributeError(f"module 'atm.ui' has no attribute {name!r}")
