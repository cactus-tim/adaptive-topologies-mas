"""Regression test — importing atm.analysis.oracle must NOT pull in matplotlib.

This is a guard against import bloat: oracle.py is used by the runtime router
(M8.7) and must never add matplotlib to the process's module set.

The TYPE_CHECKING guard pattern used in oracle.py ensures that the annotation
``"matplotlib.figure.Figure"`` is evaluated only during type-checking, never
at module import time.

Test strategy
-------------
To verify that oracle.py itself does not import matplotlib, we spawn a fresh
subprocess with a clean Python environment.  The subprocess:
  1. Imports ``atm.analysis.oracle`` directly (bypassing __init__.py is NOT
     possible via normal import, so we use importlib to load the file directly).
  2. Asserts ``"matplotlib"`` is absent from ``sys.modules``.
  3. Exits 0 on success, 1 on failure (with an explanatory message to stderr).

Using a subprocess guarantees zero cross-contamination with other tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_ORACLE_PATH = _REPO_ROOT / "src" / "atm" / "analysis" / "oracle.py"

_CHECK_SCRIPT = """\
import importlib.util
import sys
from pathlib import Path

oracle_path = Path({oracle_path!r})
assert oracle_path.exists(), f"oracle.py not found: {{oracle_path}}"

spec = importlib.util.spec_from_file_location("atm.analysis.oracle", oracle_path)
assert spec is not None and spec.loader is not None

mod = importlib.util.module_from_spec(spec)
sys.modules["atm.analysis.oracle"] = mod

spec.loader.exec_module(mod)

matplotlib_modules = [k for k in sys.modules if k.startswith("matplotlib")]
if matplotlib_modules:
    print(f"FAIL: oracle.py imported matplotlib at module level: {{matplotlib_modules!r}}", flush=True)
    sys.exit(1)

print("OK: matplotlib not imported", flush=True)
sys.exit(0)
""".format(oracle_path=str(_ORACLE_PATH))


def _run_subprocess_check() -> subprocess.CompletedProcess[str]:
    """Run the import-isolation check in a subprocess, return result."""
    return subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(_REPO_ROOT / "src"),
            "HOME": str(Path.home()),
            "PATH": "/usr/bin:/bin",
        },
        timeout=30,
    )


class TestOracleModuleDoesNotImportMatplotlib:
    """Loading oracle.py in a clean process must not import matplotlib."""

    def test_oracle_module_does_not_import_matplotlib(self) -> None:
        """Core regression: fresh subprocess, import oracle.py, assert no matplotlib."""
        result = _run_subprocess_check()
        assert result.returncode == 0, (
            f"oracle.py imported matplotlib at module level.\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        assert "OK:" in result.stdout

    def test_oracle_exports_plot_oracle_vs_router(self) -> None:
        """plot_oracle_vs_router is a callable exported from oracle.py."""
        import atm.analysis.oracle as oracle_mod

        assert hasattr(oracle_mod, "plot_oracle_vs_router")
        assert callable(oracle_mod.plot_oracle_vs_router)

    def test_oracle_plot_function_importable(self) -> None:
        """plot_oracle_vs_router can be imported from atm.analysis.oracle."""
        from atm.analysis.oracle import plot_oracle_vs_router  # noqa: F401

        assert callable(plot_oracle_vs_router)

    def test_oracle_module_has_type_checking_guard(self) -> None:
        """oracle.py source code uses TYPE_CHECKING guard for matplotlib annotation."""
        source = _ORACLE_PATH.read_text(encoding="utf-8")
        assert "TYPE_CHECKING" in source, (
            "oracle.py should use the TYPE_CHECKING guard for matplotlib imports"
        )
        assert "if TYPE_CHECKING:" in source, "oracle.py should have 'if TYPE_CHECKING:' block"
        assert "import matplotlib" in source, (
            "oracle.py should have 'import matplotlib' under TYPE_CHECKING guard"
        )
