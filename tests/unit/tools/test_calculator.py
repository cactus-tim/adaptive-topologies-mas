"""Unit tests for CalculatorTool — AST-safe expression evaluator.

Tests verify:
- Correct arithmetic results and output shape
- Safe math function calls (sqrt, etc.)
- Rejection of unsafe inputs (__import__, arbitrary names, attributes)
"""

from __future__ import annotations

import math

import pytest

from atm.tools.global_.calculator import CalculatorTool


@pytest.fixture()
def calc() -> CalculatorTool:
    return CalculatorTool()


# ---------------------------------------------------------------------------
# Happy-path arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_simple_add(calc: CalculatorTool) -> None:
    """2+2 should return value=4 and expression='2+2'."""
    result = await calc.ainvoke({"expression": "2+2"})
    assert result.ok is True
    assert result.error is None
    assert result.output["value"] == 4
    assert result.output["expression"] == "2+2"


@pytest.mark.asyncio
async def test_calculator_precedence(calc: CalculatorTool) -> None:
    """1 + 2*3 should respect operator precedence → value=7."""
    result = await calc.ainvoke({"expression": "1 + 2*3"})
    assert result.ok is True
    assert result.output["value"] == 7


@pytest.mark.asyncio
async def test_calculator_power(calc: CalculatorTool) -> None:
    """2**10 should return 1024."""
    result = await calc.ainvoke({"expression": "2**10"})
    assert result.ok is True
    assert result.output["value"] == 1024


@pytest.mark.asyncio
async def test_calculator_sqrt(calc: CalculatorTool) -> None:
    """sqrt(16) should return 4.0."""
    result = await calc.ainvoke({"expression": "sqrt(16)"})
    assert result.ok is True
    assert result.output["value"] == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_calculator_unary_minus(calc: CalculatorTool) -> None:
    """-5 + 3 should return -2."""
    result = await calc.ainvoke({"expression": "-5 + 3"})
    assert result.ok is True
    assert result.output["value"] == -2


@pytest.mark.asyncio
async def test_calculator_pi_constant(calc: CalculatorTool) -> None:
    """pi should return math.pi."""
    result = await calc.ainvoke({"expression": "pi"})
    assert result.ok is True
    assert result.output["value"] == pytest.approx(math.pi)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_output_shape(calc: CalculatorTool) -> None:
    """Both 'value' and 'expression' keys must be present in output."""
    result = await calc.ainvoke({"expression": "3 * 4"})
    assert result.ok is True
    assert "value" in result.output
    assert "expression" in result.output


# ---------------------------------------------------------------------------
# Rejection of unsafe inputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_rejects_import(calc: CalculatorTool) -> None:
    """__import__('os') must be rejected with ok=False and 'invalid' in error."""
    result = await calc.ainvoke({"expression": "__import__('os')"})
    assert result.ok is False
    assert result.error is not None
    assert "invalid" in result.error.lower()


@pytest.mark.asyncio
async def test_calculator_rejects_name_outside_allowlist(calc: CalculatorTool) -> None:
    """foo + 1 must be rejected — 'foo' is not in the allowed name set."""
    result = await calc.ainvoke({"expression": "foo + 1"})
    assert result.ok is False
    assert result.error is not None
    assert "invalid" in result.error.lower()


@pytest.mark.asyncio
async def test_calculator_rejects_attribute(calc: CalculatorTool) -> None:
    """os.system('x') must be rejected — Attribute nodes are disallowed."""
    result = await calc.ainvoke({"expression": "os.system('x')"})
    assert result.ok is False
    assert result.error is not None
    assert "invalid" in result.error.lower()
