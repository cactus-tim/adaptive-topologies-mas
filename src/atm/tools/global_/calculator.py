"""CalculatorTool — AST-walker safe expression evaluator.

Evaluates arithmetic expressions without using eval() or exec().
Only whitelisted AST node types and math functions are allowed.

Allowed AST nodes:
    Expression, BinOp (+/-/*///%/**/FloorDiv), UnaryOp (UAdd, USub),
    Constant (int/float), Call (func must be a whitelisted Name),
    Name (only 'pi' and 'e' constants).

Everything else (Attribute, Subscript, Import, Lambda, etc.) is rejected.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from typing import Any, ClassVar

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

_Numeric = int | float

# ---------------------------------------------------------------------------
# Safe function/constant mapping — only these Names are allowed
# ---------------------------------------------------------------------------

_SAFE_FUNCS: dict[str, Callable[..., _Numeric]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "min": min,
    "max": max,
}

_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------


def _eval_node(node: ast.AST) -> _Numeric:
    """Recursively evaluate an AST node.

    Raises ValueError with a descriptive reason for any disallowed node type.
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"non-numeric constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        bin_op = node.op
        if isinstance(bin_op, ast.Add):
            return left + right
        if isinstance(bin_op, ast.Sub):
            return left - right
        if isinstance(bin_op, ast.Mult):
            return left * right
        if isinstance(bin_op, ast.Div):
            return left / right
        if isinstance(bin_op, ast.Mod):
            return left % right
        if isinstance(bin_op, ast.Pow):
            return left**right
        if isinstance(bin_op, ast.FloorDiv):
            return left // right
        raise ValueError(f"unsupported binary operator: {type(bin_op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        unary_op = node.op
        if isinstance(unary_op, ast.UAdd):
            return +operand
        if isinstance(unary_op, ast.USub):
            return -operand
        raise ValueError(f"unsupported unary operator: {type(unary_op).__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function names are allowed (no attribute access)")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"function not allowed: {func_name!r}")
        fn = _SAFE_FUNCS[func_name]
        call_args = [_eval_node(arg) for arg in node.args]
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in calculator calls")
        result: _Numeric = fn(*call_args)
        return result

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_NAMES:
            return _SAFE_NAMES[name]
        raise ValueError(f"name not allowed: {name!r}")

    raise ValueError(f"disallowed expression node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


class CalculatorTool:
    """AST-safe arithmetic calculator tool.

    Evaluates mathematical expressions without using eval() or exec().
    Uses a strict AST-walker that only allows whitelisted node types and
    math functions.

    Allowed operations:
        - Binary: +, -, *, /, %, **, //
        - Unary: +, -
        - Constants: int, float literals; 'pi', 'e'
        - Functions: sqrt, sin, cos, tan, log, log10, exp, pow,
                     floor, ceil, abs, min, max

    Rejects:
        - Attribute access (os.system, etc.)
        - Import nodes
        - Lambda expressions
        - Names not in the explicit allowlist
        - Any other AST node not listed above
    """

    name: ClassVar[str] = "calculator"

    schema: ClassVar[ToolSchema] = ToolSchema(
        name="calculator",
        description=(
            "Evaluates a mathematical expression safely. "
            "Supports +, -, *, /, %, **, // and math functions "
            "(sqrt, sin, cos, tan, log, log10, exp, pow, floor, ceil, abs, min, max) "
            "plus constants pi and e."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g. '2+2', 'sqrt(16)').",
                },
            },
            "required": ["expression"],
        },
        returns={
            "value": "number",
            "expression": "string",
        },
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Evaluate the expression and return the numeric result.

        Parameters
        ----------
        args:
            dict with key ``expression`` (str) — the math expression to evaluate.

        Returns
        -------
        ToolResult
            On success: ``ok=True``, ``output={"value": <number>, "expression": <str>}``.
            On failure: ``ok=False``, ``error="invalid expression: <reason>"``.
        """
        from uuid import uuid4

        call_id = uuid4()
        expression: str = args.get("expression", "")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=f"invalid expression: syntax error — {exc}",
                latency_ms=0,
            )

        try:
            value = _eval_node(tree)
        except (ValueError, ZeroDivisionError, TypeError, ArithmeticError) as exc:
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=f"invalid expression: {exc}",
                latency_ms=0,
            )

        return ToolResult(
            call_id=call_id,
            ok=True,
            output={"value": value, "expression": expression},
            error=None,
            latency_ms=0,
        )
