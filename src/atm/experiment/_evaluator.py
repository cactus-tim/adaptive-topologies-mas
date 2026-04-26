"""Inline substring evaluator for M6 experiment runs.

This is a temporary M6 stub that evaluates the quality of the final answer
using a simple substring check. M10 will introduce a full TaskRegistry with
proper evaluators (e.g., code execution, test running, AST analysis).

Design note: No subprocess, no sandbox, pure Python — CI-compatible.
"""

from __future__ import annotations

from typing import Any


def evaluate(task_cfg: Any, final_answer: str) -> float:
    """Evaluate the quality of the final answer for the given task.

    M6 stub: returns 1.0 if "55" is found in final_answer, else 0.0.
    The string "55" is the expected Fibonacci result: fib(10) = 55.

    This is intentionally simplistic — the full evaluator framework
    (TaskRegistry, per-task evaluators, sandbox execution) is planned for M10.

    Args:
        task_cfg:     Task configuration (TaskCfg or dict). Currently unused;
                      reserved for per-task evaluator dispatch in M10.
        final_answer: The final answer string produced by the topology.

    Returns:
        1.0 if "55" is in final_answer (case-sensitive), else 0.0.
    """
    if "55" in (final_answer or ""):
        return 1.0
    return 0.0
