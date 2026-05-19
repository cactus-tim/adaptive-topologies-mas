"""Post-graph finalize hook — synthesize a final answer when the graph leaves
``shared.final_answer`` unusable for the task type.

Why this exists
---------------
Tool-using executors (Cerebras gpt-oss-120b in agent mode) on non-code tasks
sometimes finish without ever emitting a DRAFT — they keep calling
``code_run`` / ``file_write`` and the ReAct loop exits on ``max_tool_iters``
with no text response. The topology-level extractors then fall back to the
written ``solution.py`` which contains Python intermediates, not the
human-readable / templated answer the evaluator needs.

Pre-existing fixes (chain d585bbb + star/mesh/hier/debate parity) made
extractors *prefer* a DRAFT for non-code tasks, but they cannot conjure one
when the executor never produced any. This module performs a SINGLE no-tool
LLM call after the graph terminates, asking the same model to read its prior
tool outputs and emit the answer in the required format.

When it fires
-------------
Only when ALL of the following hold:
  * ``task_id`` belongs to a non-code task (``gsm8k`` / ``commongen`` /
    ``dabench``).
  * ``current_answer`` looks unusable: empty, or Python-shaped, or — for
    DABench — missing the ``@name[value]`` template the evaluator expects.

Code tasks (``humaneval``) are never touched. The existing file-write
preference is correct for them.

Scope
-----
Runs once, outside the LangGraph compile/ainvoke cycle. Single LLM call,
budget-tracked through the same ``LLMWrapper`` that the executor used so it
counts against the run/experiment caps. On any error, returns the original
``current_answer`` unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from atm.core.types import Message, MessageKind, TaskSpec
from atm.llm.wrapper import LLMWrapper

logger = logging.getLogger(__name__)

_NON_CODE_TASK_PREFIXES: tuple[str, ...] = ("gsm8k/", "commongen/", "dabench/")

_PY_MARKERS = (
    "import ",
    "def ",
    "from ",
    "print(",
    "return ",
    "pd.",
    "np.",
    "plt.",
)

_DABENCH_TEMPLATE_RE = re.compile(r"@[A-Za-z_]\w*\[[^\]]+\]")

_COMMONGEN_META_MARKERS: tuple[str, ...] = (
    "issues identified",
    "suggested fix",
    "the executor",
    "does not contain",
    "does not provide",
    "task requirements",
)


def _is_non_code_task(task_id: str) -> bool:
    return task_id.startswith(_NON_CODE_TASK_PREFIXES)


def _looks_like_code(text: str) -> bool:
    """Heuristic: does ``text`` look like Python source rather than an answer?"""
    if "```" in text:
        return True
    hits = sum(1 for marker in _PY_MARKERS if marker in text)
    return hits >= 2


def _commongen_needs_finalize(task_spec: TaskSpec, stripped: str) -> bool:
    """CommonGen-specific finalize trigger.

    The standard "empty/code" triggers miss a common failure mode: the
    "answer" is the critic's complaint about the executor's output
    ("Issues identified: ...") rather than the requested sentence. Trigger
    finalize when:
      * any required concept is missing from the text (concept_coverage < 1.0), OR
      * the text matches a meta-commentary marker.
    """
    md = task_spec.metadata or {}
    concepts: list[str] = list(md.get("concepts") or [])
    if concepts:
        text_lower = stripped.lower()
        for concept in concepts:
            if concept.lower() not in text_lower:
                return True
    lower = stripped.lower()
    return any(marker in lower for marker in _COMMONGEN_META_MARKERS)


def _needs_finalize(task_spec: TaskSpec, current_answer: str) -> bool:
    """Decide whether to invoke the no-tool LLM finalize step.

    Triggers (any one is sufficient) for non-code tasks only:
      * empty / ``"<incomplete>"`` answer
      * answer looks like Python source
      * task is DABench AND the answer lacks any ``@name[value]`` template
      * task is CommonGen AND any required concept is missing OR the answer
        is a critic-style meta-comment (see ``_commongen_needs_finalize``)
    """
    task_id = task_spec.id
    if not _is_non_code_task(task_id):
        return False

    stripped = current_answer.strip()
    if not stripped or stripped == "<incomplete>":
        return True

    if _looks_like_code(stripped):
        return True

    if task_id.startswith("dabench/") and _DABENCH_TEMPLATE_RE.search(stripped) is None:
        return True

    return task_id.startswith("commongen/") and _commongen_needs_finalize(task_spec, stripped)


_MAX_TOOL_RESULTS = 6
_MAX_TOOL_OUTPUT_CHARS = 1200


def _format_tool_history(final_state: dict[str, Any]) -> str:
    """Render the last few executor tool results into a compact text block."""
    agents = (final_state or {}).get("agents") or {}
    executor_state = agents.get("executor") or {}
    tool_calls: list[Any] = list(executor_state.get("tool_calls") or [])
    tool_results: list[Any] = list(executor_state.get("tool_results") or [])

    by_call_id = {getattr(c, "id", None): c for c in tool_calls}

    blocks: list[str] = []
    for result in tool_results[-_MAX_TOOL_RESULTS:]:
        call = by_call_id.get(getattr(result, "call_id", None))
        tool_name = getattr(call, "tool_name", "?") if call is not None else "?"
        ok = getattr(result, "ok", False)
        output = getattr(result, "output", "") or ""
        if not isinstance(output, str):
            output = str(output)
        if len(output) > _MAX_TOOL_OUTPUT_CHARS:
            output = output[:_MAX_TOOL_OUTPUT_CHARS] + " …[truncated]"
        status = "ok" if ok else "FAIL"
        blocks.append(f"[{tool_name} {status}]\n{output}")

    return "\n\n".join(blocks)


def _format_required_format(task_spec: TaskSpec) -> str:
    """Render the task's required output format for the prompt."""
    md = task_spec.metadata or {}
    fmt = str(md.get("format") or "").strip()
    constraints = str(md.get("constraints") or "").strip()
    parts: list[str] = []
    if fmt:
        parts.append(f"Required format:\n{fmt}")
    if constraints:
        parts.append(f"Constraints:\n{constraints}")

    if task_spec.id.startswith("commongen/"):
        concepts: list[str] = list(md.get("concepts") or [])
        if concepts:
            parts.append(
                "Output requirements:\n"
                "- Write ONE complete English sentence (5-25 words).\n"
                f"- The sentence MUST contain every concept verbatim: {', '.join(concepts)}.\n"
                "- Do NOT output a list of issues, suggestions, headers, or markdown."
            )

    return "\n\n".join(parts)


async def maybe_finalize_answer(
    *,
    llm: LLMWrapper,
    task_spec: TaskSpec | None,
    final_state: dict[str, Any],
    current_answer: str,
) -> str:
    """Optionally replace ``current_answer`` with a synthesized final answer.

    Returns the original answer unchanged when:
      * ``task_spec`` is None (replay / inline-prompt path);
      * the task is a code task;
      * the answer already looks usable;
      * the LLM call raises (logged + swallowed — never block the run).
    """
    if task_spec is None:
        return current_answer

    if not _needs_finalize(task_spec, current_answer):
        return current_answer

    tool_history = _format_tool_history(final_state)
    required_format = _format_required_format(task_spec)

    system_text = (
        "You are completing the FINAL step of a task. Output ONLY the answer "
        "in the required format. Do NOT call any tools. Do NOT write code. "
        "Do NOT include reasoning, explanation, restatement of the question, "
        "Markdown, or code fences — just the answer text."
    )

    user_parts: list[str] = [f"Task:\n{task_spec.input}"]
    if required_format:
        user_parts.append(required_format)
    if tool_history:
        user_parts.append("Computation results so far:\n" + tool_history)
    if current_answer.strip():
        user_parts.append(
            "Previous (rejected) attempt — do not repeat verbatim:\n" + current_answer.strip()[:600]
        )
    user_parts.append("Now write the final answer.")
    user_text = "\n\n".join(user_parts)

    messages: list[Message] = [
        Message(sender="system", kind=MessageKind.REQUEST, content=system_text),
        Message(sender="user", kind=MessageKind.REQUEST, content=user_text),
    ]

    try:
        response = await llm.ainvoke(messages, tools=None, agent_id="finalize")
    except Exception:
        logger.warning(
            "finalize LLM call failed; keeping original answer",
            extra={"task_id": task_spec.id},
            exc_info=True,
        )
        return current_answer

    text = (response.text or "").strip()
    if not text:
        return current_answer

    logger.info(
        "finalize_answer synthesized",
        extra={
            "task_id": task_spec.id,
            "preview": text[:200],
            "original_preview": current_answer[:200],
        },
    )
    return text
