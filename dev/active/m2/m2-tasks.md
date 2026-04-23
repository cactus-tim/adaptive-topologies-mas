# M2 — LLM Layer — Tasks

## Phase 1: Core extensions + deps + scaffold  COMPLETE

- [x] 1.1 Add `LLMError(AtmError)` to `core/errors.py` — `src/atm/core/errors.py`
  - Acceptance: `from atm.core.errors import LLMError` works; attrs `provider`, `model`,
    `attempts`; `__cause__` property; chaining semantics identical to `ToolError`.

- [x] 1.2 Add `started_at: datetime` to `LLMResponse` in `core/types.py` — `src/atm/core/types.py`
  - Acceptance: `LLMResponse()` auto-populates `started_at` via `_utcnow` default factory;
    no existing call-sites broken.

- [x] 1.3 Update `llm_calls` reducer in `core/state.py` — `src/atm/core/state.py`
  - Acceptance: `dedup_by_id_reducer("id", sort_by="started_at")` wired; TODO comment removed.

- [x] 1.4 Regression unit tests for 1.1–1.3
  - Files: `tests/unit/core/test_errors.py`, `tests/unit/core/test_types.py`,
    `tests/unit/core/test_state.py` (or test_reducers.py)
  - Acceptance: `uv run pytest tests/unit/core -q` green; 138 M1 baseline still pass;
    tests cover `LLMError` attrs+chain, `LLMResponse.started_at` default, reducer sort order.

- [x] 1.5 Add runtime deps to `pyproject.toml` and lock — `pyproject.toml`, `uv.lock`
  - Acceptance: `uv run python -c "from langchain.chat_models import init_chat_model; from langchain_openai import ChatOpenAI; from langchain_anthropic import ChatAnthropic; print('ok')"` exits 0.

- [x] 1.6 Create `conf/pricing.yaml` with 5 entries — `conf/pricing.yaml`
  - Acceptance: loads via `yaml.safe_load`; contains `openai:gpt-4o`, `openai:gpt-4o-mini`,
    `anthropic:claude-3-5-sonnet-latest`, `anthropic:claude-3-5-haiku-latest`,
    `fake:deterministic`; top-level `version: 1`.

- [x] 1.7 Create fixture YAMLs in `tests/fixtures/llm/`
  - Files: `planner_simple.yaml`, `executor_code_run.yaml`, `determinism_seed.yaml`
  - Acceptance: all three load via `yaml.safe_load`.

## Phase 2: LLM module implementations  COMPLETE

- [x] 2.1 `llm/pricing.py` — `Pricing` loader + cost calculation
  - Files: `src/atm/llm/pricing.py`, `src/atm/llm/__init__.py`,
    `tests/unit/llm/__init__.py`, `tests/unit/llm/test_pricing.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_pricing.py -q` green; 6 test cases
    (loader / unknown-model / OpenAI-no-cache / OpenAI-cache-hit / Anthropic-cache / estimate).

- [x] 2.2 `llm/budget.py` — `BudgetTracker` with asyncio.Lock
  - Files: `src/atm/llm/budget.py`, `tests/unit/llm/test_budget.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_budget.py -q` green; 6 test cases pass.

- [x] 2.3 `llm/retry.py` — async retry + `is_transient` duck-typing
  - Files: `src/atm/llm/retry.py`, `tests/unit/llm/test_retry.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_retry.py -q` green; 4 test cases;
    `is_transient` also matches `getattr(exc, "status_code", None) >= 500` or `== 429`.

- [x] 2.4 `llm/providers/` — three factory functions + `inject_cache_control`
  - Files: `src/atm/llm/providers/__init__.py`, `openai.py`, `anthropic.py`, `vllm.py`,
    `tests/unit/llm/test_providers.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_providers.py -q` green;
    `inject_cache_control` returns NEW list, never mutates input.

- [x] 2.5 `llm/fake.py` — `FakeLLM` (scripted / replay / echo)
  - Files: `src/atm/llm/fake.py`, `tests/unit/llm/test_fake_llm.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_fake_llm.py -q` green; 6 test cases;
    `latency_ms=0` constant; `REPLAY_SCHEMA` maps `call_id -> LLMResponse.id`.

## Phase 3: Wrapper + Message adapters  COMPLETE

- [x] 3.1 Implement `Message.to_lc` / `Message.from_lc` in `core/types.py`
  - Files: `src/atm/core/types.py`, `tests/unit/core/test_message_lc_adapter.py`
  - Acceptance: 4-kind round-trip test passes; stubs no longer raise.

- [x] 3.2 `llm/wrapper.py` — `LLMWrapper` (full implementation)
  - Files: `src/atm/llm/wrapper.py`, `tests/unit/llm/test_wrapper_usage.py`,
    `tests/unit/llm/test_wrapper_retry.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_wrapper_*.py tests/unit/core/test_message_lc_adapter.py -q`
    green; OpenAI-shape + Anthropic-shape usage parsed; budget-exceed halts before
    `_llm.ainvoke`; astream raises `NotImplementedError`.

## Phase 4: Integration contract + exports  COMPLETE

- [x] 4.1 Integration acceptance test `test_llm_layer_contract.py`
  - Files: `tests/integration/llm/__init__.py`,
    `tests/integration/llm/test_llm_layer_contract.py`
  - Acceptance: `uv run pytest tests/integration/llm/test_llm_layer_contract.py -v` green;
    covers scripted-fake + budget-exceed-path + replay-round-trip; all exception assertions
    use `atm.core.errors` types, not bare `Exception`.

- [x] 4.2 Public exports + ruff/mypy clean
  - Files: `src/atm/llm/__init__.py`, `src/atm/core/__init__.py`
  - Acceptance: `from atm.llm import LLMWrapper, FakeLLM, BudgetTracker, Pricing` works;
    `uv run ruff check src tests` → 0 findings;
    `uv run mypy src/atm` → 0 findings;
    `uv run pytest -q` → all tests green.

---

## Stats

- Total: 14 tasks · ~7.5h
- Done: 14 / 14

## How to Update

Check off tasks with `[x]` and update `m2-context.md` SESSION PROGRESS after each milestone.
Phase header format: replace `NOT STARTED` with `IN PROGRESS` when first task is ticked,
and `COMPLETE` when all tasks in the phase carry `[x]`.
