# M2 — LLM Layer — Plan

## Executive Summary

Build the unified LLM abstraction for ATM: a `LLMWrapper` over LangChain chat models with
per-call usage/cost tracking, three-tier budget gating (call / run / experiment), retry on
transient errors (429 / 5xx / network), prompt caching support (Anthropic `cache_control`,
OpenAI automatic), provider factories (OpenAI / Anthropic / vLLM), a deterministic `FakeLLM`
(scripted / replay / echo) for tests, and a pricing table. Introduces the primary contract
`LLMWrapper.ainvoke -> LLMResponse` that every agent in M5 will consume.

## Current State

M1 is complete (138 tests green). `core/types.py` exports `LLMResponse`, `TokenUsage`,
`BudgetEvent`, `Message`, `ToolCall` and several StrEnum types. `core/errors.py` has
`AtmError`, `BudgetExceededError`, `PhaseError`, `ToolError` — `LLMError` is absent and must
be added. `LLMResponse` lacks `started_at`. The `llm_calls` reducer in `core/state.py` has a
deferred TODO comment deferring `sort_by="started_at"` to M2. `Message.to_lc` / `from_lc`
are stubs that raise `NotImplementedError`. No `langchain-*` packages are installed yet.

## Proposed Approach

Implement in four sequential waves with intra-wave parallelism. Each wave has a clear
dependency gate:

- Wave 1: core extensions + deps + pricing YAML + fixtures (no interdependencies)
- Wave 2: five disjoint modules — pricing.py, budget.py, retry.py, providers/*.py, fake.py
- Wave 3: wrapper.py + Message.to_lc/from_lc (stitches all wave-2 modules)
- Wave 4: integration contract test + public exports / linting

Key architectural decisions:
- `asyncio.Lock` in `BudgetTracker` (not `threading.RLock` — the whole stack is async)
- Retry implemented manually with `asyncio.sleep` + exponential backoff + jitter; no `tenacity`
- `cache_write_tokens` added as optional kwarg to `Pricing.cost` rather than extending frozen
  `TokenUsage` Pydantic model
- `FakeLLM` keyed by `agent_id` with `role` as fallback (matches arch.md §4.4, not task-description wording)
- `LLMWrapper` accepts optional `llm: BaseChatModel | FakeLLM | None` injection kwarg — no
  separate `FakeLLMWrapper` class needed
- `BudgetTracker.persist()` is a documented no-op stub; M3 implements SQLAlchemy bridge

## Implementation Phases

### Phase 1: Core extensions + deps + scaffold (~1.5h)
**Goal:** Establish all M1 changes and install deps before any new module is written.

- [ ] 1.1 Add `LLMError` to `core/errors.py` and export from `atm.core`
  - File: `src/atm/core/errors.py`
  - Acceptance: `from atm.core.errors import LLMError` works; shape mirrors `ToolError`
    with attrs `provider`, `model`, `attempts`; `__cause__` property.

- [ ] 1.2 Extend `LLMResponse` in `core/types.py` with `started_at: datetime`
  - File: `src/atm/core/types.py`
  - Acceptance: `LLMResponse()` (no `started_at` arg) auto-populates field via `_utcnow`
    default factory; no existing call-sites break.

- [ ] 1.3 Update `GraphState.llm_calls` reducer in `core/state.py`
  - File: `src/atm/core/state.py`
  - Acceptance: `dedup_by_id_reducer("id", sort_by="started_at")` used for `llm_calls`;
    TODO comment removed.

- [ ] 1.4 Write regression unit tests for Steps 1.1–1.3
  - Files: `tests/unit/core/test_errors.py`, `tests/unit/core/test_types.py`,
    `tests/unit/core/test_state.py` (or test_reducers.py)
  - Acceptance: `uv run pytest tests/unit/core -q` green; 138 M1 baseline tests still pass.

- [ ] 1.5 Add runtime deps to `pyproject.toml` and lock
  - File: `pyproject.toml`, `uv.lock`
  - Acceptance: `uv run python -c "from langchain.chat_models import init_chat_model; from langchain_openai import ChatOpenAI; from langchain_anthropic import ChatAnthropic; print('ok')"` exits 0.

- [ ] 1.6 Create `conf/pricing.yaml` with 4 models + fake entry
  - File: `conf/pricing.yaml`
  - Acceptance: file loads via `yaml.safe_load`; contains keys for `openai:gpt-4o`,
    `openai:gpt-4o-mini`, `anthropic:claude-3-5-sonnet-latest`,
    `anthropic:claude-3-5-haiku-latest`, `fake:deterministic`.

- [ ] 1.7 Create `tests/fixtures/llm/` with 3 YAML fixture files
  - Files: `tests/fixtures/llm/planner_simple.yaml`,
    `tests/fixtures/llm/executor_code_run.yaml`,
    `tests/fixtures/llm/determinism_seed.yaml`
  - Acceptance: all three load via `yaml.safe_load`.

### Phase 2: LLM module implementations (~3h)
**Goal:** Five disjoint modules, fully unit-tested, ready for the wrapper to compose.

- [ ] 2.1 Implement `llm/pricing.py` — `Pricing` loader + cost calculation
  - File: `src/atm/llm/pricing.py`, `src/atm/llm/__init__.py`,
    `tests/unit/llm/__init__.py`, `tests/unit/llm/test_pricing.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_pricing.py -q` green;
    `Pricing.from_yaml("conf/pricing.yaml").cost("openai:gpt-4o-mini", TokenUsage(1000,500,0,1500)) == 0.00045`;
    unknown model raises `LLMError`.

- [ ] 2.2 Implement `llm/budget.py` — `BudgetTracker` with asyncio.Lock
  - File: `src/atm/llm/budget.py`, `tests/unit/llm/test_budget.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_budget.py -q` green; 6 tests pass
    (per-call / per-run / per-experiment / warn-once / concurrent-record / callback-awaited).

- [ ] 2.3 Implement `llm/retry.py` — async retry with exponential backoff + jitter
  - File: `src/atm/llm/retry.py`, `tests/unit/llm/test_retry.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_retry.py -q` green; 4 tests pass.

- [ ] 2.4 Implement `llm/providers/openai.py`, `anthropic.py`, `vllm.py`
  - Files: `src/atm/llm/providers/__init__.py`, `src/atm/llm/providers/openai.py`,
    `src/atm/llm/providers/anthropic.py`, `src/atm/llm/providers/vllm.py`,
    `tests/unit/llm/test_providers.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_providers.py -q` green; factory kwargs
    forwarded correctly; `inject_cache_control` returns new list, never mutates input.

- [ ] 2.5 Implement `llm/fake.py` — `FakeLLM` (scripted / replay / echo)
  - File: `src/atm/llm/fake.py`, `tests/unit/llm/test_fake_llm.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_fake_llm.py -q` green; 6 tests pass;
    `latency_ms=0` constant so Pydantic `__eq__` works without field exclusion.

### Phase 3: Wrapper + Message adapters (~2h)
**Goal:** Compose all wave-2 modules into the published `LLMWrapper.ainvoke -> LLMResponse` contract.

- [ ] 3.1 Implement `Message.to_lc` and `Message.from_lc` in `core/types.py`
  - File: `src/atm/core/types.py`,
    `tests/unit/core/test_message_lc_adapter.py`
  - Acceptance: 4-kind round-trip test passes; `to_lc` / `from_lc` no longer raise.

- [ ] 3.2 Implement `llm/wrapper.py` — `LLMWrapper`
  - File: `src/atm/llm/wrapper.py`, `tests/unit/llm/test_wrapper_usage.py`,
    `tests/unit/llm/test_wrapper_retry.py`
  - Acceptance: `uv run pytest tests/unit/llm/test_wrapper_*.py tests/unit/core/test_message_lc_adapter.py -q`
    green; OpenAI-shape + Anthropic-shape usage parsed correctly; budget-exceed halts
    before calling `_llm.ainvoke`; streaming raises `NotImplementedError`.

### Phase 4: Integration contract + exports (~1h)
**Goal:** Lock the M2 published contract and clean up public API.

- [ ] 4.1 Write integration acceptance test `test_llm_layer_contract.py`
  - Files: `tests/integration/llm/__init__.py`,
    `tests/integration/llm/test_llm_layer_contract.py`
  - Acceptance: `uv run pytest tests/integration/llm/test_llm_layer_contract.py -v` green;
    covers wrapper+budget+scripted-fake + budget-exceed + replay round-trip.

- [ ] 4.2 Wire `src/atm/llm/__init__.py` public exports + ruff/mypy clean
  - File: `src/atm/llm/__init__.py`, `src/atm/core/__init__.py`
  - Acceptance: `from atm.llm import LLMWrapper, FakeLLM, BudgetTracker, Pricing` works;
    `uv run ruff check src tests` zero findings; `uv run mypy src/atm` zero findings;
    `uv run pytest -q` all green.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/core/errors.py` | Add `LLMError` | Retry exhaustion and pricing lookup need a typed error |
| `src/atm/core/types.py` | Add `LLMResponse.started_at`; implement `Message.to_lc/from_lc` | CI-1 contract requirement; wrapper needs LC message conversion |
| `src/atm/core/state.py` | Add `sort_by="started_at"` to llm_calls reducer | Ordered call log for M3 DB persistence |
| `src/atm/llm/__init__.py` | New — public export barrel | Clean import surface for M5 consumers |
| `src/atm/llm/pricing.py` | New | Cost calculation from pricing YAML |
| `src/atm/llm/budget.py` | New | Async-safe three-tier budget gating |
| `src/atm/llm/retry.py` | New | Exponential backoff retry with transient detection |
| `src/atm/llm/wrapper.py` | New | Core LLMWrapper — published M2 contract |
| `src/atm/llm/fake.py` | New | Deterministic FakeLLM for all downstream tests |
| `src/atm/llm/providers/openai.py` | New | OpenAI `init_chat_model` factory |
| `src/atm/llm/providers/anthropic.py` | New | Anthropic factory + `inject_cache_control` |
| `src/atm/llm/providers/vllm.py` | New | vLLM `ChatOpenAI(base_url=...)` factory |
| `conf/pricing.yaml` | New | Per-1K token pricing for 4 models + fake |
| `pyproject.toml` | Add 6 runtime deps | LangChain ecosystem + pyyaml + pyarrow |
| `tests/unit/llm/` | New dir — 7 test files | Unit coverage per module |
| `tests/integration/llm/` | New dir — 1 test file | M2 contract acceptance test |
| `tests/fixtures/llm/` | New dir — 3 YAML fixtures | FakeLLM scripted-mode data |

## Dependencies & Order Constraints

```
Wave 1: steps 1.1–1.7 (no cross-dependencies; run in parallel)
   |
Wave 2: steps 2.1–2.5
  2.1 (pricing.py) needs: 1.5 (pyyaml dep), 1.6 (pricing.yaml)
  2.2 (budget.py)  needs: 1.1 (LLMError / BudgetExceededError)
  2.3 (retry.py)   needs: 1.1 (LLMError), 1.5 (langchain exc types)
  2.4 (providers)  needs: 1.5 (langchain installed)
  2.5 (fake.py)    needs: 1.1 (LLMResponse.started_at), 1.5 (pyyaml, pyarrow)
  All five are disjoint modules — run in parallel within Wave 2.
   |
Wave 3: step 3.1–3.2
  3.1 (to_lc/from_lc) can start at Wave 2; touches core/types.py (same file as 1.2 so serialize)
  3.2 (wrapper.py) needs: 2.1, 2.2, 2.3, 2.4; touches no Wave 2 files
   |
Wave 4: steps 4.1–4.2 (parallel; 4.1 reads all modules, 4.2 writes __init__.py only)
```

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LangChain `usage_metadata` shape changes across 0.3.x | Medium | Medium | Implement both normalized `usage_metadata` path and provider-specific `response_metadata` fallback; normalized takes precedence |
| `openai.RateLimitError` import fails in stripped env | Low | Low | Guard with `try/except ImportError`; duck-type fallback on `.status_code == 429` |
| Parquet schema drift between M2 fake-writer and M3 real writer | Medium | High | Define `REPLAY_SCHEMA` as module constant in `atm.llm.fake`; M3 imports it |
| Off-by-1000 token math in `Pricing.cost` | Low | High | Hand-calculate 3 test cases and assert exact float values |
| LangChain transitive deps conflict with existing sqlalchemy/asyncpg | Low | Medium | Run `uv sync` immediately after editing `pyproject.toml`; resolve by relaxing upper bounds if needed |
| Anthropic cache_control on plain-string content rejected | Low | Medium | Convert content to list-of-blocks only when `cache_key` is set |

## Implementation Notes (from reviewer, non-blocking)

1. **`is_transient(exc)` duck-typing** (Step 2.3): in addition to matching the class tuple from
   `RetryPolicy.retry_on`, also match any exception with `.status_code >= 500` or
   `.status_code == 429` via `getattr(exc, "status_code", None)`. This ensures future SDK
   variants are caught without updating the tuple.

2. **`inject_cache_control` immutability** (Step 2.4): the helper must return a NEW
   `list[BaseMessage]` — never mutate the input list or any message in-place. Callers may
   pass cached prompt lists.

3. **`FakeLLM.latency_ms = 0`** (Step 2.5): set `latency_ms=0` as a constant in all
   `FakeLLM` responses so that Pydantic `__eq__` works correctly in determinism tests without
   needing field exclusion.

4. **`REPLAY_SCHEMA` key mapping** (Step 4.1): `REPLAY_SCHEMA` (defined in `atm.llm.fake`)
   must map `call_id -> LLMResponse.id` — the Parquet column name is `call_id`, not `id`, to
   match the M3 DB column convention.

## Out of Scope

- Real LLM API calls in tests (all tests use FakeLLM or AsyncMock)
- Streaming (`astream`) — stubbed with `NotImplementedError`; M4+ implements
- `BudgetTracker.persist()` DB round-trip — no-op stub; M3 implements SQLAlchemy bridge
- Live-provider smoke tests — deferred to `@pytest.mark.live` module (post-M2 follow-up)
- Cache TTL as a per-call parameter — hardcoded `"5m"` default in M2; promoted to config in M5+
- `TokenUsage.cache_write_tokens` field extension — handled via optional kwarg on `Pricing.cost`
- `BudgetPersister` protocol seam — prefer no-op stub over extra interface surface in M2

## Timeline

- Total: ~7.5h
- Phase 1: ~1.5h
- Phase 2: ~3h
- Phase 3: ~2h
- Phase 4: ~1h
- Created: 2026-04-23
