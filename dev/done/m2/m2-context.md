# M2 — LLM Layer — Context

## SESSION PROGRESS (2026-04-23)

### COMPLETED
- Steps 1.1-1.4: Added `LLMError(AtmError)` to `core/errors.py` (attrs: provider, model, attempts, cause property); added `started_at: datetime` to `LLMResponse` in `core/types.py` with `_utcnow` default factory; updated `GraphState.llm_calls` reducer to `dedup_by_id_reducer("id", sort_by="started_at")`; wrote 14 new regression tests (8 for LLMError, 3 for LLMResponse.started_at, 3 for reducer sort). Replaced the M1-era `test_llm_response_no_started_at_field` guard with M2 tests. All 150 tests green, ruff clean.
- Step 1.5: Added LangChain runtime deps to `pyproject.toml` and updated `uv.lock` via `uv add`; all 138 M1 tests green; import verification passed.
- Step 1.6: Created `conf/pricing.yaml` with `version: 1` and 5 model entries (openai:gpt-4o, openai:gpt-4o-mini, anthropic:claude-3-5-sonnet-latest, anthropic:claude-3-5-haiku-latest, fake:deterministic) using April-2026 public pricing; Anthropic entries include cache_write_per_1k distinct from cache_read_per_1k; all fields documented with source comment.
- Step 1.7: Created 3 FakeLLM fixture YAMLs in `tests/fixtures/llm/` — `planner_simple.yaml` (1 entry, plain text), `executor_code_run.yaml` (2 entries: tool_call then text), `determinism_seed.yaml` (2 entries for bit-identical replay verification); all load via `yaml.safe_load`.
- Step 2.1: Implemented `src/atm/llm/pricing.py` with `ModelPricing` (all-optional float fields, 0 defaults) and `Pricing` (version + models dict, `from_yaml`, `cost`, `estimate`). Auto-detects OpenAI vs Anthropic cache convention by checking which rate fields are non-zero. `cache_write_tokens` passed as optional kwarg to `cost`. 16 unit tests in `tests/unit/llm/test_pricing.py` — all green. Added `types-PyYAML` to dev deps for mypy strict compliance.
- Step 2.2: Implemented `src/atm/llm/budget.py` — `BudgetLevel` (StrEnum), `BudgetEvent` (Pydantic frozen model), `BudgetTracker` (asyncio.Lock, three-tier check/record, warn+exceed callbacks, sync/async callback support); wrote `tests/unit/llm/test_budget.py` (7 tests: per-call/per-run/per-experiment cutoff, record accumulation, 100-concurrent gather lock correctness, event callback for warn+exceed, sync callback). All 159 unit tests green; ruff 0 findings; mypy 0 findings.
- Step 2.3: Implemented `src/atm/llm/retry.py` — `RetryPolicy` (frozen dataclass), `is_transient` (duck-typing on status_code 429/>=500 + retry_on tuple), `with_retry` (exponential backoff with full jitter, LLMError wrapping on exhaustion with __cause__ preserved); wrote `tests/unit/llm/test_retry.py` (19 tests: is_transient variants, success/backoff/non-transient/exhaustion/jitter). All tests green; ruff 0 findings; mypy 0 findings.

- Step 2.4: Implemented `src/atm/llm/providers/` package — `build_openai` (ChatOpenAI, strips prefix, api_key="EMPTY" default), `build_anthropic` (ChatAnthropic, strips prefix), `build_vllm` (ChatOpenAI with base_url + api_key="EMPTY"), `inject_cache_control` (deep-copy immutability, converts str→list-of-blocks, multi-modal aware). 26 unit tests in `tests/unit/llm/test_providers.py` — all green. ruff 0 findings; mypy 0 findings.
- Step 2.5: Implemented `src/atm/llm/fake.py` — `FakeLLM` (scripted / echo / replay modes); `REPLAY_SCHEMA` pyarrow schema constant; `asyncio.Lock` for per-agent step_idx; `latency_ms=0` constant for Pydantic `__eq__` determinism; wrote `tests/unit/llm/test_fake_llm.py` (8 tests). ruff 0 findings; mypy 0 findings.

- Step 3.1: Implemented `Message.to_lc` and `Message.from_lc` in `src/atm/core/types.py`. Mapping: REQUEST/DRAFT/CRITIQUE → HumanMessage, DECISION → AIMessage, BROADCAST → HumanMessage(additional_kwargs={channel:'broadcast'}), PHASE_EMIT → SystemMessage. `from_lc` stores LC extras in `payload['_lc']`. Updated M1 stub-check tests in `test_types.py` to reflect M2 implementation. 18 new tests in `test_message_lc_adapter.py` — all green.
- Step 3.2: Created `src/atm/llm/wrapper.py` — `LLMWrapper` with `ainvoke` + `astream` (stub). Supports OpenAI (`usage_metadata`) and Anthropic (`response_metadata.usage`) usage shapes. Pre-call budget estimate + three-level BudgetExceededError guard. Retry via `with_retry`. Token counting via tiktoken (openai) or heuristic (anthropic). Cache injection for Anthropic when `cache_key` set. 14 new tests (9 usage, 5 retry) — all green. ruff 0 findings; mypy 0 findings.

- Step 4.1: Created `tests/integration/llm/__init__.py` (empty) and `tests/integration/llm/test_llm_layer_contract.py` with 3 acceptance tests (scripted-fake end-to-end, budget-exceed-before-call, replay round-trip). Fixed a bug in `wrapper.py` where `agent_id` was not forwarded to `FakeLLM.ainvoke` and where `LLMResponse` returned by FakeLLM was not handled in the usage-parsing step (short-circuit added). All 3 integration tests green; all 260 unit tests green; ruff 0 findings.

- Fix mode: Resolved 2 blocking findings — (1) widened `FakeLLM.ainvoke` signature to accept `list[BaseMessage]` and removed 2 unused `# type: ignore` from `wrapper.py`; (2) renamed M2 runtime `BudgetEvent` → `BudgetSignal` in `budget.py`, `llm/__init__.py`, and `test_budget.py` to eliminate name collision with M1 DB schema class in `core/types.py`. ruff 0 findings; mypy 0 findings; 263 tests green.

### IN PROGRESS
- All blocking findings resolved; M2 complete.

### BLOCKERS
- None

## Quick Resume

1. Read this file
2. Check `m2-tasks.md` for what's next
3. Read `m2-plan.md` Phase 1 for strategy
4. Start with: Step 1.1 — add `LLMError` to `src/atm/core/errors.py`

## Key Files

**`src/atm/core/errors.py`**
- Role: ATM exception hierarchy (`AtmError`, `BudgetExceededError`, `PhaseError`, `ToolError`)
- Planned change: add `LLMError(AtmError)` with attrs `provider`, `model`, `attempts`; export from `atm.core`
- Status: NOT STARTED

**`src/atm/core/types.py`**
- Role: Pydantic domain models (`LLMResponse`, `TokenUsage`, `Message`, `ToolCall`, etc.)
- Planned change: (a) add `started_at: datetime` to `LLMResponse` with `_utcnow` default; (b) implement `Message.to_lc` / `Message.from_lc` (currently raise `NotImplementedError`)
- Status: NOT STARTED

**`src/atm/core/state.py`**
- Role: LangGraph `GraphState` TypedDict + reducer wiring
- Planned change: update `llm_calls` field to `dedup_by_id_reducer("id", sort_by="started_at")`; remove deferred TODO comment
- Status: NOT STARTED

**`src/atm/llm/pricing.py`**
- Role: Load `conf/pricing.yaml` and compute per-call USD cost from `TokenUsage`
- Planned change: new file; `PriceRow` Pydantic model, `Pricing` class with `from_yaml`, `cost`, `estimate_cost`
- Status: NOT STARTED

**`src/atm/llm/budget.py`**
- Role: Async-safe three-tier budget gating (call / run / experiment)
- Planned change: new file; `BudgetLimits`, `BudgetTracker` with `asyncio.Lock`, `check_before_call`, `record`, `persist` (stub)
- Status: NOT STARTED

**`src/atm/llm/retry.py`**
- Role: Async exponential backoff retry with transient-error detection
- Planned change: new file; `RetryPolicy`, `with_retry`, `is_transient`
- Status: NOT STARTED

**`src/atm/llm/wrapper.py`**
- Role: Primary M2 contract — `LLMWrapper.ainvoke(messages, ...) -> LLMResponse`
- Planned change: new file; wires pricing + budget + retry + providers; parses OpenAI and Anthropic usage metadata shapes
- Status: NOT STARTED

**`src/atm/llm/fake.py`**
- Role: Deterministic `FakeLLM` for tests (scripted / replay / echo modes)
- Planned change: new file; `FakeLLM` with seeded RNG; `REPLAY_SCHEMA` constant; `latency_ms=0`
- Status: NOT STARTED

**`src/atm/llm/providers/anthropic.py`**
- Role: Anthropic `init_chat_model` factory + `inject_cache_control`
- Planned change: new file; `inject_cache_control` returns NEW list, never mutates input
- Status: NOT STARTED

**`src/atm/llm/providers/openai.py`**
- Role: OpenAI `init_chat_model` factory
- Planned change: new file
- Status: NOT STARTED

**`src/atm/llm/providers/vllm.py`**
- Role: vLLM `ChatOpenAI(base_url=...)` factory
- Planned change: new file
- Status: NOT STARTED

**`conf/pricing.yaml`**
- Role: Per-1K-token price table for 4 models + `fake:deterministic`
- Planned change: new file; `version: 1`; Anthropic and OpenAI price fields
- Status: NOT STARTED

**`pyproject.toml`**
- Role: Project manifest + dep pinning
- Planned change: add `langchain>=0.3,<0.4`, `langchain-core>=0.3,<0.4`, `langchain-openai>=0.3,<0.4`, `langchain-anthropic>=0.3,<0.4`, `pyyaml>=6`, `pyarrow>=16`, `tiktoken>=0.8`
- Status: DONE (Step 1.5)

## Decisions

**asyncio.Lock in BudgetTracker**
- Decision: use `asyncio.Lock`, not `threading.RLock`
- Rationale: the entire wrapper stack is async; there is no cross-thread use in M2. Arch.md §4.2 uses `RLock` as an illustrative stub only.

**Retry without tenacity**
- Decision: implement retry manually with `asyncio.sleep` + exponential backoff + jitter
- Rationale: avoids an extra dependency; keeps the retry predicate trivial to unit-test; backoff formula is a one-liner.

**cache_write_tokens as Pricing.cost kwarg**
- Decision: add `cache_write_tokens: int = 0` optional kwarg to `Pricing.cost` rather than extending `TokenUsage`
- Rationale: `TokenUsage` is a frozen M1 Pydantic contract; extending it would require all M1 call-sites to update. Default-factory keeps backward compatibility but adds surface area unnecessarily.

**FakeLLM fixture key = agent_id with role fallback**
- Decision: key scripted fixture by `agent_id`; fall back to `role` if `agent_id` absent from YAML
- Rationale: arch.md §4.4 is explicit about `agent_id`; task description wording ("role") was loose.

**LLMWrapper llm injection kwarg**
- Decision: `LLMWrapper.__init__` accepts optional `llm: BaseChatModel | FakeLLM | None = None`; when provided, skip `init_chat_model`
- Rationale: simpler than a parallel `FakeLLMWrapper` class; single point of test injection.

**BudgetTracker.persist stub**
- Decision: `persist()` is a no-op for M2 (docstring + `return None`)
- Rationale: M3 implements the SQLAlchemy `UPDATE ... RETURNING` bridge. Introducing a `BudgetPersister` Protocol now would be premature surface area.

**REPLAY_SCHEMA column naming**
- Decision: Parquet schema constant `REPLAY_SCHEMA` in `atm.llm.fake` maps `call_id -> LLMResponse.id`
- Rationale: M3 DB column convention uses `call_id`; keeping it consistent here allows M3 to import the same schema without renaming.

**inject_cache_control immutability**
- Decision: always return a new `list[BaseMessage]`; never mutate the input list or its messages
- Rationale: callers may pass cached prompt lists that are reused across invocations.

**is_transient duck-typing**
- Decision: `is_transient(exc)` matches the class tuple from `RetryPolicy.retry_on` AND additionally checks `getattr(exc, "status_code", None) in (429,)` or `>= 500`
- Rationale: future SDK variants may expose status codes without being in the explicit tuple; duck-typing covers them automatically.

**FakeLLM latency_ms = 0**
- Decision: always set `latency_ms=0` in `FakeLLM` responses
- Rationale: allows determinism tests to use plain Pydantic `__eq__` without excluding `latency_ms` from comparison.

## Constraints

- Python 3.11+, Pydantic v2, strict mypy, ruff — must pass at Phase 4 close
- All 138 M1 tests must remain green throughout; Step 1 is the only phase touching M1 files
- No real LLM API calls in tests — all tests use `FakeLLM` or `AsyncMock`
- Streaming (`astream`) raises `NotImplementedError` for all of M2
- LangChain 0.3.x line only — pin `<0.4` to avoid silent 1.x surprises
- `conf/pricing.yaml` values must be annotated with source and check-date comment
