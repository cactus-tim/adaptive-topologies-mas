# m10-merge-m8 — Context

## SESSION PROGRESS (2026-05-13)

### COMPLETED
- Step 1.1: Backup branch `backup/feat-m10-pre-m8-merge` + tag `wave-1-pre` + `strategy.md` created
- Step 2.1: HITL infra brought from origin/feat/m8 via checkout-paths (64 files, 16552 insertions). Commit: `d2f57bc` — `bring(m9+m9.2): import HITL infra from feat/m8`. Import sanity: `from atm.human import HumanGateway, ...` — exit 0.

### IN PROGRESS
- Wave 3: Parallel Code Expansion (Steps 3.1, 3.2, 3.3, 3.4, 3.5) — NOT STARTED

### BLOCKERS
- None

## Quick Resume

1. Read this file
2. Check `m10-merge-m8-tasks.md` for what's next
3. Read `m10-merge-m8-plan.md` Phase 1 for strategy
4. Start with: Step 1.1 — создать backup branch `backup/feat-m10-pre-m8-merge`, tag `wave-1-pre`, зафиксировать стратегию в `dev/active/m10-merge-m8/strategy.md`

**Critical invariants before starting any work:**
- Working directory must be clean (`git status` empty) — confirmed at plan time.
- Before each wave N>=2: `git tag -f wave-N-pre HEAD` (mandatory).
- Do NOT run `git merge origin/feat/m8` until Step 9 (Wave 9).

## Key Files

**`src/atm/experiment/runner.py`**
- Role: центральный оркестратор run_one(); M11-версия (804 строки) с compute_quality, seed_all, model_version_snapshot, sandbox_image_digest в finally
- Planned change: добавить _build_role_router, _build_human_gateway_llm, dynamic_human_role SELECT, cognitive_load_proxy write, timeout_policy branches; расширить _update_run_success/_update_run_failed сигнатуры
- Status: NOT STARTED

**`src/atm/experiment/config.py`**
- Role: Pydantic v2 конфиг ExperimentConfig; уже содержит EvaluationCfg (M11)
- Planned change: добавить класс HumanCfg + поле `ExperimentConfig.human: HumanCfg | None = None`
- Status: NOT STARTED

**`src/atm/evaluation/metrics.py`**
- Role: M11-метрики (aggregate_quality, humaneval_pass_at_k, cost_per_quality, time_per_quality, aggregate_human_load)
- Planned change: добавить human_sim_cognitive_load_proxy, _load_weights, WEIGHTS_PATH, DEFAULT_WEIGHTS
- Status: NOT STARTED

**`src/atm/evaluation/__init__.py`**
- Role: публичный API пакета evaluation; M11-список __all__
- Planned change: добавить human_sim_cognitive_load_proxy в импорт и __all__
- Status: NOT STARTED

**`src/atm/storage/models.py`**
- Role: SQLAlchemy ORM — Run, HumanInteraction, и др.; Run уже имеет human_role, model_version_snapshot, sandbox_image_digest
- Planned change: добавить Run.cognitive_load_proxy: Mapped[float | None]
- Status: NOT STARTED

**`alembic/versions/0002_add_cognitive_load_proxy.py`**
- Role: новая Alembic-миграция для Run.cognitive_load_proxy
- Planned change: создать новую ИЛИ взять verbatim с feat/m8 (определяется pre-check'ом Step 3.3)
- Status: NOT STARTED

**`src/atm/human/__init__.py`**
- Role: публичный API human-пакета; re-exports HumanGateway, LLMSimulatedGateway, CLIGateway, routers
- Status: DONE (Step 2.1)

**`src/atm/human/gateway.py`** (from feat/m8)
- Role: HumanGateway base class + HumanContext, HumanResponse dataclasses
- Status: DONE (Step 2.1)

**`src/atm/human/llm_simulated.py`** (from feat/m8)
- Role: LLMSimulatedGateway — fallback при timeout_policy="llm_fallback"
- Status: DONE (Step 2.1)

**`src/atm/human/cli_gateway.py`** (from feat/m8)
- Role: CLIGateway — интерактивный CLI gateway
- Status: DONE (Step 2.1)

**`src/atm/human/role_router.py`** (from feat/m8)
- Role: FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter, HumanRoleRouter
- Status: DONE (Step 2.1)

**`conf/evaluation/cognitive_load.yaml`** (from feat/m8)
- Role: веса alpha/beta/gamma для формулы human_sim_cognitive_load_proxy
- Status: DONE (Step 2.1)

**`src/atm/topology/*.py`**
- Role: 5 topology-классов (star, chain, mesh, debate, hierarchical/adaptive); текущие — m7-версии без HITL-узлов
- Planned change: принять m8-версии с HITL-узлами через final merge (Step 9); НЕ трогать до Step 9
- Status: NOT STARTED

**`dev/codebase-map.md`**
- Role: живая карта кодовой базы для developer-агентов; датирован "2026-05-13 (M11-resync)"
- Planned change: полная перезапись под M9/M9.1/M9.2/M10/M11 после Step 9
- Status: NOT STARTED

**`dev/active/m10-merge-m8/strategy.md`** (new)
- Role: документация выбора phased-merge стратегии + wave-tagging invariant
- Planned change: создать в Step 1.1
- Status: NOT STARTED

**`tests/unit/experiment/test_run_update_helpers.py`** (new)
- Role: TDD unit-тест для _update_run_success/_update_run_failed с новыми kwargs
- Planned change: создать в Step 5.1
- Status: NOT STARTED

**`tests/integration/experiment/test_run_one_full_contract.py`** (new)
- Role: acceptance e2e — проверяет полный контракт run_one() с HITL, value-range assertion на cognitive_load_proxy
- Planned change: создать в Step 8.1; становится зелёным после Step 9
- Status: NOT STARTED

## Decisions

**Merge стратегия: phased-additive, не rebase**
- Decision: поэтапные правки на feat/m10 + один git merge --no-commit --no-ff в конце (Step 9)
- Rationale: PR#10 уже открыт с review-историей (rebase сломал бы все SHA и комментарии); на feat/m8 два merge-commit'а после merge-base → rebase дал бы ~30 конфликтов вместо 6

**runner.py канон: M11-версия + M9.x addons**
- Decision: не заменять runner.py целиком m8-версией; добавлять M9.1/M9.2 фичи поверх M11-канона
- Rationale: M11-каноничные фичи (seed_all, compute_quality, model_version_snapshot/sandbox_image_digest в finally) не должны быть потеряны; m8-runner их не содержит

**Option A (locked-in): conditional kwargs spread для topology.build**
- Decision: `if cfg.human and cfg.human.enabled: human_kwargs = {...}` + `topology.build(..., **human_kwargs)`
- Rationale: без cfg.human kwargs не передаются → TypeError не возникает; старые топологии работают нетронутыми до Step 9 (final merge)

**Alembic pre-check обязателен перед созданием миграции**
- Decision: сначала проверить `git ls-tree -r origin/feat/m8 -- alembic/versions/` — взять m8-миграцию verbatim если она уже добавляет cognitive_load_proxy
- Rationale: избежать 2 Alembic heads после final merge; если heads всё же расходятся — merge-migration

**m9.2-тесты: adapt mock evaluate → compute_quality**
- Decision: заменить `mocker.patch("atm.experiment.runner.evaluate", return_value=1.0)` на `mocker.patch("atm.experiment.runner.compute_quality", new=AsyncMock(return_value=(1.0, {})))`
- Rationale: M6-stub `_evaluator.evaluate(cfg.task, final_answer)` не существует на feat/m10; новый pipeline — async compute_quality(spec, final_answer, sandbox, judge_llm, run_seed)

**cognitive_load_proxy value-range assertion в acceptance test**
- Decision: детерминированная фикстура (K=2 interrupts, фиксированный context_len/latency) + `pytest.approx(expected, rel=0.05)` или bounds-check; НЕ `is not None`
- Rationale: reviewer M5 explicit — `is not None` недостаточно как acceptance criterion

**Wave-tagging hygiene**
- Decision: перед стартом каждой wave N≥2 обязателен `git tag -f wave-N-pre HEAD`
- Rationale: позволяет за один `git reset --hard wave-N-pre` откатить wave если что-то пошло не так

## Constraints

- `FinishReason.HUMAN_TIMEOUT` уже присутствует в enum на feat/m10 HEAD — используем для timeout_policy="fail" без правок core/types
- Алиас `origin/feat/m8` должен существовать и быть up-to-date (`git fetch origin feat/m8` при необходимости)
- postgres:16 нужен для integration-тестов (Steps 8.1, 9.1); `@pytest.mark.requires_postgres` маркер
- Тесты с `cfg.human.enabled=true` + реальной HITL-топологией — запускаются только ПОСЛЕ Step 9 (final merge принесёт m8-топологии с HITL-узлами)
- HumanCfg.role — Literal (не StrEnum), чтобы не конфликтовать с Run.human_role (StrEnum если есть в core/types); сверить при Step 3.1
- `_build_role_router` для role_router="llm": ассертировать что `cfg.human.role_router_model is not None` (иначе mypy strict падает)
- НЕ удалять `bc5f66dd0897_initial.py` (duplicate-initial Alembic revision) — out of scope
- НЕ добавлять human_role/cognitive_load_proxy в RunResult.metrics dict — только в runs колонки
