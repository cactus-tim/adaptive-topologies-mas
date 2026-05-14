# m10-merge-m8 — Plan

## Executive Summary

Интегрировать ветку `feat/m8` (содержащую `feat/m9` HITL + `feat/m9.2` Adaptive Role Router) в текущую ветку `feat/m10` (содержащую `feat/m11` Evaluation Framework + M10 task-mix), не нарушая Exit Criteria milestones M9/M9.1/M9.2/M10/M11 из `arch/PLAN.md`. PR#10 должен перестать быть `CONFLICTING` и стать ready-for-review.

Ключевая архитектурная позиция: `runner.py` берётся в каноничной M11-версии (с `compute_quality`, `seed_all`, `model_version_snapshot`, `sandbox_image_digest` в finally), а поверх неё накладываются M9.1/M9.2-добавки (HumanGateway wiring, `_build_role_router`, post-run агрегация `human_role` из `human_interactions`, `cognitive_load_proxy`). M6-stub `src/atm/experiment/_evaluator.py` удаляется. Конфликты в 6 файлах разрешаются вручную поэтапными коммитами на `feat/m10`.

## Current State

- `feat/m10` HEAD = `2a11d23` (M10-resync завершён); `feat/m8` существует как отдельная ветка.
- `runner.py` (804 строки) — M11-версия: `seed_all`, `compute_quality`, `model_version_snapshot`/`sandbox_image_digest` в finally. **Не содержит**: HumanGateway wiring, `_build_role_router`, `cognitive_load_proxy` write.
- `src/atm/human/__init__.py` — пустой; вся HITL-инфраструктура (`gateway.py`, `llm_simulated.py`, `cli_gateway.py`, `role_router.py`) живёт только на `feat/m8`.
- `storage/models.py` — `Run` имеет `human_role`, `model_version_snapshot`, `sandbox_image_digest`; **не имеет** `cognitive_load_proxy`.
- `experiment/config.py` — содержит `EvaluationCfg`; **не содержит** `HumanCfg` и `ExperimentConfig.human`.
- `evaluation/metrics.py` — содержит M11-функции; **не содержит** `human_sim_cognitive_load_proxy`.
- Alembic chain на m10 HEAD: `bc5f66dd0897_initial.py` → `0001_initial_business_schema.py` (head = `0001`).
- `src/atm/experiment/_evaluator.py` — отсутствует на `feat/m10` HEAD (уже удалён).
- Все тесты на `feat/m10` HEAD зелёные (1233 unit, M11-resync).

## Proposed Approach

**Стратегия**: phased-additive on-branch — поэтапные правки на `feat/m10` + один merge-commit `feat/m8` в самом конце (Step 13). **Не rebase** (причины: PR#10 уже открыт с review-историей; на `feat/m8` два merge-commit'а после merge-base, что дало бы ~30 конфликтов вместо 6).

**Wave-tagging hygiene**: перед стартом каждой wave N≥2 обязателен `git tag -f wave-N-pre HEAD` для per-wave отката.

**Ключевые решения**:
- `runner.py` — M11-канон + M9.1/M9.2 addons поверх (не merge m8-версии runner'а целиком).
- **Option A (locked-in)**: conditional kwargs spread для topology.build — без `cfg.human` kwargs не передаются; старые топологии работают нетронутыми до Step 13.
- Alembic pre-check перед созданием новой миграции (взять m8-миграцию verbatim, если она уже добавляет `cognitive_load_proxy`).
- `_evaluator.py` — удалить если воскреснет при финальном merge; транзитивные импорты в топологиях почистить post-merge audit'ом (до commit).
- m9.2-тесты, мокающие старый `evaluate(cfg.task, final_answer)`, переписываются под `compute_quality(...)` (AsyncMock).

## Implementation Phases

### Phase 1: Foundation (~0.5h)
**Goal:** Зафиксировать стратегию, создать backup-ветку и wave-1-pre tag.

- [ ] 1.1 Выбрать стратегию слияния + создать backup branch + wave-tagging hygiene
  - File: `dev/active/m10-merge-m8/strategy.md` (новый), ref: `backup/feat-m10-pre-m8-merge`, ref: `wave-1-pre`
  - Acceptance: `git branch | grep backup/feat-m10-pre-m8-merge` — строка; `git tag -l wave-1-pre` → `wave-1-pre`; `strategy.md` содержит обоснование + wave-tagging invariant.

### Phase 2: HITL Infrastructure Bring-in (~1h)
**Goal:** Принести всю HITL-инфраструктуру с `feat/m8` в не-конфликтные пути.

- [ ] 2.1 Принести HITL-инфраструктуру с feat/m8 через checkout-paths
  - File: `src/atm/human/{gateway,llm_simulated,cli_gateway,role_router}.py` (new), `src/atm/human/__init__.py` (update), `conf/human/`, `conf/evaluation/cognitive_load.yaml`, `tests/unit/human/**`, `tests/integration/human/**`, `tests/unit/evaluation/test_cognitive_load_proxy.py`
  - Acceptance: `uv run python -c "from atm.human import HumanGateway, LLMSimulatedGateway, FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter"` — exit 0.

### Phase 3: Parallel Code Expansion (~2h)
**Goal:** Параллельно расширить config, metrics, models, провести grep audit и обновить lockfile.

- [ ] 3.1 Расширить `experiment/config.py` — добавить `HumanCfg` + `ExperimentConfig.human` (TDD)
  - File: `src/atm/experiment/config.py`, `tests/unit/experiment/test_config.py`
  - Acceptance: `uv run pytest tests/unit/experiment/test_config.py tests/unit/experiment/test_config_evaluation_cfg.py -q` — зелёное.

- [ ] 3.2 Слить add/add `evaluation/metrics.py` — добавить `human_sim_cognitive_load_proxy` + weights loader (TDD)
  - File: `src/atm/evaluation/metrics.py`, `tests/unit/evaluation/test_cognitive_load_proxy.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_cognitive_load_proxy.py tests/unit/evaluation/test_metrics.py -q` — зелёное; `uv run mypy src/atm/evaluation/metrics.py --strict` — clean.

- [ ] 3.3 Добавить `Run.cognitive_load_proxy` + Alembic-миграция (с pre-check на feat/m8, TDD)
  - File: `src/atm/storage/models.py`, `alembic/versions/0002_add_cognitive_load_proxy.py` (или verbatim m8)
  - Acceptance: `uv run alembic upgrade head` проходит; `uv run alembic heads | wc -l` → `1`; `uv run pytest tests/unit/storage/ -q` — зелёное.

- [ ] 3.4 Подтвердить отсутствие `_evaluator.py` + audit импортов (grep)
  - File: `src/atm/experiment/_evaluator.py` (удалить если воскрес)
  - Acceptance: `grep -r "from atm.experiment._evaluator" src tests/unit/experiment tests/integration` — пусто (или findings задокументированы).

- [ ] 3.5 Перегенерировать `uv.lock`
  - File: `uv.lock`, `pyproject.toml` (проверка diff с m8)
  - Acceptance: `uv sync --frozen` exit 0; `uv run python -c "import atm; print('ok')"` exit 0.

### Phase 4: evaluation/__init__.py Re-export (~0.25h)
**Goal:** Добавить `human_sim_cognitive_load_proxy` в публичный API пакета evaluation.

- [ ] 4.1 Обновить `evaluation/__init__.py` — re-export `human_sim_cognitive_load_proxy`
  - File: `src/atm/evaluation/__init__.py`
  - Acceptance: `uv run python -c "from atm.evaluation import human_sim_cognitive_load_proxy, compute_quality, persist_tlx, NasaTLX, score_ground_truth; print('ok')"` — exit 0; `uv run ruff check src/atm/evaluation/` — clean.

### Phase 5: Runner Signatures (~0.5h)
**Goal:** Атомарно расширить сигнатуры helper'ов runner'а под новые поля (TDD, без логики).

- [ ] 5.1 Расширить сигнатуры `_update_run_success` и `_update_run_failed` — kwargs `human_role` + `cognitive_load_proxy` (TDD)
  - File: `src/atm/experiment/runner.py`, `tests/unit/experiment/test_run_update_helpers.py` (новый)
  - Acceptance: `uv run pytest tests/unit/experiment/test_run_update_helpers.py -q` — зелёное; `uv run pytest tests/unit/experiment/test_runner.py tests/unit/experiment/test_runner_evaluation_wiring.py -q` — без регрессии; `uv run mypy src/atm/experiment/runner.py --strict` — clean.

### Phase 6: Runner HITL Wiring (~2.5h)
**Goal:** Подключить HumanGateway, _build_role_router, dynamic_human_role SELECT, cognitive_load_proxy write, timeout_policy branches в run_one().

- [ ] 6.1 Добавить `_build_role_router`, `_build_human_gateway_llm`, wire в `run_one` + timeout_policy branches (TDD)
  - File: `src/atm/experiment/runner.py`
  - Acceptance: `uv run pytest tests/unit/experiment/test_runner.py tests/unit/experiment/test_runner_evaluation_wiring.py -q` — зелёное (regression); `uv run mypy src/atm/experiment/runner.py --strict` — clean.

### Phase 7: Test Suite Sanity + m9.2 Test Adaptation (~1.5h)
**Goal:** Параллельно: sanity check топологий после Option A wiring и адаптация m9.2-тестов под compute_quality.

- [ ] 7.1 Sanity check топологий — убедить что Option A не сломал ни один topology test (read-only)
  - File: (read-only) `tests/unit/topology/*`, `tests/integration/topology/*`
  - Acceptance: `uv run pytest tests/unit/topology -q && uv run pytest tests/integration/topology -q` — зелёное.

- [ ] 7.2 Адаптировать m9.2-тесты runner'а + audit integration/human-тестов под `compute_quality` pipeline (TDD)
  - File: `tests/unit/experiment/{test_runner_role_router,test_runner_human_role,test_runner_initial_state}.py`, `tests/integration/human/**` (selective)
  - Acceptance: `uv run pytest tests/unit/experiment/ -q` — зелёное; `uv run pytest tests/integration/human/ -m "not requires_postgres" -q` — зелёное.

### Phase 8: Acceptance Test Scaffolding (~1h)
**Goal:** Написать e2e acceptance test (будет красным до Step 13, зелёным после).

- [ ] 8.1 Написать integration acceptance test `test_run_one_full_contract.py` с value-range assertion (TDD)
  - File: `tests/integration/experiment/test_run_one_full_contract.py` (новый), `tests/fixtures/experiment/` (опц. YAML)
  - Acceptance: файл существует; `uv run pytest tests/integration/experiment/test_run_one_full_contract.py -m requires_postgres -q` — после Step 13 зелёное с value-range assertion на `cognitive_load_proxy`.

### Phase 9: Final Merge (~1h)
**Goal:** Выполнить `git merge origin/feat/m8 --no-commit --no-ff`, file-by-file ресолюшен 6 защищаемых файлов, post-merge audit, commit.

- [ ] 9.1 Финальный merge `origin/feat/m8` — file-by-file ресолюшен + post-merge `_evaluator` audit + alembic single-head check
  - File: `src/atm/topology/*.py` (m8-версия), `alembic/versions/*` (ресолюшен), все auto-resolved m8 additions
  - Acceptance: `uv run pytest tests/unit -q` — зелёное; `uv run pytest tests/integration -m requires_postgres -q` — зелёное (включая тест из Step 8); `uv run alembic heads | wc -l` → `1`; `! grep -rn "_evaluator" src/` — пусто; `git log --merges -1` — merge-commit.

### Phase 10: Codebase Map + Final Audit (~0.75h)
**Goal:** Параллельно: перезаписать codebase-map и выполнить финальный integrity sweep + push.

- [ ] 10.1 Перезаписать `dev/codebase-map.md` под целостную M9/M9.1/M9.2/M10/M11 картину
  - File: `dev/codebase-map.md`
  - Acceptance: `grep -E "M9|M9.1|M9.2|cognitive_load|human/gateway|role_router" dev/codebase-map.md` — все термины присутствуют.

- [ ] 10.2 Final integrity audit + push + reopen PR#10 review
  - File: (no file changes; CI-validation + push)
  - Acceptance: `gh pr view 10 --json mergeable` → `"MERGEABLE"`; CI зелёный.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/experiment/runner.py` | Extend: `_build_role_router`, `_build_human_gateway_llm`, dynamic_human_role SELECT, cognitive_load_proxy write, timeout_policy branches в `run_one()` | M9/M9.1/M9.2 wiring поверх M11-канона |
| `src/atm/experiment/config.py` | Add: `HumanCfg` class + `ExperimentConfig.human: HumanCfg \| None = None` | Конфиг-контракт для HITL |
| `src/atm/evaluation/metrics.py` | Add: `human_sim_cognitive_load_proxy`, `_load_weights`, `WEIGHTS_PATH`, `DEFAULT_WEIGHTS` | M9.2 cognitive load формула |
| `src/atm/evaluation/__init__.py` | Add to `__all__`: `human_sim_cognitive_load_proxy` | Публичный API пакета |
| `src/atm/storage/models.py` | Add: `Run.cognitive_load_proxy: Mapped[float \| None]` | Хранение метрики |
| `alembic/versions/0002_add_cognitive_load_proxy.py` | New (или verbatim из m8) | DB-схема |
| `src/atm/human/__init__.py` | Rewrite: re-exports HumanGateway, LLMSimulatedGateway, CLIGateway, routers | Публичный API human-пакета |
| `src/atm/human/gateway.py` | New (from m8) | HumanGateway base + HumanContext/HumanResponse |
| `src/atm/human/llm_simulated.py` | New (from m8) | LLMSimulatedGateway |
| `src/atm/human/cli_gateway.py` | New (from m8) | CLIGateway |
| `src/atm/human/role_router.py` | New (from m8) | FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter, HumanRoleRouter |
| `conf/evaluation/cognitive_load.yaml` | New (from m8) | Веса alpha/beta/gamma для формулы |
| `conf/human/role_table.yaml` | New (from m8, если существует) | Таблица ролей для RuleBasedRoleRouter |
| `src/atm/topology/*.py` | Update (after Step 13 merge): m8-версии с HITL-узлами | M9.1 HITL в топологиях |
| `dev/active/m10-merge-m8/strategy.md` | New | Документация стратегии merge |
| `dev/codebase-map.md` | Rewrite | Отражение целостной M9+M10+M11 картины |
| `uv.lock` | Regen if needed | Lockfile consistency |
| `tests/unit/experiment/test_run_update_helpers.py` | New | TDD для Step 5 |
| `tests/unit/experiment/test_runner_role_router.py` | Adapt: mock evaluate → compute_quality | M6→M11 API |
| `tests/unit/experiment/test_runner_human_role.py` | Adapt: mock evaluate → compute_quality | M6→M11 API |
| `tests/unit/experiment/test_runner_initial_state.py` | Adapt: mock evaluate → compute_quality | M6→M11 API |
| `tests/integration/experiment/test_run_one_full_contract.py` | New | Acceptance e2e |
| `tests/integration/human/**` | Selective adapt: _evaluator → compute_quality | M6→M11 API |

## Dependencies & Order Constraints

```
Step 1 (strategy/backup)
  └→ Step 2 (HITL bring-in)
       ├→ Step 3 (config.py HumanCfg)     ─┐
       ├→ Step 4 (metrics.py proxy func)   ─┤ parallel
       ├→ Step 6 (models + alembic)        ─┤ (Wave 3)
       ├→ Step 8 (grep audit)              ─┤
       └→ Step 10 (uv.lock)               ─┘
            Step 4 → Step 5 (evaluation/__init__ re-export)
            Step 6 → Step 7a (runner signatures)
            Steps 7a + 2 + 3 + 4 + 5 → Step 7b (runner wiring)
                 Step 7b → Step 9 (topology sanity, read-only)  ─┐ parallel
                 Step 7b → Step 11 (m9.2 test adapt)            ─┘ (Wave 7)
                      Steps 7b + 11 → Step 12 (acceptance test)
                           Step 12 → Step 13 (final merge)
                                Step 13 → Step 14 (codebase-map)  ─┐ parallel
                                Step 13 → Step 15 (audit + push)   ─┘ (Wave 10)
```

**Note**: Step 8 (grep audit) и Step 10 (uv.lock) — can-parallel с 3/4/6 на Wave 3.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Wave-tag пропущен перед wave N — откат невозможен one-liner'ом | Medium | Medium | Документировать как mandatory pre-wave action в `strategy.md`; orchestrator checklist |
| `_evaluator.py` «воскресает» в Step 13 и его транзитивные импорты ломают topology tests | Medium | High | Post-merge audit (grep `_evaluator` в `src/atm/topology/`) ДО commit; `git rm` если воскрес |
| 2 Alembic heads после final merge | Medium | Medium | Mandatory pre-check в Step 6: взять m8-миграцию verbatim или создать merge-migration |
| mypy strict: `cfg.human.role_router_model` → `None` → `build_llm` падает | Medium | Low | `assert cfg.human.role_router_model is not None` под условием `role_router == "llm"` |
| m9.2-тесты зависят от HITL-топологии (не принесена до Step 13) — fail при run | Medium | Medium | Тесты, требующие реальной HITL-топологии, маркировать `@pytest.mark.skip(reason="re-enable after Step 13")` |
| FakeLLM не эмитирует `last_model_version` → `model_version_snapshot={}` | Low | Low | Acceptance test явно допускает пустой dict |
| CI на GH использует pinned-versions, локально проходит — удалённо нет | Low | Medium | `uv run --frozen` для локальной симуляции |
| Топологии с m8 имеют транзитивные зависимости, не принесённые до Step 13 | Low | Medium | Option A locked-in: без `cfg.human` kwargs не передаются → TypeError не возникает |

## Out of Scope

1. PR#9 (`feat/m8` → `main`) — не трогаем, остаётся как есть.
2. Удаление `bc5f66dd0897_initial.py` (duplicate-initial Alembic revision) — не трогаем.
3. Добавление `human_role`/`cognitive_load_proxy` ключей в `RunResult.metrics` dict — не делаем (только в `runs` колонки).
4. CLI smoke `atm run --config <yaml-with-human-section>` как acceptance criterion — достаточно direct call в integration-тесте.
5. Bulk topology checkout (Option B, rejected) — топологии приедут через final merge в Step 13.
6. Rebase стратегия — отклонена в Step 1.

## Timeline

- Total: ~10h
- Created: 2026-05-13
