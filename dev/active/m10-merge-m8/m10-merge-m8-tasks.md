# m10-merge-m8 — Tasks

<!-- Wave-tagging invariant: before starting ANY wave N>=2, run `git tag -f wave-N-pre HEAD`. -->

## Wave 1: Foundation (Step 1) COMPLETE

- [x] 1.1 Создать backup branch + установить wave-1-pre tag + зафиксировать стратегию в strategy.md
  - Type: simple
  - Depends On: none
  - Can-Parallel-With: none (всё остальное зависит от backup-точки)
  - Files: `dev/active/m10-merge-m8/strategy.md` (new), backup branch `backup/feat-m10-pre-m8-merge`, tag `wave-1-pre`
  - Acceptance: `git branch | grep backup/feat-m10-pre-m8-merge` — строка; `git tag -l wave-1-pre` → `wave-1-pre`; `cat dev/active/m10-merge-m8/strategy.md` содержит обоснование + wave-tagging invariant
  - Pre-wave tag: `wave-1-pre` устанавливается ЗДЕСЬ (первый шаг)
  - Verification:
    - `git branch backup/feat-m10-pre-m8-merge feat/m10`
    - `git tag -f wave-1-pre HEAD`
    - В `strategy.md`: phased-additive chosen; rebase rejected (reasons); wave-N-pre invariant documented; final merge is Step 9 only

---

## Wave 2: HITL Bring-in (Step 2) COMPLETE

<!-- PRE-WAVE: `git tag -f wave-2-pre HEAD` -->

- [x] 2.1 Принести HITL-инфраструктуру с feat/m8 через checkout-paths (no-commit до конца шага)
  - Type: simple
  - Depends On: 1.1
  - Can-Parallel-With: none (foundation для waves 3-10)
  - Files: `src/atm/human/__init__.py`, `src/atm/human/gateway.py`, `src/atm/human/llm_simulated.py`, `src/atm/human/cli_gateway.py`, `src/atm/human/role_router.py`, `conf/human/role_table.yaml` (if exists), `conf/evaluation/cognitive_load.yaml`, `tests/unit/human/**`, `tests/integration/human/**`, `tests/unit/evaluation/test_cognitive_load_proxy.py`
  - Acceptance: `uv run python -c "from atm.human import HumanGateway, LLMSimulatedGateway, FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter"` — exit 0
  - Pre-wave tag: `git tag -f wave-2-pre HEAD`
  - Implementation notes:
    - `git diff --name-only --diff-filter=A feat/m10..origin/feat/m8 -- src/atm/human/ conf/ tests/unit/human/ tests/integration/human/` — список добавленных файлов
    - Для каждого: `git ls-tree origin/feat/m8 -- <path>` (проверить существование), затем `git checkout origin/feat/m8 -- <path>`
    - НЕ переносить `src/atm/experiment/_evaluator.py`
    - `src/atm/human/__init__.py` должен re-export: HumanGateway, HumanContext, HumanResponse, LLMSimulatedGateway, CLIGateway, FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter, HumanRoleRouter
    - НЕ запускать pytest — принесённые тесты временно красные (import _evaluator); это нормально

---

## Wave 3: Parallel Code Expansion (Steps 3-6, 8, 10) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-3-pre HEAD` -->

- [ ] 3.1 Расширить experiment/config.py — добавить HumanCfg + ExperimentConfig.human (TDD)
  - Type: tdd
  - Depends On: 2.1
  - Can-Parallel-With: 3.2, 3.3, 3.4, 3.5
  - Files: `src/atm/experiment/config.py`, `tests/unit/experiment/test_config.py`
  - Acceptance: `uv run pytest tests/unit/experiment/test_config.py tests/unit/experiment/test_config_evaluation_cfg.py -q` — зелёное
  - Pre-wave tag: `git tag -f wave-3-pre HEAD`
  - Implementation notes:
    - `git show origin/feat/m8:src/atm/experiment/config.py | grep -A 30 "class HumanCfg"` — сверить форму
    - Поля HumanCfg: enabled, role (Literal[...]), gateway, gateway_model, timeout_s=900, timeout_policy, role_router="fixed", role_table, role_router_model
    - `ExperimentConfig.human: HumanCfg | None = None` — strict optional
    - НЕ трогать EvaluationCfg и _check_topology validator
    - TDD: сначала написать test_experiment_config_with_human_section (fail), потом добавить класс

- [ ] 3.2 Добавить human_sim_cognitive_load_proxy + weights loader в evaluation/metrics.py (TDD)
  - Type: tdd
  - Depends On: 2.1
  - Can-Parallel-With: 3.1, 3.3, 3.4, 3.5
  - Files: `src/atm/evaluation/metrics.py`, `tests/unit/evaluation/test_cognitive_load_proxy.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_cognitive_load_proxy.py tests/unit/evaluation/test_metrics.py -q` — зелёное; `uv run mypy src/atm/evaluation/metrics.py --strict` — clean
  - Implementation notes:
    - `git show origin/feat/m8:src/atm/evaluation/metrics.py` — сверить сигнатуру (ожидается async)
    - Импорты: Path, UUID, yaml, sqlalchemy as sa, AsyncSession, HumanInteraction
    - Формула: `α·count(interrupts) + β·mean(context_len) + γ·mean(latency_s)` (SELECT из human_interactions WHERE run_id=...)
    - `_load_weights()` — читает conf/evaluation/cognitive_load.yaml; fallback на DEFAULT_WEIGHTS
    - TDD: (а) формула на mock-данных, (б) fallback на DEFAULT_WEIGHTS, (в) zero-interactions → 0.0
    - НЕ трогать существующие функции (aggregate_quality, humaneval_pass_at_k, ...)

- [ ] 3.3 Добавить Run.cognitive_load_proxy колонку + Alembic-миграция (с mandatory pre-check на feat/m8, TDD)
  - Type: tdd
  - Depends On: 2.1
  - Can-Parallel-With: 3.1, 3.2, 3.4, 3.5
  - Files: `src/atm/storage/models.py`, `alembic/versions/0002_add_cognitive_load_proxy.py` (new or verbatim m8)
  - Acceptance: `uv run alembic upgrade head` проходит; `uv run alembic heads | wc -l` → `1`; `uv run alembic downgrade -1 && uv run alembic upgrade head` — round-trip clean; `uv run pytest tests/unit/storage/ -q` — зелёное
  - Implementation notes:
    - MANDATORY pre-check: `git ls-tree -r origin/feat/m8 -- alembic/versions/` → для каждой m8-only миграции `git show origin/feat/m8:alembic/versions/<file>` → проверить adds cognitive_load_proxy
    - Если m8 уже добавляет: `git checkout origin/feat/m8 -- alembic/versions/<file>` (verbatim)
    - Если нет: `uv run alembic revision -m "add cognitive_load_proxy to runs"` с `down_revision = "0001"` и distinct revision_id
    - Колонку добавить после quality_score: `cognitive_load_proxy: Mapped[float | None] = mapped_column(sa.Double(), nullable=True)`
    - Зафиксировать результат pre-check + финальный head id в strategy.md
    - TDD: `assert hasattr(Run, 'cognitive_load_proxy') and Run.__table__.columns['cognitive_load_proxy'].nullable is True`

- [ ] 3.4 Подтвердить отсутствие _evaluator.py + audit всех импортов (read-only grep)
  - Type: simple
  - Depends On: 2.1
  - Can-Parallel-With: 3.1, 3.2, 3.3, 3.5
  - Files: `src/atm/experiment/_evaluator.py` (удалить если воскрес)
  - Acceptance: `grep -r "from atm.experiment._evaluator" src tests/unit/experiment tests/integration` — пусто или findings задокументированы для Step 7.2
  - Implementation notes:
    - `grep -r "from atm.experiment._evaluator" src tests` — должно быть пусто
    - `grep -r "from atm.experiment import _evaluator" src tests` — должно быть пусто
    - Если что-то найдено в принесённых тестах Step 2 — собрать список, передать в Step 7.2
    - Если файл существует — `git rm src/atm/experiment/_evaluator.py`

- [ ] 3.5 Перегенерировать uv.lock (если pyproject.toml менялся)
  - Type: simple
  - Depends On: 2.1
  - Can-Parallel-With: 3.1, 3.2, 3.3, 3.4
  - Files: `uv.lock`, `pyproject.toml` (проверка diff)
  - Acceptance: `uv sync --frozen` exit 0; `uv run python -c "import atm; print('ok')"` exit 0
  - Implementation notes:
    - `git show origin/feat/m8:pyproject.toml > /tmp/m8_pyproject.toml && diff pyproject.toml /tmp/m8_pyproject.toml`
    - Если есть новые deps (например, httpx для CLI gateway) — смержить вручную в pyproject.toml, затем `uv lock`
    - Если pyproject не менялся — `uv lock` no-op; закоммитить отдельным "chore" коммитом

---

## Wave 4: evaluation/__init__.py Re-export (Step 5) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-4-pre HEAD` -->

- [ ] 4.1 Добавить human_sim_cognitive_load_proxy в evaluation/__init__.py __all__
  - Type: simple
  - Depends On: 3.2 (импорт должен резолвиться)
  - Can-Parallel-With: 3.3 (разные файлы)
  - Files: `src/atm/evaluation/__init__.py`
  - Acceptance: `uv run python -c "from atm.evaluation import human_sim_cognitive_load_proxy, compute_quality, persist_tlx, NasaTLX, score_ground_truth; print('ok')"` — exit 0; `uv run ruff check src/atm/evaluation/` — clean
  - Pre-wave tag: `git tag -f wave-4-pre HEAD`
  - Implementation notes:
    - Добавить в from-import из .metrics: human_sim_cognitive_load_proxy (в алфавитном порядке между humaneval_pass_at_k и persist_quality)
    - Добавить в __all__ аналогично
    - Проверить циркулярных импортов: metrics.py → storage.models → не должен импортить evaluation

---

## Wave 5: Runner Signatures (Step 7a) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-5-pre HEAD` -->

- [ ] 5.1 Расширить сигнатуры _update_run_success / _update_run_failed — добавить human_role + cognitive_load_proxy kwargs (TDD)
  - Type: tdd
  - Depends On: 3.3 (колонка должна существовать в модели)
  - Can-Parallel-With: none (центральный файл; Step 6.1 строго после)
  - Files: `src/atm/experiment/runner.py`, `tests/unit/experiment/test_run_update_helpers.py` (new)
  - Acceptance: `uv run pytest tests/unit/experiment/test_run_update_helpers.py -q` — зелёное; `uv run pytest tests/unit/experiment/test_runner.py tests/unit/experiment/test_runner_evaluation_wiring.py -q` — без регрессии; `uv run mypy src/atm/experiment/runner.py --strict` — clean
  - Pre-wave tag: `git tag -f wave-5-pre HEAD`
  - Implementation notes:
    - TDD first: написать падающий тест в test_run_update_helpers.py (AsyncMock(spec=AsyncSession), capture stmt через session.execute.call_args, assert "human_role" and "cognitive_load_proxy" in str(compiled))
    - Только после красного теста: расширить сигнатуры `(session, run_id, ..., human_role: str | None = None, cognitive_load_proxy: float | None = None)`
    - Расширить .values(...) clause в обоих helper'ах — включить новые колонки (передавать None если не указаны — safer)
    - Существующие call-sites продолжают работать благодаря defaults = None
    - НА ЭТОМ ШАГЕ: никакой логики SELECT из human_interactions, никакого вызова human_sim_cognitive_load_proxy

---

## Wave 6: Runner HITL Wiring (Step 7b) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-6-pre HEAD` -->

- [ ] 6.1 Добавить _build_role_router, _build_human_gateway_llm, wire в run_one, timeout_policy branches
  - Type: tdd
  - Depends On: 5.1 (runner signatures), 2.1 (human package), 3.1 (HumanCfg), 3.2+4.1 (human_sim_cognitive_load_proxy), 3.3 (cognitive_load_proxy column)
  - Can-Parallel-With: none (центральный файл)
  - Files: `src/atm/experiment/runner.py`
  - Acceptance: `uv run pytest tests/unit/experiment/test_runner.py tests/unit/experiment/test_runner_evaluation_wiring.py -q` — зелёное (regression); `uv run mypy src/atm/experiment/runner.py --strict` — clean
  - Pre-wave tag: `git tag -f wave-6-pre HEAD`
  - Implementation notes:
    - Импорты: `from atm.human import HumanGateway, LLMSimulatedGateway, CLIGateway, FixedRoleRouter, RuleBasedRoleRouter, LLMRoleRouter, HumanRoleRouter`; `from atm.evaluation.metrics import human_sim_cognitive_load_proxy`
    - `_build_role_router(human_cfg, llms)`: switch по role_router: fixed/rule/llm; return None если human_cfg is None
    - `_build_human_gateway_llm(human_cfg, llms)`: switch по gateway: llm/cli; return None если human_cfg is None
    - В run_one: построить human_gateway + role_router ЕСЛИ cfg.human and cfg.human.enabled
    - Option A (locked-in): `human_kwargs = {"human_cfg": cfg.human, "human_gateway_llm": human_gateway, "role_router": role_router} if (cfg.human and cfg.human.enabled) else {}`; `topology.build(..., **human_kwargs)`
    - После compute_quality, ДО _update_run_success: SELECT role FROM human_interactions WHERE run_id=:rid ORDER BY requested_at DESC LIMIT 1 → dynamic_human_role; fallback cfg.human.role или None
    - `cognitive_load_proxy = await human_sim_cognitive_load_proxy(session, run_id)` (try/except, fallback None)
    - Передать оба значения в _update_run_success и в BudgetExceeded/Exception ветки
    - timeout_policy branches (canonical from m8): fail → raise → FinishReason.HUMAN_TIMEOUT; llm_fallback → LLMSimulatedGateway fallback; skip → bypass human-node
    - Reference: `git show origin/feat/m8:src/atm/experiment/runner.py | grep -A 20 timeout_policy`
    - НЕ удалять seed_all, НЕ удалять finally-блок
    - mypy: assert cfg.human.role_router_model is not None под условием role_router == "llm"

---

## Wave 7: Sanity + Test Adaptation (Steps 9, 11) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-7-pre HEAD` -->

- [ ] 7.1 Sanity check топологий после Option A wiring (read-only)
  - Type: simple
  - Depends On: 6.1
  - Can-Parallel-With: 7.2
  - Files: (read-only) `tests/unit/topology/*`, `tests/integration/topology/*`
  - Acceptance: `uv run pytest tests/unit/topology -q && uv run pytest tests/integration/topology -q` — зелёное
  - Pre-wave tag: `git tag -f wave-7-pre HEAD`
  - Implementation notes:
    - НЕ модифицировать топологии
    - Если что-то красное — регрессия от Step 6.1; вернуться в Step 6.1

- [ ] 7.2 Адаптировать m9.2-тесты runner'а + audit integration/human-тестов под compute_quality (TDD)
  - Type: tdd
  - Depends On: 6.1
  - Can-Parallel-With: 7.1
  - Files: `tests/unit/experiment/test_runner_role_router.py`, `tests/unit/experiment/test_runner_human_role.py`, `tests/unit/experiment/test_runner_initial_state.py`, `tests/integration/human/**` (selective)
  - Acceptance: `uv run pytest tests/unit/experiment/ -q` — зелёное; `uv run pytest tests/integration/human/ -m "not requires_postgres" -q` — зелёное; `uv run pytest tests/unit -q` — общий регресс зелёный
  - Implementation notes:
    - Заменить `mocker.patch("atm.experiment.runner.evaluate", return_value=1.0)` → `mocker.patch("atm.experiment.runner.compute_quality", new=AsyncMock(return_value=(1.0, {})))`
    - integration/human audit: `grep -rn "atm.experiment._evaluator\|atm.experiment.runner.evaluate\|evaluate(cfg.task" tests/integration/human/` → для matched файлов переписать
    - test_runner_human_role.py: insert HumanInteraction(role="Reviewer", run_id=...) через fixture-session, assert Run.human_role == "Reviewer"
    - Тесты с cfg.human.enabled=true + реальной HITL-топологией: `@pytest.mark.skip(reason="re-enable after Step 9 final merge")`
    - Все три unit-файла должны иметь @pytest.mark.asyncio

---

## Wave 8: Acceptance Test Scaffolding (Step 12) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-8-pre HEAD` -->

- [ ] 8.1 Написать acceptance integration test test_run_one_full_contract.py
  - Type: tdd
  - Depends On: 6.1 (runner wiring), 7.2 (test suite green)
  - Can-Parallel-With: none
  - Files: `tests/integration/experiment/test_run_one_full_contract.py` (new), `tests/fixtures/experiment/` (опц.)
  - Acceptance: файл существует; после Step 9 `uv run pytest tests/integration/experiment/test_run_one_full_contract.py -m requires_postgres -q` — зелёное
  - Pre-wave tag: `git tag -f wave-8-pre HEAD`
  - Implementation notes:
    - Маркер: @pytest.mark.requires_postgres
    - Конфиг: cfg.human.enabled=True, role=Reviewer, gateway=llm, role_router=fixed, topology=chain; FakeLLM фикстуры
    - SELECT из runs: quality_score is not None; human_role == "Reviewer"; cognitive_load_proxy — value-range; model_version_snapshot non-empty or {}; sandbox_image_digest is None (acceptable)
    - SELECT из human_interactions WHERE run_id=...: count >= 1
    - M5 value-range assertion: фиксированная фикстура K=2 interrupts, mean(context_len)=128, mean(latency_s)=1.5; комментарий с вычислением: `expected = alpha*2 + beta*128 + gamma*1.5`; assert pytest.approx(expected, rel=0.05) ИЛИ bounds [lower, upper]
    - Тест ИЗНАЧАЛЬНО КРАСНЫЙ (Chain-топология без HITL-узлов); зеленеет после Step 9
    - cleanup: await engine.dispose() в fixture

---

## Wave 9: Final Merge (Step 13) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-9-pre HEAD` -->

- [ ] 9.1 Финальный merge origin/feat/m8 в feat/m10 — file-by-file ресолюшен 6 файлов + post-merge audit
  - Type: simple
  - Depends On: 7.2 (test suite ready), 8.1 (acceptance test exists)
  - Can-Parallel-With: none
  - Files: `src/atm/topology/*.py` (m8 version), `alembic/versions/*` (ресолюшен), все auto-resolved m8 additions
  - Acceptance: `uv run pytest tests/unit -q` — зелёное; `uv run pytest tests/integration -m requires_postgres -q` — зелёное (Step 8 test зелёный); `uv run alembic heads | wc -l` → `1`; `! grep -rn "_evaluator" src/` — exit code != 0; `git log --merges -1` — merge-commit
  - Pre-wave tag: `git tag -f wave-9-pre HEAD`
  - Implementation notes:
    - `git fetch origin feat/m8` (если не up-to-date)
    - `git merge origin/feat/m8 --no-commit --no-ff`
    - Для 6 protected files: `git checkout HEAD -- <path>` (взять нашу m10-версию):
      - `src/atm/experiment/runner.py`
      - `src/atm/experiment/config.py`
      - `src/atm/evaluation/__init__.py`
      - `src/atm/evaluation/metrics.py`
      - `dev/codebase-map.md`
      - `uv.lock` (или `uv lock` повторно)
    - Для новых m8-файлов (топологии с HITL-узлами и др.) — accept default merge (auto-resolve "added by them")
    - _evaluator resurrection check: `ls src/atm/experiment/_evaluator.py` → если существует → `git rm src/atm/experiment/_evaluator.py`
    - Post-merge transitive audit (M1): `grep -rn "_evaluator" src/atm/topology/` → исправить любые найденные импорты ДО commit
    - `grep -rn "_evaluator" src/` — должно быть пусто
    - Alembic heads check: если 2 heads → `uv run alembic merge -m "merge m10 cognitive_load_proxy with m8 head" <rev1> <rev2>`
    - Снять @pytest.mark.skip с тестов, помеченных "re-enable after Step 9"
    - `git add -A && git commit -m "merge: integrate feat/m8 (m9+m9.2) into feat/m10"`
    - После commit: `uv run pytest tests/unit -q` + `uv run pytest tests/integration -m requires_postgres -q`

---

## Wave 10: Codebase Map + Final Audit (Steps 14, 15) NOT STARTED

<!-- PRE-WAVE: `git tag -f wave-10-pre HEAD` -->

- [ ] 10.1 Перезаписать dev/codebase-map.md под целостную M9/M9.1/M9.2/M10/M11 картину
  - Type: simple
  - Depends On: 9.1
  - Can-Parallel-With: 10.2
  - Files: `dev/codebase-map.md`
  - Acceptance: `grep -E "M9|M9.1|M9.2|cognitive_load|human/gateway|role_router" dev/codebase-map.md` — все термины присутствуют
  - Pre-wave tag: `git tag -f wave-10-pre HEAD`
  - Implementation notes:
    - Сохранить структуру оригинала (Tech Stack / Project Structure / Key Modules / Patterns / External Deps / Constraints)
    - Добавить секцию «M10-m8-merge (2026-05-13)»
    - В Project Structure: `human/` → «M9 complete + M9.1 (HITL в 5 топологиях) + M9.2 (Adaptive Role Router fixed/rule/llm)»
    - В Experiment Runner: _build_role_router, _build_human_gateway_llm, dynamic_human_role SELECT, cognitive_load_proxy write, timeout_policy branches
    - Обновить test-count актуальным числом

- [ ] 10.2 Final integrity audit + push + reopen PR#10 review
  - Type: simple
  - Depends On: 9.1
  - Can-Parallel-With: 10.1
  - Files: (no file changes; CI-validation + git push)
  - Acceptance: `gh pr view 10 --json mergeable` → `"MERGEABLE"`; CI зелёный
  - Implementation notes:
    - `grep -r "from atm.experiment._evaluator" src tests` — пусто
    - `uv run alembic upgrade head` на чистой БД → single head
    - `uv run pytest tests/unit -q` — зелёное
    - `uv run pytest tests/integration -m "not live" -q` — зелёное
    - `uv run mypy src/atm --strict` — clean
    - `uv run ruff check src/atm tests` — clean
    - `git push origin feat/m10`
    - На GitHub: ack PR#10 стал MERGEABLE
    - (опц.) `git branch -D backup/feat-m10-pre-m8-merge` — только если CI зелёный; wave-*-pre tags оставить для аудита

---

## Stats

- Total: 18 tasks across 10 waves · ~10h
- Done: 2 / 18

## Wave Summary

| Wave | Steps | Parallelism | ~Time |
|------|-------|-------------|-------|
| 1 | 1.1 | sequential | 0.5h |
| 2 | 2.1 | sequential | 1h |
| 3 | 3.1, 3.2, 3.3, 3.4, 3.5 | ALL parallel | 2h |
| 4 | 4.1 | sequential | 0.25h |
| 5 | 5.1 | sequential | 0.5h |
| 6 | 6.1 | sequential | 2.5h |
| 7 | 7.1, 7.2 | parallel | 1.5h |
| 8 | 8.1 | sequential | 1h |
| 9 | 9.1 | sequential | 1h |
| 10 | 10.1, 10.2 | parallel | 0.75h |

## How to Update

After each step completes:
1. Mark task with [x] in this file
2. Update phase header: if all tasks in wave done → add "COMPLETE"; if some → add "IN PROGRESS"
3. Update Stats "Done" count
4. Update `m10-merge-m8-context.md` SESSION PROGRESS section
