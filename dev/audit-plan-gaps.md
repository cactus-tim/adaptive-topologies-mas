# PLAN.md gap audit (2026-05-12)

## Summary

Gap-аудит охватил PLAN.md (M0–M14+), arch.md (§1–18), и experiment_plan.md. Найдено 4 критических пробела, 7 важных и 5 минорных. Большинство критических связаны с тремя темами: (1) datasets M10 не покрывают task-mix из experiment_plan.md; (2) инфраструктура воспроизводимости (seed_all, git_sha, model_version_snapshot) не имеет явного milestone-owner; (3) E4 (adaptive role switching — RQ4) требует компонентов, которых нет ни в одном milestone. Остальные пробелы — отсутствие явных задач для CLI-команд `atm resume`/`atm replay`, CI-pipeline, raw_tlx_score-колонки, и несоответствие между tools_policy.yaml и списком задач M4.

---

## Critical gaps

### G1 — Task-mix M10 vs experiment_plan.md несовместимы

- **Where:** PLAN.md §8 M10 (tasks); experiment_plan.md §0
- **Issue:** M10 объявляет 4 типа задач: `humaneval`, `mmlu`, `creative`, `analysis`. Experiment plan (§0) использует другой набор: `HumanEval (programming)`, **`GSM8K (reasoning)`**, **`CommonGen (creative)`**, **`InfiAgent-DABench (decision)`**. MMLU-Pro из M10 не фигурирует в experiment_plan вообще. `tasks/analysis.py` (PLAN.md) — собственный датасет, а experiment_plan говорит про DABench. `tasks/mmlu.py` (PLAN.md) — MMLU-Pro, но в experiment_plan задача типа "reasoning" решается через GSM8K (exact match). CommonGen и GSM8K вообще не упоминаются в PLAN.md.
- **Risk:** После M10 окажется, что написанные `tasks/*.py` не соответствуют тому, что E1–E4 будут реально гонять. Придётся переписывать перед E1, что задержит старт экспериментов и нарушит Exit criteria M10 ("evaluator на известных примерах"). Также тип `"qa"` в `TaskSpec.type` (arch.md §3.1) ≠ `"reasoning"` в experiment_plan — несоответствие типов создаст проблему для oracle_table и heatmap-анализа.
- **Suggested fix:** Синхронизировать M10 с experiment_plan.md §0. Добавить задачи: `tasks/gsm8k.py` (GSM8K, exact match), пересмотреть `tasks/creative.py` как CommonGen-based (ROUGE + concept coverage), `tasks/analysis.py` переориентировать на DABench. `TaskSpec.type` расширить на `"reasoning"` и `"decision"` или зафиксировать маппинг типов в одном месте. MMLU либо оставить как дополнительный датасет, либо убрать из M10.

---

### G2 — Reproducibility bundle: нет milestone-owner для seed_all, git_sha, model_version_snapshot

- **Where:** PLAN.md §8 M6, M12; arch.md §14.4
- **Issue:** arch.md §14.4 декларирует reproducibility bundle: `seed_all(seed)`, `git_sha` запись при старте, `model_version_snapshot` из первого LLMWrapper-вызова, `sandbox_image_digest` при старте DockerSandbox. Ни один milestone не содержит задачи для реализации этих механизмов. M6 создаёт `runner.py` (минимальная версия), M12 делает полный `experiment/runner.py` — но ни там, ни там нет чеклист-пункта про `git rev-parse HEAD`, `seed_all`, или запись `model_version_snapshot`/`sandbox_image_digest` при инициализации run-а. `structlog` с `filter_secrets` (arch.md §14.5) тоже нигде не milestone-ится.
- **Risk:** Runner будет написан без этих полей, `runs` будет содержать пустые `model_version_snapshot`/`sandbox_image_digest`. Это блокирует `atm replay` (arch.md §14.4, п. 1 — deterministic replay требует `model_version_snapshot` для pin-а модели). Для диплома "воспроизводимость" — это NFR (arch.md §1.3), и без него аргументация в тексте диплома про воспроизводимость экспериментов будет голословной.
- **Suggested fix:** Добавить в M12 (или M6 для минимальной версии) явные задачи: (a) `seed_all(seed)` вызов в runner при старте run-а; (b) `git rev-parse HEAD` → `experiments.git_sha`; (c) `DockerSandbox` возвращает/записывает `sandbox_image_digest` при инициализации; (d) `LLMWrapper` первый вызов → `runs.model_version_snapshot`; (e) `structlog` init с `filter_secrets` процессором.

---

### G3 — E4 (RQ4: adaptive role switching) не имеет milestone

- **Where:** PLAN.md §8; experiment_plan.md §5 (E4)
- **Issue:** experiment_plan.md §5 описывает E4 — "Adaptive topology + adaptive role" с 3 стратегиями смены роли: `role_fixed_best`, `role_by_phase_rule`, `role_by_phase_llm`. Это требует компонента, который по фазе меняет активную `HumanRole` — назовём его `AdaptiveRoleRouter`. Ни в одном milestone PLAN.md такого компонента нет. M9/M9.1 дают HITL-инфраструктуру с фиксированной ролью. `HumanCfg` (arch.md §12.1) содержит только `roles: list[str]` — список фиксированных ролей, без динамики. E4 метрика `human_sim_cognitive_load_proxy` (кол-во interrupt'ов, длина контекста, latency ответа) тоже нигде не milestone-ится.
- **Risk:** E4 невозможно провести без нового компонента. Если обнаружится это перед E4, придётся либо дописывать роль-роутер экстренно (без тестов, без дизайна), либо упрощать E4 до "только role_fixed_best" — что делает RQ4 неотвечаемым. Это центральный вопрос диплома.
- **Suggested fix:** Добавить milestone M9.2 (или расширить M9.1) с задачами: (a) `HumanRoleRouter` Protocol: `decide(phase, state) → HumanRole`; (b) `FixedRoleRouter` (совместимость с M9/M9.1); (c) `RuleBasedRoleRouter` (таблица фаза→роль из E2); (d) `LLMRoleRouter` (LLM выбирает роль); (e) `HumanCfg` расширить полем `role_router: Literal["fixed", "rule", "llm"] = "fixed"`; (f) `cognitive_load_proxy` функция в `evaluation/metrics.py`.

---

### G4 — Exit criteria M8 не проверяют интеграцию с M9.1 (Adaptive + HITL)

- **Where:** PLAN.md §8 M8, M9.1
- **Issue:** M8 Exit criteria проверяют adaptive meta-graph корректность (transitions, monotonicity, dedup) — но только без HITL. M9.1 добавляет HITL в Adaptive: human как side-input для TopologyRouter/TransitionGate. Возникает специфический риск: `interrupt()` внутри subgraph Adaptive мета-графа при LangGraph checkpointer + subgraph + interrupt — флаг "subgraph-HITL-test важно" вынесен в риски M9.1, но отсутствует в Exit criteria M9.1. Exit M9.1 говорит "все 5 топологий имеют HITL-вариант с тестами" — но тест Adaptive+HITL особый: interrupt прилетает изнутри subgraph-а мета-графа, что принципиально отличается от flat-топологий. `human_can_override_router=true` флаг создаёт state-race: если TopologyRouter уже решил switch, а human_override пришёл — кто выигрывает? Это не специфицировано.
- **Risk:** Баг "двойная запись в human_interactions при resume из Adaptive subgraph" обнаружится в E2/E3. Неспецифицированный приоритет human_override vs guard vs router может дать непредсказуемое поведение при E4, делая `topology_transitions` неинтерпретируемыми для RQ2.
- **Suggested fix:** Добавить в Exit M9.1: "Adaptive + HITL тест: interrupt внутри active subgraph (не в parent-графе) — ровно одна запись в `human_interactions`; resume корректен." Добавить в задачи M9.1 (Adaptive + HITL): явное дизайн-решение о приоритете `human_override vs guard vs router` (рекомендуется: human_override → guard → router, с записью в `TopologyTransition.decided_by='human_override'`).

---

## Important gaps

### G5 — CLI команды `atm resume` и `atm replay` не milestone-ятся

- **Where:** PLAN.md §8 M12; arch.md §12.5, §14.4
- **Issue:** arch.md §12.5 декларирует 6 CLI-команд: `atm run`, `atm grid`, `atm estimate`, `atm status`, `atm resume`, `atm analyse`. Также arch.md §14.4 добавляет `atm replay`. M12 содержит только: `atm run`, `atm grid`, `atm estimate`, `atm status`. `atm resume` (продолжить с checkpoint'а) и `atm replay` (deterministic replay через FakeLLM) отсутствуют в чеклисте M12. Reconcile-логика (arch.md §14.2: "если runs.status='running' и процесс не жив → сбросить в 'failed' или allow resume через --force-resume") тоже не milestone-ится.
- **Risk:** `atm resume` критична для HITL-экспериментов E2/E5 (реальный человек ответил, но процесс упал). `atm replay` критична для детерминированного воспроизводства. Без reconcile — после краша runner-а в grid часть runs навсегда остаётся в `status='running'`.
- **Suggested fix:** Добавить в M12 задачи: (a) `atm resume --run-id <uuid> [--force-resume]` с reconcile-логикой; (b) `atm replay <run_id> [--mode deterministic|semantic]`; (c) reconcile в runner: при `atm grid` после crash проверять `runs WHERE status='running' AND exp_id=...`.

---

### G6 — `raw_tlx_score` колонка в `human_interactions` не milestone-ится в M3 или M11

- **Where:** PLAN.md §8 M3, M11; arch.md §13.3
- **Issue:** arch.md §13.3 фиксирует: "Агрегированный `raw_tlx_score` — в той же строке отдельным колонкой для быстрого фильтра." Но в описании таблицы `human_interactions` (PLAN.md §5.1) есть только `tlx_scores (jsonb)` — нет колонки `raw_tlx_score DOUBLE PRECISION`. M3 создаёт миграцию по §5.1 (без этой колонки). M11 реализует `evaluation/tlx.py` с агрегацией, но не говорит "добавить колонку в PG" и не добавляет Alembic-миграцию.
- **Risk:** M13 строит "human cognitive load boxplots" — он будет читать из `human_interactions.raw_tlx_score`, которой нет → нужна либо in-pandas агрегация (медленно для больших датасетов, неудобно), либо hotfix миграция перед M13. E5 анализ через PG-фильтр ("выбери sessions с raw_tlx > 70") не работает без колонки.
- **Suggested fix:** Добавить `raw_tlx_score DOUBLE PRECISION` в `human_interactions` либо в M3 (вместе с остальными таблицами), либо отдельной Alembic-миграцией в M11. Добавить UPDATE в aggregator (M11): после вычисления `NasaTLX.raw_score` — `UPDATE human_interactions SET raw_tlx_score=...`.

---

### G7 — `conf/tools_policy.yaml` не упомянут в задачах M4

- **Where:** PLAN.md §8 M4; arch.md §5.3
- **Issue:** arch.md §5.3 декларирует `conf/tools_policy.yaml` — центральный конфиг, определяющий `global` и `per_role` tools для `ToolRegistry`. PLAN.md §4 (структура репозитория) не включает этот файл. M4 не содержит задачи "создать `conf/tools_policy.yaml`". `ToolRegistry.tools_for(role)` (arch.md §5.1) читает этот конфиг при инициализации — без файла Registry не знает ни о каком из local tools.
- **Risk:** Без явной задачи в M4 разработчик либо создаст файл неправильно, либо захардкодит tool-assignment в `ToolRegistry.__init__`. Оба варианта нарушают extensibility §15.2 (добавить новую роль = добавить в policy файл).
- **Suggested fix:** Добавить в M4 задачу "создать `conf/tools_policy.yaml` по спеке arch.md §5.3 и убедиться, что `ToolRegistry` читает его, а не хардкодит". Добавить файл в структуру §4 PLAN.md.

---

### G8 — `EvaluatorRegistry` не milestone-ится

- **Where:** PLAN.md §8 M10, M11; arch.md §15.4
- **Issue:** arch.md §15.4 ("Новая задача") описывает `EvaluatorRegistry` как отдельный объект: "зарегистрировать evaluator в `EvaluatorRegistry`". `TaskSpec.evaluator_key` (arch.md §3.1) матчится с ключом в этом реестре. `Evaluator Protocol` (arch.md §13.1) имеет поле `key`. Но ни M10, ни M11 не содержат задачи "создать `EvaluatorRegistry`". M11 добавляет `judges.py`, `ground_truth.py`, `metrics.py`, `tlx.py` — но не реестр. M10 добавляет `tasks/registry.py` (для TaskRegistry), но не EvaluatorRegistry.
- **Risk:** Без реестра `TaskSpec.evaluator_key` не матчится ни с чем → evaluator выбирается либо хардкодом в runner, либо через условные конструкции if/elif. Extensibility §15.4 нарушена: добавить новую задачу без реестра невозможно паттерн-совместимо.
- **Suggested fix:** Добавить в M11 задачу "создать `evaluation/registry.py`: `EvaluatorRegistry` аналогично `TaskRegistry`; зарегистрировать `HumanEvalRunner`, `MMLUMatcher`, `CreativeRubricJudge`, `AnalysisEvaluator`". Обновить M10: каждый task-файл регистрирует свой evaluator в `EvaluatorRegistry`.

---

### G9 — Integation с M8 (Adaptive) для analysis/oracle.py зависит от E1, но M8.7 помечен как условный

- **Where:** PLAN.md §8 M8.7, M13; arch.md §8bis.2
- **Issue:** M8.7 помечен как "условно после E1 pilot" и имеет статус `[ ]` (не реализован). Но `OracleTopologyRouter` (M8.3) уже реализован и читает `oracle_table.json` — значит таблица должна существовать к моменту E3. M13 (analysis tooling) не содержит задачи на "фаза-transition timeline для Adaptive" анализ через `topology_transitions`-специфику, хотя именно это нужно для RQ2-анализа E3 ("где switch'и реально помогают"). `analysis/oracle.py` — в структуре репозитория, но не в задачах M13.
- **Risk:** Перед E3 понадобится M8.7 — но к тому времени M12 уже закончен. Если M8.7 не сделан, E3 с oracle-режимом невозможен. M13 без `topology_transitions`-специфичных лоадеров не покрывает главный RQ2-аналитический вопрос.
- **Suggested fix:** Убрать условность из M8.7 — фиксировать его как обязательный (выполнить до M12, то есть до старта грид-экспериментов). Добавить в M13 задачу: `analysis/oracle.py` лоадер и plot "topology_switch timeline × quality correlation" для RQ2.

---

### G10 — Нет milestone для CI/CD pipeline

- **Where:** PLAN.md §8 M0; весь план
- **Issue:** M0 создаёт "pytest скелет + 1 smoke-тест". На этом история CI заканчивается. Нигде в плане нет задачи для настройки GitHub Actions (или аналога) CI. Между тем к M7 накоплено несколько integration-тестов с `@pytest.mark.requires_postgres`, которые нельзя гонять в lightweight CI без Docker. Маркер `@pytest.mark.live` (arch.md §4.4) тоже требует решения "как мы гоняем live-тесты vs unit vs integration".
- **Risk:** При разработке после M6-M7 не будет автоматической проверки регрессий. Тесты с `requires_postgres` могут молча не гоняться в CI. К M11 накопится ~20-30 тестов разных уровней без стратегии запуска.
- **Suggested fix:** Добавить в M0 (или выделить M0.1) задачи: (a) `.github/workflows/ci.yml` с тремя jobs: `lint` (ruff+mypy), `unit` (pytest -m "not requires_postgres and not live"), `integration` (pytest -m requires_postgres, с postgres-service); (b) зафиксировать в dev/docs, как гонять каждый тип тестов локально; (c) `@pytest.mark.live` тесты — только через явный `--run-live` флаг.

---

### G11 — `atm analyse` CLI и `analysis/loaders.py::load_runs()` не покрывают topology_transitions

- **Where:** PLAN.md §8 M13; experiment_plan.md §4 (E3 метрики)
- **Issue:** M13 описывает `analysis/loaders.py`: `load_experiment(exp_id)`, `load_llm_calls()`, `load_runs()`. E3 требует специфичных метрик из `topology_transitions`: `topology_switch_count`, `guard_override_rate`, `router_cost_share`, `time_per_topology`, `oracle_gap_loo`. Функция `load_topology_transitions(run_id)` не упоминается в M13. Plots M13 перечисляет "phase-transition timelines для Adaptive" — но не "topology-transition timelines", что принципиально разные вещи (особенно для RQ2).
- **Risk:** M13 будет реализован без `load_topology_transitions()`, и при написании аналитического ноутбука для E3 придётся добавлять этот лоадер на лету — что не задокументировано и не протестировано.
- **Suggested fix:** Добавить в M13 задачи: (a) `analysis/loaders.py::load_topology_transitions(exp_id)` → pandas; (b) plot "topology switch timeline × quality" (RQ2-специфичный); (c) Pareto для adaptive vs static с confidence bands.

---

## Minor gaps

### G12 — `conf/tools_policy.yaml` отсутствует в структуре репозитория §4

- **Where:** PLAN.md §4
- **Issue:** §4 детально описывает структуру репозитория, но `conf/tools_policy.yaml` (arch.md §5.3) не указан. Разработчик, читающий только PLAN.md §4, не знает, что файл нужен и где он лежит.
- **Risk:** Файл создаётся в неожиданном месте, или вообще не создаётся в M4.
- **Suggested fix:** Добавить `conf/tools_policy.yaml` в структуру §4.

---

### G13 — `conf/oracle/` директория не упомянута в структуре §4

- **Where:** PLAN.md §4, §8 M8.7
- **Issue:** M8.7 создаёт `conf/oracle/type_level_manual.yaml` и `data/oracle/e1_leave_one_out.json`. Ни `conf/oracle/`, ни `data/oracle/` не фигурируют в §4.
- **Risk:** Минорно — разработчик создаст директорию сам, но структура будет расходиться с документацией.
- **Suggested fix:** Добавить в §4: `conf/oracle/type_level_manual.yaml` и `data/oracle/` в data-директорию.

---

### G14 — `AgentRole.COORDINATOR` не включён в список из 5 ролей в §7 PLAN.md

- **Where:** PLAN.md §7 (Agent roles); arch.md §3.1
- **Issue:** PLAN.md §7 говорит "5 ролей — 5 Agent-ов" и перечисляет Planner/Researcher/Executor/Critic/Debater. Но arch.md §3.1 уже содержит `AgentRole.COORDINATOR` в enum. M8 (Adaptive) требует Coordinator как отдельную роль (superset 7 ролей). Нигде в PLAN.md нет упоминания конфига `conf/agents/coordinator.yaml` (системный промпт, tools для Coordinator в Hierarchical/Adaptive).
- **Risk:** В M8.6 (Superset agent roster) Coordinator инстанциируется, но его конфиг не задекларирован явно в PLAN.md. Возможен пропуск при реализации.
- **Suggested fix:** Обновить §7 PLAN.md: "6 ролей (+ Coordinator для Hierarchical/Adaptive)". Добавить `conf/agents/coordinator.yaml` в структуру §4 и в задачи M7 (Hierarchical) или M8 (Adaptive).

---

### G15 — `Monitor` роль HITL не имеет ни одного topology-owner кроме "любая"

- **Where:** PLAN.md §8 M9.1; arch.md §9.4
- **Issue:** arch.md §9.4 описывает Monitor роль: "cron-like: каждые N итераций, только observation-payload; topology passes shared+iteration". Это единственная роль, не привязанная к конкретной топологии. M9.1 описывает Star/Mesh/Debate/Hierarchical/Adaptive — Monitor там не упоминается (по умолчанию входит в каждую?). Activation rule "каждые N итераций" не специфицирована конфигом — нет поля `monitor_interval` в `HumanCfg`.
- **Risk:** Monitor не реализуется нигде, так как нет явного owner. `HumanCfg` без `monitor_interval` не позволяет контролировать частоту. При E4 Monitor не будет в quality-экспериментах.
- **Suggested fix:** Добавить в M9 (или M9.1) явную задачу "Monitor роль: реализовать cron-вставку каждые N итераций в базовой топологии (Chain или Star), конфиг `monitor_interval_iters`". Добавить `monitor_interval_iters: int | None = None` в `HumanCfg`.

---

### G16 — Нет явной задачи на `pricing.yaml` обновление для новых моделей

- **Where:** PLAN.md §8 M2; arch.md §4.3
- **Issue:** M2 создаёт `conf/pricing.yaml` с ценами для GPT-4o, GPT-4o-mini, Claude. Experiment plan §0 добавляет confirmation runs на `gpt-4o` (с апгрейдом) и упоминает LiveCodeBench-mini. Между M2 и финальными runs может пройти несколько месяцев; цены меняются. Нет задачи "проверить pricing перед E1" или "обновить pricing при старте экспериментов".
- **Risk:** Dry-run estimator даёт неверную оценку из-за устаревших цен → перерасход бюджета.
- **Suggested fix:** Добавить в M12 (dry-run estimator) задачу "проверить актуальность `conf/pricing.yaml` перед запуском; добавить `last_verified_date` поле в yaml". Добавить в Risks таблицу PLAN.md §10 риск "устаревшие цены".

---

## Confirmed-clean areas

Следующие части плана проверены и покрыты достаточно хорошо:

- **M9 HITL Chain** — полностью специфицирован: Protocol, idempotency, timeout/fallback, 3 тест-сценария. Exit criteria точны.
- **M8 Adaptive meta-graph (M8.1–M8.6)** — детальная декомпозиция с явными тестами. Exit criteria верифицируемы. SwitchGuards, SignalBus, TransitionGate — все имеют unit-тесты.
- **M3 Storage** — полное покрытие: все 6 таблиц, две connection pools, flush-инварианты. Exit criteria проверяемы.
- **M2 LLMWrapper** — FakeLLM, retry, budget, pricing — детально и тестами покрыто.
- **arch.md §3.3bis dedup_by_id_reducer** — resolved (§17 deviation #14), implementation spec полная.
- **Stopping criteria precedence** — единая во всех топологиях, §7.1 arch.md. Закрыт глобально.
- **Budget three-tier** — race condition решена (§18 open question #4, resolved). PG SELECT...UPDATE semantics.
- **Subgraph + Checkpointer** — риск идентифицирован в M9, M9.1 (subgraph-HITL-test отдельно). Не закрыт полностью (см. G4), но риск зафиксирован.
- **Parallelism критического пути** — M9 после M6 (не M7), M10 после M5, M9.1 после M9 — все зависимости логически верны.
- **M13 анализ (базовая часть)** — loaders + plots адекватны для RQ1 (topology × task_type heatmap, Pareto quality×cost).
