# m8-adaptive — TransitionGate + Adaptive meta-graph + integration test

## Goal
Финальный сборочный блок волны M8: собрать meta-граф адаптивной топологии L2, который на каждом тике вызывает PhaseRouter и TopologyRouter, выполняет TransitionGate (state-transfer), и делегирует исполнение в один из 5 subgraph-узлов (статические топологии M6/M7). Проверить всё end-to-end на FakeLLM integration-тесте.

После этого блока выполняются exit-criteria M8 из PLAN (строки 517–521).

## Scope
- `src/atm/topology/adaptive.py`:
  - meta-граф (LangGraph) со следующими узлами:
    - `phase_router_node` — вызывает `PhaseRouter.decide(state) -> PhaseDecision`; если `next_phase != current_phase` → dispatch `phase_transition` event, обновить `state.phase`, `state.phase_started_at_iter`
    - `topology_router_node` — вызывает `TopologyRouter.decide(state) -> TopologyDecision`; **dispatch `topology_transition` event каждый тик** (включая no-change); если topology поменялась — обновить counters и `topology_history`, записать `TopologyTransition` в PG
    - `transition_gate_node` — чистая функция, реализует state-transfer таблицу из `arch.md §7.7` / §8bis (что переносить между топологиями: messages, draft, plan, и т.д.)
    - 5 subgraph-узлов: linear / supervisor / mesh / debate / hierarchical — оборачивают компилированные графы из M6/M7 через helper `node_for_topology(name) -> RunnableNode`
  - Superset agent roster: при `topology.name == 'adaptive'` инстанциируются все 7 ролей (Planner, Executor, Critic, Verifier, Researcher, Debater, Coordinator — точный список сверить с M6/M7 inventory) — даже если в текущем тике активна топология, использующая только подмножество
- `conf/experiments/adaptive_smoke.yaml`:
  - 1 задача
  - topology.name == 'adaptive'
  - PhaseRouter: Rule
  - TopologyRouter: Rule (для smoke-теста детерминированно)
  - SwitchGuards с разумными дефолтами (min_dwell=1, max_per_run=10, …)
  - FakeLLM либо реальный (на усмотрение — для CI должен быть FakeLLM)
- Integration-тест (FakeLLM) в `tests/integration/test_adaptive_m8.py`:
  - Задача со скриптованной последовательностью: после первой execute-итерации FakeLLM возвращает ответы, которые заставят Executor выставить `stuck=True` → TopologyRouter должен переключить на mesh
  - Дальше эмитятся reject'ы (rejected_count ≥ 3) → переключение на debate
  - Проверки:
    - последовательность `topology_transitions` (in-memory state.topology_history) соответствует ожиданиям: ['linear', …, 'mesh', …, 'debate', …] (точный список зависит от phase × signals)
    - `phase` монотонна по всему run'у (`SELECT phase FROM phases ORDER BY at` строго возрастает)
    - `messages` без дублей по `message_id` (dedup reducer из m8-foundation сработал)
    - guards срабатывают: специально провоцируем thrashing → в transitions появляется хотя бы одна запись с `decided_by='guard_override'`

## Out-of-scope
- Любые изменения PhaseManager, TopologyRouter, SwitchGuards, SignalBus, агентов, типов, reducer'ов — должны быть готовы в `m8-foundation` / `m8-routing` / `m8-signals`.
- Oracle pipeline (M8.7) — после E1 pilot.
- Real-LLM smoke (с настоящей моделью) — не часть exit-criteria M8.

## Inputs / Preconditions
- **Все три блока завершены и смержены**: `m8-foundation`, `m8-routing`, `m8-signals`.
- Импорты доступны:
  - `TopologyTransition`, `TopologyDecision`, `PhaseDecision` из `atm.core.types`
  - `SharedState` с полями `signals`, `iter_total`, `topology_history`, etc.
  - `dedup_by_id_reducer` для merge messages/transitions
  - `RuleBasedPhaseRouter`, `LLMPhaseRouter`
  - `RuleBasedTopologyRouter`, `LLMTopologyRouter`, `OracleTopologyRouter`, `GuardedRouter`, `SwitchGuards`
  - `emit_signal` + сигнальные ключи; агенты Critic/Executor/Planner эмитят сигналы корректно
- Компилированные субграфы из M6 (linear, supervisor) и M7 (mesh, debate, hierarchical) доступны через какой-то фабричный API.

## Outputs / Contract
- Точка входа: `atm run --config conf/experiments/adaptive_smoke.yaml` выполняется и:
  - делает ≥1 реальный topology switch внутри execution-фазы
  - все transitions попадают в PG (`topology_transitions`) и parquet (артефакты run'а)
  - monotonicity invariant держится
  - `messages` уникальны по `message_id`
- `from atm.topology.adaptive import build_adaptive_graph` — фабрика, возвращающая скомпилированный meta-граф.

## References
- `arch/PLAN.md` строки 500–521 (M8.6 чек-лист + exit-criteria M8)
- `arch/arch.md` §7.7 — phase × signals → topology
- `arch/arch.md` §8 — Adaptive L2
- `arch/arch.md` §8bis — TransitionGate state-transfer таблица
- `dev/dec/m8-foundation.md`
- `dev/dec/m8-routing.md`
- `dev/dec/m8-signals.md`
- Существующие конфиги в `conf/experiments/*.yaml` — взять за образец стиля

## Depends On
- `m8-foundation`
- `m8-routing`
- `m8-signals`

(все три блока ОБЯЗАТЕЛЬНЫ — wave 3.)

## Suggested run-task class
**complex** — сборка LangGraph meta-графа с условной маршрутизацией, 5 subgraph-узлов, dispatch событий двух типов, persistence в PG, integration-тест на FakeLLM со скриптованным сценарием, новый конфиг.

## Notes
- Integration-тест на FakeLLM **обязателен** (см. M8.6 exit-criteria в PLAN строки 506–510 и 517–521): топология должна сделать ≥1 реальный switch внутри execution-фазы, transitions должны лечь в PG и в parquet, messages без дублей.
- `topology_transition` event эмитится **каждый тик**, включая no-change — это важно для аналитики (M11 будет считать distribution и dwell-time).
- `phase_transition` event — только при advance.
- TransitionGate — чистая функция (state → state'), не имеет side-effects; легко юнит-тестить отдельно от meta-графа (рекомендую вынести 2–3 чистых юнит-теста на саму функцию TransitionGate в добавление к integration-тесту).
- Для теста guard_override (`SELECT COUNT(*) FROM topology_transitions WHERE decided_by='guard_override' > 0`) — отдельный integration-сценарий с thrashing-провокацией (FakeLLM возвращает чередующиеся сигналы быстрее min_dwell).
- Superset agent roster: если `node_for_topology(name)` возвращает граф, который ожидает определённых агентов в state — нужно убедиться, что все 7 ролей инстанциированы заранее.
