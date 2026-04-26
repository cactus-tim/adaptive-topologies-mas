# M7 — Mesh, Debate, Hierarchical Topologies — Plan

## Executive Summary

Реализовать milestone M7 — три новые статические топологии (Mesh, Debate, Hierarchical) поверх готового топологического фреймворка из M6 (`Topology` Protocol, `TopologyConfig`, `TopologyRegistry`, `_should_stop`). Цель — все 5 статических топологий работают end-to-end, регистрируются в `TopologyRegistry`, имеют конфиги, unit-тесты на `build()/routing/stopping` (FakeLLM-driven) и integration-тесты (e2e на одной задаче).

**Final Class**: complex — три новые публичные топологии, каждая со своей нетривиальной control-flow семантикой (broadcast-bus + voting/quorum в Mesh; параллельный fan-out двух Debater-ов + judge-loop в Debate; вложенные compiled subgraph-ы с двумя уровнями координации в Hierarchical). Hierarchical вводит технику subgraph-в-узле, которая будет фундаментом M8 Adaptive.

**Needs Integration Tests**: yes — три топологии регистрируются в `TopologyRegistry` под публичными именами `mesh`, `debate`, `hierarchical`, их `build()` потребляется `experiment/runner.py` и будущим M8.

## Current State

M6 поставил:
- `topology/base.py`: `Topology` Protocol, `TopologyConfig` (Pydantic), `TopologyRegistry`, `_should_stop`.
- `topology/star.py`, `topology/chain.py` — работающие топологии, следуют паттерну post-process нод.
- `core/state.py:67`: `SharedState.broadcast_bus: list[Message]` уже определён.
- `core/types.py`: `MessageKind` уже содержит `DRAFT`, `DECISION`; `Debater` принимает `stance: pro|contra`.
- Тесты: `tests/unit/topology/test_star.py`, `test_chain.py`, `tests/integration/topology/test_star_e2e.py`, `test_chain_e2e.py`.

## Proposed Approach

Три новых модуля (`mesh.py`, `debate.py`, `hierarchical.py`) добавляются строго по M6-паттерну:
- Каждый модуль регистрирует свой класс через `@TopologyRegistry.register("name")` сайд-эффектом импорта.
- Централизованная регистрация в `topology/__init__.py` выполняется **один раз** в Step 0 через `contextlib.suppress(ImportError): importlib.import_module(...)`.
- Steps 2/3/4 трогают только строго disjoint файлы — нет конфликтов при параллельной разработке в worktrees.
- `core/state.py` не модифицируется (broadcast_bus уже есть).
- Все stopping проходят через существующий `_should_stop` helper.
- Тесты: TDD-style — сначала unit-тесты (patch StateGraph + isolated routing-function тесты), затем integration e2e (FakeLLM scripted fixtures).

**Mesh**: broadcast-bus + dispatcher нода + round-robin/priority активация + voting/quorum.
**Debate**: Planner → параллельный fan-out к двум Debater-ам → Critic-as-judge → loop.
**Hierarchical**: Top-Coordinator (rule-based) → 2 compiled subgraph-а (по одному на sub-team) → workers.

## Implementation Phases

### Phase 1: Pre-flight + Register (~0.5h)
**Goal:** Верифицировать SharedState и зарегистрировать 3 новые топологии в `__init__.py`.

- [ ] 0.1 Verify `SharedState.broadcast_bus` exists in `core/state.py` (grep, read-only)
  - File: `src/atm/core/state.py`
  - Acceptance: `grep -n "broadcast_bus" src/atm/core/state.py` возвращает строку на ~line 67; если поле отсутствует — остановиться и эскалировать.

- [ ] 0.2 Extend `topology/__init__.py` with guarded imports for mesh, debate, hierarchical
  - File: `src/atm/topology/__init__.py`
  - Acceptance: файл содержит `contextlib.suppress(ImportError)` для mesh, debate, hierarchical + обновлённый комментарий "Registration list finalized at M6+M7. Future topologies must be added explicitly here."; `uv run pytest tests/ -q` остаётся зелёным.

### Phase 2: Configs + Three Topologies (parallel) (~4.5h)
**Goal:** Создать конфиги и реализовать все три топологии с unit- и integration-тестами.

- [ ] 1.1 Create `conf/topology/mesh.yaml`
  - File: `conf/topology/mesh.yaml`
  - Acceptance: `TopologyConfig(**yaml.safe_load(...))` без ошибок; поля `name=mesh`, `max_iterations=20`, `extra.max_rounds=6`, `extra.consensus_threshold=3`, `extra.broadcast_bus_cap=200`.

- [ ] 1.2 Create `conf/topology/debate.yaml`
  - File: `conf/topology/debate.yaml`
  - Acceptance: `TopologyConfig(**yaml.safe_load(...))` без ошибок; поля `name=debate`, `max_iterations=12`, `extra.max_rounds=4`.

- [ ] 1.3 Create `conf/topology/hierarchical.yaml`
  - File: `conf/topology/hierarchical.yaml`
  - Acceptance: `TopologyConfig(**yaml.safe_load(...))` без ошибок; поля `name=hierarchical`, `max_iterations=20`, `extra.sub_teams` (2 команды), `extra.final_answer_strategy="json_concat"`.

- [ ] 2.1 Implement `MeshTopology` in `mesh.py`
  - File: `src/atm/topology/mesh.py`
  - Acceptance: `@TopologyRegistry.register("mesh")` декоратор присутствует; `build()` возвращает `CompiledStateGraph`; граф содержит узлы `dispatcher`, `mesh_broadcast`, `mesh_postprocess` + 4 agent-узла; `uv run ruff check` + `uv run mypy src/atm/topology/mesh.py` clean.

- [ ] 2.2 Write unit tests for MeshTopology
  - File: `tests/unit/topology/test_mesh.py`
  - Acceptance: 10 тестов (перечислены в плане) зелёные, включая `test_broadcast_bus_does_not_unbound` (MC-5) и `test_consensus_vote_payload_str_format`.

- [ ] 2.3 Write integration test for MeshTopology
  - File: `tests/integration/topology/test_mesh_e2e.py`
  - Acceptance: 2 e2e-теста (consensus-path + max_rounds-path) зелёные с FakeLLM scripted fixtures.

- [ ] 2.4 Create FakeLLM fixtures for Mesh
  - Files: `tests/fixtures/llm/m7_mesh_consensus.yaml`, `tests/fixtures/llm/m7_mesh_max_rounds.yaml`
  - Acceptance: fixtures загружаются FakeLLM без ошибок; consensus-fixture продуцирует 3 идентичных `vote_for` ответа.

- [ ] 3.1 Implement `DebateTopology` in `debate.py`
  - File: `src/atm/topology/debate.py`
  - Acceptance: `@TopologyRegistry.register("debate")` присутствует; `build()` возвращает `CompiledStateGraph` с параллельным fan-out (`planner → debater_pro` и `planner → debater_contra`); `build()` бросает `ValueError` при `debater_pro_id == debater_contra_id`; ruff/mypy clean.

- [ ] 3.2 Write unit tests for DebateTopology
  - File: `tests/unit/topology/test_debate.py`
  - Acceptance: 10 тестов (перечислены в плане) зелёные, включая `test_build_rejects_same_debater_id_for_pro_and_contra` и `test_judge_postprocess_malformed_decision_treated_as_rejected`.

- [ ] 3.3 Write integration test for DebateTopology
  - File: `tests/integration/topology/test_debate_e2e.py`
  - Acceptance: 2 e2e-теста (judge-decides-after-loop + max_rounds) зелёные; assert `Message.id` уникальности после fan-in (MC-3).

- [ ] 3.4 Create FakeLLM and agent fixtures for Debate
  - Files: `tests/fixtures/llm/m7_debate_judge_decides.yaml`, `tests/fixtures/llm/m7_debate_max_rounds.yaml`, `tests/fixtures/agents/debater_contra.yaml` (если не существует)
  - Acceptance: fixtures загружаются без ошибок; judge-decides-fixture продуцирует approved=True в round 2.

- [ ] 4.1 Implement `HierarchicalTopology` in `hierarchical.py`
  - File: `src/atm/topology/hierarchical.py`
  - Acceptance: `@TopologyRegistry.register("hierarchical")` присутствует; `build()` компилирует 2 subgraph-а и top-level граф; `top_coord` и `sub_coord` — rule-based closures, не Agent-обёртки; `final_answer = json.dumps({"team_a": ..., "team_b": ...})`; `ValueError` при попытке определить 3-й уровень; ruff/mypy clean.

- [ ] 4.2 Write unit tests for HierarchicalTopology
  - File: `tests/unit/topology/test_hierarchical.py`
  - Acceptance: 11 тестов (перечислены в плане) зелёные, включая `test_no_compiled_subgraph_in_workers` (MC-1), `test_strict_two_levels_invariant`, `test_top_coord_and_sub_coord_are_not_in_agents_dict` (MC-4), `test_final_answer_json_concat_format` (MC-6).

- [ ] 4.3 Write integration test for HierarchicalTopology
  - File: `tests/integration/topology/test_hierarchical_e2e.py`
  - Acceptance: e2e-тест с 4 worker-ами (2 per team) и rule-based координаторами зелёный; assert `json.loads(shared.final_answer) == {"team_a": ..., "team_b": ...}`.

- [ ] 4.4 Create FakeLLM fixture for Hierarchical
  - File: `tests/fixtures/llm/m7_hierarchical_finalize.yaml`
  - Acceptance: fixture загружается без ошибок; fixture продуцирует DRAFT-сообщения от обоих worker-team-ов.

### Phase 3: Cross-topology Sanity (~1h)
**Goal:** Smoke-проход и integration-level precedence-проверка для всех 5 топологий.

- [ ] 5.1 Write cross-topology sanity test file
  - File: `tests/integration/topology/test_all_topologies_sanity.py`
  - Acceptance: `test_all_5_topologies_registered` зелёный; параметризированный `test_topology_builds_and_runs_smoke` для всех 5 имён зелёный; `test_should_stop_precedence_consistent_integration` (MC-2) зелёный — для каждой топологии integration-уровень подтверждает что `finish_reason == "max_iter"` при `iter_total == max_iterations - 1` + success-сигнал одновременно.

- [ ] 5.2 Final full suite check
  - Acceptance: `uv run pytest tests/ -q` green; `uv run ruff check && uv run mypy src/atm` clean.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/topology/__init__.py` | Добавить 3 guarded import | Централизованная регистрация mesh/debate/hierarchical |
| `src/atm/topology/mesh.py` | Создать | MeshTopology implementation |
| `src/atm/topology/debate.py` | Создать | DebateTopology implementation |
| `src/atm/topology/hierarchical.py` | Создать | HierarchicalTopology implementation |
| `conf/topology/mesh.yaml` | Создать | Конфиг Mesh |
| `conf/topology/debate.yaml` | Создать | Конфиг Debate |
| `conf/topology/hierarchical.yaml` | Создать | Конфиг Hierarchical |
| `tests/unit/topology/test_mesh.py` | Создать | Unit-тесты Mesh |
| `tests/unit/topology/test_debate.py` | Создать | Unit-тесты Debate |
| `tests/unit/topology/test_hierarchical.py` | Создать | Unit-тесты Hierarchical |
| `tests/integration/topology/test_mesh_e2e.py` | Создать | E2E-тест Mesh |
| `tests/integration/topology/test_debate_e2e.py` | Создать | E2E-тест Debate |
| `tests/integration/topology/test_hierarchical_e2e.py` | Создать | E2E-тест Hierarchical |
| `tests/integration/topology/test_all_topologies_sanity.py` | Создать | Cross-topology sanity + precedence |
| `tests/fixtures/llm/m7_mesh_consensus.yaml` | Создать | FakeLLM fixture для Mesh consensus-path |
| `tests/fixtures/llm/m7_mesh_max_rounds.yaml` | Создать | FakeLLM fixture для Mesh max_rounds-path |
| `tests/fixtures/llm/m7_debate_judge_decides.yaml` | Создать | FakeLLM fixture для Debate judge-decides |
| `tests/fixtures/llm/m7_debate_max_rounds.yaml` | Создать | FakeLLM fixture для Debate max_rounds |
| `tests/fixtures/agents/debater_contra.yaml` | Создать (если нет) | Agent-fixture для Debater(contra) |
| `tests/fixtures/llm/m7_hierarchical_finalize.yaml` | Создать | FakeLLM fixture для Hierarchical |
| `src/atm/core/state.py` | Только чтение (verify) | Подтвердить broadcast_bus на ~line 67 |

## Dependencies & Order Constraints

```
Wave 1: [Step 0]  — foundation, sequential
Wave 2: [Steps 1, 2, 3, 4]  — parallel, disjoint files, depends on Step 0
Wave 3: [Step 5]  — gate, depends on Steps 2, 3, 4
```

- Steps 2/3/4 не трогают `__init__.py` вообще — регистрация активируется через guarded import из Step 0.
- Steps 2, 3, 4 можно запустить в отдельных worktrees параллельно: каждый трогает строго свои файлы.
- Step 5 должен идти строго после merge всех трёх топологий.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LangGraph compiled subgraph требует совместимость reducer-ов | Medium | High | Оба level используют GraphState — overlap полный. Fallback: обернуть subgraph в node-функцию |
| broadcast_bus unbounded growth (MC-5) | Low | Medium | Hard cap `bus[-cap:]`, cap=200 в конфиге и в коде |
| Параллельный super-step в Debate теряет сообщение (Message.id collision) | Low | High | Каждый Message имеет uuid4 id; add_messages dedup-by-id |
| Опечатка в `import_module("atm.topology.mesh")` — silently ImportError | Medium | Medium | contextlib.suppress + тест `test_all_5_topologies_registered` ловит |
| Voting tally: vote_for=None или нестроковый тип | Low | Low | `continue` при нестроковом; unit-тест `test_consensus_vote_payload_str_format` |
| broadcast_bus cap обрезает legitimate vote-сообщения | Low | Medium | cap=200 >> 4 агента × 6 rounds × ~3 msg = 72 msg max |
| LangGraph subgraph checkpointer propagation | Low | Medium | Верифицировано в 2026-04; checkpointer наследуется автоматически |
| Two-topology name collision в registry (последний побеждает) | Low | High | Step 5 `test_all_5_topologies_registered` + assert `len(set(names)) == len(names)` |

## Out of Scope

- Adaptive topology (M8) — `TopologyTransition`, `PhaseRouter`, `TopologyRouter` — не в M7.
- HITL integration (M9) — `human_gateway` принимается опционально в `build(**kwargs)`, но не используется в нодах M7.
- `Send()` API / динамический fan-out — не нужен, число Debater-ов/sub-team-ов фиксировано.
- 3+ уровней в Hierarchical — строго 2 уровня (arch.md §7.6); `ValueError` при попытке нарушить.
- Manual smoke-runs через `atm run --config ...` — желательны, но вне scope автоматических тестов M7.

## Timeline
- Total: ~6h (Wave1 0.5h + Wave2 4.5h parallel + Wave3 1h)
- Created: 2026-04-26
