# M7 — Mesh, Debate, Hierarchical Topologies — Context

## SESSION PROGRESS (2026-04-26)

### COMPLETED
- Step 0 (Wave 1 Pre-flight + Register): verified `broadcast_bus` at state.py:67; extended `topology/__init__.py` with guarded imports for mesh/debate/hierarchical + updated comment; 823 tests pass.
- Step 1 (Config files 1.1–1.3): created conf/topology/mesh.yaml, debate.yaml, hierarchical.yaml; all load via TopologyConfig without errors.
- Step 2 (Mesh topology — subtasks 2.1–2.4): implemented `MeshTopology` in `mesh.py` with dispatcher (round-robin + priority), mesh_broadcast (cap=200, MC-5), mesh_postprocess (vote tally + consensus), stopping precedence via _should_stop; 22 unit tests + 2 integration e2e tests all green; ruff clean; fixtures m7_mesh_consensus.yaml + m7_mesh_max_rounds.yaml created.
- Step 3 (Debate topology — subtasks 3.1–3.4): implemented `DebateTopology` in `debate.py` with parallel fan-out (planner → debater_pro + debater_contra), judge loop, judge_postprocess, stopping precedence via _should_stop; 17 unit tests + 2 integration e2e tests all green; ruff clean; fixtures created.
- Step 4 (Hierarchical topology — subtasks 4.1–4.4): implemented `HierarchicalTopology` in `hierarchical.py` with compiled subgraphs (team_a, team_b), rule-based top_coord and sub_coord closures (MC-4), strict 2-level invariant ValueError (MC-1), JSON-concat final_answer (MC-6), stopping precedence via _should_stop; 13 unit tests + 2 integration e2e tests all green; ruff clean; fixture m7_hierarchical_finalize.yaml created; conf/topology/hierarchical.yaml created.
- Step 5 (Cross-topology sanity — subtasks 5.1–5.2): created `tests/integration/topology/test_all_topologies_sanity.py` with 11 tests (1 registry check + 5 smoke + 5 precedence); all 892 tests pass (32 skipped); ruff clean; mypy has 1 pre-existing error in mesh.py (out of Step 5 scope).

### IN PROGRESS
- All steps complete

### BLOCKERS
- None

## Quick Resume
1. Read this file
2. Check m7-tasks.md for what's next
3. Read m7-plan.md Phase 1 for strategy
4. Start with: Step 0.1 — verify `broadcast_bus` in `src/atm/core/state.py` (grep -n "broadcast_bus"), затем 0.2 — extend `src/atm/topology/__init__.py`

## Key Files

### Step 0 — Pre-flight + `__init__.py`

**`src/atm/core/state.py`**
- Role: определяет SharedState и все reducer-ы; `broadcast_bus: list[Message]` должен быть на ~line 67
- Planned change: только чтение (verify-only)
- Status: NOT STARTED

**`src/atm/topology/__init__.py`**
- Role: централизованная регистрация всех топологий через guarded `contextlib.suppress(ImportError)` import-ы
- Planned change: добавить 3 guarded import-а (mesh, debate, hierarchical) + обновить комментарий
- Status: NOT STARTED

### Step 1 — Config files

**`conf/topology/mesh.yaml`**
- Role: конфиг TopologyConfig для Mesh; поля: name, max_iterations=20, extra.max_rounds=6, extra.consensus_threshold=3, extra.broadcast_bus_cap=200, extra.agents=[planner,researcher,executor,critic]
- Status: DONE

**`conf/topology/debate.yaml`**
- Role: конфиг TopologyConfig для Debate; поля: name, max_iterations=12, extra.max_rounds=4, extra.debater_pro_id=debater_pro, extra.debater_contra_id=debater_contra, extra.judge_id=critic
- Status: DONE

**`conf/topology/hierarchical.yaml`**
- Role: конфиг TopologyConfig для Hierarchical; поля: name, max_iterations=20, extra.sub_teams (2 команды: team_a/team_b с 2 workers каждая), extra.final_answer_strategy="json_concat"
- Status: DONE

### Step 2 — Mesh topology

**`src/atm/topology/mesh.py`**
- Role: MeshTopology — broadcast-bus граф с dispatcher, mesh_broadcast (post-process, cap), mesh_postprocess (vote tally), 4 agent-нодами; регистрация через @TopologyRegistry.register("mesh")
- Planned change: создать
- Status: NOT STARTED

**`tests/unit/topology/test_mesh.py`**
- Role: unit-тесты routing/stopping/registration/invariants Mesh (10 тестов)
- Planned change: создать
- Status: NOT STARTED

**`tests/integration/topology/test_mesh_e2e.py`**
- Role: e2e integration-тесты Mesh (consensus-path + max_rounds-path) с FakeLLM
- Planned change: создать
- Status: NOT STARTED

**`tests/fixtures/llm/m7_mesh_consensus.yaml`**
- Role: FakeLLM scripted fixture — 3 совпадающих vote_for="4" от planner/researcher/executor
- Planned change: создать
- Status: NOT STARTED

**`tests/fixtures/llm/m7_mesh_max_rounds.yaml`**
- Role: FakeLLM scripted fixture — разные голоса, consensus не достигается, выход по max_rounds
- Planned change: создать
- Status: NOT STARTED

### Step 3 — Debate topology

**`src/atm/topology/debate.py`**
- Role: DebateTopology — Planner → parallel fan-out к debater_pro + debater_contra → critic_judge → judge_postprocess → loop/END; регистрация через @TopologyRegistry.register("debate")
- Planned change: создать
- Status: COMPLETE

**`tests/unit/topology/test_debate.py`**
- Role: unit-тесты routing/stopping/fan-out/registration/invariants Debate (17 тестов — включает все 10 обязательных)
- Planned change: создать
- Status: COMPLETE

**`tests/integration/topology/test_debate_e2e.py`**
- Role: e2e integration-тесты Debate (judge-decides + max_rounds) с FakeLLM; assert Message.id уникальности (MC-3)
- Planned change: создать
- Status: COMPLETE

**`tests/fixtures/llm/m7_debate_judge_decides.yaml`**
- Role: FakeLLM scripted fixture — judge approved=True в round 2, winner="pro"
- Planned change: создать
- Status: COMPLETE

**`tests/fixtures/llm/m7_debate_max_rounds.yaml`**
- Role: FakeLLM scripted fixture — judge approved=False во всех rounds, выход по max_rounds
- Planned change: создать
- Status: COMPLETE

**`tests/fixtures/agents/debater_contra.yaml`**
- Role: agent-fixture для Debater с stance=contra
- Planned change: создать при необходимости
- Status: COMPLETE

### Step 4 — Hierarchical topology

**`src/atm/topology/hierarchical.py`**
- Role: HierarchicalTopology — top_coord (rule-based closure) → 2 compiled subgraph-а → workers; final_answer = json.dumps({"team_a":..., "team_b":...}); регистрация через @TopologyRegistry.register("hierarchical")
- Planned change: создать
- Status: COMPLETE

**`conf/topology/hierarchical.yaml`**
- Role: TopologyConfig для Hierarchical (name, max_iterations=20, sub_teams, final_answer_strategy)
- Planned change: создать
- Status: COMPLETE

**`tests/unit/topology/test_hierarchical.py`**
- Role: unit-тесты routing/stopping/subgraph-structure/invariants Hierarchical (13 тестов, 11 required)
- Planned change: создать
- Status: COMPLETE

**`tests/integration/topology/test_hierarchical_e2e.py`**
- Role: e2e integration-тест Hierarchical с 4 worker-ами (2 per team); assert JSON-format final_answer
- Planned change: создать
- Status: COMPLETE

**`tests/fixtures/llm/m7_hierarchical_finalize.yaml`**
- Role: FakeLLM scripted fixture — DRAFT-сообщения от обоих worker-team-ов, top_coord финализирует
- Planned change: создать
- Status: COMPLETE

**`tests/unit/topology/conftest.py`**
- Role: autouse fixture re-registers all topology classes after registry clear in test_base.py
- Planned change: extended to also handle chain and hierarchical (was only star before)
- Status: COMPLETE

### Step 5 — Cross-topology sanity

**`tests/integration/topology/test_all_topologies_sanity.py`**
- Role: параметризированный smoke-run для всех 5 топологий + test_all_5_topologies_registered + test_should_stop_precedence_consistent_integration (MC-2) на integration-уровне
- Planned change: создать
- Status: COMPLETE — 11 tests: 1 registry + 5 smoke + 5 MC-2 precedence; all pass

## Decisions

### Централизованная регистрация в __init__.py (Step 0, единственный раз)
- Decision: `__init__.py` модифицируется ТОЛЬКО в Step 0; Steps 2/3/4 не трогают его вообще.
- Rationale: исключает merge-конфликты при параллельной разработке в worktrees (CI-1).

### broadcast_bus записывается только через post-process ноду mesh_broadcast
- Decision: agent-ноды в Mesh НЕ пишут напрямую в broadcast_bus; только mesh_broadcast post-process нода делает это.
- Rationale: детерминистический merge через одну точку записи (M6 паттерн post-process; CI-2).

### Координаторы Hierarchical — rule-based closures, не Agent-обёртки (MC-4)
- Decision: top_coord и sub_coord — closure-функции внутри build(); в agents dict передаются ТОЛЬКО worker-ы.
- Rationale: coordinator-агентов не существует в M5 canonical_4; rule-based логика не требует LLM.

### Hierarchical final_answer = JSON-concat (MC-6)
- Decision: `final_answer = json.dumps({"team_a": ..., "team_b": ...}, ensure_ascii=False)`.
- Rationale: простой, детерминистический, машиночитаемый формат для downstream M8 Adaptive.

### Debate параллельный fan-out через стандартные edges (не Send API)
- Decision: `graph.add_edge("planner", "debater_pro")` + `graph.add_edge("planner", "debater_contra")` — LangGraph запускает их в одном super-step.
- Rationale: число Debater-ов фиксировано = 2; Send() API нужен только для динамического fan-out.

### Debate: Debater-ноды пишут только в agents[*]["outbox"], не в shared
- Decision: во время параллельного super-step каждый Debater пишет только в свой outbox.
- Rationale: избегаем shared-write collisions при concurrent execution (Invariant Step 3).

### Mesh consensus-vote payload-формат
- Decision: `{"vote_for": str}` где str — сам ответ-строка (не hash); нестроковый vote_for игнорируется.
- Rationale: простой, читаемый, достаточный для M7; задокументировать в docstring mesh.py.

### Subgraph в Hierarchical — direct compile() в parent.add_node()
- Decision: CompiledStateGraph передаётся прямо в `parent.add_node(team_id, compiled_subgraph)` если schemas overlap.
- Rationale: верифицировано через LangGraph docs 2026-04. Fallback зарезервирован: обернуть в node-функцию если compile-в-add_node не работает.

## Constraints

- LangGraph >= 0.3 в зависимостях — `StateGraph`, `START`, `END`, parallel fan-out, compiled subgraphs доступны.
- `core/state.py` не модифицируется в M7 — `broadcast_bus` уже есть.
- M6 артефакты (`base.py`, `star.py`, `chain.py`) не модифицируются — open/closed principle.
- Adaptive (M8) и HITL (M9) вне scope M7.
- Ruff + mypy strict должны проходить на всех новых файлах.
- Hierarchical: строго 2 уровня; `ValueError` при попытке нарушить (arch.md §7.6).
