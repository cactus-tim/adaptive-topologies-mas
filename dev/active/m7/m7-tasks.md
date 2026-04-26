# M7 — Mesh, Debate, Hierarchical Topologies — Tasks

## Wave 1: Pre-flight + Register [Step 0] COMPLETE

- [x] 0.1 Verify `SharedState.broadcast_bus` exists in `core/state.py` — `src/atm/core/state.py`
  - Type: simple (read-only verify)
  - Acceptance: `grep -n "broadcast_bus" src/atm/core/state.py` возвращает строку на ~line 67; если поле отсутствует — эскалировать, не продолжать.

- [x] 0.2 Extend `topology/__init__.py` with 3 guarded imports — `src/atm/topology/__init__.py`
  - Type: simple
  - Acceptance: файл содержит `contextlib.suppress(ImportError)` для mesh, debate, hierarchical; обновлён комментарий "Registration list finalized at M6+M7..."; `uv run pytest tests/ -q` green; `python -c "from atm.topology import TopologyRegistry; print(TopologyRegistry.list_names())"` содержит star и chain (mesh/debate/hierarchical появятся после Wave 2).

---

## Wave 2: Configs + Three Topologies [Steps 1-4] NOT STARTED

### Step 1 — Config files (parallel-safe)

- [x] 1.1 Create `conf/topology/mesh.yaml` — `conf/topology/mesh.yaml`
  - Type: simple
  - Acceptance: `TopologyConfig(**yaml.safe_load(open("conf/topology/mesh.yaml")))` без ошибок; поля name=mesh, max_iterations=20, extra.max_rounds=6, extra.consensus_threshold=3, extra.broadcast_bus_cap=200.

- [x] 1.2 Create `conf/topology/debate.yaml` — `conf/topology/debate.yaml`
  - Type: simple
  - Acceptance: `TopologyConfig(**yaml.safe_load(open("conf/topology/debate.yaml")))` без ошибок; поля name=debate, max_iterations=12, extra.max_rounds=4, extra.debater_pro_id, extra.debater_contra_id, extra.judge_id.

- [x] 1.3 Create `conf/topology/hierarchical.yaml` — `conf/topology/hierarchical.yaml`
  - Type: simple
  - Acceptance: `TopologyConfig(**yaml.safe_load(open("conf/topology/hierarchical.yaml")))` без ошибок; поля name=hierarchical, max_iterations=20, extra.sub_teams (2 команды), extra.final_answer_strategy="json_concat".

### Step 2 — Mesh topology (parallel-safe; disjoint files)

- [ ] 2.1 Implement `MeshTopology` — `src/atm/topology/mesh.py`
  - Type: tdd
  - Acceptance: `@TopologyRegistry.register("mesh")` декоратор; `build()` → `CompiledStateGraph`; узлы dispatcher, mesh_broadcast, mesh_postprocess + 4 agent-ноды; broadcast_bus cap применяется (MC-5); docstring документирует vote payload-формат; ruff/mypy clean.

- [ ] 2.2 Write unit tests for MeshTopology — `tests/unit/topology/test_mesh.py`
  - Type: tdd
  - Acceptance: 10 тестов зелёные: test_dispatcher_round_robin_cycles_agents, test_priority_activation_routes_to_critic_on_draft, test_consensus_threshold_writes_winner_signal, test_max_rounds_routes_to_end, test_global_max_iterations_overrides_topology_max, test_build_returns_compiled_graph_with_expected_nodes, test_register_under_name_mesh, test_mesh_broadcast_writes_outbox_to_bus, test_broadcast_bus_does_not_unbound (MC-5), test_consensus_vote_payload_str_format.

- [ ] 2.3 Write e2e integration tests for MeshTopology — `tests/integration/topology/test_mesh_e2e.py`
  - Type: integration
  - Acceptance: 2 теста зелёные (consensus-path: `shared.final_answer == "4"`, consensus reached; max_rounds-path: нет consensus, выход по max_rounds); FakeLLM scripted.

- [ ] 2.4 Create FakeLLM fixtures for Mesh — `tests/fixtures/llm/m7_mesh_consensus.yaml`, `tests/fixtures/llm/m7_mesh_max_rounds.yaml`
  - Type: simple
  - Acceptance: fixtures загружаются FakeLLM без ошибок; consensus-fixture имеет 3+ совпадающих vote_for="4"; max_rounds-fixture имеет разные vote_for у разных агентов.

### Step 3 — Debate topology (parallel-safe; disjoint files)

- [x] 3.1 Implement `DebateTopology` — `src/atm/topology/debate.py`
  - Type: tdd
  - Acceptance: `@TopologyRegistry.register("debate")` декоратор; параллельный fan-out planner → debater_pro + debater_contra; debate_round_start no-op нода для loop; `ValueError` при debater_pro_id == debater_contra_id; judge_postprocess парсит approved + winner; ruff/mypy clean.

- [x] 3.2 Write unit tests for DebateTopology — `tests/unit/topology/test_debate.py`
  - Type: tdd
  - Acceptance: 10 тестов зелёные: test_planner_fans_out_to_both_debaters, test_judge_postprocess_writes_signals_on_approve, test_judge_postprocess_writes_final_answer_from_winner, test_route_from_judge_returns_end_on_approved, test_route_from_judge_returns_loop_on_not_approved_within_max_rounds, test_route_from_judge_returns_end_on_max_rounds_exceeded, test_global_max_iterations_overrides_topology_max, test_register_under_name_debate, test_judge_postprocess_malformed_decision_treated_as_rejected, test_build_rejects_same_debater_id_for_pro_and_contra.

- [x] 3.3 Write e2e integration tests for DebateTopology — `tests/integration/topology/test_debate_e2e.py`
  - Type: integration
  - Acceptance: 2 теста зелёные (judge-decides-after-loop + max_rounds); assert `signals["judge_decided"] == True`; assert все Message.id уникальны после fan-in (MC-3); `iter_total` корректно инкрементируется.

- [x] 3.4 Create FakeLLM and agent fixtures for Debate — `tests/fixtures/llm/m7_debate_judge_decides.yaml`, `tests/fixtures/llm/m7_debate_max_rounds.yaml`, `tests/fixtures/agents/debater_contra.yaml` (если не существует)
  - Type: simple
  - Acceptance: fixtures загружаются без ошибок; judge-decides-fixture: round 1 approved=False, round 2 approved=True winner="pro"; max_rounds-fixture: все rounds approved=False.

### Step 4 — Hierarchical topology (parallel-safe; disjoint files)

- [ ] 4.1 Implement `HierarchicalTopology` — `src/atm/topology/hierarchical.py`
  - Type: tdd
  - Acceptance: `@TopologyRegistry.register("hierarchical")` декоратор; build() компилирует 2 subgraphs + top-level; top_coord и sub_coord — rule-based closures (не Agent, не в agents dict; MC-4); final_answer = json.dumps({"team_a":..., "team_b":...}; MC-6); ValueError при sub_teams[i].sub_teams (MC-1, strict 2-level); ruff/mypy clean.

- [ ] 4.2 Write unit tests for HierarchicalTopology — `tests/unit/topology/test_hierarchical.py`
  - Type: tdd
  - Acceptance: 11 тестов зелёные: test_build_creates_two_subgraphs_via_compile_call, test_top_coord_route_to_subgraphs_on_first_iter, test_top_coord_finalize_signal_sets_signals_and_routes_end, test_top_coord_max_rounds_reached_routes_end_with_topology_max, test_global_max_iterations_overrides_topology_max, test_subgraph_runs_workers_in_order, test_register_under_name_hierarchical, test_strict_two_levels_invariant, test_no_compiled_subgraph_in_workers (MC-1), test_final_answer_json_concat_format (MC-6), test_top_coord_and_sub_coord_are_not_in_agents_dict (MC-4).

- [ ] 4.3 Write e2e integration test for HierarchicalTopology — `tests/integration/topology/test_hierarchical_e2e.py`
  - Type: integration
  - Acceptance: тест с 4 worker-ами (executor_a1, executor_a2, executor_b1, executor_b2) и rule-based координаторами зелёный; assert `json.loads(shared.final_answer) == {"team_a": ..., "team_b": ...}`; assert `signals["top_coord_finalize"] == True`; assert `iter_total` consistent.

- [ ] 4.4 Create FakeLLM fixture for Hierarchical — `tests/fixtures/llm/m7_hierarchical_finalize.yaml`
  - Type: simple
  - Acceptance: fixture загружается без ошибок; DRAFT-сообщения от worker-ов обоих teams присутствуют; top_coord успешно финализирует.

---

## Wave 3: Cross-topology Sanity [Step 5] NOT STARTED

- [ ] 5.1 Write cross-topology sanity test file — `tests/integration/topology/test_all_topologies_sanity.py`
  - Type: integration
  - Acceptance: test_all_5_topologies_registered: `set(TopologyRegistry.list_names()) >= {"star","chain","mesh","debate","hierarchical"}` + `len(set(names)) == len(names)` (нет дубликатов); параметризированный test_topology_builds_and_runs_smoke для всех 5 имён: `iter_total > 0`, нет исключений; test_should_stop_precedence_consistent_integration (MC-2): для каждой топологии `finish_reason == "max_iter"` при iter_total==max_iterations-1 + success-сигнал.

- [ ] 5.2 Final full suite check
  - Type: simple
  - Acceptance: `uv run pytest tests/ -q` green; `uv run ruff check && uv run mypy src/atm` clean; все 5 топологий присутствуют в `TopologyRegistry.list_names()`.

---

## Stats
- Total: 19 tasks · ~6h
- Done: 0 / 19

## How to Update
Отмечать выполненные задачи `[x]`, обновлять заголовки фаз (NOT STARTED / IN PROGRESS / COMPLETE), обновлять SESSION PROGRESS в m7-context.md после каждого milestone.

Шаблон статусов:
- `NOT STARTED` — ни одна задача фазы не начата
- `IN PROGRESS` — хотя бы одна задача начата, не все завершены
- `COMPLETE` — все задачи фазы завершены и acceptance criteria выполнены
