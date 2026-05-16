# Namespace topology.extra.<topology_name> Schema — Plan

## Executive Summary

Refactor `topology.extra` from a single flat untyped `dict[str, Any]` into a typed,
namespaced sub-schema (`TopologyExtras`) where each topology consumes only its own
sub-namespace (`star`, `chain`, `debate`, `hierarchical`, `mesh`, `adaptive`). The
goal is to permanently eliminate shared-key collisions (e.g. `max_rounds` meaning
debate-rounds vs hierarchical-team-rounds vs mesh-round-robin-activations) that
previously broke the mesh pilot. Refactor must be bw-compat-safe: flat-form configs
continue working with a deprecation warning, so the in-flight E1 launch on the new
machine is not blocked.

## Current State

Two `TopologyConfig` shells exist in the codebase:

1. `atm.experiment.config.TopologyCfg` (`src/atm/experiment/config.py:85`) — Pydantic
   schema read from YAML. Currently `extra: dict[str, Any]`.
2. `atm.topology.base.TopologyConfig` (`src/atm/topology/base.py:29`) — internal data
   class passed to `topology.build(agents, cfg, **kwargs)`. Currently also
   `extra: dict[str, Any]`.

The runner (`src/atm/experiment/runner.py:1093` and `:2046`) copies fields from the
former into the latter. The internal `TopologyConfig` is also referenced inside
`adaptive.py` to construct sub-topology configs (line 449), and inside all topology
unit tests.

Six topology builders each read flat keys from `cfg.extra`:

| Topology     | Keys read from `cfg.extra`                                                                          |
|--------------|-----------------------------------------------------------------------------------------------------|
| chain        | (none)                                                                                               |
| star         | `planning_max_iter`, `exec_max_iter`, `verify_max_iter`                                             |
| debate       | `max_rounds`, `debater_pro_id`, `debater_contra_id`, `judge_id`                                     |
| hierarchical | `max_rounds`, `final_answer_strategy`, `finalize_signal`, `sub_teams`                               |
| mesh         | `mesh_max_rounds`, `consensus_threshold`, `activation_policy`, `agent_order`, `broadcast_bus_cap`   |
| adaptive     | `phase_router`, `topology_router`, `switch_guards`, `switch_guards_config`, `subgraph_max_iterations`, `planning_max_iter`, `exec_max_iter`, `verify_max_iter`, `run_id` |

Shared key collisions (`max_rounds`, `planning_max_iter`, etc.) caused the mesh pilot
failure and will cause further bugs as topology counts grow.

## Proposed Approach

1. Define six Pydantic sub-models (`StarExtras`, `ChainExtras`, `DebateExtras`,
   `HierarchicalExtras`, `MeshExtras`, `AdaptiveExtras`) and a container
   `TopologyExtras` in `src/atm/experiment/config.py`.
2. Replace `TopologyCfg.extra: dict[str, Any]` with
   `extra: TopologyExtras = Field(default_factory=TopologyExtras)`.
3. Add a `@model_validator(mode="before")` on `TopologyCfg` that detects flat-shaped
   extras and remaps them into the correct namespace with a `DeprecationWarning`.
4. Add a `get_topology_extras(cfg, topology_name)` helper in `base.py` that handles
   both the new namespaced shape and the legacy flat shape, so topology builders need
   only one mechanical substitution each.
5. Update all five topology builders to call `get_topology_extras` instead of reading
   `cfg.extra` directly.
6. Fix `adaptive._get_subgraph` to forward per-topology namespace buckets to dispatched
   sub-topologies (previously forwarded adaptive's own keys, breaking sub-topology reads).
7. Update `runner.py` to serialise the typed `TopologyExtras` back to a dict using
   `model_dump(exclude_none=True)` before passing to the internal `TopologyConfig`.
8. Migrate all production YAML configs and test fixtures to the namespaced shape.

**Key design decisions:**

- `adaptive.run_id` is the only `Optional[X] = None` field. Using `exclude_none=True`
  in Step 6 ensures the field is absent from the dumped dict, preserving the builder's
  `str(uuid.uuid4())` fallback (if `None` were present, `str(None)` = literal `"None"`
  would silently corrupt every adaptive run's run_id).
- `AdaptiveExtras` intentionally uses phase-limit defaults of 3/10/4 (not star's 2/5/3)
  because adaptive runs longer reasoning phases; these match the inline literals at
  `adaptive.py:394-396`, not any named constants.
- `max_rounds` in legacy flat form scatters ONLY to `debate` and `hierarchical`. It
  does NOT scatter to `mesh` (mesh uses 12 to avoid starvation) or `adaptive` (adaptive
  does not read `max_rounds` at all).
- `_resolve_dotpath` requires no changes: it already descends into `BaseModel`
  subclasses, so `topology.extra.mesh.max_rounds` resolves naturally post-refactor.
- `ChainExtras` is included as an empty sub-model for schema symmetry and future
  extensibility.

## Implementation Phases

### Phase 1: Schema + Helper (~2.5h) — Wave 1, parallel
**Goal:** Establish the typed schema and the indirection helper that all builders will use.

- [ ] 1.1 Add `TopologyExtras` typed sub-schemas and bw-compat validator in `config.py`
  - File: `src/atm/experiment/config.py`
  - Acceptance: `TopologyCfg(extra={"mesh":{"max_rounds":12}})` succeeds; `TopologyCfg(extra={"max_rounds":2})` emits `DeprecationWarning` and routes to `debate` + `hierarchical`; unknown key raises `ValidationError`.

- [ ] 1.2 Add `get_topology_extras` helper in `base.py` + export from `__init__.py`
  - File: `src/atm/topology/base.py`, `src/atm/topology/__init__.py`
  - Acceptance: Helper returns correct bucket for namespaced shape, full flat dict for legacy shape, and `{}` for empty/missing extras.

### Phase 2: Tests + Runner + Fixtures (~2h) — Wave 2, after Phase 1
**Goal:** Verify the schema, migrate breaking tests, update the runner, and prepare configs.

- [ ] 2.1 Unit tests for `TopologyExtras` schema + bw-compat + migrate existing breakers
  - File: `tests/unit/experiment/test_config.py`, `tests/unit/experiment/test_config_grid.py`
  - Acceptance: All 15+ new/migrated test cases pass; `pytest tests/unit/experiment/ -v` green; schema-vs-source parity guard test passes.

- [ ] 2.2 Update `runner.py` to serialise typed extras with `model_dump(exclude_none=True)`
  - File: `src/atm/experiment/runner.py` (2 call sites at lines 1093 and 2046)
  - Acceptance: Both call sites use `cfg.topology.extra.model_dump(exclude_none=True)`; `adaptive.run_id` absent from dumped dict when unset.

- [ ] 2.3 Migrate canonical test YAML fixtures
  - File: `tests/fixtures/experiment/*.yaml`
  - Acceptance: Fixtures load cleanly under new schema; at least one legacy-flat fixture retained to exercise bw-compat path.

- [ ] 2.4 Migrate all topology builders to use `get_topology_extras`
  - File: `src/atm/topology/star.py`, `src/atm/topology/debate.py`, `src/atm/topology/hierarchical.py`, `src/atm/topology/mesh.py`, `src/atm/topology/adaptive.py`
  - Acceptance: `pytest tests/unit/topology/ -v` green without any test file edits; mesh builder reads `max_rounds` (not `mesh_max_rounds`) from its namespace bucket.

- [ ] 2.5 Migrate production configs in `conf/experiments/`
  - File: `conf/experiments/e1_pilot.yaml`, `e1_pilot_sanity.yaml`, `e1_mini.yaml`, `e1_mini_fixes.yaml`, `e1_sanity_hier.yaml`, `e1_full.yaml`, `e2_full.yaml`, `e3_full.yaml`, `e4_full.yaml`, `adaptive_smoke.yaml`, `adaptive_thrashing.yaml`, `adaptive_guard_override.yaml` (verify `smoke.yaml` needs no edit)
  - Acceptance: All configs load without `DeprecationWarning`; `e3_full` and `e4_full` have `mesh.max_rounds == 12` (not 2).

### Phase 3: Adaptive subgraph + Validation tests (~1.5h) — Wave 3, after Phase 2
**Goal:** Fix the critical adaptive subgraph forwarding bug and add explicit refactor-validation tests.

- [ ] 3.1 Fix `adaptive._get_subgraph` namespace forwarding + delete `_meta_keys`
  - File: `src/atm/topology/adaptive.py`
  - Acceptance: Adaptive dispatches sub-topology with `{"<topo_name>": {...}}` shaped extras; existing `test_adaptive_human.py` tests remain green; new `test_adaptive_namespace_forwarding.py` passes.

- [ ] 3.2 Add 2 refactor-validation tests (namespaced mesh build + bw-compat warning)
  - File: `tests/unit/topology/test_mesh.py`, `tests/unit/experiment/test_config.py` (or new `test_topology_extras.py`)
  - Acceptance: Namespaced `MeshTopology` build resolves correct `max_rounds`; bw-compat fixture emits exactly one `DeprecationWarning` for `max_rounds` and produces correct `debate.max_rounds` + `hierarchical.max_rounds`.

### Phase 4: Housekeeping + Final smoke (~1h) — Wave 4 + 5
**Goal:** Clean up docstrings and verify end-to-end acceptance criteria.

- [ ] 4.1 Update docstrings in topology modules and `config.py`
  - File: `src/atm/topology/debate.py`, `star.py`, `hierarchical.py`, `mesh.py`, `adaptive.py`, `src/atm/experiment/config.py`
  - Acceptance: No references to flat `extra.key` patterns remain; `ruff check` clean.

- [ ] 4.2 End-to-end smoke run + final pytest
  - File: (no files edited — verification only)
  - Acceptance: `pytest tests/unit/ -v` green (308+ existing + 16+ new); `pytest tests/unit/topology/ tests/unit/experiment/ -W error::DeprecationWarning` green; e3/e4 mesh starvation guard passes; adaptive `run_id` resolves to a real UUID string (not `"None"`); `atm grid run -c conf/experiments/e1_mini.yaml` startup succeeds.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/experiment/config.py` | Add 6 sub-schemas + `TopologyExtras` + retype `TopologyCfg.extra` + bw-compat pre-validator | Core schema refactor |
| `src/atm/topology/base.py` | Add `get_topology_extras` helper | Single indirection point for all builders |
| `src/atm/topology/__init__.py` | Export `get_topology_extras` | Consistent with existing exports |
| `src/atm/topology/star.py` | Switch to `get_topology_extras(cfg, "star")` | Eliminate flat reads |
| `src/atm/topology/debate.py` | Switch to `get_topology_extras(cfg, "debate")` | Eliminate flat reads |
| `src/atm/topology/hierarchical.py` | Switch to `get_topology_extras(cfg, "hierarchical")` | Eliminate flat reads |
| `src/atm/topology/mesh.py` | Switch to `get_topology_extras(cfg, "mesh")` + rename `mesh_max_rounds` read to `max_rounds` | Namespace eliminates collision alias |
| `src/atm/topology/adaptive.py` | Switch to `get_topology_extras(cfg, "adaptive")` + fix `_get_subgraph` forwarding + delete `_meta_keys` | Critical correctness: sub-topologies must receive their own bucket |
| `src/atm/experiment/runner.py` | Both call sites: `model_dump(exclude_none=True)` | Convert typed schema to dict for internal `TopologyConfig` |
| `conf/experiments/*.yaml` (10-12 files) | Migrate flat `extra:` to namespaced shape | Eliminate DeprecationWarning in production |
| `tests/unit/experiment/test_config.py` | Migrate 1 broken test + add 15 new cases | Cover new schema + bw-compat |
| `tests/unit/experiment/test_config_grid.py` | Rewrite 1 broken test + add 2 new tests | New dotpath contract |
| `tests/unit/topology/test_mesh.py` | Add namespaced-form build test | Explicit refactor-validation |
| `tests/unit/topology/test_adaptive_namespace_forwarding.py` | New file | Verify subgraph forwarding |

## Dependencies & Order Constraints

- **Wave 1** (parallel): Step 1.1, Step 1.2
- **Wave 2** (after Wave 1): Steps 2.1, 2.2, 2.3, 2.4, 2.5 (all depend on 1.1 or 1.2; 2.4 depends on 1.2)
- **Wave 3** (after Wave 2): Step 3.1 (depends on 1.2 + 2.4), Step 3.2 (depends on 1.1 + 2.4 + 3.1)
- **Wave 4** (after Wave 3): Step 4.1 (depends on 2.4 + 3.1)
- **Wave 5** (final): Step 4.2 (depends on everything)

Note: Step 2.4 modifies 5 independent topology files — the orchestrator may execute
them in sub-parallel since the file sets are disjoint.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Default-value drift between schema and topology builder constants | Medium | Silent incorrect defaults in sweep experiments | Schema-vs-source parity guard test (Step 2.1 test 15) fails immediately on drift |
| Adaptive phase-limit drift (no named `_DEFAULT_*` constants in adaptive.py) | Medium | Adaptive runs truncated silently | Test 15 asserts raw integer parity (3/10/4) against `adaptive.py:394-396` inline literals |
| `exclude_none=False` inadvertently used in runner, producing `"None"` run_id | Low | Every adaptive run gets a literal `"None"` as run_id | Step 4.2 smoke test explicitly asserts `run_id not in adaptive_bucket` and UUID validity |
| `max_rounds` flat-remap scattered to `mesh` in e3/e4 migration | Medium | Mesh starvation bug reintroduced | Step 4.2 asserts `cfg.topology.extra.mesh.max_rounds == 12` for e3_full and e4_full |
| `_get_subgraph` forwarding omission (adaptive sub-topologies fall back to defaults) | High before fix | Sub-topology experiments silently use wrong params | Step 3.1 adds dedicated forwarding test with monkeypatched `MeshTopology.build` |
| Mixed namespaced+flat shape in legacy test fixtures surprises new contributors | Low | Confusing test failures | `get_topology_extras` detection invariant documented in `base.py`; mixed shape treated as legacy-flat |
| A future Optional-typed field added to a sub-model with `.get(key, fallback)` semantics | Low | Same `"None"` injection bug for new field | Audit table in Codebase Context; reviewer re-audits on any future field addition |

## Out of Scope

- `HumanCfg.extra` namespace collision risk — separate schema, separate refactor.
- Promoting `DeprecationWarning` to a hard error — deferred to a follow-up release.
- Wiring `phase_router` / `topology_router` into the router construction code path
  (currently dead schema fields kept for forward-compatibility).
- Integration tests (no HTTP/gRPC/pub-sub contract; Pydantic schema unit tests + smoke
  run are sufficient).

## Timeline

- Total: ~7h
- Created: 2026-05-16
