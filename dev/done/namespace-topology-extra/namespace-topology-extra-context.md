# Namespace topology.extra.<topology_name> Schema — Context

## SESSION PROGRESS (2026-05-16)

### COMPLETED
- Step 1.1: Added `StarExtras`, `ChainExtras`, `DebateExtras`, `HierarchicalExtras`, `MeshExtras`, `AdaptiveExtras`, `TopologyExtras` sub-schemas to `src/atm/experiment/config.py`; replaced `TopologyCfg.extra: dict[str,Any]` with `extra: TopologyExtras`; added `@model_validator(mode="before")` bw-compat pre-validator with correct scattering rules (`max_rounds` → debate+hierarchical only, `mesh_max_rounds` → mesh.max_rounds, etc.); added `_FLAT_TO_NAMESPACE` and `_TOPOLOGY_EXTRA_NAMES` helpers; mypy + ruff clean; 20/21 tests pass (1 expected failure: `test_load_config_topology_extra_passthrough` uses flat star keys that now trigger `DeprecationWarning`, fixed in Step 2.1).
- Step 1.2: Added `get_topology_extras(cfg, topology_name) -> dict[str, Any]` helper to `src/atm/topology/base.py`; added `_TOPOLOGY_NAMES` frozenset constant; exported `get_topology_extras` from `src/atm/topology/__init__.py` (import + `__all__`); added 11 new test cases in `tests/unit/topology/test_base.py` covering all four branches (namespaced, flat, empty, mixed); all smoke checks pass; mypy + ruff clean; 256 topology tests pass (6 pre-existing `test_mesh_human` failures unrelated to this step).
- Step 2.1: Migrated `test_load_config_topology_extra_passthrough` to use `pytest.warns(DeprecationWarning)` + attribute access (`cfg.topology.extra.star.*`); split `test_grid_cfg_rejects_topology_extra_keys` into 3 tests (`test_grid_cfg_rejects_unknown_topology_extra_keys`, `test_grid_cfg_accepts_namespaced_topology_extra_dotpath`, `test_grid_cfg_rejects_namespaced_dict_leaf`); added 14 new TopologyExtras tests (T1-T14) including schema-vs-source parity guard; 49 tests pass in `tests/unit/experiment/`; ruff clean; no new mypy errors (1 pre-existing error in test_config_grid.py unrelated to this step).
- Step 2.2: Both call sites in `src/atm/experiment/runner.py` (lines 1096 and 2049) changed from `extra=cfg.topology.extra` to `extra=cfg.topology.extra.model_dump(exclude_none=True)`; ruff clean; mypy requires Wave 1 merge to resolve `dict` vs `TopologyExtras` type.
- Step 2.3: All 4 existing fixtures (`grid_minimal.yaml`, `grid_runner_failfast.yaml`, `grid_runner_mini.yaml`, `valid_minimal.yaml`) already had empty `extra: {}` or no `extra` block — no migration needed. Added `star_flat_extras_bwcompat.yaml` as the deliberate bw-compat fixture (flat star keys: `planning_max_iter`, `exec_max_iter`, `verify_max_iter`) that emits exactly one `DeprecationWarning` and remaps to `extra.star.*`. `pytest tests/unit/experiment/test_smoke_yaml_loads.py -v` green; all 5 fixtures load without errors (flat one loads with expected DeprecationWarning).
- Step 2.4: Migrated all 5 topology builders (`star.py`, `debate.py`, `hierarchical.py`, `mesh.py`, `adaptive.py`) to use `get_topology_extras(cfg, "<name>")` instead of `cfg.extra or {}`; mesh now reads `max_rounds` (not `mesh_max_rounds`) from its namespaced bucket (bw-compat validator remaps legacy `mesh_max_rounds` at schema level); `adaptive.py` retains `_raw_extra` for `_get_subgraph` (Step 3.1 replaces it); `run_id` now reads from `extras` with uuid4() fallback; 271 topology unit tests pass; mypy + ruff clean; all 5 namespaced + legacy flat smoke checks pass.

- Step 3.1: Fixed `adaptive._get_subgraph` to forward full namespaced `cfg.extra` dict to sub-topology builders (instead of filtering via `_meta_keys`); deleted `_meta_keys` frozenset and `_raw_extra` variable; added 6 new tests in `tests/unit/topology/test_adaptive_namespace_forwarding.py` covering mesh namespace forwarding, debate namespace forwarding, no adaptive key leakage, star alias forwarding, namespaced dict shape verification, and subgraph_max_iter propagation; 277/277 topology tests pass; mypy + ruff clean.
- Step 3.2: Added `TestMeshNamespacedExtras` class to `tests/unit/topology/test_mesh.py` with `test_mesh_reads_max_rounds_from_namespaced_extra` and `test_mesh_falls_back_to_default_when_namespace_empty`; added `test_legacy_flat_extras_remap_does_not_pollute_other_buckets` to `tests/unit/experiment/test_config.py`; all 3 new tests pass; 493 total tests pass in tests/unit/topology + tests/unit/experiment.

- Step 4.1: Updated docstrings in all 5 topology modules (`debate.py`, `star.py`, `hierarchical.py`, `mesh.py`, `adaptive.py`) to document the namespaced `extras.<topology_name>.*` schema with defaults, DeprecationWarning note, and mesh starvation rationale; added 15-line comment block above `TopologyExtras` in `config.py` explaining schema rationale and single-source-of-truth convention; 499 tests pass; ruff clean.

### IN PROGRESS
- Step 2.5: Migrate production configs in conf/experiments/

### BLOCKERS
- None

## Quick Resume

1. Read this file
2. Check `namespace-topology-extra-tasks.md` for what is next
3. Read `namespace-topology-extra-plan.md` Phase 1 for strategy
4. Start with: Task 1.1 — add `TopologyExtras` typed sub-schemas and bw-compat validator in `src/atm/experiment/config.py`

## Key Files

**`src/atm/experiment/config.py`**
- Role: Pydantic schema read from YAML; contains `TopologyCfg` with `extra: TopologyExtras`
- Planned change: DONE — six sub-models + `TopologyExtras` + bw-compat validator implemented; `_FLAT_TO_NAMESPACE`, `_TOPOLOGY_EXTRA_NAMES` helpers added
- Status: COMPLETE (Step 1.1)

**`src/atm/topology/base.py`**
- Role: Internal `TopologyConfig` dataclass passed to all topology builders; `extra: dict[str, Any]` stays loose (not retyped)
- Planned change: Add `get_topology_extras(cfg, topology_name) -> dict[str, Any]` helper and `_TOPOLOGY_NAMES` frozenset constant
- Status: NOT STARTED

**`src/atm/topology/__init__.py`**
- Role: Public re-export surface for `atm.topology`
- Planned change: Add `get_topology_extras` to the `from atm.topology.base import (...)` block and to `__all__`
- Status: NOT STARTED

**`src/atm/topology/star.py`**
- Role: Star topology builder; reads `planning_max_iter`, `exec_max_iter`, `verify_max_iter` from `cfg.extra`
- Planned change: Replace flat `cfg.extra or {}` read with `get_topology_extras(cfg, "star")`
- Status: NOT STARTED

**`src/atm/topology/debate.py`**
- Role: Debate topology builder; reads `max_rounds`, `debater_pro_id`, `debater_contra_id`, `judge_id` from `cfg.extra`
- Planned change: Replace flat read with `get_topology_extras(cfg, "debate")`
- Status: NOT STARTED

**`src/atm/topology/hierarchical.py`**
- Role: Hierarchical topology builder; reads `max_rounds`, `final_answer_strategy`, `finalize_signal`, `sub_teams` from `cfg.extra`
- Planned change: Replace flat read with `get_topology_extras(cfg, "hierarchical")`
- Status: NOT STARTED

**`src/atm/topology/mesh.py`**
- Role: Mesh topology builder; reads `mesh_max_rounds` (aliased key), `consensus_threshold`, `activation_policy`, `agent_order`, `broadcast_bus_cap` from `cfg.extra`
- Planned change: Replace flat read with `get_topology_extras(cfg, "mesh")`; rename `mesh_max_rounds` lookup to `max_rounds` (bw-compat validator handles the remapping at schema level)
- Status: NOT STARTED

**`src/atm/topology/adaptive.py`**
- Role: Adaptive topology builder; reads 9 keys from `cfg.extra`; dispatches sub-topologies via `_get_subgraph`; strips `_meta_keys` before forwarding extras
- Planned change: Replace flat read with `get_topology_extras(cfg, "adaptive")`; rewrite `_get_subgraph` to forward the per-sub-topology namespace bucket (not the adaptive bucket); delete `_meta_keys` filter; add defensive comment near `run_id` read about `exclude_none=True` dependency
- Status: NOT STARTED

**`src/atm/experiment/runner.py`**
- Role: Experiment runner; two call sites (lines 1096 and 2049) copy `cfg.topology.extra` into internal `TopologyConfig`
- Planned change: Both call sites change from `extra=cfg.topology.extra` to `extra=cfg.topology.extra.model_dump(exclude_none=True)`
- Status: COMPLETE (Step 2.2)

**`conf/experiments/*.yaml` (10-12 files)**
- Role: Production experiment configs with flat `topology.extra:` blocks
- Planned change: Migrate all to namespaced shape; add yaml comment pointing to dev docs; e3_full + e4_full require special-case: `max_rounds` goes ONLY to `debate` + `hierarchical`, NOT to `mesh` (mesh stays at default 12)
- Status: NOT STARTED

**`tests/unit/experiment/test_config.py`**
- Role: Config schema unit tests; line 259-261 uses `cfg.topology.extra["planning_max_iter"]` (breaks post-refactor)
- Planned change: Migrate line 259-261 to attribute access; add 15 new schema + bw-compat test cases including schema-vs-source parity guard test
- Status: NOT STARTED

**`tests/unit/experiment/test_config_grid.py`**
- Role: Grid config tests; line 150-153 asserts old "topology.extra.* rejected" contract (wrong post-refactor)
- Planned change: Rewrite `test_grid_cfg_rejects_topology_extra_keys` into 3 separate tests reflecting new contract
- Status: NOT STARTED

**`tests/fixtures/experiment/star_flat_extras_bwcompat.yaml`** (NEW — Step 2.3)
- Role: Deliberate bw-compat fixture with flat star extras (`planning_max_iter`, `exec_max_iter`, `verify_max_iter`); exercises the DeprecationWarning + remapping code path
- Status: COMPLETE (Step 2.3)

**`tests/unit/topology/test_mesh.py`**
- Role: Mesh topology unit tests
- Planned change: Add namespaced-form build test using `{"mesh": {"max_rounds": 12}}` extra shape
- Status: NOT STARTED

**`tests/unit/topology/test_adaptive_namespace_forwarding.py`**
- Role: New file; 6 tests covering _get_subgraph namespace forwarding
- Status: COMPLETE (Step 3.1)

## Decisions

**Typed schema in `experiment.config`, loose dict in `topology.base`**
- Decision: `TopologyCfg.extra` becomes `TopologyExtras`; internal `TopologyConfig.extra` stays `dict[str, Any]`
- Rationale: Keeps existing topology unit test fixtures working without migration (they construct `TopologyConfig(extra={"flat_key": val})`); the `get_topology_extras` helper handles both shapes at the builder call site

**`exclude_none=True` in runner serialisation**
- Decision: `model_dump(exclude_none=True)` at both runner call sites
- Rationale: `AdaptiveExtras.run_id` defaults to `None`. The adaptive builder reads `extra.get("run_id", str(uuid.uuid4()))`. `dict.get(key, default)` returns `default` only when the key is ABSENT — not when it is `None`. Using `exclude_none=False` would produce `{"run_id": None}` and `str(None)` = the literal string `"None"`, silently corrupting every adaptive run's run_id. `exclude_none=True` omits the key entirely, preserving the UUID fallback. Per audit, `run_id` is the only affected field.

**`max_rounds` flat-remap target namespaces**
- Decision: Legacy `max_rounds` scatters to `debate` + `hierarchical` ONLY
- Rationale: `mesh.max_rounds` default is 12 (starvation-safe); scattering a flat `max_rounds: 2` to mesh would re-introduce the original mesh pilot failure. `adaptive` does not read `max_rounds` at all (no such field on `AdaptiveExtras`).

**`AdaptiveExtras` phase-limit defaults: 3/10/4, not star's 2/5/3**
- Decision: `planning_max_iter=3`, `exec_max_iter=10`, `verify_max_iter=4`
- Rationale: `adaptive.py:394-396` uses inline fallbacks of 3/10/4, deliberately larger than star's 2/5/3. Mirroring star's defaults would silently truncate adaptive runs. There are no named `_DEFAULT_*` constants in adaptive.py for these — the schema-vs-source parity guard test (test 15) asserts raw integer parity against the inline literals.

**`get_topology_extras` detection invariant**
- Decision: A `cfg.extra` dict is treated as NAMESPACED iff every top-level key is a member of `{star, chain, debate, hierarchical, mesh, adaptive}`. Otherwise treated as LEGACY-FLAT and returned verbatim.
- Rationale: Simple, cheap O(n) check; the bw-compat pre-validator in `config.py` normalises all incoming YAML before the runner reaches this helper, so the mixed-shape edge case only arises in hand-constructed test fixtures.

**`_meta_keys` deletion in adaptive.py**
- Decision: Delete the `_meta_keys` filter after Step 3.1
- Rationale: Under the namespaced model, the per-topology bucket is pre-isolated; meta-key stripping is no longer needed. Each sub-topology receives only its own namespace bucket.

**`ChainExtras` included despite chain reading no extras**
- Decision: Include as empty sub-model
- Rationale: Schema symmetry; reserves space for future chain knobs; cheap to remove later.

**`phase_router` and `topology_router` in `AdaptiveExtras`**
- Decision: Keep as schema fields, marked with `# NOTE: parsed but NOT consumed`
- Rationale: `adaptive.build()` does not call `extra.get("phase_router", ...)` — they appear in `_meta_keys` but are dead fields. Kept for forward-compatibility with planned router pluggability. Out of scope to wire or remove in this refactor.

## Constraints

- Pydantic v2 in use; `model_validator(mode="before")` accepted.
- `pytest -W error::DeprecationWarning` is NOT currently enabled in `pyproject.toml` — deprecations don't break the suite.
- `HumanCfg.extra` is out of scope — do not touch.
- Sweep dotpaths like `topology.extra.<key>` (flat) will now fail `_resolve_dotpath`; this is correct and intentional.
- No integration tests warranted (Pydantic schema unit tests + smoke run are sufficient).

## Amendments Applied

- Reviewer nit (non-blocking): Step 7 in the draft has `Can-Parallel-With: 5` which is incorrect — Step 5 is in Wave 3, making cross-wave parallelism with Step 7 (Wave 2) meaningless. The wave graph is authoritative; the orchestrator should follow wave ordering, not the `Can-Parallel-With` field for cross-wave entries. This nit is noted here; no changes to the plan content were required.
