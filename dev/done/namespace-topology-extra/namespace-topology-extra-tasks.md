# Namespace topology.extra.<topology_name> Schema — Tasks

## Phase 1: Schema + Helper (Wave 1 — parallel) NOT STARTED

- [x] 1.1 Add `TopologyExtras` typed sub-schemas and bw-compat validator in `config.py` — `src/atm/experiment/config.py`
  - Acceptance: `TopologyCfg(extra={"mesh":{"max_rounds":12}})` succeeds without warning; `TopologyCfg(extra={"max_rounds":2})` emits `DeprecationWarning` and routes value to `extra.debate.max_rounds` and `extra.hierarchical.max_rounds`; unknown key raises `ValidationError`; `TopologyExtras` exported from module.

- [x] 1.2 Add `get_topology_extras` helper in `base.py` + export from `__init__.py` — `src/atm/topology/base.py`, `src/atm/topology/__init__.py`
  - Acceptance: Helper returns named-namespace bucket for all-namespace-keys shape; returns full flat dict for legacy shape; returns `{}` for empty or missing `extra`; mixed shape (both namespace and non-namespace keys) treated as legacy-flat; unit test in `tests/unit/topology/test_base.py` covers all four branches.

## Phase 2: Tests + Runner + Configs + Builders (Wave 2 — after Phase 1) NOT STARTED

- [x] 2.1 Unit tests for `TopologyExtras` schema + bw-compat + migrate existing breakers — `tests/unit/experiment/test_config.py`, `tests/unit/experiment/test_config_grid.py`
  - Acceptance: Migrated test at line 259-261 uses attribute access and passes; `test_grid_cfg_rejects_topology_extra_keys` rewritten into 3 tests covering new contract; 15 new cases pass (including schema-vs-source parity guard test asserting adaptive phase defaults are 3/10/4 vs inline literals); `pytest tests/unit/experiment/ -v` fully green.

- [x] 2.2 Update `runner.py` to serialise typed extras — `src/atm/experiment/runner.py`
  - Acceptance: Both call sites (lines ~1093 and ~2046) changed to `extra=cfg.topology.extra.model_dump(exclude_none=True)`; `adaptive.run_id` absent from dumped dict when unset in YAML.

- [x] 2.3 Migrate canonical test YAML fixtures — `tests/fixtures/experiment/*.yaml`
  - Acceptance: Fixtures that previously used flat `extra:` converted to namespaced shape; at least one fixture deliberately left on flat shape to exercise bw-compat; `pytest tests/unit/experiment/test_smoke_yaml_loads.py -v` green.

- [x] 2.4 Migrate all 5 topology builders to use `get_topology_extras` — `src/atm/topology/star.py`, `src/atm/topology/debate.py`, `src/atm/topology/hierarchical.py`, `src/atm/topology/mesh.py`, `src/atm/topology/adaptive.py`
  - Acceptance: Each builder calls `get_topology_extras(cfg, "<name>")` instead of `cfg.extra or {}`; mesh builder reads `max_rounds` (not `mesh_max_rounds`) from its bucket; `pytest tests/unit/topology/ -v` green with no test file edits required; rename comment added in `mesh.py` explaining the `mesh_max_rounds` alias change.

- [ ] 2.5 Migrate production configs in `conf/experiments/` — `conf/experiments/*.yaml` (10-12 files)
  - Acceptance: All configs load without `DeprecationWarning`; each migrated `extra:` block has a comment pointing to dev docs; `e3_full.yaml` and `e4_full.yaml` have `mesh.max_rounds` absent (defaults to 12), `debate.max_rounds: 2`, `hierarchical.max_rounds: 2`; `smoke.yaml` verified to need no edit.

## Phase 3: Adaptive Subgraph + Validation Tests (Wave 3 — after Phase 2) NOT STARTED

- [x] 3.1 Fix `adaptive._get_subgraph` namespace forwarding + delete `_meta_keys` — `src/atm/topology/adaptive.py`
  - Acceptance: `_get_subgraph` forwards `{"<topo_name>": sub_bucket}` for namespaced shape; falls back to flat dict for legacy shape; `_meta_keys` set deleted; defensive comment added near `run_id` read; `pytest tests/unit/topology/test_adaptive_human.py -v` still green; new `test_adaptive_namespace_forwarding.py` passes.

- [x] 3.2 Add 2 refactor-validation tests — `tests/unit/topology/test_mesh.py`, `tests/unit/experiment/test_config.py` (or `tests/unit/topology/test_adaptive_namespace_forwarding.py`)
  - Acceptance: Namespaced mesh build test: `TopologyConfig(extra={"mesh":{"max_rounds":12}})` passed to `MeshTopology().build(...)` resolves `max_rounds` to 12; bw-compat test: legacy flat fixture emits `DeprecationWarning` matching `"max_rounds"` and produces correct `debate.max_rounds` + `hierarchical.max_rounds`.

## Phase 4: Housekeeping + Final Smoke (Waves 4-5 — after Phase 3) NOT STARTED

- [x] 4.1 Update docstrings in topology modules and `config.py` — `src/atm/topology/debate.py`, `star.py`, `hierarchical.py`, `mesh.py`, `adaptive.py`, `src/atm/experiment/config.py`
  - Acceptance: No flat `extra.key` references remain in docstrings; mesh bug-comment at lines 174-181 replaced with pointer to dev docs; `ruff check` clean; `pytest` still green.

- [ ] 4.2 End-to-end smoke run + final pytest — (no files edited, verification only)
  - Acceptance:
    1. `pytest tests/unit/ -v` green (308+ existing + 16+ new tests)
    2. `pytest tests/unit/topology/ tests/unit/experiment/ -W error::DeprecationWarning` green
    3. `atm grid run -c conf/experiments/e1_mini.yaml` startup succeeds (schema validates, first topology constructs)
    4. Legacy flat-form YAML loads with `DeprecationWarning`
    5. `e3_full.yaml` and `e4_full.yaml`: `cfg.topology.extra.mesh.max_rounds == 12` (not 2)
    6. Adaptive `run_id` smoke: `"run_id" not in get_topology_extras(sub_cfg, "adaptive")` AND `uuid.UUID(resolved)` does not raise AND `resolved != "None"`

---

## Stats

- Total: 11 tasks (~7h)
- Done: 0 / 11

## How to Update

Check off tasks with `[x]` and update `namespace-topology-extra-context.md` SESSION PROGRESS after each milestone.
Phase headers: if all tasks in a phase done, mark `COMPLETE`; if some done, mark `IN PROGRESS`.
