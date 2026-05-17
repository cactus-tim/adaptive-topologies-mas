# Post-fix E3 grids

These grids accompany `fix/adaptive-router-correctness` and verify the
behaviour of the router after the 5 correctness fixes.  They are deliberately
small so the diff can be exercised end-to-end before launching `e3_full`.

| Grid | Cells | Budget | Purpose |
|------|-------|--------|---------|
| `e3_postfix_smoke.yaml`      | 48  | ~$1   | Sanity: adaptive vs chain/star/mesh on 2 tasks |
| `e3_postfix_thrashing.yaml`  | 12  | ~$0.5 | Guards on/off — should show fewer switches under guards-on |
| `e3_postfix_advisor.yaml`    | 8   | ~$0.5 | Advisory hint consumption — check reasons in topology_transitions |
| `e3_postfix_full.yaml`       | 384 | ~$8   | Head-to-head adaptive(rule) vs static on 4 tasks × 8 shuffles × 3 seeds |
| `e3_postfix_full_llm.yaml`   | 96  | ~$2   | Adaptive with LLM topology router; pair with `e3_postfix_full.yaml` for statics |

## Recommended order

1. `e3_postfix_smoke.yaml` (15 min) — fail-fast smoke; check no regressions
2. `e3_postfix_thrashing.yaml` (~5 min) — open the parquet, plot
   `topology_switch_count` per cell; the guards-on cells should have
   notably fewer switches than guards-off
3. `e3_postfix_advisor.yaml` (~5 min) — inspect the `reason` column of
   `topology_transitions`: in the HITL-on cells you should now see entries
   containing `human_advisor_hint=...` proving the hint reached the router
4. `e3_postfix_full.yaml` + `e3_postfix_full_llm.yaml` together (~3 h) — the
   head-to-head data for diploma update

## Commands

```bash
# Smoke
uv run atm grid -c conf/experiments/e3_postfix_smoke.yaml --parallelism 4

# Thrashing — base run (guards on/off swept)
uv run atm grid -c conf/experiments/e3_postfix_thrashing.yaml --parallelism 4

# Advisor mode (advisory)
uv run atm grid -c conf/experiments/e3_postfix_advisor.yaml --parallelism 4 \
  +human.extra.human_can_override_router=false

# Advisor mode (override) — separate run because human.extra can't be in sweep
uv run atm grid -c conf/experiments/e3_postfix_advisor.yaml --parallelism 4 \
  +human.extra.human_can_override_router=true

# Full head-to-head (rule + statics)
uv run atm grid -c conf/experiments/e3_postfix_full.yaml --parallelism 12

# Companion: adaptive(llm)
uv run atm grid -c conf/experiments/e3_postfix_full_llm.yaml --parallelism 12
```

## What changed under the hood

| Fix | Where | Effect |
|-----|-------|--------|
| #1 | `adaptive.py` dispatch_topology_node | iter_total no longer double-incremented; cfg.max_iterations now has consistent semantics across topologies |
| #2 | `guards.py` _violates_cooldown        | Cooldown correctly blocks immediate return to the just-left topology |
| #3 | `adaptive.py` apply_transition_gate   | `signals['phase_switch_count']` is tracked; SwitchGuards.max_per_phase no longer falls back to len(history) |
| #4 | `adaptive.py` apply_transition_gate   | pre-subgraph phase threaded through so subgraph-internal phase advances are detected |
| #5 | `topology_router.py` RuleBased + LLM  | `signals['human_advisor_hint']` consumed; single-use cleanup in transition_gate |

See the PR description for full diagnostic detail and regression tests.
