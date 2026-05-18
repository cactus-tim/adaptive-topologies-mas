# E4 post-fix runbook

Re-run of E4 (adaptive topology + adaptive role, RQ4) against the corrected
adaptive implementation from PR #18.

| Config | Cells | Wall (p=8) | Budget | Purpose |
|--------|-------|-----------|--------|---------|
| `e4_postfix_smoke.yaml`           | 3   | ~2 min  | ~$0.10 | Verify all 3 role_router modes still discriminate |
| `e4_postfix_full.yaml`            | 540 | ~5-6 h  | ~$35   | Full RQ4 head-to-head: fixed vs rule vs llm × 4 tasks |
| `e4_postfix_full_dabench.yaml`    | 135 | ~3-4 h  | ~$10   | Catchup for dabench cells (cascade-prone), p=4 |
| `e4_postfix_full_commongen.yaml`  | 135 | ~3 h    | ~$10   | Catchup for commongen (Cerebras-timeout-prone), p=6 |

**Total compute envelope** if everything runs once cleanly: ~$35 + ~$5 judge ≈ $40.
With both catchups added: ~$55-60 (matches prior E4 plan budget of $45+$5).

---

## Recommended order

```bash
# 0. Pre-flight: PG up, .env populated with PG_DSN, ANTHROPIC/OPENAI/CEREBRAS keys
docker compose up -d
docker compose ps              # postgres healthy

# 1. SMOKE — sanity that 3 role_router modes produce distinct role distributions
uv run atm grid -c conf/experiments/e4_postfix_smoke.yaml --parallelism 3
# Acceptance: exit=0, 3/3 completed, role distribution check (see below)

# 2. FULL — main 540-cell head-to-head
uv run atm grid -c conf/experiments/e4_postfix_full.yaml --parallelism 8
# Expect: 4-6 hours wall.  Will likely cascade on dabench-llm cells.
# A "partial" exit with 70-80% completion is acceptable — catchups fill gaps.

# 3a. If dabench cells dominate the failures:
uv run atm grid -c conf/experiments/e4_postfix_full_dabench.yaml --parallelism 4

# 3b. If commongen cells produced finish_reason=error (LLM timeout):
uv run atm grid -c conf/experiments/e4_postfix_full_commongen.yaml --parallelism 6

# 4. If a cascade kills the main grid mid-flight before status=completed:
uv run atm grid -c conf/experiments/e4_postfix_full.yaml --resume-incomplete
# (only re-runs cells with status != completed; idempotent thanks to run_id).
```

---

## Smoke acceptance check

```python
import pandas as pd
df = pd.read_parquet("data/experiments/experiments/<smoke_exp_id>/_runs.parquet")
# Open human_interactions parquet if separate, or inspect role_router_decision
# events from observability.

# Pseudocode (real query depends on schema):
for mode in ["fixed", "rule", "llm"]:
    sub = df[df["human_role_router"] == mode]
    # fixed: all rows share single role; rule: 2-3 distinct; llm: cost > 0
```

If the role distribution check fails for any mode, **do not run the full grid**
— investigate role_router wiring first.

---

## Infrastructure notes (lessons from E3 postfix)

Already applied in the fix branch:

1. **`mp_context=spawn`** in `src/atm/experiment/grid.py` (commit 5b05899) —
   workers no longer inherit parent PG socket FDs.  Without this, workers
   that fork off after the checkpointer warmup or reconcile share FDs with
   the parent; any concurrent write garbles the PG wire protocol → connection
   loss → cascade death.

2. **PG WAL caps** in `docker-compose.yml` (5b05899):
   - `max_wal_size=8GB`
   - `min_wal_size=2GB`
   - `checkpoint_completion_target=0.9`
   Default 1GB caused "checkpoints occurring too frequently" warnings under
   24-worker adaptive load.

3. **Lower parallelism** for cascade-prone task types:
   - Full grid p=8 (down from prior E4=12)
   - DABench catchup p=4 (down further)
   - CommonGen catchup p=6 with bumped HITL timeout (1200 s vs 900 s)

What we still **don't** fully understand: dabench+llm-router 0/41 zombies
in E3 postfix despite mp_context=spawn.  Suspects (not investigated):
- psycopg pipeline race under specific connection-pool patterns
- ProcessPoolExecutor + per-worker async event loop interaction
- Cerebras-side connection-throttling under sustained load (kills the
  worker via TCP RST while it's mid-graph-execution)

If E4 dabench reproduces the cascade, lower parallelism is the workaround
that proved most reliable in E3.

---

## What to compare against (apples-to-apples)

Prior E4 numbers (from `arch/diploma/results/e4_analysis.md` §3, n=180 per mode):

| Mode    | mean_q | mean_cost  | mean_wall_s |
|---------|--------|------------|-------------|
| `fixed` | 0.6531 | $0.01186   | 381.3       |
| `rule`  | 0.6799 | $0.01193   | 355.8       |
| `llm`   | 0.6432 | $0.01168   | 285.2       |

Per-task winner table (prior, §4):

| Task      | fixed q       | rule q        | llm q         | Winner        |
|-----------|---------------|---------------|---------------|---------------|
| humaneval | 0.911 (0.288) | **0.978**     | 0.844         | rule (+0.067) |
| gsm8k     | 0.978         | 0.978         | 0.956         | tie           |
| commongen | 0.568         | 0.586         | **0.591**     | llm (+0.023)  |
| dabench   | 0.156         | 0.178         | **0.182**     | llm (+0.026)  |

Prior RQ4 verdict: rule beats fixed by Δ=+0.027 overall, but bootstrap CI
[-0.055, +0.110] includes 0 → **not statistically significant**.  «RQ4 not
replicated» in the sense that adaptive role gives no detectable lift.

### Expected direction after fixes

Based on E3 postfix observations:
- **All 3 modes' baselines lift** because adaptive's underlying topology
  routing got cleaner (fix #1 iter_total + #5 advisor hint biggest sources).
- **DABench column moves most** (E3 rule dabench: 0.174 → 0.423 = +0.249).
- **Rule lead over fixed**: could either preserve, shrink, or grow.  Hard
  to predict — depends on whether role_router was being undermined by the
  same bugs that hurt topology_router.
- **LLM mode**: prior was hurt by HITL-hint-not-consumed; now the hint is
  consumed.  Lift likely larger than rule mode.

**Possible scientific outcomes**:
- A. RQ4 still not replicated (verdict unchanged, just bigger numbers).
- B. Rule mode pulls ahead with significance: +0.04-0.06 Δ over fixed,
     bootstrap CI excludes 0.  This **flips** the RQ4 verdict.
- C. LLM mode wins: previously the weakest, now the strongest because
     the hint actually feeds back into router decisions.

Any of A/B/C is publishable.  Reporting A confidently is also progress —
it strengthens the prior conclusion («adaptive role doesn't help even
when the underlying adaptive code is correct»).

---

## After completion

1. Open the resulting parquets, compute the same tables as
   `arch/diploma/results/e4_analysis.md` §3-§4.
2. Re-run bootstrap-CI on `rule − fixed` and `llm − fixed` (overall and
   per-task).  Use the same 10000-resample method as prior analysis.
3. If verdict **same** as prior → write a 1-paragraph addendum to
   `e4_analysis.md` confirming RQ4 robustness under fixes.
4. If verdict **flips** → larger rewrite + new bootstrap tables; pre-fix
   numbers stay in main body, post-fix as §10 «Postfix re-analysis».
5. Update `arch/diploma/results/synthesis.md` if either RQ2 or RQ4 verdict
   changed direction.

---

## Diff vs prior E4

Identical:
- Sub-graph caps (max_iterations=6, subgraph_max_iterations=4, etc.)
- 3 role_router modes (fixed / rule / llm)
- 4 task × 15 shuffle × 3 seed sweep
- Worker / judge models
- HITL config (gateway=llm_simulated, timeout_policy=llm_fallback)

Changed:
- `parallelism: 12 → 8` (PG-cascade insurance)
- Catchup configs introduced for dabench (p=4) and commongen (p=6) tail risks
- Implicit: runs against PR #18 code (5 router/state fixes vs prior pre-fix)
