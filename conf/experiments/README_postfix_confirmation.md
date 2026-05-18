# Post-fix confirmation runbook

Cross-family validation of E3/E4 head-line claims on `cerebras:qwen-3-235b-a22b-instruct-2507`,
against the **corrected** adaptive implementation (PR #18).

These runs replace the prior `confirmation_e3_{rule,llm}.yaml` and
`confirmation_e4.yaml` which were executed on pre-fix code (commits
`9510b99`, `64c43ae`).  Without re-running confirmation, the post-fix E3/E4
results have no cross-family backing — which is fine if RQ verdicts are
unchanged, but the post-fix data shows **direction changes**:
- E3: adaptive (rule) reaches parity with best static; llm-router jumps
  +0.219 mean quality.  This is a **new claim** vs prior «adaptive proigryvaet».
- E4: all three role_router modes drop slightly vs prior; rule no longer
  shows even the small advantage it had.  This **strengthens** RQ4
  «role-router not helpful», but the strengthening is new.

Both new claims need cross-family backing to be defensible at the diploma
level.

### Full confirmations (apples-to-apples with pre-fix confirmation)

| Config                                | Cells | Wall (p=8) | Budget | Purpose |
|---------------------------------------|-------|-----------|--------|---------|
| `confirmation_e3_postfix_rule.yaml`   | 180   | ~1.5-2 h  | ~$15   | E3 rule cross-family — validate RQ2 (adaptive ≈ best static) |
| `confirmation_e3_postfix_llm.yaml`    | 180   | ~2-3 h    | ~$15   | E3 llm cross-family — validate Pareto (llm at fraction of cost) |
| `confirmation_e4_postfix.yaml`        | 540   | ~6-8 h    | ~$45   | E4 cross-family — validate strengthened RQ4 |

Total compute envelope for full: ~$75 worker + ~$10 judge ≈ $85, ~10-12 hours
wall on the deploy machine if run sequentially.  Comparable to prior
confirmation runs ($6.50 + $6.50 + $50 ≈ $63).

### Mini confirmations (time-constrained, single-night runs)

| Config                                     | Cells | Wall (p=4) | Budget | n per cell |
|--------------------------------------------|-------|-----------|--------|------------|
| `confirmation_e3_postfix_rule_mini.yaml`   | 60    | ~1-1.5 h  | ~$4    | n=15 per task — **matches prior confirmation_e3 density** |
| `confirmation_e3_postfix_llm_mini.yaml`    | 60    | ~1.5 h    | ~$4    | n=15 per task — same as rule companion |
| `confirmation_e4_postfix_mini.yaml`        | 72    | ~1-1.5 h  | ~$5    | n=6 per (mode, task) — direction-only (formal significance already settled by prior full confirmation_e4 on n=15) |

Total mini envelope: **~$13 worker + ~$3 judge ≈ $16, ~3.5-4 h wall sequentially.**
Can be started in the evening and complete overnight.

**When mini is enough**:
- E3 rule/llm mini: n=15 per task **matches the density that was used in
  the prior pre-fix confirmation chapter** — same statistical weight as
  what `confirmation_e3_analysis.md` already calls «sufficient cross-family
  backing».  Bootstrap-CI on principal effects (LLM gsm8k/humaneval lifts,
  ~0.4 magnitude on gpt-oss postfix) will be informative.
- E4 mini: direction-only.  Significance was not achievable on prior full
  confirmation_e4 (n=15 per (mode, task), CI included 0) and won't appear
  at smaller n either.  Mini just confirms direction holds cross-family.

**When you need full**:
- Defensible tight bootstrap-CI for confirmation chapter
- Cells you want to claim parity for specific shuffles
- Replicating prior `confirmation_e3` 360-cell density exactly

---

## Recommended order

### Mini overnight (time-constrained — $16, ~3.5h)

```bash
# 1. E3 rule mini (60 cells, ~1-1.5h)
uv run atm grid -c conf/experiments/confirmation_e3_postfix_rule_mini.yaml --parallelism 4

# 2. E3 llm mini (60 cells, ~1.5h)
uv run atm grid -c conf/experiments/confirmation_e3_postfix_llm_mini.yaml --parallelism 4

# 3. E4 mini (72 cells, ~1-1.5h)
uv run atm grid -c conf/experiments/confirmation_e4_postfix_mini.yaml --parallelism 4
```

### Full confirmation (when time permits — $85, ~10-12h)

```bash
# Wait until e4_postfix_full.yaml main grid has completed.
# Then run confirmation in this order to keep TPM contention low:

# 1. E3 rule confirmation (180 cells, ~1.5-2h)
uv run atm grid -c conf/experiments/confirmation_e3_postfix_rule.yaml --parallelism 8

# 2. E3 llm confirmation (180 cells, ~2-3h) — more cascade-prone, p=6
uv run atm grid -c conf/experiments/confirmation_e3_postfix_llm.yaml --parallelism 6

# 3. E4 confirmation (540 cells, ~6-8h) — longest single grid
uv run atm grid -c conf/experiments/confirmation_e4_postfix.yaml --parallelism 6
```

If a cascade kills any grid mid-flight, use `--resume-incomplete`:

```bash
uv run atm grid -c conf/experiments/confirmation_e4_postfix.yaml --resume-incomplete
```

---

## What to compare against (prior pre-fix confirmation)

### Prior `confirmation_e3` (commits `9510b99`)

| Router | mean_q | mean_cost | mean_wall |
|--------|--------|-----------|-----------|
| rule   | 0.8664 | $0.0190   | 272.6 s   |
| llm    | 0.5599 | $0.0171   | 218.1 s   |

Per-task on qwen (from `confirmation_e3_analysis.md`):
| Router | commongen | dabench | gsm8k | humaneval |
|--------|-----------|---------|-------|-----------|
| rule   | 0.582     | 0.920   | 0.991 | 0.978     |
| llm    | 0.789     | 0.916   | 0.000 | 0.553     |

Key prior observation: **family-divergence is huge**.  Qwen solves dabench
at 0.92 (vs 0.17 on gpt-oss); llm-router collapses on gsm8k+humaneval to
0.0/0.55 (vs 0.91/0.87 on gpt-oss).

### Prior `confirmation_e4` (commit `64c43ae`)

«RQ4 NOT replicated» on qwen, broadly matching gpt-oss verdict (rule ≈ fixed
≈ llm within noise).  Tables in `confirmation_e4_analysis.md`.

---

## Expected direction after fixes

Based on E3 postfix observations (gpt-oss family):

**For E3 rule on qwen**:
- Rule mean q likely stays high (qwen already at 0.866 prior; minor uptick
  from fix #1 on dabench-style tasks).
- DABench column: prior was 0.92, so iter_total fix has little headroom —
  expect ~0.95.

**For E3 llm on qwen** (most uncertain):
- Prior was 0.56 mean (collapsed on gsm8k/humaneval).
- On gpt-oss, postfix llm jumped +0.219 mean from advisor-hint + iter_total
  fixes.  On qwen the same fixes should help similar mechanisms — but
  qwen-llm-router's collapse on verifiable tasks might have a different
  root cause (model-specific JSON-parsing failure, observed in
  `confirmation_e3_analysis.md §6`).  If post-fix llm on qwen still hovers
  around 0.5-0.6 on gsm8k → the family-effect is robust, not an artefact.

**For E4 on qwen**:
- Prior confirmation: rule ≈ fixed ≈ llm within noise.
- Postfix on gpt-oss: rule and llm drop −0.083..−0.113 vs fixed.
- If on qwen postfix shows **same drop direction** → strong claim:
  «role-router doesn't help on either family under corrected implementation».
- If on qwen postfix shows **flat or positive** → limitation:
  «role-router has family-specific behaviour, gpt-oss penalises it».

---

## After completion

For each confirmation grid:
1. Open `data/experiments/experiments/<exp_id>/_runs.parquet`.
2. Compute per-router (or per-role_router) × per-task mean quality.
3. Bootstrap-CI on key comparisons:
   - E3 rule: `qwen vs gpt-oss postfix` per task
   - E4: `rule − fixed` and `llm − fixed` on qwen
4. Write addendum to existing `confirmation_e3_analysis.md` /
   `confirmation_e4_analysis.md` (§N — Post-fix re-run).  Keep the prior
   §-sections intact as audit trail; new addendum points to post-fix
   numbers as primary.
5. Update `synthesis.md` if any RQ-verdict's direction changes
   cross-family.

---

## Cost ceiling guard

`per_experiment_usd: 50.0` (E3 confirmations) and `100.0` (E4 confirmation)
are conservative caps.  Actual prior spend was $3-6 per E3 grid and $25-30
for E4.  If atm reports cost > 80% of cap during run, that's a red flag —
investigate before continuing.

---

## Diff vs prior confirmation configs

Same:
- Worker (cerebras:qwen-3-235b-a22b-instruct-2507)
- Judge (openai:gpt-4.1-mini)
- HITL (llm_simulated, role=reviewer, role_router=fixed)
- Sweep dimensions (180/180/540 cells)
- provider_opts nulls (qwen rejects reasoning_effort)

Changed:
- Base include: `e3_postfix_full.yaml` / `e4_postfix_full.yaml` (post-fix
  defaults) instead of `e3_full.yaml` / `e4_full.yaml`
- parallelism: 8/6/6 instead of 30/30/8 (cascade safety)
- All numeric results from these configs reflect PR #18 corrected code
