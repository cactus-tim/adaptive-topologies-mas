# Agent Briefing Template — Experiment Analysis (E1-E5)

Шаблон промпта для запуска general-purpose агента на анализ одного
эксперимента. Используется при подготовке reviews в
`arch/diploma/results/`. Накоплен по итогам E1/E2/E3 сессий — отражает
все косяки которые повторять не хочется.

---

## Стандартная структура промпта

Скопируй блок ниже, подставь переменные `{EX}`, `{EXP_ID}`, `{RQ}`,
`{PLAN_SECTION}` и т.п., прикинь — нужны ли E-specific дополнения.

```
Write a thorough analytical review of experiment {EX} results for the
diploma. Save as `/home/cactustim/agents/adaptive-topologies-mas/arch/diploma/results/{ex_lower}_analysis.md`
(Russian prose, English for code/config/metric names).

## Context

You are writing for a master's diploma on "adaptive topologies in
multi-agent LLM systems". {EX} is experiment number {N} of five.

### Sources of truth (read in this order)

1. **Experiment plan** — `/home/cactustim/agents/adaptive-topologies-mas/arch/experiment_plan.md`,
   focus on **{PLAN_SECTION}** (the {EX}-specific section) plus global §0
   (task-mix). This is the canonical source for RQ, hypothesis, design,
   smart-cuts, exclusions.

2. **Raw aggregate parquet** — `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/{EXP_ID}/_runs.parquet`.
   One row per Run. Also `experiment.json` in the same dir for experiment
   metadata.

3. **Pre-computed artefacts** (be skeptical — cross-check against your own
   pandas aggregations from the parquet; flag discrepancies):
   - `/home/cactustim/agents/adaptive-topologies-mas/analysis/{EX}_results.json` (if exists)
   - {OPTIONAL: e1_top3.json для E2/E3/E4, oracle file для E3, и т.п.}

4. **Config** — `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/{EX}_full.yaml`
   (+ `scripts/run_{EX}.py` if wrapper exists).

### Parquet schema reminders (DO NOT skip)

Columns in `_runs.parquet`:
  id, exp_id, topology, task_id, agent_set, human_role, seed, model,
  status, finish_reason, budget_spent_usd, quality_score, wall_time_s,
  iterations, started_at, finished_at, error, cognitive_load_proxy,
  replay_of, host, process_pid, models_by_role_json,
  model_version_snapshot, sandbox_image_digest.

KNOWN LIMITATIONS:
  - `task_id` is just the bench short name ("humaneval" / "gsm8k" /
    "commongen" / "dabench") — NO `/0`, `/1` suffix. The shuffle_seed
    dimension is NOT in the schema, so per-shuffle analysis is impossible
    from this parquet alone. State this as a limitation if needed.
  - `human_role` is set only when HITL was enabled (E2/E3/E4); NaN
    for E1 baseline.
  - `replay_of` is non-NULL only for explicit replay runs (rare).

### Pre-aggregation hygiene (CRITICAL — do not skip)

Some dumps contain retry / resume / zombie-recovered rows. Always:

```python
import pandas as pd
df = pd.read_parquet('data/experiments/experiments/{EXP_ID}/_runs.parquet')

# 1. Status filter
df = df[df.status == 'completed']
df = df[df.quality_score.notna()]

# 2. Dedupe by canonical key — keep latest by started_at
#    Adapt key columns based on experiment dimensionality.
key_cols = ['topology', 'task_id', 'seed']
if 'human_role' in df.columns and df.human_role.notna().any():
    key_cols.append('human_role')
# Add other sweep dimensions present in your experiment.

df = df.sort_values('started_at').drop_duplicates(subset=key_cols, keep='last')

# 3. Sanity report — print expected vs actual cell count.
print(f'after dedupe: n={len(df)}')
print(f'per (topology, task_id): {df.groupby(["topology","task_id"]).size().describe()}')
```

If post-dedupe count does NOT match the expected sweep size from the
config (e.g. 5 topo × 4 task × 15 shuffle × 3 seeds = 900 for E1):

  - Flag the discrepancy in the report
  - Try a broader key (e.g. without shuffle_seed since it's not in schema)
  - If still off — DO NOT silently average over duplicates; ask for clarification

### What the analysis should cover (~5-6 pages markdown)

1. **Постановка** — RQ, hypothesis, what {EX} measures.
2. **Дизайн** — sweep, smart-cuts applied, exclusions. State actual n after
   dedupe vs expected.
3. **Результаты — overall** — overall mean_q, mean_cost, wall-time.
4. **Per-dimension breakdowns** — tables for each swept dim (topology,
   task, role, etc).
5. **Cross-dim matrices** — (topology × task), (role × task), etc.
6. **Ответ на RQ** — direct, numerical. If RQ is rejected, frame it
   constructively (negative result is a finding too).
7. **Аномалии и ограничения** — outliers, broken benchmarks, ceiling
   effects, dedup discrepancies you flagged.
8. **Стоимость и время** — actual vs planned cost, wall, per-cell mean.
9. **Связь со следующими экспериментами** — how this {EX}'s output feeds
   {EX+1}..{EX+N}. Reference concrete artifacts produced.

### Style

- Russian prose, English for code/config/metric names
- Cite concrete numbers (no hand-waving)
- Markdown tables for numerical data
- If pre-computed JSON disagrees with your raw aggregation, FLAG it
  explicitly — list both numbers, propose which one is right
- Length: ~5-6 pages when rendered

### Known issues to flag if observed (DO NOT re-discover)

- **DABench** is broken / extremely hard for current LLM stack — all
  topologies score ~0.2 on it across E1/E2/E3. If you see this on {EX},
  document as limitation, do not interpret as topology-specific.
- **GSM8K** is saturated — most topologies achieve ≥0.9. Low
  discriminative power; flag if relevant.
- **Mesh** is degenerate on programming/text tasks (voting on long
  strings) — high zero-rate on humaneval/commongen.
- **LLM-based meta-routing** (E3 result) has comparable quality to
  rule-based at 1/3 cost, but loses ~9pp on gsm8k specifically.

### DO NOT

- Do not run any grids.
- Do not modify configs.
- Do not commit the analysis file (user reviews first).
- Do not silently average over dedupe duplicates — flag.

### Return value

3-5 sentence summary of key findings when done. Focus on what changed vs
expectations and any methodological caveats discovered.
```

---

## Sanity checklist before launching agent

Перед `Agent({...})` убедись:

- [ ] `{EXP_ID}` корректный UUID — проверь через `ls data/experiments/experiments/`
- [ ] `_runs.parquet` файл реально существует и не пустой (`wc -c`)
- [ ] Известны актуальные `mean_q` / `n` для cross-check (через `head -50` парquetа)
- [ ] Известны какие related artefacts передавать (top3, oracle, results.json от
      других этапов и т.п.)
- [ ] Указан experiment_plan §-номер
- [ ] Pre-flagged known issues добавлены если они применимы

## Что добавлять для конкретных экспериментов

| Экс | Дополнительно передать |
|---|---|
| E1 | top3.json output expected, oracle file generation hint |
| E2 | top3.json как фильтр-входной, e2_report.py для понимания aggregation |
| E3 | e1_top3.json + oracle file + warning про retry duplicates (как в E3 был!) |
| E4 | E3 champion config + E2 best-role-per-task таблица |
| E5 | E4 champion + survey schema + Latin square design |

## Lessons learned (по итогам сессий)

| Bug | Где впервые | Как избежать |
|---|---|---|
| Считал raw rows без status='completed' filter | E3 session | Pre-aggregation hygiene block (выше) |
| Считал raw rows без dedupe → дубликаты от retry | E3 session | Dedupe by canonical key |
| Не передал oracle файл E1 агенту — он нашёл сам, но потратил time | E1 session | Explicit list of artefacts |
| Не предупредил про shuffle_seed теряется в task_id | E1 session | Schema cheat-sheet в промпте |
| Не зафиксировал что DABench broken cross-всем-экспам | E1+E2+E3 sessions | "Known issues" блок |
| Confused "Pareto winner" claim для llm router без dedupe | E3 session | Always dedupe BEFORE aggregating |
