# Synthesis — итоги шести экспериментов

Документ объединяет результаты E1–E4 (original, gpt-oss-120b) и
confirmation_e3 / confirmation_e4 (cross-family validation на qwen-3-235b)
в единый нарратив для дипломной работы. Источники по каждому эксперименту —
отдельные файлы `arch/diploma/results/{e1,e2,e3,e4,confirmation_e3,confirmation_e4}_analysis.md`.

**Дата подготовки:** 2026-05-17.
**Worker для original:** `cerebras:gpt-oss-120b` (OpenAI gpt-oss family).
**Worker для confirmation:** `cerebras:qwen-3-235b-a22b-instruct-2507` (Alibaba Qwen family, cross-family).
**Judge (константа во всех):** `openai:gpt-4.1-mini`.

---

## 1. Что построено

- **Adaptive multi-agent infrastructure** на LangGraph с поддержкой 6 топологий
  (Star, Chain, Mesh, Debate, Hierarchical, Adaptive) и meta-graph для runtime
  switching по фазам plan→exec→verify.
- **3 topology-router'а**: rule-based, LLM-based, oracle (leave-one-out from E1).
- **3 role-router'а**: fixed, rule-based, LLM-based; роли —
  coordinator/reviewer/judge/peer/monitor.
- **HITL gateway** (`LLMSimulatedGateway`) с timeout-fallback policy.
- **SwitchGuards** (anti-thrashing): min_dwell, cooldown, max_per_run/phase.
- **Storage layer**: PostgreSQL + Parquet dual-write, `atm export-exp` для
  cross-machine snapshot'ов.
- **Pipeline**: grid runner (`atm grid run`), budget tracking, judge с
  self-consistency × 3.

**Объём прогонов:**

| Этап | Cells | Status |
|---|---|---|
| E1 full | 900 | 900 completed |
| E2 full | 540 | 540 completed |
| E3 (rule + llm + oracle) | 540 | 540 completed |
| E4 full | 540 | 540 completed |
| confirmation_e3 (rule + llm) | 360 | 360 completed |
| confirmation_e4 | 540 | 540 completed |
| **Итого** | **3420** | **0 fails в финальных грид-прогонах** |

Совокупный бюджет: ~$60 (под планом ~$200). Все ключи API уложились в free /
PAYG лимиты Cerebras и OpenAI.

---

## 2. Original results (gpt-oss-120b)

### 2.1. E1 — best-static-per-task baseline

E1 установил верхнюю границу качества для статических топологий:

| Task | Best topology | mean_q | mean_cost (per run) |
|---|---|---:|---:|
| humaneval | chain  | 0.933 | $0.0101 |
| gsm8k     | star   | 1.000 | $0.0030 |
| commongen | debate | 0.584 | $0.0041 |
| dabench   | debate | 0.348 | $0.0367 |
| **AVG (per-task)** | — | **0.716** | **$0.0135** |

Это **ceiling** для последующих exp'ов: чтобы адаптивная стратегия имела смысл,
она должна обгонять best-per-task.

**Аномалии:** mesh degenerate на humaneval/commongen (voting на длинных
строках); dabench broken (~0.35 даже у winner).

### 2.2. E2 — HITL + role per task

Незначительные приросты от добавления симулированного human-reviewer'а.
Per-task role-winners разные (reviewer на одних, coordinator на других).
Никакой role не выигрывает универсально.

### 2.3. E3 — adaptive topology (RQ2)

**Формальный verdict: RQ2 НЕ supported.** Oracle (**in-sample per-task top-1
lookup из E1**, ранее ошибочно назывался «leave-one-out» — см. e3_analysis §6)
даёт q=0.6975, что **ниже E1 ceiling 0.7164** на 1.9 п.п. —
adaptive обвязка добавляет organizational overhead, который окупается **только
на CommonGen** (+7 п.п. oracle над best-static).

Но **внутри E3 обнаружен Pareto-эффект**:

| Router | mean_q | mean_cost | Pareto-наблюдение |
|---|---:|---:|---|
| rule   | 0.6446 | $0.0236 | baseline |
| llm    | 0.6313 | $0.0099 | **−1.3 пп quality, 2.38× cheaper** |
| oracle | 0.6975 | $0.0227 | upper bound |

**Cost ratio rule:llm = 2.38× стабилен по всем 4 задачам** (диапазон 2.1–3.0×) —
это не шум на одной задаче, а structural pattern. **Llm-router systematically
выбирает дешёвые под-топологии без quality-loss.**

Caveat (важный): vs **E1 best-static** llm-router всё-таки **теряет −8.5 п.п.
quality** за 26% cost economy. То есть Pareto-эффект существует **внутри
adaptive-вариантов**, но vs static-baseline это уже trade-off, не free lunch.

### 2.4. E4 — adaptive role (RQ4)

| role_router | mean_q | mean_cost | verdict |
|---|---:|---:|---|
| fixed (baseline) | 0.6531 | $0.01186 | — |
| rule  | **0.6799** | $0.01193 | direction +0.0268, но **p=0.53 NOT statsig** |
| llm   | 0.6432 | $0.01168 | −0.0099, NOT confirmed |

**Methodological catch:** rule-router фактически collapsed в **98.3% coordinator
(177/180 phases)** — не настоящая "phase-adaptive" работа, а единичный пик.
По сути E4 переоткрыл E2 prior "coordinator > reviewer как single-pick".

**Cost neutrality:** все три mode в диапазоне $0.0117–$0.0119 — gain от
rule (если он есть) "free" по бюджету.

---

## 3. Cross-family validation (qwen-3-235b)

### 3.1. confirmation_e3 — Pareto-нарратив развалился

| Router | gpt-oss mean_q | qwen mean_q | Δ |
|---|---:|---:|---:|
| rule | 0.6446 | **0.866** | +22 пп |
| llm | 0.6313 | 0.560 | −7 пп |

**На qwen rule-router доминирует на 30.6 п.п. quality**, а **cost-ratio rule:llm
схлопнулся с 2.38× до 1.11×.** На DABench rule даже *дешевле* llm. Тезис
"llm-router structurally cheaper" — **gpt-oss-specific артефакт**.

Per-task:
- **GSM8K-регрессия llm усилилась** с −8.9 пп до **−68.9 пп** (llm=0.22 vs rule=0.91)
- **HumanEval**: знак flip'нулся — llm=0.40, rule=1.00 (на gpt-oss было обратное)
- **CommonGen** — единственная задача где llm на qwen ещё выигрывает (по 3/3 cells)
- **DABench solved**: q≈0.92 для обоих роутеров (vs ~0.17 на gpt-oss)

### 3.2. confirmation_e4 — RQ4 REJECTED

| role_router | gpt-oss mean_q | qwen mean_q | gpt-oss delta vs fixed | qwen delta vs fixed |
|---|---:|---:|---:|---:|
| fixed | 0.6531 | 0.582 | — | — |
| rule | 0.6799 | 0.5013 | **+0.0268** | **−0.0807** (sign flip!) |
| llm | 0.6432 | 0.5277 | −0.0099 | −0.0543 (5× усиление) |

**Sign reversal на rule** (Welch p=0.081, ближе к significance чем оригинальный
p=0.53 — но **в противоположную сторону**).

**Role-distribution shift — самая глубокая находка**:

| router | gpt-oss top role | qwen top role |
|---|---|---|
| fixed | reviewer (180) | reviewer (180) |
| **rule** | coordinator 98.3% | **peer 94%** |
| **llm** | coordinator 88% + reviewer 12% | **reviewer 73% + judge 18% + peer 6% + coord 3%** |

Та же `DEFAULT_ROLE_TABLE`, тот же `phase_router=rule` — но output полностью
другой. **Rule-policy НЕ model-invariant** — она зависит от base-model через
phase-classifier'ы, которые делают разные предсказания на разных семьях.

### 3.3. Cross-family novelties

**⚠ Внимание: reasoning_effort confound.** На gpt-oss workers использовался
`reasoning_effort=low` (critic — `high`); qwen API не принимает этот параметр,
поэтому он отключён. Поэтому все per-task cross-family delta ниже —
смесь family-effect и reasoning_effort-effect, и без отдельного ablation
(qwen с reasoning_effort или gpt-oss без) их **нельзя разделить**.

- **DABench на qwen solved** (q=0.93 vs 0.18) → переопределение: не "broken
  benchmark", а **либо gpt-oss-specific weakness, либо reasoning_effort
  artifact** (без ablation нельзя сказать какое из двух).
- **GSM8K на qwen collapses** (q=0.27 vs 0.98) → qwen плох в reasoning
  без reasoning_effort. Это **с большой вероятностью confound**: на gpt-oss
  reasoning_effort на gsm8k давал самый большой прирост в E2.
- **CommonGen стабилен** на обеих семьях (q~0.6) → text generation
  transfers well.
- **HumanEval halved** на qwen (1.0 → 0.36 на best rule cell) → code
  generation harder for qwen без proper reasoning configuration.

---

## 4. Synthesis — central finding

**Family-dependence в adaptive multi-agent routing — единственный structurally
sound вывод этой работы.**

Конкретно:

1. **Adaptive routing strategies не переносимы между LLM families** без
   re-tuning. Оптимум на gpt-oss (llm-router для cost, rule-router для role) —
   неоптимум на qwen (там rule доминирует на topology, никакой role-router не
   beats fixed).
2. **Rule-based policies, выглядящие как "model-independent" tables,
   фактически model-dependent** через свои input classifiers — что демонстрирует
   role-distribution shift (rule выдал coordinator 98% на gpt-oss и peer 94%
   на qwen).
3. **"Saturated/broken benchmark" claim'ы тоже family-dependent.** DABench
   ('broken' на gpt-oss) — solved на qwen. GSM8K (saturated на gpt-oss) —
   collapses на qwen. Любые benchmark-выводы из single-family исследования
   эпистемологически слабы.

### 4.1. Cost-aware Pareto — дизайн-dimension worth studying (но per-family)

На gpt-oss мы наблюдали **structural** Pareto-эффект: llm-router systematic
2.38× cheaper при −1.3 пп quality, **стабильно по 4 задачам**. На qwen этот
эффект коллапсирует. Но **сама гипотеза** что в multi-agent системах
существует exploitable cost/quality Pareto-фронт — рабочая. Она не отменяется
тем, что наш конкретный механизм (llm-router) family-specific. Будущим работам
стоит:

1. Изучать cost/quality Pareto **per family**, не как universal claim
2. Co-designing routers с target family — например, train router на target
   family до deployment
3. Multi-family ensemble routing (если cost-efficient model в family A для
   некоторых задач, switch на model в family B для других)

### 4.2. Дизайн-принцип: family-aware adaptive routing

Из результата следует, что **adaptive multi-agent systems должны проектироваться
под конкретную family моделей**, на которой будут deployment'иться:

- Rule tables, обнаруженные на family X, нельзя перенести на family Y
- Optimal router policies — function of (task, model_family), не только task
- "Universal adaptive system" — over-claim без cross-family evidence

### 4.3. Methodology — cross-family validation должен быть mandatory step

Без confirmation runs мы бы защищали:
- "llm-router выигрывает по cost" — **gpt-oss-specific**
- "adaptive role-routing improves quality" — **family-specific, не replicated**
- "DABench broken" — **wrong, был gpt-oss-specific**

Cross-family validation — не optional bonus к multi-agent работе, а
**обязательный шаг** перед любым claim'ом о generalizability.

---

## 5. Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | n=180 per cell — статистически underpowered | E4 original RQ4 не достигает p<0.05 (p=0.53); confirmation E4 — ближе (p=0.081) но всё ещё borderline |
| 2 | 2 families только (gpt-oss + qwen) | Family-dependence claim был бы сильнее на 3–5 семей |
| 3 | `reasoning_effort` отключён на qwen-preview | Confound — qwen может быть значительно лучше с правильной настройкой, особенно на reasoning tasks (gsm8k, humaneval) |
| 4 | E5 (real-human validation) не выполнено | LLM-simulated HITL не валидирован на настоящих людях |
| 5 | `analysis/e4_report.py:170` typo — `total_cost_usd: 0` | Косметика, per-cell numbers корректны |
| 6 | Oracle на confirmation не запускался | Cross-family upper-bound не измерен (намеренно: oracle table собран на gpt-oss данных, transfer был бы смещён) |
| 7 | E3 rule-router на gpt-oss дал 98% coordinator | Реальная phase-adaptivity не достигнута — фактически single-role pick |
| 8 | Все эксперименты — single judge family (openai gpt-4.1-mini) | Cross-judge-family validation не сделана; potential judge bias |

---

## 6. Future work

1. **Broader cross-family study** — добавить 1–2 семьи (claude, gemini, llama-4)
   чтобы укрепить family-dependence claim или найти universal patterns
2. **Family-aware design experiments** — train router'ы на target family до
   deployment (например, маленький fine-tune на trace'ах с целевой модели)
3. **Resolve reasoning_effort confound** — повторить confirmation на qwen с
   правильной конфигурацией для reasoning tasks
4. **E5 real-human validation** — Latin-square study на 8–12 участниках с
   curated 12-task subset из champion configs
5. **Cost-aware Pareto per-family** — как separate design dimension; explicit
   Pareto-optimization вместо single-objective quality
6. **Statistical power** — n=300–500 per cell для reliable p-values при типичных
   effect sizes в этой области (~0.03 quality delta)
7. **Cross-judge-family validation** — добавить anthropic claude-judge как
   second judge для проверки judge-family bias

---

## 7. Bootstrap-CI verification (added 2026-05-17, post-review)

Все headline-deltas пересчитаны через non-parametric bootstrap (10 000 iters, percentile method, RNG seed=42) — это правильнее Welch t-test при бимодальных q-распределениях (std≈0.4 при mean≈0.65). Сводка:

| Сравнение | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| **E3 rule − llm (gpt-oss)** | +0.013 | [−0.072, +0.097] | **includes 0** |
| **E3 oracle − rule (gpt-oss)** | +0.053 | [−0.033, +0.137] | **includes 0** |
| **E3 oracle − llm (gpt-oss)** | +0.066 | [−0.018, +0.151] | **includes 0** |
| **E3 cost-ratio rule/llm (gpt-oss)** | 2.375× | [1.943, 2.938] | **EXCLUDES 1.0** ✓ |
| **E4 rule − fixed (gpt-oss)** | +0.027 | [−0.055, +0.110] | **includes 0** |
| **E4 llm − fixed (gpt-oss)** | −0.010 | [−0.094, +0.073] | includes 0 |
| **E4 humaneval driver (rule−fixed)** | +0.067 | [−0.022, +0.156] | **includes 0** |
| **conf_E3 rule − llm (qwen)** | +0.307 | [+0.233, +0.378] | **EXCLUDES 0** ✓ |
| **conf_E3 cost-ratio rule/llm (qwen)** | 1.108× | [0.980, 1.254] | **includes 1.0** |
| **conf_E4 rule − fixed (qwen)** | −0.081 | [−0.173, +0.010] | borderline (upper edge near 0) |
| **conf_E4 llm − fixed (qwen)** | −0.054 | [−0.144, +0.035] | includes 0 |

**Что устойчиво.** (a) E3 cost-saving llm-router на gpt-oss (2.4× cheaper, CI ниже 1.94× никогда не падает); (b) conf_E3 family-flip rule>>llm на qwen (Δ+0.31 уверенно отстоит от 0); (c) conf_E4 rule-loss на qwen — почти устойчив (CI верх +0.010).

**Что НЕ устойчиво.** Все quality-delta на gpt-oss: оригинальный E4 «rule wins +0.027» и его humaneval-driver +0.067 — обе CI включают 0; E3 «oracle > rule» и «rule ≈ llm» — тоже. То есть **внутри gpt-oss данные совместимы с нулевым role/topology-router-эффектом по качеству.** Cost-эффект — единственный устойчивый positive finding на gpt-oss.

**Implication для defence claim'ов.**
- «Rule role-router даёт +0.027 quality» → **некорректно** в одиночку; правильно: «direction совпадает на 4/4 задач; magnitude в пределах 95% CI [−0.055, +0.110] совместим с нулём».
- «LLM-router 2.38× cheaper at quality parity» → **корректно** (cost-ratio CI устойчиво, quality CI включает 0 = parity).
- «Family-dependence» — устойчиво для **cost-структуры** (1.11× vs 2.38×) и для **conf_E3 quality-flip** (CI исключает 0). Для **conf_E4 role-router** — borderline, нельзя сделать сильное утверждение.

---

## 8. Defense one-liner (rewritten)

> "Мы построили adaptive multi-agent topology infrastructure и систематически
> протестировали ключевые гипотезы (RQ2: adaptive topology, RQ4: adaptive role)
> на двух семьях LLM. Самый устойчивый positive finding — **cost-saving
> llm-router на gpt-oss (2.38× cheaper, 95% CI [1.94, 2.94]) при quality-parity**.
> Quality-delta role-router'а на gpt-oss оригинального E4 — direction-only
> результат (95% CI delta включает 0). Cross-family валидация на qwen
> показала, что cost-структура llm-router'а не переносится (ratio 1.11×
> вместо 2.38×) и role-policies, выглядящие model-independent, фактически
> зависят от base-model через phase-classifier-ы. Главный вклад —
> single-counterexample к universality adaptive-routing claim'ов и
> methodological argument о mandatory cross-family validation."

---

## Appendix: списочный обзор exp_id'ов

| Эксп | exp_id | Парquet |
|---|---|---|
| E1 full | `f153400f-f8ba-4ca3-8dc2-66f31ba15236` | `data/experiments/experiments/f153400f-.../_runs.parquet` |
| E2 full | `783f4b1e-8888-461e-82fe-07c700f789bc` | `data/experiments/experiments/783f4b1e-.../_runs.parquet` |
| E3 rule | `a03dbe54-db29-4641-a332-d822ef223207` | (тот же шаблон) |
| E3 llm  | `2a0a23a3-ec05-4762-9c59-8f217a6de91b` | |
| E3 oracle | `b147b95d-969e-464e-b2d5-7c5ae1641c78` | |
| E4 full | `272bb2cf-e1c2-4e0d-96c8-432f4e5f9a29` | |
| conf_E3 rule | `6ee94e1f-7b3b-4c99-8066-272e1946a666` | |
| conf_E3 llm  | `e8077e39-e9b5-42dd-98cd-5e307dfb135f` | |
| conf_E4 | `d7f7d1eb-32c0-4315-a8dc-49b65a4d6894` | |

---

## Appendix B: Orphan UUID-директории

В `data/experiments/experiments/` лежат 9 UUID-директорий без `manifest.json` и
без `_runs.parquet` — только `runs/<run_uuid>/{phases,llm_calls,topology_transitions}.parquet`.
Это **pre-instrumentation runs** из периода до миграции на dual-write
PG↔Parquet writer (вероятно ранние пилоты E1/E3/E4). К финальной экспериментальной
сетке (3420 runs в синтезе) они **не относятся** и в анализ не входят:

| UUID (8) | n run subdirs | sample inner files |
|---|---:|---|
| 2345a05a | 12 | phases + llm_calls + topology_transitions (старый full-instrumentation pilot) |
| 2b21f5aa | 10 | то же |
| 309cd330 | 111 | только llm_calls.parquet (early-stage trace) |
| 3da1078e | 27 | только llm_calls.parquet |
| 9caf7924 | 378 | только llm_calls.parquet — крупнейший pre-instrumentation batch |
| b77e7da7 | 12 | только llm_calls.parquet |
| d89e74db | 2 | только llm_calls.parquet |
| e11a2b9d | 30 | full-instrumentation pilot |
| ffa34fdf | 30 | только llm_calls.parquet |

Эти директории следует либо архивировать в `data/experiments/_orphans/`, либо
удалить — они не имеют связанного эксперимента в pipeline и сбивают сканер.
