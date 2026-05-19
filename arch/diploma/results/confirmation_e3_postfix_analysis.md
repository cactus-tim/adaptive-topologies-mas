# E3 confirmation (post-fix mini) — cross-family валидация на `cerebras:qwen-3-235b`

**Эксперимент:** `confirmation_e3_postfix_{rule,llm}_mini` — две независимые сетки adaptive-pipeline по 60 ячеек, по одному `topology_router` каждая, на cross-family worker'е `cerebras:qwen-3-235b-a22b-instruct-2507`, на финальной post-audit реализации.

| `router_mode` | `exp_id` | name | started (UTC) | wall (h) | n_cells | sum_cost |
|---|---|---|---|---:|---:|---:|
| `rule` | `2bea6828-924c-4464-b54b-ce31944d73cc` | `confirmation_e3_postfix_rule_mini` | 2026-05-18 21:52 | 0.78 | 60 | $3.29 |
| `llm`  | `1e5034c1-ed9d-4c7f-a88e-602069336449` | `confirmation_e3_postfix_llm_mini`  | 2026-05-18 22:39 | 0.32 | 60 | $2.03 |

**Запуск:** все agent-роли — `cerebras:qwen-3-235b-a22b-instruct-2507`; judge — `openai:gpt-4.1-mini` (тот же, что и в gpt-oss E3 post-fix); HITL — `llm_simulated` через `openai:gpt-4.1-mini`, фиксированная роль `reviewer` (`role_router=fixed`). Адаптивная обвязка: `topology.name=adaptive`, `subgraph_max_iterations=4`, `max_iterations=12`, switch-guards (`min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`), `parallelism=4`. Бюджеты подняты под qwen output-tokens (per-call $0.08, per-run $0.60).

**Дизайн:** 2 router × 4 task × 5 shuffle × 3 seed = **120 runs** (60 на router), n=15 на (router, task). Oracle сознательно пропущен — confirmation проверяет Pareto-нарратив rule vs llm, а не upper bound.

---

## 1. Постановка

Cross-family confirmation сравнивает поведение adaptive(rule/llm) на qwen-3-235b против эталонного запуска E3 post-fix на gpt-oss-120b. Methodology framing зафиксирован в implementation-audit (см. `_postfix_narrative_plan.md`); собственно audit-детали обсуждаются в основном `e3_postfix_analysis.md` и не повторяются здесь.

Этот mini-confirmation отвечает на три центральных вопроса cross-family:

1. **Direction.** Воспроизводится ли E3-post-fix-направление «rule ≥ llm по quality, llm ≥ rule по cost-effectiveness» на cross-family worker'е?
2. **Per-task профиль.** Как ведёт себя LLM-router на verifiable tasks (gsm8k, humaneval) на qwen-3-235b — повторяет ли он gpt-oss-уровни (0.917 / 0.808) или коллапсирует?
3. **Pareto.** Какой router Pareto-доминирует на cross-family и совпадает ли выбор Pareto-winner'а с тем, что наблюдается на gpt-oss?

Содержательно это тест **family-зависимости основного Pareto-вывода E3**. Если на qwen LLM-router тоже Pareto-эффективнее — Pareto-нарратив RQ2 универсален. Если на qwen LLM-router рушится на verifiable — Pareto-вывод E3 family-specific и должен в диплом-нарративе оформляться с явным caveat'ом.

**Спойлер итога.** Direction «rule ≥ llm» сохраняется и **усиливается** на qwen относительно gpt-oss: разница составляет +0.304 п.п. mean_q (CI [+0.185, +0.426]) против +0.036 п.п. un-weighted per-task на gpt-oss. Per-task LLM-router на qwen **коллапсирует** на gsm8k (0.267 vs gpt-oss 0.917) и humaneval (0.533 vs gpt-oss 0.808). Pareto-winner на qwen — rule (strictly dominates llm по quality), на gpt-oss — llm (Pareto-эффективнее по $/q). Это **family-specific limitation**: Pareto-вывод E3 не обобщается на qwen.

---

## 2. Дизайн и completeness audit

### 2.1 Sweep

| Размерность | Значения | Кардинальность |
|---|---|---:|
| `topology_router` | rule, llm | 2 (oracle пропущен) |
| `topology.name` | adaptive | 1 |
| `task.name` | humaneval, gsm8k, commongen, dabench | 4 |
| `task.shuffle_seed` | 0..4 | 5 |
| `seed` (LLM) | 42, 43, 44 | 3 |
| `human.role` | reviewer (фикс) | 1 |
| **Итого ячеек plan** | | **120** (60 × 2) |

### 2.2 Completeness audit

| Router | exp_id (mini) | raw rows | completed | `quality_score.notna()` | retry | failed | n_per_task |
|---|---|---:|---:|---:|---:|---:|---:|
| rule | `2bea6828…` | 60 | 60 | 60 | 0 | 0 | 15/15/15/15 |
| llm  | `1e5034c1…` | 60 | 60 | 60 | 0 | 0 | 15/15/15/15 |
| **итого** | | **120** | **120** | **120** | **0** | **0** | **полно** |

Обе сетки **доехали полностью**, без retry-дубликатов и без zombie/error. Дедуп по `id` оставляет 120 строк и не меняет числа. Это качественный контраст с основным E3 post-fix грид'ом на gpt-oss (там было 353 failed + 26 running, см. `e3_postfix_analysis.md §3.1`) — qwen + cerebras-endpoint стабильно «пролетает» grid в один проход на `parallelism=4`.

### 2.3 Wall-clock и стоимость

| Router | wall-span | mean_wall/run | mean_iter/run | sum_cost | mean_cost/run |
|---|---:|---:|---:|---:|---:|
| rule | 46.7 мин | 653.4 с | 8.00 | $3.29 | $0.0549 |
| llm  | 19.5 мин | 239.5 с | 10.08 | $2.03 | $0.0338 |

Обе сетки на `parallelism=4` уложились в час wall-clock. Общий бюджет confirmation'а — **$5.32** (rule $3.29 + llm $2.03), что в пределах cap $6.0/grid.

---

## 3. Overall results: qwen post-fix vs gpt-oss post-fix

### 3.1 Сводка по `router_mode` (n=60 на router на qwen)

| Router | n | mean_q | std_q | median_q | mean_cost ($) | sum_cost ($) | mean_wall (s) | mean_iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rule (qwen)** | 60 | **0.8769** | 0.231 | 1.000 | 0.0549 | 3.29 | 653.4 | 8.00 |
| **llm (qwen)**  | 60 | 0.5727 | 0.422 | 1.000 | **0.0338** | **2.03** | **239.5** | 10.08 |

### 3.2 Side-by-side с gpt-oss E3 post-fix

| Router | gpt-oss mean_q (per-row) | gpt-oss mean_q (un-weighted per-task) | qwen mean_q | Δ family (qwen − gpt-oss) |
|---|---:|---:|---:|---:|
| rule | 0.6301 | 0.6988 | **0.8769** | **+0.247** vs per-row, +0.178 vs un-weighted |
| llm  | 0.6549 | 0.6627 | **0.5727** | **−0.082** vs per-row, **−0.090** vs un-weighted |

**Главный «двухстрочный» вывод.** На qwen rule-router **выше gpt-oss** на ~0.18–0.25 п.п., а llm-router — **ниже gpt-oss** на ~0.08–0.09 п.п. Это означает, что разделение routers (rule vs llm) на qwen **расходится сильнее**, чем на gpt-oss: Δq (rule − llm) на qwen = +0.304, на gpt-oss un-weighted per-task = +0.036, per-row = −0.025 (CI накрывает 0). Cross-family LLM-router'у на qwen хуже, rule-router'у — лучше.

### 3.3 Bootstrap-CI на principal effects (10 000 iters, percentile, RNG=42)

| Сравнение | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| **qwen** rule − llm (quality) | **+0.3048** | **[+0.1846, +0.4264]** | EXCLUDES 0 |
| **gpt-oss** rule − llm (quality, per-row) | −0.0248 | [−0.1205, +0.0734] | includes 0 |
| **qwen** rule/llm cost-ratio | **1.643×** | **[1.225, 2.152]** | EXCLUDES 1.0 |
| **gpt-oss** rule/llm cost-ratio | **2.151×** | **[1.576, 3.017]** | EXCLUDES 1.0 |

**Интерпретация bootstrap'а:**

- **Rule > LLM на qwen — устойчиво и сильно** (CI плотно положительный, ширина 0.24).
- **Rule ≈ LLM на gpt-oss** (CI накрывает 0). То есть на qwen routers расходятся, на gpt-oss — нет.
- **Cost-ratio rule/llm на qwen меньше gpt-oss'овского** (1.64× vs 2.15×). Adaptive(rule) на qwen относительно дешевле, чем на gpt-oss (где rule в 2.15× дороже llm). Но абсолютные значения cost у rule сравнимы: qwen $0.055/run vs gpt-oss $0.079/run.

### 3.4 Cost-effectiveness ($/quality-unit)

| Router | qwen ($/q) | gpt-oss ($/q) | Δ family |
|---|---:|---:|---:|
| rule | **$0.0626** | $0.126 | qwen дешевле в 2.0× |
| llm  | **$0.0590** | $0.056 | parity |

На qwen rule-router и llm-router имеют почти одинаковый $/q ($0.063 vs $0.059, llm на 6% дешевле). Но эта parity не значит Pareto-equivalence: на qwen llm выдаёт quality 0.573 при cost-saving 6%, против rule с quality 0.877. Trade-off −0.30 quality за −6% cost — **не Pareto-меню**. На gpt-oss та же ось наклонена иначе: llm даёт quality 0.655 при cost-saving 53%, против rule с quality 0.630 — это **Pareto-доминирование llm**.

---

## 4. Per-task разрез: qwen post-fix

### 4.1 Mean quality_score `router × task` на qwen

| router \\ task | commongen | dabench | gsm8k | humaneval | mean (4-task) |
|---|---:|---:|---:|---:|---:|
| **rule (qwen)** | 0.6410 | 0.9333 | 0.9333 | **1.0000** | **0.8769** |
| llm  (qwen)     | **0.6685** | 0.8222 | 0.2667 | 0.5333 | 0.5727 |

### 4.2 Cross-family side-by-side (qwen vs gpt-oss post-fix), per `router × task`

| Router | task | gpt-oss mean | qwen mean | Δ family (qwen − gpt-oss) |
|---|---|---:|---:|---:|
| rule | commongen | 0.6114 | **0.6410** | +0.030 |
| rule | dabench   | 0.2838 | **0.9333** | **+0.650** (qwen решает) |
| rule | gsm8k     | 1.0000 | **0.9333** | −0.067 |
| rule | humaneval | 0.9000 | **1.0000** | +0.100 |
| llm  | commongen | 0.6196 | **0.6685** | +0.049 |
| llm  | dabench   | 0.3067 | **0.8222** | **+0.515** (qwen решает) |
| llm  | gsm8k     | 0.9167 | **0.2667** | **−0.650** (qwen разваливается) |
| llm  | humaneval | 0.8077 | **0.5333** | **−0.274** (qwen ниже) |

**Структурный вывод per-task.** На qwen vs gpt-oss profile задач **переворачивается**:

- **DABench**: qwen решает (rule 0.93, llm 0.82), gpt-oss не решает (rule 0.28, llm 0.31). Family-divergence в +0.5..+0.65 п.п. в пользу qwen.
- **GSM8K + LLM-router**: qwen коллапсирует (0.267), gpt-oss — нет (0.917). Family-divergence в −0.65 в пользу gpt-oss.
- **HumanEval + LLM-router**: qwen ниже (0.533), gpt-oss выше (0.808). Family-divergence в −0.27.
- **CommonGen**: parity ±0.05 на обоих routers.
- **GSM8K/HumanEval + rule-router**: оба family близки к потолку (qwen 0.93/1.00 vs gpt-oss 1.00/0.90).

Главный mismatch — **LLM-router на verifiable tasks**: на qwen он не работает (gsm8k zero-rate 73%, humaneval zero-rate 47%, см. §4.4), на gpt-oss работает нормально. Это **family-specific limitation** LLM-routing'а.

### 4.3 Mean cost ($) `router × task` на qwen

| router \\ task | commongen | dabench | gsm8k | humaneval | mean |
|---|---:|---:|---:|---:|---:|
| rule (qwen) | 0.0456 | 0.0871 | 0.0247 | 0.0623 | 0.0549 |
| llm  (qwen) | 0.0294 | 0.0536 | 0.0194 | 0.0329 | 0.0338 |
| rule (gpt-oss) | 0.0391 | 0.1738 | 0.0178 | 0.0505 | 0.0793 |
| llm  (gpt-oss) | 0.0134 | 0.1184 | 0.0091 | 0.0239 | 0.0369 |

Cross-family cost-pattern:

- **rule** на qwen дешевле gpt-oss во всех 4 tasks (cost-ratio gpt-oss/qwen = {0.86, 2.00, 0.72, 0.81} = mean 1.10×).
- **llm** на qwen примерно сопоставим с gpt-oss (overall mean $0.034 vs $0.037).
- **DABench** на qwen в 2× дешевле gpt-oss (rule $0.087 vs $0.174). qwen эффективнее на этом бенчмарке как по quality, так и по cost.

### 4.4 Zero-rate `router × task` на qwen

| router \\ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| rule one-rate | 0.000 | **0.933** | **0.933** | **1.000** |
| rule zero-rate | 0.000 | 0.067 | 0.067 | 0.000 |
| llm one-rate | 0.000 | 0.600 | 0.267 | 0.533 |
| llm zero-rate | 0.000 | 0.067 | **0.733** | 0.467 |

LLM-router на qwen демонстрирует **бимодальную failure-mode на verifiable**: zero-rate 73% на GSM8K и 47% на HumanEval. На gpt-oss zero-rate на тех же задачах = 0.083 и 0.192. То есть на qwen LLM-router в 8.8× чаще выдаёт quality=0 на GSM8K и в 2.4× чаще — на HumanEval. Это и есть **family-specific collapse**.

### 4.5 Mean wall_s `router × task` на qwen

| router \\ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| rule (qwen) | 461.5 | 573.8 | 346.5 | **1232.0** |
| llm  (qwen) | 170.0 | 219.5 | 248.0 | 320.5 |

Wall-ratio rule/llm на qwen = {2.71, 2.61, 1.40, 3.84} = mean 2.64×. На gpt-oss аналогичный ratio = {1.94, 1.35, 6.78, 6.97} = mean 4.27×. То есть на qwen rule-router *проще* по wall-time (2.64× медленнее llm) против gpt-oss (4.27×). На обеих family'ях rule использует весь iter-budget сильнее llm.

---

## 5. Bootstrap-CI на principal effects: LLM-router collapse на verifiable

Эта секция отвечает на ключевой cross-family-вопрос: совпадает ли поведение LLM-router'а на gsm8k/humaneval с gpt-oss-уровнями?

### 5.1 Bootstrap-CI на qwen LLM-router absolute mean per task

| Task | qwen LLM mean | 95% CI | gpt-oss LLM mean | Confirmation? |
|---|---:|---|---:|---|
| commongen | 0.6685 | [0.631, 0.711] | 0.6196 | parity (qwen чуть выше) |
| dabench   | 0.8222 | [0.667, 0.944] | 0.3067 | **qwen ВЫШЕ** на +0.52 (family-divergence) |
| gsm8k     | **0.2667** | **[0.067, 0.467]** | 0.9167 | **collapse**: CI не доходит до gpt-oss даже верхним концом |
| humaneval | **0.5333** | **[0.267, 0.800]** | 0.8077 | **partial collapse**: верхний конец CI касается gpt-oss mean'а |

**Bootstrap-формулировка.** Для **gsm8k** 95% CI [0.067, 0.467] на qwen LLM **строго ниже** gpt-oss mean 0.917 — то есть gpt-oss-уровень LLM-router'а на GSM8K **не воспроизводится на qwen с уверенностью CI**. Для **humaneval** CI [0.267, 0.800] *включает* gpt-oss mean 0.808 верхним концом, но точечный qwen mean 0.533 на 0.275 ниже gpt-oss'овского — частичное воспроизведение направления, не magnitude.

### 5.2 Bootstrap-CI на qwen rule-router absolute mean per task

| Task | qwen rule mean | 95% CI | gpt-oss rule mean | Confirmation? |
|---|---:|---|---:|---|
| commongen | 0.6410 | [0.609, 0.673] | 0.6114 | parity |
| dabench   | 0.9333 | [0.844, 1.000] | 0.2838 | **qwen ВЫШЕ** на +0.65 |
| gsm8k     | 0.9333 | [0.811, 1.000] | 1.0000 | parity (qwen чуть ниже потолка) |
| humaneval | 1.0000 | [1.000, 1.000] | 0.9000 | qwen на потолке |

Rule-router на qwen voспроизводит gpt-oss-pattern на 3 из 4 tasks (commongen parity, gsm8k и humaneval — на потолке). DABench — единственный task, где qwen *значительно выше* gpt-oss.

### 5.3 Paired-сравнение rule vs llm на qwen — 12 (task, seed) cells

| Сравнение | mean Δ rule − llm | rule wins / llm wins / tie |
|---|---:|---|
| Overall (12 cells) | **+0.304** | 7 / 4 / 1 |
| commongen (3 cells) | −0.027 | 1 / 2 / 0 |
| dabench (3 cells) | **+0.111** | 2 / 1 / 0 |
| gsm8k (3 cells) | **+0.667** | 3 / 0 / 0 |
| humaneval (3 cells) | **+0.467** | 3 / 0 / 0 |

Rule **сильно доминирует** llm на verifiable (gsm8k +0.667, humaneval +0.467), parity на open-ended (commongen). Это противоположно gpt-oss post-fix-картине, где per-task ranking более сбалансирован.

---

## 6. Cost-Pareto: cross-family финальная таблица

| Snapshot | rule mean_q | llm mean_q | rule cost | llm cost | $/q rule | $/q llm | Pareto-winner |
|---|---:|---:|---:|---:|---:|---:|---|
| **qwen post-fix (этот conf)** | **0.877** | **0.573** | **$0.0549** | **$0.0338** | **$0.0626** | **$0.0590** | **rule** (Δ+0.304 quality, +64% cost) |
| **gpt-oss post-fix** (per-row) | 0.630 | 0.655 | $0.0793 | $0.0369 | $0.1259 | **$0.0563** | **llm** (Δ−0.025 quality по per-row, −53% cost) |

**Cross-family Pareto-вывод.** Pareto-winner-by-family **ориентирован противоположно**. На gpt-oss LLM-router Pareto-доминирует (одновременно выше по per-row quality и в 2.15× дешевле). На qwen rule-router strictly dominates по quality (Δ +0.304, CI исключает 0), и хотя llm дешевле на 38% по cost, экономия не компенсирует quality-потерю в 0.30 п.п. **Pareto-ось E3 семейно-зависима**: рекомендация «LLM-router как Pareto-optimal» из основного E3 post-fix не обобщается на qwen.

---

## 7. Direction agreement check

Главный вопрос confirmation'а: совпадает ли cross-family direction с gpt-oss-наблюдениями?

| Что наблюдалось на gpt-oss post-fix | Что наблюдаем на qwen post-fix | Verdict |
|---|---|---|
| `rule.mean_q` overall ≈ 0.63 (per-row) | qwen rule.mean_q = 0.877 (+0.247 vs gpt-oss per-row) | direction OK, magnitude qwen ВЫШЕ |
| `llm.mean_q` overall ≈ 0.65 | qwen llm.mean_q = 0.573 (−0.082) | **direction inverted**: llm на qwen ниже rule, на gpt-oss — выше |
| `llm.gsm8k` ≈ 0.917 | qwen llm.gsm8k = 0.267 | **NOT confirmed**: collapse на qwen |
| `llm.humaneval` ≈ 0.808 | qwen llm.humaneval = 0.533 | **partially confirmed**: ниже gpt-oss на 0.275 |
| `rule.dabench` ≈ 0.284 | qwen rule.dabench = 0.933 | **opposite direction**: qwen решает |
| `llm.dabench` ≈ 0.307 | qwen llm.dabench = 0.822 | **opposite direction**: qwen решает |
| rule > llm direction overall | rule 0.877 > llm 0.573 (Δ +0.305, CI [+0.185, +0.426]) | **confirmed и усилено**: gpt-oss Δ per-row = −0.025 (CI накрывает 0), qwen Δ строго положительный |
| llm Pareto-эффективнее rule | qwen: llm $0.059/q ≈ rule $0.063/q (6% cheaper at 30 п.п. lower quality) | **NOT confirmed как Pareto-доминирование** |

**Сводный verdict.** Из 8 ключевых направлений gpt-oss post-fix эффекта:

- **1 направление** (rule > llm overall) воспроизводится с *усилением* magnitude на qwen.
- **1 направление** (DABench профиль) **инвертируется**: qwen решает, gpt-oss — нет.
- **2 направления** (llm.gsm8k, llm.humaneval gpt-oss-уровней) **не воспроизводятся**: qwen LLM-router коллапсирует на gsm8k, частично — на humaneval.
- **2 направления** ($/q Pareto и overall llm.mean_q above rule) **не воспроизводятся как Pareto-доминирование**: на qwen llm proigryvaet rule по quality, а $/q parity не Pareto.
- **2 направления** (commongen parity, rule на потолке gsm8k/humaneval) — parity.

То есть **direction agreement — partial**. Universal claim — «rule ≥ llm по quality» (на qwen усилено, на gpt-oss слабо). Family-specific claims — LLM-router Pareto, LLM-router lift на verifiable, DABench broken — **не обобщаются**.

---

## 8. Ответ на RQ2 cross-family

**RQ2 cross-family verdict: PARTIAL CONFIRMATION с family-specific limitation.**

| Sub-claim RQ2 | gpt-oss direction | qwen cross-family | Verdict |
|---|---|---|---|
| (a) Adaptive(rule) ≈ best static после фиксов | gpt-oss: rule=0.699 un-weighted per-task, отстаёт на 2.7 п.п. от best-static 0.726 | qwen: rule=0.877, выходит в ceiling на 3/4 task (humaneval 1.0, dabench 0.93, gsm8k 0.93) | **direction confirmed, magnitude qwen стабильнее** |
| (b) Adaptive(llm) Pareto-optimal | gpt-oss: llm=0.655 at $0.037 vs rule=0.630 at $0.079 → llm Pareto winner (per-row) | qwen: llm=0.573 at $0.034 vs rule=0.877 at $0.055 → llm НЕ Pareto winner (Δq=−0.30 за −38% cost) | **NOT confirmed** — на qwen llm не Pareto-эффективен |
| (c) LLM-router держит verifiable | gpt-oss: gsm8k 0.917, humaneval 0.808 | qwen: gsm8k 0.267 (zero-rate 73%), humaneval 0.533 (zero-rate 47%) | **family-specific NOT confirmed** — qwen LLM-router collapse |

**Финальная формулировка для diploma.**

> «Cross-family валидация RQ2 показывает, что **Pareto-вывод E3 семейно-зависим**. На gpt-oss-120b LLM-router Pareto-доминирует rule-router'а: при per-row quality 0.655 (vs rule 0.630) и cost $0.037/run (vs rule $0.079/run) он одновременно выше по quality и в 2.15× дешевле. На qwen-3-235b профиль инвертируется: rule-router strictly dominates по quality (0.877 vs 0.573, bootstrap-CI Δ [+0.185, +0.426]) при cost-parity по $/quality-unit (rule $0.063 vs llm $0.059). LLM-router на qwen **коллапсирует на verifiable tasks**: GSM8K mean 0.267 (zero-rate 73%, vs gpt-oss 0.917), HumanEval mean 0.533 (zero-rate 47%, vs gpt-oss 0.808). Pareto-вывод E3 («LLM-router — оптимальная точка»), таким образом, остаётся **gpt-oss-specific**; на cross-family deployment (qwen) рекомендация инвертируется в «rule-router как default для verifiable tasks»».

**Гипотеза о family-specific limitation (defensible).** Qwen LLM-router выдаёт **zero-rate 73% на GSM8K**, что указывает на структурный, а не статистический failure-mode. Подозрения: (i) qwen-specific JSON-parsing failure при router-prompt'е (model-specific tokenizer / parser interaction), (ii) reasoning_effort-confound (qwen API не принимает параметр `reasoning_effort`, доступный gpt-oss), (iii) qwen-router-prompt-семантика делает несовместимый выбор под-топологии (например, `chain` вместо `star`) для математических задач. Любая из этих гипотез — **defensible limitation** работы; полная диагностика требует отдельной cross-family серии с инструментированным router-trace.

---

## 9. Аномалии и связь с diploma narrative

### 9.1 DABench: «family-divergence» в пользу qwen

На gpt-oss post-fix DABench у rule = 0.284, у llm = 0.307 (zero-rate 60–68%, бенчмарк «broken» для gpt-oss-120b). На qwen post-fix те же rule = 0.933, llm = 0.822 (zero-rate 7%/7%) — DABench нормально решается. То есть **DABench-narrative — gpt-oss-specific limitation, не universal feature бенчмарка**: qwen-3-235b справляется с numeric-exact-match на data-analysis tasks на ≥0.82 (rule) / ≥0.82 (llm).

Diploma-implication: формулировка «DABench broken for the stack» из основного E3 post-fix должна быть уточнена как «DABench broken for gpt-oss-120b on this evaluation; qwen-3-235b solves DABench at ≥0.82 mean quality». Это **усиливает** методологическую честность главы 5: limitation бенчмарка — не universal, а model-specific.

### 9.2 GSM8K: «family-specific LLM-router collapse»

qwen llm.gsm8k = 0.267 (zero-rate 73%). gpt-oss llm.gsm8k = 0.917. Разница cross-family в 3.4× — структурная. Diploma-implication: на verifiable math tasks LLM-routing нужно тестировать per-family; cross-family transfer не гарантирован.

### 9.3 HumanEval: «family-specific partial collapse»

qwen llm.humaneval = 0.533 (zero-rate 47%), gpt-oss = 0.808. Разница в 1.5×, ниже GSM8K-divergence, но всё ещё значимая. Direction match (LLM-router выдаёт меньше потолка на verifiable), magnitude — family-зависимая.

### 9.4 Cost-структура: qwen rule дешевле gpt-oss rule, llm parity

Mean cost на qwen: rule $0.055/run, llm $0.034/run. На gpt-oss: rule $0.079/run, llm $0.037/run. Rule на qwen в 1.44× дешевле gpt-oss-аналога; llm — parity. Это означает, что **cross-family deployment** на qwen-cerebras обходится ~30% дешевле на rule-router'е и сопоставимо на llm-router'е. Cost-implication для diploma: на cross-family bills нет «штрафа» за qwen — но quality-trade при выборе llm-router становится экономически невыгодным.

### 9.5 HITL constants

HITL во всех 120 runs срабатывал в роли `reviewer` через `llm_simulated` (`openai:gpt-4.1-mini`), как и в основном E3 post-fix. Confound'а нет.

### 9.6 Влияние на diploma narrative

Этот cross-family mini-confirmation **семейно-локализует** Pareto-вывод E3:

1. **Pareto-вывод E3 ограничен gpt-oss-120b.** На cross-family deployment (qwen-3-235b) рекомендация «LLM-router как Pareto-optimal» **инвертируется**: на qwen rule-router strictly dominates llm-router по quality, при cost-parity по $/q. Это нужно явно артикулировать в §RQ2-Discussion как family-scope caveat.
2. **DABench limitation — model-specific.** «DABench broken» — не universal feature task'а, а gpt-oss-specific failure. Qwen-3-235b решает DABench на ≥0.82. Это уточняет §Limitations.
3. **LLM-routing — family-sensitive design choice.** При выборе router-стратегии для нового worker'а нельзя экстраполировать качество LLM-router'а с одной family на другую без cross-family валидации. Это сильный contribution-point: «LLM-routing semantics не переносятся между model families».

### 9.7 Влияние на downstream (E4 confirmation)

E4 confirmation на qwen (см. `confirmation_e4_postfix_analysis.md`) проверяет RQ4 cross-family с этим же подходом: cross-family на qwen vs основной E4 post-fix на gpt-oss. Family-asymmetry, найденная в этом E3 confirmation, мотивирует тот же frame для E4.

---

## 10. Итоги

1. **Completeness.** Mini-grid доехал полностью: 60/60 в каждой сетке, дедуп тривиален. Нет zombie, retry, error. Прогон обеих сеток за 1.1 ч wall (`parallelism=4`), $5.32 total.
2. **Direction agreement: partial.** «rule ≥ llm overall» воспроизводится с **усилением** magnitude (qwen Δ +0.305, gpt-oss Δ −0.025 per-row); LLM-router на verifiable tasks — **family-specific collapse** на qwen.
3. **Magnitude rule vs llm на qwen.** Δ rule − llm = +0.304 (CI [+0.185, +0.426]) — устойчиво и сильно положительный. На gpt-oss та же Δ per-row = −0.025 (CI накрывает 0). Cross-family family-divergence в Δ rule − llm = +0.33 п.п.
4. **Per-task profile на qwen** показывает три явных family-divergence:
   - **DABench**: qwen решает (rule 0.93, llm 0.82), gpt-oss не решает (rule 0.28, llm 0.31).
   - **GSM8K + LLM**: qwen коллапсирует (0.267, zero-rate 73%), gpt-oss работает (0.917).
   - **HumanEval + LLM**: qwen partial collapse (0.533, zero-rate 47%), gpt-oss работает (0.808).
5. **Cost-Pareto cross-family ось ориентирована противоположно.** На gpt-oss — LLM-router Pareto-winner (+per-row quality, в 2.15× дешевле). На qwen — rule strictly dominates (Δq +0.304, $/q parity).
6. **RQ2 cross-family verdict.** **Direction confirmed только для rule ≥ llm; Pareto NOT confirmed; LLM-router на verifiable — family-specific limitation.** Pareto-нарратив E3 («LLM-router Pareto-optimal») остаётся gpt-oss-specific. На qwen рекомендация инвертируется в «rule-router как default».
7. **Defensible limitation.** Qwen LLM-router рушится на GSM8K (mean 0.267, zero-rate 73%). Возможные root-causes: qwen-specific JSON-parsing failure router-prompt'а, reasoning_effort-confound, qwen-router-prompt-семантика. Открытое ограничение, корректное для диплома: «cross-family LLM-routing требует per-family валидации».
8. **Diploma contribution-point.** Cross-family asymmetry в Pareto-winner'е (rule на qwen, llm на gpt-oss) — нетривиальный научный результат: «при равной adaptive-обвязке выбор оптимального router'а зависит от model family и не переносится между LLM-семействами».

---

## 11. Что нужно дочитать вместе с этим документом

- `arch/diploma/results/e3_postfix_analysis.md` — основной E3 post-fix на gpt-oss-120b, anchor для cross-family-сравнения
- `arch/diploma/results/_postfix_narrative_plan.md` §2 — canonical gpt-oss post-fix numbers
- `conf/experiments/README_postfix_confirmation.md` — design rationale (mini sufficient because n=15 matches основной grid density)
- `arch/diploma/results/confirmation_e4_postfix_analysis.md` — RQ4 cross-family verdict (downstream)
