# E3 confirmation — Cross-family валидация на `cerebras:qwen-3-235b`

**Эксперимент:** `confirmation_e3` — две независимые сетки adaptive-pipeline, по одному `topology_router` каждая, на cross-family worker'е:

| `router_mode` | `exp_id` | name | started |
|---|---|---|---|
| `rule` | `6ee94e1f-7b3b-4c99-8066-272e1946a666` | `confirmation_e3_rule` | 2026-05-17 17:03 UTC |
| `llm`  | `e8077e39-e9b5-42dd-98cd-5e307dfb135f` | `confirmation_e3_llm`  | 2026-05-17 17:51 UTC |

**Запуск:** git `fab69da`, worker / router / critic / planner / executor / researcher / summarizer — все `cerebras:qwen-3-235b-a22b-instruct-2507` (Alibaba Qwen семейство, cross-family vs `cerebras:gpt-oss-120b` оригинального E3). Judge — **тот же** `openai:gpt-4.1-mini`, `self_consistency_n=3`, `temperature=0`. HITL — `llm_simulated` через `openai:gpt-4.1-mini`, фиксированная роль `reviewer` (`role_router=fixed`). Адаптивная обвязка идентична E3: `adaptive` topology, `subgraph_max_iterations=4`, `max_iterations=6`, switch-guards включены (`min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`), `parallelism` — 30 (rule) и 25 (llm).
**Дизайн:** 2 router × 4 task × 15 shuffle × 3 seed = **360 runs** (180 на router). Oracle-режим сознательно пропущен (см. handoff §6, plan §6) — confirmation проверяет *Pareto-нарратив rule vs llm*, а не upper bound.

---

## 1. Постановка confirmation'а

Оригинальный E3 на `cerebras:gpt-oss-120b` сформулировал двухчастный нарратив (см. `arch/diploma/results/e3_analysis.md §6–7`):

1. **Pareto.** `llm`-router даёт качество, сопоставимое с `rule` (Δ = −1.3 п.п., в пределах seed-шума ~5 п.п.) при **в 2.38× меньшей стоимости** ($0.0099 vs $0.0236 per run). Cost-ratio rule/llm удерживается между 2.1× и 3.0× на всех 4 задачах — *структурный* эффект, не one-task-артефакт.
2. **Task-specific regression.** Единственная заметная регрессия `llm` против `rule` — на GSM8K (−8.9 п.п.); на HumanEval `llm` даже обгоняет `rule` (+4.5 п.п.).

Confirmation-эксперимент проверяет, насколько эти выводы воспроизводятся, если поменять *семейство модели* worker'а, удерживая всё остальное (инфраструктура — тот же Cerebras endpoint, judge — тот же `gpt-4.1-mini`, адаптивная обвязка, smart-cuts, judge protocol, сетка). Цель — **decouple «инфраструктура / judge / architecture» от «model family»**: если выводы устойчивы к смене семьи весов worker'а, это аргумент в пользу того, что E3 нарратив описывает свойство *router-стратегий внутри adaptive*, а не идиосинкразию одной модели.

Per template `arch/diploma/AGENT_BRIEFING_TEMPLATE.md §«Особый фокус для confirmation experiments»` вопросы:

- **Direction.** Сохраняется ли знак `rule.mean_q − llm.mean_q > 0`?
- **Magnitude.** Удерживается ли разрыв в пределах ±50 % от исходного (т. е. −0.5…−2 п.п. для `llm − rule`)?
- **Cost-ratio.** Удерживается ли `rule_cost / llm_cost ≈ 2.38×` (как структурный эффект router-стратегии)?
- **Per-task rank-order.** Воспроизводится ли регрессия `llm` именно на GSM8K?
- **Cross-family novelties.** Где qwen ведёт себя качественно иначе, чем gpt-oss, и что это говорит о *task-family-model interactions*?

**Спойлер итога.** Direction знака сохраняется только в смысле «`rule.mean_q ≥ llm.mean_q` overall», но **magnitude взрывается** (−30.6 п.п. вместо −1.3 п.п.), а **cost-ratio коллапсирует** (1.11× вместо 2.38×). На уровне per-task rank-order — оригинальный нарратив *не воспроизводится*: на qwen `llm`-router *катастрофически* проваливается на verifiable tasks (HumanEval, GSM8K), но *обгоняет* `rule` на CommonGen. DABench, наоборот, решается обоими роутерами на 0.92 — то есть E1/E2/E3 наблюдение «DABench broken» оказалось **gpt-oss-specific weakness**, а не свойством бенчмарка. Эти находки переписывают трактовку оригинального E3 (см. §5, §6, §9).

---

## 2. Дизайн и pre-aggregation hygiene

### 2.1 Sweep

| Размерность | Значения | Кардинальность |
|---|---|---:|
| `topology_router` | rule, llm | 2 (oracle пропущен) |
| `topology.name` | adaptive | 1 |
| `task.name` | humaneval, gsm8k, commongen, dabench | 4 |
| `task.shuffle_seed` | 0..14 | 15 |
| `seed` (LLM) | 42, 43, 44 | 3 |
| `human.role` | reviewer (фикс) | 1 |
| **Итого ячеек plan** | | **360** (180 × 2) |

### 2.2 Pre-aggregation hygiene

В отличие от оригинального E3 (где raw rows раздувались в 1.4–2.7× за счёт retry/zombie — см. `e3_analysis.md §2.3`), сырой парquet confirmation'а **чист**: 180 / 180 в каждой из двух сеток, все 360 строк `status=='completed'` и `finish_reason=='success'`, `quality_score` ненулевой везде:

| Router | raw rows | completed & `quality_score.notna()` | retry | failed |
|---|---:|---:|---:|---:|
| rule | 180 | 180 | 0 | 0 |
| llm | 180 | 180 | 0 | 0 |
| **итого** | **360** | **360** | 0 | 0 |

Канонический ключ `(router_mode, topology, task_id, seed, human_role)` даёт 24 группы (2 router × 4 task × 3 seed), и каждая ровно по 15 строк (≡ 15 shuffle-seeds, скрытые от parquet-схемы). То есть дедуп de facto не требуется, но **факт чистоты подтверждён**. Это — отдельный методологический выигрыш по сравнению с оригинальным E3, где «time-rank-15» дедуп пришлось вводить как костыль (см. `e3_analysis.md §9.4`).

### 2.3 Wall-clock и стоимость

Confirmation прогон занял **~30 мин wall-clock на router** (`parallelism=25–30`), против ~3.7–4.5 ч на router в оригинальном E3 (`parallelism=12`). Это сочетание: (а) более высокий parallelism, (б) bigger Cerebras throughput для qwen-235b в момент запуска. Suma cost: **$3.42 (rule) + $3.09 (llm) = $6.50** — что *ниже* оригинального E3 при том, что per-token qwen-235b ≈ 1.7× дороже gpt-oss-120b ($0.60/$1.20 vs $0.35/$0.75 per 1M). Это уже намёк (подтверждённый в §7): qwen на этих задачах генерирует *меньше токенов на run* — короткие, уверенные ответы.

---

## 3. Overall results: qwen vs gpt-oss

### 3.1 Сводка по `router_mode` (n=180 на router)

| Router | n | mean_q | std_q | median_q | mean_cost ($) | sum_cost ($) | mean_wall_s | mean_iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rule (qwen)** | 180 | **0.8664** | 0.237 | 1.000 | 0.0190 | 3.42 | 272.6 | 6.03 |
| **llm  (qwen)** | 180 | 0.5599 | 0.441 | 1.000 | **0.0171** | **3.09** | **218.1** | 6.06 |

Side-by-side с оригинальным E3 (`e3_analysis.md §3.1`):

| Router | qwen mean_q | gpt-oss mean_q | Δq (qwen−gpt) | qwen mean_cost | gpt-oss mean_cost | cost ratio qwen/gpt |
|---|---:|---:|---:|---:|---:|---:|
| **rule** | 0.8664 | 0.6446 | **+0.2218** | $0.0190 | $0.0236 | 0.81× |
| **llm**  | 0.5599 | 0.6313 | **−0.0714** | $0.0171 | $0.0099 | 1.73× |

**Главный «двухстрочный» вывод (negative for original narrative).** Картина роутер-стратегий на qwen качественно иная: `rule` *резко* отрывается от `llm` (Δ = +30.6 п.п. вместо +1.3 п.п. на gpt-oss), а cost-ratio `rule/llm` коллапсирует до **1.11×** (vs 2.38× на gpt-oss). Иными словами, на qwen:

1. Качество `rule` **выросло** на +22.2 п.п. — adaptive с rule-router'ом «понравился» qwen.
2. Качество `llm` **упало** на −7.1 п.п.
3. `llm` уже **не на ~3× дешевле**, а почти равен `rule` по cost ($0.0171 vs $0.0190).

Это **не confirmation** оригинального Pareto-нарратива, а его *opposite reading*: на qwen `rule`-router strictly доминирует `llm` и по качеству, и по wall-time, и почти не уступает по цене. Подробный per-task разбор — §4.

### 3.2 Cost-effectiveness ($/quality-unit)

| Router | qwen | gpt-oss |
|---|---:|---:|
| **rule** | **$0.0219** | $0.0366 |
| llm | $0.0306 | $0.0157 |

На qwen `rule` *Pareto-dominates* по $/q (на 28 % дешевле за единицу качества), тогда как на gpt-oss было ровно наоборот (`llm` был на 57 % дешевле за единицу качества). **Pareto-фронт переехал между семействами моделей.**

### 3.3 Bootstrap-CI (added 2026-05-17, post-review)

Non-parametric bootstrap (10 000 iters, percentile, RNG=42) для side-by-side с E3 (gpt-oss):

| Сравнение | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| **qwen** rule − llm (quality) | **+0.3065** | **[+0.2334, +0.3778]** | **EXCLUDES 0** ✓ |
| **qwen** rule_cost / llm_cost | **1.108×** | **[0.980, 1.254]** | **includes 1.0** |
| (vs E3 gpt-oss) rule − llm | +0.0134 | [−0.0723, +0.0972] | includes 0 |
| (vs E3 gpt-oss) rule_cost / llm_cost | 2.375× | [1.943, 2.938] | EXCLUDES 1.0 |

**Bootstrap решительно подтверждает family-divergence:**
- **Quality-flip устойчив** — на qwen rule>llm с Δ≈0.31 (CI исключает 0); на gpt-oss «rule≈llm» (CI включает 0).
- **Cost-ratio collapse устойчив** — на qwen CI [0.98, 1.25] плотно вокруг 1.0; на gpt-oss CI [1.94, 2.94] никогда не падает ниже 2× cheaper.

Это **самый сильный bootstrap-confirmed cross-family finding** в работе: оба эффекта (quality и cost) переворачиваются с устойчивостью CI.

⚠ **Reasoning_effort confound.** На gpt-oss workers использовался `reasoning_effort=low` (critic — `high`); qwen API не принимает этот параметр, поэтому он отключён. Часть наблюдаемого Δ может быть reasoning_effort-effect, а не чистым family-effect. Особенно подозрительны GSM8K-collapse (llm-router на qwen падает с 0.91 → 0.22) и HumanEval-collapse (0.87 → 0.40) — на gpt-oss reasoning_effort давал наибольший прирост именно на этих задачах. Для строгого family-claim нужен дополнительный ablation: либо gpt-oss без reasoning_effort, либо qwen с reasoning_effort (когда API позволит). DABench-reframing («unbroken на qwen», +0.74 п.п.) тоже нужно читать как «family-effect ИЛИ reasoning_effort-effect», не как чистое family-finding.

---

## 4. Per-task разрез

### 4.1 Mean quality_score: `router × task` на qwen (с qwen−gpt-oss дельтой)

| router \\ task | commongen | dabench | gsm8k | humaneval | mean (4-task) |
|---|---:|---:|---:|---:|---:|
| **rule (qwen)** | 0.6358 | 0.9185 | 0.9111 | **1.0000** | **0.8664** |
| llm  (qwen)     | **0.6914** | **0.9259** | 0.2222 | 0.4000 | 0.5599 |
| rule (gpt-oss)  | 0.5823 | 0.1741 | 1.0000 | 0.8222 | 0.6446 |
| llm  (gpt-oss)  | 0.5769 | 0.1704 | 0.9111 | 0.8667 | 0.6313 |

Жирным — победитель столбца среди двух qwen-роутеров.

**По каждой задаче qwen vs gpt-oss:**

| task | Δ rule (qwen−gpt) | Δ llm (qwen−gpt) | комментарий |
|---|---:|---:|---|
| humaneval | **+0.178** | **−0.467** | rule вышел в потолок 1.0; llm провалился вдвое |
| gsm8k     | −0.089 | **−0.689** | rule почти не пострадал, llm обрушился с 0.91 до 0.22 |
| commongen | +0.054 | +0.114 | оба чуть выше, llm обгоняет rule на qwen |
| dabench   | **+0.744** | **+0.756** | **gpt-oss-specific limitation: «DABench broken» на qwen НЕ воспроизводится** |
| **mean**  | +0.222 | −0.071 | |

### 4.2 Mean cost ($): `router × task` на qwen

| router \\ task | commongen | dabench | gsm8k | humaneval | mean |
|---|---:|---:|---:|---:|---:|
| rule (qwen) | 0.01417 | 0.02463 | 0.01412 | 0.02301 | 0.0190 |
| llm  (qwen) | 0.01251 | 0.02485 | 0.01154 | 0.01964 | 0.0171 |
| rule (gpt-oss, ref) | 0.01359 | 0.05061 | 0.00850 | 0.02155 | 0.0236 |
| llm  (gpt-oss, ref) | 0.00481 | 0.02387 | 0.00381 | 0.00719 | 0.0099 |

### 4.3 Cost-ratio rule/llm — per task (структурная стабильность narrative'а)

| task | qwen rule/llm | gpt-oss rule/llm | сохранилось? |
|---|---:|---:|---|
| commongen | **1.13×** | 2.83× | нет (−60 %) |
| dabench   | **0.99×** | 2.12× | нет (rule даже дешевле llm) |
| gsm8k     | **1.22×** | 2.23× | нет (−45 %) |
| humaneval | **1.17×** | 3.00× | нет (−61 %) |
| **mean**  | **1.11×** | 2.38× | **narrative collapse** |

На gpt-oss `rule/llm` колебался 2.1–3.0× и был *структурным* (см. `e3_analysis.md §7.1`). На qwen он сжимается в 0.99–1.22× и **больше не выглядит как router-стратегия-driven**, а скорее как ~uniform overhead в пределах одной модели. Это означает, что router-strategy *по-разному распределяет под-топологии* в зависимости от семьи модели: на gpt-oss `llm`-router склонялся к «дешёвым» под-топологиям (hierarchical с `max_rounds=2`), на qwen — нет.

### 4.4 Mean wall_s: `router × task`

| router \\ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| rule (qwen) | 188.4 | 200.1 | 202.6 | 499.3 |
| llm  (qwen) | 219.3 | 199.2 | 195.9 | 258.2 |

Интересный качественный эффект: rule-router на HumanEval тратит **499 с** (vs 258 с у llm), но при этом получает 1.0 vs 0.4. Длинный wall ≡ chain/hierarchical с верификацией → правильный код. На gpt-oss было обратное соотношение (см. `e3_analysis.md §4.3`: rule humaneval = 847 с, *тоже* > llm). То есть rule-router *структурно* тратит больше wall на programming-task — это переносится между семействами, и на qwen это окупается, а на gpt-oss — нет.

### 4.5 Zero-rate и one-rate: `router × task` на qwen

| router \\ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| rule one-rate | 0.000 | **0.889** | 0.911 | **1.000** |
| rule zero-rate | 0.000 | 0.044 | 0.089 | 0.000 |
| llm one-rate | 0.000 | **0.911** | 0.222 | 0.400 |
| llm zero-rate | 0.000 | 0.067 | **0.778** | **0.600** |

Распределение действительно бимодальное (как и на gpt-oss). Но на qwen `llm`-router выдаёт *zero* в 78 % cells на GSM8K и 60 % cells на HumanEval — это не «llm чуть слабее», это **системный failure mode** llm-router'а на verifiable tasks для qwen worker'а.

---

## 5. Кросс-роутерное сравнение: paired (task, seed) cells

12 ячеек `(task × seed)`, в каждой 15 shuffle-replications:

| Сравнение | mean Δ rule − llm | rule wins / llm wins / tie |
|---|---:|---|
| Overall (12 cells) | **+0.307** | 7 / 4 / 1 |
| commongen (3 cells) | −0.056 | 0 / 3 / 0 |
| dabench (3 cells) | −0.007 | 1 / 1 / 1 |
| gsm8k (3 cells) | **+0.689** | 3 / 0 / 0 |
| humaneval (3 cells) | **+0.600** | 3 / 0 / 0 |

То есть paired-сравнение даёт два чётких кластера задач:

- **Verifiable / programming (gsm8k + humaneval).** rule безоговорочно доминирует (6/6 cells, Δ +60–69 п.п.).
- **Open-ended (commongen).** llm безоговорочно доминирует (3/3 cells, Δ −5.6 п.п.).
- **DABench.** Шум, ничья.

Сравнение с gpt-oss (`e3_analysis.md §5.1`): там `rule − llm` ≈ +0.013, доминация rule в 7/12. На qwen знак тот же, но *magnitude* на verifiable tasks в **40–50× больше**. Это не «magnitude в пределах ±50 % от исходной»; это качественно новая регрессия llm-router'а, появляющаяся при смене worker'а на qwen.

---

## 6. Replication verdict

| Утверждение оригинального E3 | qwen confirmation | вердикт |
|---|---|---|
| `rule.mean_q ≥ llm.mean_q` overall (direction) | rule 0.866 > llm 0.560 | **direction confirmed** |
| `\|Δq\| ≈ 1.3 п.п.` (magnitude) | `\|Δq\| ≈ 30.6 п.п.` | **magnitude not confirmed** (×23 больше) |
| `cost_ratio rule/llm ≈ 2.38×, стабильно по 4 задачам` | qwen ratio 1.11× (range 0.99–1.22) | **Pareto narrative NOT replicated** |
| `llm Pareto-optimal: −1 п.п. quality за 2.38× cheaper` | на qwen llm: −30 п.п. quality за 1.11× cheaper | **Pareto narrative inverts** — rule доминирует |
| `llm имеет −8.9 п.п. регрессию специально на GSM8K` | qwen llm: −69 п.п. на GSM8K (vs rule) | **regression direction confirmed, magnitude ×8** |
| `llm обгоняет rule на HumanEval (+4.5 п.п.)` | qwen llm: −60 п.п. на HumanEval | **NOT confirmed, sign flipped** |
| `DABench broken: mean_q ~0.17–0.24 у всех router'ов` | qwen DABench: mean_q ~0.92 у обоих | **NOT confirmed — broken был gpt-oss-specific** |

**Сводный вердикт.** Из 7 ключевых утверждений оригинального E3 на qwen-confirmation воспроизводится только одно (overall sign `rule ≥ llm`). Magnitude и Pareto-нарратив — **не воспроизводятся**. Per-task профиль `llm`-router'а — *качественно* другой: на gpt-oss он был «дешёвый и почти не хуже», на qwen — «дешёвый, но в 2× хуже». Per-task ranking меняется.

Это **honest negative result для confirmation**'а — но не «бракованный confirmation», а ценная находка о *family-зависимости* router-стратегий: см. §9.

---

## 7. Cross-family novelties

Confirmation на qwen вскрыл четыре эффекта, которых в оригинальном E3 не было.

### 7.1 DABench «unbroken» на qwen

Самая громкая находка. На gpt-oss-120b DABench была hard-bottleneck: mean_q 0.17–0.24 у всех топологий E1, у всех ролей E2, у всех роутеров E3. В template'е это закреплено как «known issue: DABench is broken / extremely hard for current LLM stack».

На qwen-235b: **mean_q 0.918 у rule, 0.926 у llm**, zero-rate 4–7 %, one-rate 89–91 %. Иными словами, **qwen решает DABench почти бинарно успешно**, а gpt-oss не решает её почти никогда.

**Интерпретация.** Re-read оригинального DABench-нарратива: не «бенчмарк сломан», а **«gpt-oss-120b слаба на структурированных numeric/dataframe-задачах с tool-вызовами»**. Это переопределяет статус DABench из «выкидываем как noise» в «valid discriminator между семьями моделей» — DABench на этом стеке **детектор data-analysis-способностей**, и Alibaba Qwen 235B на нём качественно сильнее.

Практическое следствие для диплома: все цитаты «DABench broken» в `e1/e2/e3_analysis.md §«Аномалии и ограничения»` должны быть переформулированы как «DABench broken on gpt-oss-120b worker; on qwen-3-235b solved at ~0.92».

### 7.2 GSM8K «broken» на qwen для llm-router

Симметричная находка. На gpt-oss-120b GSM8K — saturated benchmark с mean_q ≥ 0.91 у всех роутеров, главная проблема — discriminative power, но не работоспособность. На qwen-235b: `rule` остаётся на 0.911 (≈ gpt-oss), но **`llm` падает до 0.222** с zero-rate 78 %.

Это **не «qwen слабее на математике»** в чистом виде — `rule`-router держит 0.911. Это **«qwen-LLM-router принимает плохие решения на numeric tasks»**: видимо, qwen LLM-router-prompt выбирает менее подходящую под-топологию (вероятно `chain`/`debate` вместо `star`), которая на qwen-worker'е не сходится к правильному ответу. То есть это **interaction effect модель × router-prompt-семантика**, а не модель-as-a-worker эффект.

Это согласуется с пре-флагнутым signal'ом из E4 confirmation: «GSM8K on qwen drops to ~0.2 — qwen is bad at math». Но точнее: **qwen-LLM-router плох в выборе под-топологии для math**, а qwen-worker с *правильно подобранной* под-топологией (как у rule) держит ~0.91.

### 7.3 HumanEval: тот же эффект — rule = 1.0, llm = 0.40

Структурно повторяет GSM8K-effect, но ещё ярче: `rule`-router выводит qwen-worker'а на **идеальное** прохождение HumanEval (45/45 ones), а `llm`-router сваливается на 0.40 (40 % one-rate, 60 % zero-rate). Совокупно с §7.2 это означает: **llm-router-prompt на qwen систематически проваливает verifiable tasks**, а rule-router их *выводит в ceiling*. Это новое явление, отсутствующее в gpt-oss-нарративе.

### 7.4 CommonGen: inversion ranking

Единственная задача, где `llm` на qwen *обгоняет* `rule` (0.691 vs 0.636, 3/3 cells, Δ −5.6 п.п. для rule). На gpt-oss `llm` ≈ `rule` на CommonGen (0.577 vs 0.582, ничья). То есть qwen LLM-router-prompt действительно подбирает *лучшую* под-топологию для open-ended generation, чем rule-tree. Это согласуется с интуицией: rule-tree оптимизирован под определённые task-signals (numeric, programming), и для CommonGen у него нет специфичной ветки, тогда как LLM-prompt может «прочесть» семантическую новизну задачи. Но этот выигрыш на CommonGen — единственный успешный домен для llm-router на qwen, и совершенно недостаточный для компенсации провалов на verifiable tasks.

### 7.5 Сводный паттерн

```
                Verifiable (HumanEval, GSM8K)   Open-ended (CommonGen)   Numeric/tool (DABench)
gpt-oss rule    хорошо (0.82–1.00)              средне (0.58)            плохо (0.17)
gpt-oss llm     хорошо (0.87–0.91)              средне (0.58)            плохо (0.17)
qwen rule       идеально (0.91–1.00)            средне (0.64)            отлично (0.92)
qwen llm        катастрофично (0.22–0.40)       хорошо (0.69)            отлично (0.93)
```

То есть **смена семьи worker'а смещает Pareto-фронт и переопределяет, какие task'и для какого router'а сложные**. Это центральная cross-family находка confirmation'а.

---

## 8. Cost analysis: пере-проверка структурного эффекта

Оригинальный E3 (§7.1) утверждал, что rule-router *структурно* тратит в 2.1–3.0× больше, потому что предпочитает «прожорливые» под-топологии (chain/debate с длинными цепочками). На qwen эта связка ослабевает почти до нуля:

| | gpt-oss | qwen |
|---|---:|---:|
| `rule_mean_cost / llm_mean_cost` overall | 2.38× | **1.11×** |
| per-task range | 2.12–3.00× | 0.99–1.22× |
| `rule_mean_cost` per run | $0.0236 | $0.0190 |
| `llm_mean_cost` per run | $0.0099 | $0.0171 |

Note: qwen-235b per-token ≈ 1.7× *дороже* gpt-oss-120b ($0.60/$1.20 vs $0.35/$0.75 per 1M). Если бы router-decisions были инвариантны к модели и средняя длина ответа одинакова, мы бы ожидали:

- `rule_qwen_cost ≈ rule_gpt_cost × 1.7 ≈ $0.040` (наблюдаем $0.019 — в **2.1× ниже** ожидания) — qwen генерирует *короче*;
- `llm_qwen_cost ≈ llm_gpt_cost × 1.7 ≈ $0.017` (наблюдаем $0.017 — соответствует ожиданию) — llm-router-стратегия на qwen генерирует *столько же токенов*, что и на gpt-oss.

То есть **rule-router на qwen генерирует драматически меньше токенов на run** (на DABench даже rule_cost 0.0246 < llm_cost 0.0249), а llm-router — *столько же*. Возможное объяснение: rule-router выбирает на qwen более «короткие» под-топологии (например, star/hierarchical с малым числом raunds), и qwen на них даёт уверенные короткие ответы; llm-router же на qwen-prompt-driven решениях сваливается в долгие debate-циклы и/или повторные iter'ы (cap=6 достигается у обоих, но при iter==6 запросы могут быть разной длины).

**Главный вывод по cost-структуре.** *«llm-router 2.4× дешевле rule из-за выбора более дешёвых под-топологий»* — было gpt-oss-specific. На qwen эта корреляция исчезает: cost-ratio ≈ 1.1×, и rule-router выходит в Pareto-победители по обоим осям одновременно (выше качество, ниже cost-per-quality-unit).

---

## 9. Аномалии и связь с diploma narrative

### 9.1 Никаких retry-дубликатов

В отличие от оригинального E3 (где raw rows раздувались в 1.4–2.7×, см. `e3_analysis.md §2.3`), confirmation чист: 360 raw rows = 360 completed = 360 после дедупа. Косвенно это — продукт более стабильного запуска (свежий git commit `fab69da`, parallelism подобран под пропускную способность Cerebras endpoint'а). Методологически это укрепляет confidence в числах §3–§8: они не зависят от выбора стратегии дедупа.

### 9.2 Wall-time различия не аномальны

mean_wall 218–273 с на qwen vs 880–1077 с на gpt-oss — это **3–4× ускорение**, и оно объясняется: (а) qwen-235b на Cerebras быстрее inference, (б) parallelism 25–30 vs 12 в оригинальном E3, (в) меньшая длина ответов rule-router'а (см. §8). Это не «качество дешёвое в скорости» — quality_score одного из роутеров (rule) даже **выше** на qwen.

### 9.3 std_q на verifiable tasks высокий именно у llm

`std_q[humaneval, llm] = 0.495` — это бимодальное (40 % единиц, 60 % нулей). У rule на humaneval std=0.000 (все 1.0). Это **разные failure-modes**, не разные «уровни шума»: rule consistently работает, llm consistently выбирает плохо. Не аномалия данных, а *подпись* router-effect'а.

### 9.4 HITL constants

HITL во всех 360 runs срабатывал в роли `reviewer` через `llm_simulated` (`openai:gpt-4.1-mini`), как и в оригинальном E3. То есть HITL-gateway, который мог бы быть confound'ом (если бы qwen reagировал на HITL-feedback иначе), здесь стандартизирован: тот же judge-family даёт тот же reviewer-prompt и в gpt-oss, и в qwen прогоне. Confound'а нет.

### 9.5 Влияние на diploma narrative

Confirmation **переписывает интерпретацию оригинального E3** в трёх местах:

1. **`e3_analysis.md §7 («cost-aware Pareto: основная находка E3»)`** теперь нужно квалифицировать как **gpt-oss-specific**. Тезис «llm — Pareto-optimal точка E3» справедлив только для gpt-oss-120b worker'а; на qwen-235b Pareto-победитель — *rule*. В тексте диплома это формулируется как *«router-strategy Pareto-optimality model-family-dependent»*.
2. **`e3_analysis.md §7.2 («известная регрессия llm на GSM8K»)`** усиливается: direction регрессии воспроизводится на qwen, magnitude — в 8× больше, и распространяется на HumanEval. То есть **«llm-router-prompt систематически плох в выборе под-топологии для verifiable tasks»** — это структурное свойство llm-router-стратегии, наблюдаемое в обеих family'ях; gpt-oss его «прятал» (потому что и rule, и llm одинаково высокого качества на этих задачах), qwen его *обнажил*.
3. **DABench-«broken»**-нарратив (E1/E2/E3 §«known issues») должен быть переформулирован: **не свойство бенчмарка, а свойство gpt-oss-worker'а**. Qwen решает DABench с mean_q ~0.92. Это меняет статус DABench в diploma: больше не «выкинуть из агрегатов как noise», а «valid task; per-model results vary widely».

### 9.6 Влияние на E4 и downstream выбор

Оригинальный E4 на gpt-oss зафиксировал `topology_router=llm` как E3 Pareto-победителя; см. `e4_analysis.md` и `conf/experiments/e4_full.yaml`. Confirmation подтверждает, что **этот выбор справедлив только для gpt-oss worker'а**: на qwen аналогичная конфигурация E4 (`topology_router=llm`) стартует от mean_q 0.560, а не от 0.631, и теряет 30+ п.п. на verifiable tasks. То есть **`confirmation_e4` (где топология-роутер тоже фиксирован как `llm`) тестируется на ёмкой downstream-кривой**: ожидаемая deterioration role-router-эффекта на qwen логически предсказуема из confirmation_e3 результатов.

Для текста диплома: confirmation_e3 — конкретный аргумент в защиту тезиса *«адаптивные топологии MAS обладают семейно-зависимой Pareto-структурой»*, что само по себе становится частью научного вклада, а не «провалом confirmation'а».

---

## 10. Итоги

1. **Direction confirmed, magnitude — нет, Pareto-нарратив — collapsed.** На qwen `rule.mean_q − llm.mean_q = +0.307` (vs +0.013 на gpt-oss); cost-ratio `rule/llm = 1.11×` (vs 2.38×). Pareto-винеер сменился: на qwen *rule* доминирует по обоим осям ($/q = 0.022 vs 0.031).
2. **Per-task ranking перевернулся.** На gpt-oss llm-router был «−1 п.п. на 3 task, −9 п.п. на gsm8k»; на qwen — «−69 п.п. на gsm8k, −60 п.п. на humaneval, −1 п.п. на dabench, +6 п.п. на commongen». То есть **llm-router systematically failed на verifiable tasks на qwen**, и эта регрессия — *усиление* оригинального gsm8k-flaw, распространяющееся на humaneval.
3. **DABench на qwen работает (mean_q ~0.92).** Это переписывает gpt-oss-era нарратив «DABench broken» в «DABench — gpt-oss weakness», что даёт DABench-у статус valid discriminator между семьями моделей.
4. **CommonGen — единственная «правильная» зона llm-router'а на qwen.** Δ −5.6 п.п. в пользу llm (3/3 cells). Open-ended generation — единственная task family, где llm-router-prompt систематически делает лучший выбор под-топологии, чем rule-tree.
5. **Cost-структура router-стратегий не инвариантна к семье модели.** Оригинальный тезис «llm-router выбирает дешёвые под-топологии → стабильное 2.4× cost-saving» работает только для gpt-oss-120b. На qwen-235b cost-ratio коллапсирует, и rule-router генерирует *меньше* токенов на run, чем llm.
6. **Methodological win.** Confirmation-парquet чист: 360 raw rows, 0 retry, 0 zombie, 0 error — никаких «time-rank-15 dedupe» костылей не требуется, в отличие от оригинального E3.
7. **Budget.** Confirmation потратил $6.50 (rule $3.42 + llm $3.09) при wall-clock ~30 мин на router (parallelism 25–30). Per-token qwen-235b в 1.7× дороже gpt-oss, но за счёт более коротких ответов rule абсолютная стоимость сравнима или ниже.
8. **Для diploma narrative.** Confirmation — это **honest negative result для Pareto-claim'а**, но *конструктивный позитивный* для thesis «adaptive topology router-strategies — family-dependent». Original §7 e3 анализа теряет универсальный статус и становится gpt-oss-specific observation; DABench-«broken»-цитаты должны быть переформулированы; rule-router становится strong cross-family default для verifiable tasks при cross-family deployment'е.
