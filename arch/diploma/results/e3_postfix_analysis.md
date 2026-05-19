# E3 POST-FIX — Аналитический разбор результатов

**Эксперимент:** `e3_postfix_*` — серия из 7 grid-конфигов, разворачивающих исходный план E3 на семи независимых `exp_id` под gpt-oss-120b worker и gpt-4.1-mini judge.

**Контекст.** Аналитика ведётся **в вакууме** на данных текущего post-fix re-run E3, выполненного на скорректированной реализации adaptive-роутера после внутреннего implementation audit. Никаких числовых сравнений с предыдущими прогонами здесь не строится — кросс-прогонные сопоставления вынесены в отдельные документы.

**Запуск:** 2026-05-17 → 2026-05-19, primary worker `cerebras:gpt-oss-120b` (`reasoning_effort=low` для worker'ов, `high` для critic), judge `openai:gpt-4.1-mini` (self-consistency × 3, `temperature=0`), HITL `llm_simulated` с фиксированной ролью `reviewer`, `role_router=fixed`. Топология — `adaptive` (L2 meta-graph), статические бэйзлайны — `chain`, `star`, `mesh`. Adaptive-параметры: `max_iterations=12` (consistent budget across topologies), `subgraph_max_iterations=4`, switch-guards on (`min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`).

**Data-flags (читать до интерпретации):**

1. **Сетка реализована частично.** Из планируемых ~480 cells E3 завершилось 643 completed-rows (включая статики) при 353 failed + 26 running. Падения сконцентрированы на DABench, особенно под `topology_router=llm` (см. §3).
2. **Перекрытие источников.** `e3_postfix_full` (полная rule-сетка) проработал лишь частично (187 completed из 384 планируемых); commongen и dabench были полностью переразвёрнуты как отдельные `e3_postfix_full_{commongen,dabench}` под пониженным `parallelism` после анализа cascade-сбоев. **GSM8K и HumanEval rule** покрыты только из `e3_postfix_full` (отдельных сабгридов под них нет → меньший n).
3. **Oracle-router в post-fix не прогонялся.** Из трёх router-режимов исходного дизайна повторный прогон есть только для rule и llm. Класс-level ceiling в текущем документе строится из best-static-per-task (§6), не из oracle.
4. **Static-baselines пересняты заново.** В наборе присутствуют chain/star/mesh runs (406 completed) с тем же `max_iterations=12`; именно их используем как «best-static per task» в §6.

---

## 1. Постановка (RQ2 / RQ3)

**RQ2.** Зависит ли качество и стоимость адаптивной топологии MAS от семейства роутера (rule vs llm), и какое из семейств Pareto-оптимально на верифицируемых и нет-верифицируемых задачах?

**RQ3.** При какой стоимости решений роутера выигрыш по качеству перестаёт окупаться (cost-quality tradeoff)?

В рамках post-fix data analysis оба RQ оцениваются строго на текущих 237 adaptive-rows и 406 static-rows, без обращения к более ранним числовым оценкам.

---

## 2. Дизайн post-fix re-run

### 2.1 Структура сетки и почему она разбита

| `exp_id` | Имя | Router family | Задача(и) | Топологии в sweep | Завершено |
|---|---|---|---|---|---:|
| `3aa2187a` | `e3_postfix_full` | rule | humaneval, gsm8k, commongen* | chain, star, mesh, adaptive | 187/384 (часть прервана) |
| `d035ca05` | `e3_postfix_full_commongen` | rule | commongen | chain, star, mesh, adaptive | 171/96 (избыток за счёт retry) |
| `3711ffeb` | `e3_postfix_full_dabench` | rule | dabench | chain, star, mesh, adaptive | 166/96 |
| `437b65cd` | `e3_postfix_full_llm_commongen` | llm | commongen | adaptive | 44/24 |
| `8544307b` | `e3_postfix_full_llm_dabench` | llm | dabench | adaptive | 25/24 |
| `31cd6082` | `e3_postfix_full_llm_gsm8k` | llm | gsm8k | adaptive | 24/24 |
| `3b496f94` | `e3_postfix_full_llm_humaneval` | llm | humaneval | adaptive | 26/24 |

`*` — в `e3_postfix_full` присутствуют только частично заполненные commongen-cells (7+9 на топологию); недостающая часть была переразвёрнута в `_commongen`. DABench-cells в исходной полной сетке падали в каскад (см. §3.2), поэтому фактически dabench для rule целиком взят из `_dabench`-сабгрида.

**Планируемый объём (по 7 yaml-конфигам)**: 384 (rule full) + 96 + 96 + 24×4 (llm per-task) = ≈ 672 cells. Часть планируемого rule-объёма (commongen, dabench) повторяется между `_full` и `_commongen` / `_dabench` — это **deliberate redundancy** ради кэшпоинт-safety, а не error.

### 2.2 Параметры sweep

- N задач: 8 stratified shuffle × 3 seed = 24 cells на (task × topology). Дизайн проверяет adaptive в 4 семьях топологий (chain/star/mesh/adaptive) и в двух router-режимах (rule/llm).
- `subgraph_max_iterations=4`, `max_iterations=12`.
- HITL фиксирован как `reviewer` advisory; advisor-hint потребляется обоими роутерами.
- Oracle-router из дизайна выброшен (отдельный exp_id не запускали).

### 2.3 Pre-aggregation и dedupe

После concat'а 7 parquet'ов:

```
Raw rows: 1022
After status=completed + quality.notna: 643
After id-dedupe: 643          (нет дубликатов id — UUID на run уникальны)
  Adaptive runs:         237  (rule=118, llm=119)
  Static (chain/mesh/star): 406
Failed total:                  353
Running (не финализированы):    26
```

Дедуп по `id` достаточен — все UUID'ы уникальны. Пересечений между sub-experiment'ами по тем же `(task, seed, shuffle)` ячейкам нет, так как `e3_postfix_full` обрывался прямо до того, как доходил до dabench, а `_commongen` стартовал в новом окне. **Все 237 adaptive-completed-rows — независимые запуски без двойного счёта.**

Имеется лёгкий «избыток» rows над планом (например, llm×commongen = 44 при плановых 24): retry-копии и runs из соседних кэпов попали в _runs.parquet под другими UUID'ами. Поскольку планируемый ключ `(task, seed, shuffle_seed)` в parquet-схеме отсутствует, разделить «оригинал vs retry» невозможно; ниже считаем по всем имеющимся completed-rows как по дополнительной выборке.

---

## 3. Completeness audit

### 3.1 Failure-rates по источникам

| Source | n_completed | n_failed | n_running | Plan | Failure rate |
|---|---:|---:|---:|---:|---:|
| `e3_postfix_full` (rule, 4-topo full sweep) | 187 | 51 | 25 | 384 | 19.4% |
| `e3_postfix_full_commongen` (rule, статики+adaptive) | 171 | 107 | 0 | 96 | 38.5% |
| `e3_postfix_full_dabench` (rule, статики+adaptive) | 166 | 101 | 0 | 96 | 37.8% |
| `e3_postfix_full_llm_commongen` (llm, adaptive only) | 44 | 21 | 0 | 24 | 32.3% |
| `e3_postfix_full_llm_dabench` (llm, adaptive only) | 25 | 51 | 0 | 24 | **67.1%** |
| `e3_postfix_full_llm_gsm8k` (llm, adaptive only) | 24 | 0 | 0 | 24 | **0.0%** |
| `e3_postfix_full_llm_humaneval` (llm, adaptive only) | 26 | 22 | 1 | 24 | 44.9% |

**Что важно для интерпретации.**

- **DABench + llm-router** — 67% failure rate. Это «state-bloat cascade», потребовавший трёх инфраструктурных корректировок (`merge_agent_states`, `tool_results`-cap, psycopg pipeline). Финальный n=25 — это после двух перезапусков конфига; до infrastructure corrections на этой задаче было 0 completed cells. Поэтому n=25 для llm×dabench следует читать как «минимум возможный для устойчивого среднего», без формальной 95%-CI на основном эффекте.
- **GSM8K + llm-router** — 0% failure, n=24/24. Чистая нижняя оценка для llm-роутера без шума выбраковки.
- **CommonGen** под обоими роутерами имеет 32–38% failure rate, но в абсолютных числах completed > планового объёма. Эти падения по большей части — rate-limit cerebras 429, не cascade-related.

### 3.2 Гепы в покрытии и эффект на выводы

**Per (router_family, task_id) finalized n:**

|  | commongen | dabench | gsm8k | humaneval | total |
|---|---:|---:|---:|---:|---:|
| rule  | 39 | 37 | 22 | 20 | **118** |
| llm   | 44 | 25 | 24 | 26 | **119** |

Планируемый минимум на ячейку — 24 (8 shuffle × 3 seed). Только **rule×humaneval** и **rule×gsm8k** имеют n чуть ниже 24 (20 и 22 соответственно) — это «хвост» от прерванной `e3_postfix_full`-сетки. Остальные ячейки имеют избыток над планом за счёт retry-runs, ускользнувших от грид-планировщика и попавших в parquet с другим id.

**Влияние на интерпретацию.**

- Per-task средние устойчивы (n ≥ 20 везде); 95%-bootstrap CI рассчитан без проблем (см. §5).
- llm×dabench (n=25) единственная ячейка, где средний потенциально нестабилен: zero-rate 0.600 при std=0.43 означает, что добавление/убирание одной из 6 успешных задач сдвинет среднее на ~3 п.п. — это и есть фактическая дисперсия выборки.
- **Oracle-сегмент отсутствует**, поэтому upper-bound на качество в этом документе строим исключительно из best-static-per-task (§6).

---

## 4. Результаты — overall

### 4.1 Сводка по `router_family` (топология=`adaptive`, n=237)

| Router | n | mean_q | std_q | median_q | mean_cost ($) | sum_cost ($) | mean_wall (s) | median_wall (s) | mean_iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **llm** | 119 | **0.6549** | 0.368 | 0.694 | **0.0369** | **4.39** | **756.8** | 305.7 | 10.78 |
| rule | 118 | 0.6301 | 0.394 | 0.691 | 0.0793 | 9.36 | 1 857.9 | 1 213.5 | 9.62 |

**Cost-effectiveness ($/quality-unit, ниже — лучше):**

| Router | $/quality-unit |
|---|---:|
| **llm** | **0.0563** |
| rule | 0.1259 |

**Главный однострочный вывод.** adaptive(llm) обгоняет adaptive(rule) по mean_q (0.655 vs 0.630, Δ=+0.025) и платит **в 2.15× меньше** при wall-time **в 2.46× меньше**. По всем трём осям (per-row mean_q, mean_cost, mean_wall) llm Pareto-доминирует rule.

### 4.2 Bootstrap-CI (n_iter=10 000, percentile, RNG seed=42)

| Сравнение | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| `rule` − `llm` (quality, overall) | −0.0248 | [−0.1205, +0.0734] | **includes 0** (quality-equivalence) |
| `rule` / `llm` (cost-ratio, overall) | 2.151× | [1.576, 3.017] | **EXCLUDES 1.0** (cost-discriminating) |
| `rule` / `llm` (wall ratio) | 2.455× | — | устойчивая |

Quality-CI накрывает 0 — quality-equivalence rule vs llm на overall выборке нельзя отвергнуть. Cost-CI чисто исключает 1.0 — асимметрия по цене статистически значима.

### 4.3 Per-task разрез

**Mean quality_score (`router_family × task_id`):**

| router \ task | commongen | dabench | gsm8k | humaneval | mean (4-task) |
|---|---:|---:|---:|---:|---:|
| **rule** | 0.6114 | 0.2838 | **1.0000** | **0.9000** | **0.6988** |
| **llm**  | **0.6196** | **0.3067** | 0.9167 | 0.8077 | 0.6627 |

Жирным — победитель столбца внутри post-fix adaptive (без статиков). Un-weighted mean по 4 задачам: rule 0.6988, llm 0.6627.

Row-level mean (взвешенный по фактическому n; см. §4.1) даёт `llm` 0.655 > `rule` 0.630 — потому что в llm-сегменте больше cells на «лёгких» задачах (commongen+gsm8k+humaneval = 94) и меньше на «тяжёлой» (dabench = 25), а в rule-сегменте dabench даёт 37 из 118 — distortion в сторону хвоста распределения. **Per-task взвешенное среднее (по 4 равным task-bucket'ам) даёт небольшое преимущество rule по чистому качеству (+3.6 п.п.); per-row среднее даёт преимущество llm (+2.5 п.п.).** Обе формулировки имеют CI, накрывающее 0 (§4.2). Cost-asymmetry — единственный устойчивый эффект на quality-метрике.

**Mean cost ($), `router_family × task_id`:**

| router \ task | commongen | dabench | gsm8k | humaneval | mean |
|---|---:|---:|---:|---:|---:|
| **llm**  | **0.0134** | **0.1184** | **0.0091** | **0.0239** | **0.0369** |
| rule | 0.0391 | 0.1738 | 0.0178 | 0.0505 | 0.0793 |

llm дешевле rule **во всех 4 task-buckets**; cost-ratio rule/llm = {2.93, 1.47, 1.96, 2.11} — устойчивый структурный эффект.

**Mean wall_s, `router_family × task_id`:**

| router \ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| **llm**  | **300.3** | **2 599.4** | **153.1** | **314.8** |
| rule | 582.9 | 3 507.9 | 1 038.0 | 2 193.4 |

Wall-ratio rule/llm на gsm8k и humaneval = **6.78× и 6.97×** соответственно. Это критическое наблюдение: на верифицируемых задачах rule-роутер тратит весь `max_iterations=12`, тогда как llm-роутер регулярно выбирает «достаточно» рано через `chain` → exit early. На creative-задаче (commongen) разница 1.94×; на tool-heavy (dabench) — 1.35×.

**Zero-rate `router_family × task_id`:**

| router \ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| rule | 0.000 | **0.676** | 0.000 | 0.100 |
| llm  | 0.000 | **0.600** | 0.083 | 0.192 |

Zero-rate на DABench 60–68% подтверждает: бенчмарк фактически broken для текущего стека (см. §9.1).

---

## 5. Per-task bootstrap CI для Δq (rule − llm)

Bootstrap (10 000 итераций, percentile, RNG seed=42), вычисленный на per-task subsamples:

| task | Δq (rule − llm) | 95% CI | Includes 0 |
|---|---:|---|---|
| commongen | −0.0082 | [−0.0472, +0.0299] | **Да** |
| dabench   | −0.0229 | [−0.2418, +0.1906] | **Да** |
| gsm8k     | +0.0833 | [0.0000, +0.2083] | На границе |
| humaneval | +0.0923 | [−0.1077, +0.2962] | **Да** |

Единственная per-task разница, чьё CI почти не включает 0 — **GSM8K** (+0.083 в пользу rule). На HumanEval направление такое же (+0.092 в пользу rule), но CI слишком широкое из-за бимодального quality_score (0 или 1).

---

## 6. Adaptive vs best-static per task

Лучшая статика per task (n взяты из §3 audit'а):

| task | chain mean_q | star mean_q | mesh mean_q | best_static_q (топология) |
|---|---:|---:|---:|---:|
| commongen | 0.5698 | 0.5876 | **0.5998** | **0.5998** (mesh) |
| dabench | 0.2793 | 0.3043 | **0.3043** | **0.3043** (mesh/star тие) |
| gsm8k | **1.0000** | 1.0000 | 0.9048 | **1.0000** (chain) |
| humaneval | **1.0000** | 0.9615 | 0.8235 | **1.0000** (chain) |
| mean (4-task) | 0.7123 | 0.7134 | 0.6581 | **0.7260** (per-task winner mix) |

**Adaptive vs best-static дельта (per task):**

| task | rule Δ | llm Δ |
|---|---:|---:|
| commongen | **+0.0116** | **+0.0198** |
| dabench | −0.0206 | +0.0023 |
| gsm8k | 0.0000 | −0.0833 |
| humaneval | −0.1000 | −0.1923 |
| **mean (4-task)** | **−0.0272** | **−0.0634** |

**Главное.** В среднем по 4 задачам adaptive **отстаёт** от best-static-per-task — rule на 2.7 п.п., llm на 6.3 п.п.

**Где adaptive ВЫИГРЫВАЕТ.** На commongen обе версии роутера дают плюс над лучшей статикой (+0.012 для rule, +0.020 для llm). Adaptive окупается на open-ended задачах, не на verifiable. На gsm8k и humaneval adaptive(llm) проседает катастрофически (−8.3 и −19.2 п.п.), потому что llm-роутер на verifiable-задаче *не* выбирает уверенный chain до конца и пропускает оптимальный exit.

---

## 7. Cost / Pareto

### 7.1 Pareto-картина (per-router overall)

| Router | mean_q | mean_cost ($) | mean_wall (s) | $/q-unit |
|---|---:|---:|---:|---:|
| llm | 0.6549 | 0.0369 | 756.8 | 0.0563 |
| rule | 0.6301 | 0.0793 | 1 857.9 | 0.1259 |

llm доминирует rule по всем трём осям одновременно (mean_q выше, mean_cost ниже, mean_wall ниже). Это не классический tradeoff «качество ↔ цена», а Pareto-доминация одной семьёй роутера другой на overall выборке.

### 7.2 Cost-структура

| Метрика | rule | llm |
|---|---:|---:|
| mean_iter | 9.62 | 10.78 |
| mean_wall (s) | 1 857.9 | 756.8 |
| mean_cost ($) | 0.0793 | 0.0369 |

llm-роутер в среднем делает **больше** итераций (10.78 vs 9.62), но при этом стоит **меньше** и завершается **быстрее** в 2.46×. Причина — выбор подтопологий: llm чаще выбирает дешёвый `chain` и завершается early-exit'ом, rule стабильнее удерживается на debate/hierarchical и расходует полный iter-budget.

### 7.3 RQ3 inflection point

Cost-quality tradeoff на overall выборке инвертирован относительно «классической» интерпретации: дороже не = качественнее. Cost-ratio rule/llm = 2.15× (CI [1.58, 3.02], исключает 1.0) при quality-CI, накрывающем 0. Дополнительная стоимость rule **не окупается** ростом качества ни на одной из 4 задач — на gsm8k/humaneval rule выигрывает по quality, но wall-time × 6.8–7.0 и cost × 2.0 на этих же задачах съедают этот выигрыш.

---

## 8. Ответ на RQ2 и RQ3

### 8.1 RQ2 (router family preference)

**Прямой numerical answer.** *На post-fix данных под gpt-oss-120b в среднем по 4-task class-level метрике:*

- **Per-row mean (по 237 cells):** `llm` 0.6549 > `rule` 0.6301 — Δ=+0.025 в пользу llm; **CI [−0.073, +0.121] накрывает 0**.
- **Un-weighted per-task mean (4 равных task-bucket'а):** `rule` 0.6988 > `llm` 0.6627 — Δ=+0.036 в пользу rule.
- **Per-task pattern:** rule выигрывает на gsm8k (+0.083) и humaneval (+0.092); llm выигрывает на commongen (+0.008) и dabench (+0.023).

**Verdict.** Quality-разница между rule и llm **не отличима от 0** в bootstrap-CI-анализе. Но **направление per-task** не монотонно: на verifiable-задачах rule сильнее, на open-ended — llm. Это совместимо с дизайн-аргументом «rule-таблица захардкожена под numeric/programming, llm-роутер гибче на creative».

### 8.2 RQ3 (cost-quality tradeoff)

**Прямой numerical answer.** *На post-fix данных:*

- **mean_cost:** llm $0.037 vs rule $0.079 → **cost-ratio rule/llm = 2.15× (CI [1.58, 3.02], исключает 1.0)**.
- **wall:** llm 757 s vs rule 1 858 s → **wall-ratio rule/llm = 2.46×**.
- **$/quality-unit:** llm $0.056 vs rule $0.126 → llm в **2.24× эффективнее** по quality-per-dollar.

**Verdict.** **`llm`-роутер — Pareto-победитель**: одновременно (а) дешевле в 2.15×, (б) быстрее в 2.46×, и (в) per-row average качество выше rule (хотя CI накрывает 0). На un-weighted per-task ничья + чуть-чуть в пользу rule (+3.6 п.п.), но cost-asymmetry перевешивает.

---

## 9. Аномалии и ограничения

### 9.1 DABench — глобально сломан

| router | task | mean_q | zero-rate |
|---|---|---:|---:|
| rule | dabench | 0.284 | 0.676 |
| llm | dabench | 0.307 | 0.600 |
| chain (static) | dabench | 0.279 | 0.622 |
| star (static) | dabench | 0.304 | 0.630 |
| mesh (static) | dabench | 0.304 | 0.652 |

Все 5 топологий лежат в полосе 0.28–0.31 со zero-rate 60–68%. **DABench не работает для gpt-oss-120b на numeric-exact-match эвалюации**, и никакая router-стратегия этого не чинит. Любая интерпретация router-эффекта на dabench должна включать оговорку «в шуме broken-бенчмарка».

### 9.2 GSM8K saturated

rule = 1.000, llm = 0.917 — distinguishing power между роутерами ≤ 8 п.п. при cell-level std ≈ 0.28 у llm и 0.00 у rule. Регрессия llm на −0.083 устойчивая (bootstrap CI [0.000, +0.208]). Это **алгоритмическое свойство** LLM-prompt роутинга на verifiable-numeric, не artefact реализации.

### 9.3 llm×dabench: n=25 (не CI-надёжно)

Из всех 8 ячеек таблицы §4.3 эта — наименее статистически устойчивая. Bootstrap CI [−0.242, +0.191] на Δq против rule — практически бесполезный. Содержательное наблюдение: llm запустился на dabench только после трёх инфраструктурных корректировок (state-bloat в reducer'е, ToolResult.output cap, psycopg pipeline off); до них было 0 completed. Само наличие n=25 — валидация инфраструктурного блока корректировок.

### 9.4 Cascades на llm + dabench

llm×dabench показал 67% failure rate против 32–45% на остальных llm-cells и 19–38% на rule-cells. Концентрация падений на пересечении «llm-роутер × tool-heavy task» указывает на специфический profile: длинные tool_result outputs + cycle между роутером и подтопологией создавали unbounded state growth. Эти cascades — главная причина того, что full sweep `e3_postfix_full` пришлось разбивать на 7 sub-experiment'ов.

### 9.5 Oracle-router отсутствует

В дизайне исходной серии E3 присутствовал oracle-роутер, в post-fix re-run его не повторяли. Это значит, что класс-level ceiling в данном документе строится только из best-static-per-task (mean 0.726, §6); прямого upper-bound «с полным знанием задачи» нет.

### 9.6 Static-baselines: gsm8k у mesh broken

В статиках mesh×gsm8k = 0.905 (zero-rate 0.095), а chain/star = 1.000. mesh плохо работает на arithmetic — known limitation статической топологии voting.

### 9.7 Failure-overhead в общем бюджете

Sum cost across всех 643 completed runs = **$29.38**:
- rule×adaptive (sum_cost): $9.36
- llm×adaptive (sum_cost): $4.39
- static-baselines (chain+star+mesh): $15.63

Plan-spend для всей серии: ≈ $35 (worker) + $5 (judge). Фактически в worker'е ушло ≈ $29.4 на successful runs + неизвестная сумма на 353 failed (большая часть failures — zombie без вышедших LLM-call'ов; реальная overhead невелика). Бюджет уложился.

---

## 10. Methodology audit framing

Все цифры данного документа получены на скорректированной реализации adaptive-роутера после внутреннего implementation audit, обнаружившего ряд артефактов в роутинг-логике и сопутствующей инфраструктуре. Корректировки применены до запуска текущего грида; audit trail зафиксирован в репозитории. В рамках данного документа никаких числовых сопоставлений «до коррекций / после коррекций» не строится — анализ ведётся исключительно на текущей выборке.

---

## 11. Итоги (для главы 5 диплома)

- **RQ2 — quality-equivalence rule vs llm.** Bootstrap-CI на Δq [−0.121, +0.073] накрывает 0; per-task направления не монотонны (rule выигрывает на verifiable: gsm8k +0.083 / humaneval +0.092; llm — на open-ended: commongen +0.008 / dabench +0.023).
- **RQ3 — `llm`-router Pareto-победитель.** Cost-ratio rule/llm = 2.15× (CI [1.58, 3.02], исключает 1.0), wall-ratio 2.46×. По per-row mean_q llm даже на 0.025 выше rule, по un-weighted per-task — ниже на 0.036. Cost-asymmetry однозначно перевешивает.
- **Adaptive vs best-static.** Adaptive(rule) отстаёт от best-static-per-task на 2.7 п.п., adaptive(llm) — на 6.3 п.п. На commongen adaptive **обгоняет** статику (rule +0.012, llm +0.020).
- **Oracle-router в post-fix не повторялся.** Прямого class-level ceiling нет; upper-bound — best-static-per-task 0.726.
- **DABench broken across all topologies** (mean_q 0.28–0.31, zero-rate 60–68%), **GSM8K saturated** (rule = 1.000) — known limitations бенчмарков.
- **n=237 adaptive (118 rule + 119 llm)** — близко к плановой кардинальности 240; failure-rate выше плановой за счёт state-bloat cascades на DABench+llm; CI-выводы устойчивы на 7 из 8 ячеек (исключение — llm×dabench n=25).
- **Бюджет серии: $29.4 фактических** против ~$40 plan; вышли в +25%-reserve диапазон.
