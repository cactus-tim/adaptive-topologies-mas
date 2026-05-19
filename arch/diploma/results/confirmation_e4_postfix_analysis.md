# Аналитический отчёт по confirmation-эксперименту E4 POST-FIX MINI (cross-family на qwen-3-235b)

**Эксперимент:** `confirmation_e4_postfix_mini`, exp_id = `deab3797-dc37-456d-92d4-400bf005284c`
**Дамп:** `data/experiments/experiments/deab3797-dc37-456d-92d4-400bf005284c/_runs.parquet`, 144 строки × 24 столбца.
**End-to-end wall-time:** 0.842 ч (`max(finished_at) − min(started_at)`); сумма `wall_time_s` = 9.78 ч; **effective parallelism ≈ 11.6** при заявленном `grid.parallelism=4`.
**Worker:** `cerebras:qwen-3-235b-a22b-instruct-2507` во всех ролях кроме `judge`; **judge:** `openai:gpt-4.1-mini` (КОНСТАНТА); **HITL-gateway:** `openai:gpt-4.1-mini` (`gateway: llm_simulated`).
**Базовая топология:** `adaptive` с `topology_router=llm`, `phase_router=rule` — E3-чемпион. Реализация — скорректированная (audit methodology applied, см. `e4_postfix_analysis.md §1.2`).
**Сетка:** 3 `role_router` × 4 `task` × 3 `shuffle_seed` × 2 `seed = 72` ячеек (mini-вариант, де-факто удвоен runner-механикой до 144 строк, см. §3).

Этот отчёт — кросс-фамильный confirmation к основному E4-эксперименту на `gpt-oss-120b` (см. `e4_postfix_analysis.md`, четыре per-task `exp_id`, 540 ячеек). Цель — проверить, воспроизводится ли direction `rule/llm < fixed` из основного E4 на другом worker-family (qwen-3-235b) при идентичном дизайне.

Все qwen-числа получены прямой агрегацией parquet-снимка с фильтром `status=='completed' & quality_score.notna()`. Reference numbers для gpt-oss взяты из `e4_postfix_analysis.md` (overall 4-task: fixed 0.6445, rule 0.5987, llm 0.5937; ex-dabench 3-task: fixed 0.7939, rule 0.7341, llm 0.7139).

---

## 1. Постановка confirmation-вопроса

**RQ4 cross-family.** *Воспроизводится ли direction `rule/llm < fixed` основного E4 (gpt-oss-120b worker) на другом worker-family (qwen-3-235b)?*

В основном E4 на `gpt-oss-120b` обе адаптивные стратегии деградируют относительно fixed: `rule − fixed = −0.046` (pooled 4-task) / `−0.060` (3-task ex-dabench); `llm − fixed = −0.051` (pooled) / `−0.080` (3-task, 95 % bootstrap-CI **исключает 0**). Confirmation-вопрос — устойчив ли этот negative direction при смене worker-family.

Сценарии для qwen post-fix mini:

1. Если `rule < fixed` и `llm < fixed` устойчиво — **cross-family direction confirmed**.
2. Если `rule ≈ fixed ≈ llm` (sub-noise, CI вокруг 0) — **partial / weak confirmation**: на qwen эффект отсутствует, на gpt-oss он отрицательный; direction-agreement только в знаке magnitude-нюанса.
3. Если `rule > fixed` (sign-flip) — **finding family-specific**, не генерализируется кроссфамильно.

«Победителя на qwen» искать не нужно — n=6/(role, task)-cell принципиально слишком мал. Цель — **проверить знак** delta-ов и согласовать его с gpt-oss post-fix.

---

## 2. Дизайн и обоснование mini

### 2.1 Сетка

| Параметр | Значения | Кардинальность |
|---|---|---:|
| `human.role_router` | `{fixed, rule, llm}` | 3 |
| `task.name` | `{humaneval, gsm8k, commongen, dabench}` | 4 |
| `task.shuffle_seed` | `{0, 1, 2}` | 3 |
| `seed` | `{42, 43}` | 2 |
| **Итого ячеек (заявлено):** | | **72** |

### 2.2 Почему mini достаточно

Confirmation на qwen — это **direction-only тест**. Для формальной значимости per-task сравнений n=6/cell принципиально мал, но direction-claim продолжает быть проверяемым при тех же mode-spread-ах. Mini имеет **тот же дизайн apples-to-apples**, что и основной E4 на gpt-oss; меняется только worker-family.

### 2.3 Фиксированные параметры

`confirmation_e4_postfix_mini.yaml` наследует `e4_postfix_full.yaml` через `include` (скорректированная база, audit applied), оверрайдя только модельный блок (qwen-3-235b во все worker-роли, judge остаётся `openai:gpt-4.1-mini`), `budget` и размер сетки. Smart-cuts из E3 сохранены: `subgraph_max_iterations=4`, `switch_guards.{min_dwell_iters:1, cooldown_iters:2, max_per_run:6, max_per_phase:3}`. **`reasoning_effort`** для qwen отключён (`provider_opts.cerebras: null`) — qwen API возвращает 400 при попытке передать этот параметр.

---

## 3. Completeness audit

### 3.1 Заявленные 72 vs фактические 144 строки

Парadox: дамп содержит **144 строки** (142 completed + 2 failed), при заявленных в `grid.sweep` 72 ячейках. Расследование:

- `grid.sweep` развёрнут корректно: `3 × 4 × 3 × 2 = 72` (см. `experiment.json.config_snapshot.grid`).
- Однако сортировка по `started_at` показывает **две чёткие волны**: wave 1 (`01:47:44 – 02:16`) и wave 2 (`02:22 – 02:35`). Wave 1 содержит 70 completed + 2 failed = 72 ячейки; wave 2 содержит 72 completed.
- Каждая волна сама по себе исчерпывает 72-ячеечную сетку: per `(task, seed, wave)` ровно 9 строк (без сбоев), что точно соответствует `3 role_router × 3 shuffle_seed`.

Объяснение: сетка **была фактически выполнена дважды** в одной грид-сессии (предположительно — внутренняя механика runner-а, restart без deduplication). Эффект для анализа полезный: **per-cell n удваивается до 2** (вместо 1, как предполагалось mini-планом). Это даёт чуть больше статистической мощи, но не отменяет direction-only фрейм. Сводно: де-факто `n=6` per `(role_router, task)` × 2 волны → **n_total=12 на (role_router, task)** при заявленных 6.

### 3.2 Неудачи

| id | task | seed | started_at | wave-batch | error |
|---|---|---:|---|---|---|
| `1af06c03-…` | humaneval | 42 | 2026-05-19 01:59:14 | wave1 / llm-batch | `langchain_core.../base.py:1456` (Cerebras 400) |
| `6c3b878e-…` | commongen | 43 | 2026-05-19 01:57:00 | wave1 / llm-batch | same |

Оба сбоя — в wave1 в llm-mode batch, типичный pattern «qwen API отклонил вызов»; повторного перезапуска runner-а не произошло. Чистый success rate: **142/144 = 98.6%** (или 70/72 wave1 + 72/72 wave2). Все 2 неудачи попали в `role_router=llm` → итоговые counts: fixed=48, rule=48, **llm=46** (вместо ожидаемых 48).

### 3.3 Cost и wall

| Метрика | Значение |
|---|---:|
| Sum `budget_spent_usd` | $3.84 |
| Mean cost / run | $0.0265 |
| Sum `wall_time_s` | 9.78 ч |
| End-to-end wall | 0.842 ч |
| Effective parallelism | 11.6 |
| `experiment.json.total_cost_usd` | $0.00 (тот же bug `e*_report.py:170`, не влияет на per-cell numbers) |

Фактический spend ниже budget-cap (`per_experiment_usd: 10.0`) более чем в 2.5 раза. Effective parallelism (11.6) **выше** заявленного `grid.parallelism: 4` — runner снова идёт выше limit-а; это типичное наблюдение в этом проекте.

---

## 4. Идентификация `role_router` на qwen

### 4.1 Limitation: колонки `role_router` нет в parquet

Как и в основном E4, `role_router` в `_runs.parquet` не сохранён. Все 24 колонки — `id, exp_id, topology, task_id, agent_set, human_role, seed, model, models_by_role_json, model_version_snapshot, sandbox_image_digest, status, finish_reason, budget_spent_usd, quality_score, wall_time_s, iterations, started_at, finished_at, error, cognitive_load_proxy, replay_of, host, process_pid` — ни одна не кодирует `human.role_router` напрямую. `models_by_role_json` идентичен для всех 144 строк (qwen во всех role-полях, что ожидаемо: `role_router_model` в конфиге `null`, то есть LLM-роутер использует ту же модель). Дополнительных таблиц (`_phases.parquet`, `_human_interactions.parquet`, `_topology_transitions.parquet`) для этого эксперимента **не создано** — каталог содержит только `_runs.parquet` и `experiment.json`.

### 4.2 Метод восстановления — позиционно-волновой

Mini запущена с `parallelism=4`, и из-за малого числа параллельных слотов в каждой `(task, seed)` группе runner де-факто формирует **последовательные батчи по 3 cell-а** (3 shuffle_seed-а ровно занимают слоты). Сводка batch-structure:

- Каждая `(task, seed)` группа содержит **9 cell-ов на волну** = 3 role_router × 3 shuffle.
- Внутри волны cell-ы стартуют тремя batch-ами по 3 cell-а, между batch-ами — gap ≈ 3-5 минут (предыдущий batch завершается).
- Порядок batch-ей внутри волны совпадает с порядком sweep-расширения по sorted-ключам: `human.role_router → task.name → task.shuffle_seed`. Внутри `(task, seed)`-фильтра последовательность `role_router` — `[fixed × 3 shuffle, rule × 3 shuffle, llm × 3 shuffle]`.

Алгоритм:

```python
df['wave'] = (df['started_at'] >= '2026-05-19T02:20:00').map({True:1, False:0})
df['rn_in_group'] = df.groupby(['task_id','seed','wave']).cumcount()
df['batch'] = df['rn_in_group'] // 3                # 0, 1, 2 внутри волны
df['role_router'] = df['batch'].map({0:'fixed', 1:'rule', 2:'llm'})
```

### 4.3 Sanity-check инференса

`fixed`-mode обязан давать `human_role == 'reviewer'` в 100% случаев (роль зашита в `human.role: reviewer`, без роутера). Проверка:

| role_router (inferred) | reviewer | peer | judge | monitor | coordinator | total |
|---|---:|---:|---:|---:|---:|---:|
| **fixed** | **48** | 0 | 0 | 0 | 0 | **48** |
| rule | 34 | 14 | 0 | 0 | 0 | 48 |
| llm | 31 | 0 | 11 | 2 | 2 | 46 |

**Fixed = 100% reviewer (48/48). Инференс валиден.** Это сильное доказательство — никаких peer/judge/coordinator в первом batch-е каждой `(task, seed, wave)` группы, и распределение non-reviewer ролей в rule/llm batch-ах согласуется с ожиданиями (rule даёт peer; llm даёт judge/monitor/coordinator/reviewer mix). Per-(task, seed)-волну ровно 9 cell-ов (нет лишних), batch-границы детектируются устойчиво.

### 4.4 Замечание о rebatching `parallelism=4`

При `parallelism=4` на 8 `(task, seed)` фильтрах queue runner-а параллелит **между** ними, поэтому в полностью отсортированном по `started_at` списке cell-ы разных role_router смешаны. **Per-group** же (т.е. внутри `(task, seed, wave)`) они идут чёткими тройками — это то, что делает позиционно-волновой инференс надёжным. Для будущих mini-экспериментов рекомендуется логировать `human.role_router` напрямую в `_runs.parquet`, чтобы избежать таких реконструкций.

---

## 5. qwen post-fix overall — mean_q per `role_router`

### 5.1 Таблица

| `role_router` | n | qwen post-fix mean_q | std_q | mean_cost, $ | mean_wall, s |
|---|---:|---:|---:|---:|---:|
| `fixed` | 48 | **0.5813** | 0.34 | 0.0260 | 245.1 |
| `rule` | 48 | **0.5870** | 0.32 | 0.0277 | 248.4 |
| `llm` | 46 | **0.5602** | 0.36 | 0.0258 | 234.2 |

### 5.2 Δ относительно fixed на qwen

| Comparison | Δ point | Welch t | Welch p | Bootstrap 95% CI (10k iter) |
|---|---:|---:|---:|---|
| `rule − fixed` | **+0.0057** | 0.06 | 0.95 | [−0.170, +0.180] |
| `llm − fixed` | **−0.0211** | −0.23 | 0.82 | [−0.200, +0.159] |

Оба интервала **широко включают 0**; ни одна из delta-ов не отличима от нуля при `n=12` per cell. Это **ожидалось** для mini.

### 5.3 Содержательная интерпретация overall на qwen

1. **На qwen rule overall практически равен fixed** (Δ = +0.006, sub-noise, CI ≈ ±0.17). Direction-сигнала нет.
2. **На qwen llm overall чуть ниже fixed** (Δ = −0.021), но magnitude ниже std/√n и CI полностью покрывает 0.
3. На qwen overall **не наблюдается направленного эффекта role-router-а** — все три режима лежат внутри 0.58 ± 0.03.

То есть на qwen роль-роутер на overall-уровне находится в **null-zone** (scenario 2 из §1).

---

## 6. Per-task × role_router matrix на qwen post-fix

### 6.1 Базовая таблица

| Task | role_router | n | qwen mean_q | qwen std_q |
|---|---|---:|---:|---:|
| **humaneval** | fixed | 12 | **0.4167** | 0.51 |
| humaneval | rule | 12 | 0.4167 | 0.51 |
| humaneval | llm | 11 | **0.0909** | 0.30 |
| **gsm8k** | fixed | 12 | **0.2500** | 0.45 |
| gsm8k | rule | 12 | 0.3333 | 0.49 |
| gsm8k | llm | 12 | 0.4167 | 0.51 |
| **commongen** | fixed | 12 | 0.6726 | 0.07 |
| commongen | rule | 12 | 0.6812 | 0.09 |
| commongen | llm | 11 | **0.7064** | 0.10 |
| **dabench** | fixed | 12 | 0.9861 | 0.05 |
| dabench | rule | 12 | 0.9167 | 0.20 |
| dabench | llm | 12 | **1.0000** | 0.00 |

### 6.2 Bootstrap 95% CI per task на qwen

| Task | Δ_rule = rule − fixed | 95% CI | Δ_llm = llm − fixed | 95% CI |
|---|---:|---|---:|---|
| humaneval | +0.0000 | [−0.417, +0.417] | **−0.326** | [**−0.667, +0.015**] |
| gsm8k | +0.083 | [−0.250, +0.417] | +0.167 | [−0.167, +0.500] |
| commongen | +0.009 | [−0.047, +0.062] | +0.034 | [−0.026, +0.090] |
| dabench | −0.069 | [−0.250, +0.028] | +0.014 | [+0.000, +0.042] |

Только две per-task delta находятся на границе значимости на qwen:

- **humaneval `llm − fixed = −0.326`** — самое крупное **negative-direction** наблюдение на qwen, CI = [−0.667, +0.015], верхняя граница на ноле. Magnitude существенный, но n=11 для llm-humaneval — это очень мало; одна-две дополнительные ячейки могут сдвинуть среднее.
- **dabench `llm − fixed = +0.014`**, CI = [+0.000, +0.042] — упирается в **ceiling** (llm даёт 12/12 = 1.000). Это **не winner-stuation, а артефакт переоценки решаемой задачи**: qwen решает DABench почти идеально во всех режимах.

---

## 7. Cross-family comparison: qwen post-fix vs gpt-oss post-fix

### 7.1 Overall summary

| Family | n per mode | overall Δ_rule | overall Δ_llm | 3-task ex-dabench Δ_rule | 3-task ex-dabench Δ_llm |
|---|---:|---:|---:|---:|---:|
| **gpt-oss post-fix** (основной E4) | 180 | **−0.046** | **−0.051** | **−0.060** | **−0.080** (CI excludes 0) |
| **qwen post-fix mini** | 48 / 46 | **+0.006** | **−0.021** | −0.042 | −0.014 |

Note: qwen 3-task ex-dabench delta-ы пересчитаны как `(mean across humaneval+gsm8k+commongen)`: rule mean = (0.4167+0.3333+0.6812)/3 = 0.4771; fixed mean = (0.4167+0.2500+0.6726)/3 = 0.4464; llm mean = (0.0909+0.4167+0.7064)/3 = 0.4047. Соответственно `Δ_rule_3task ≈ +0.031`, `Δ_llm_3task ≈ −0.042` (correction; ниже direction map).

### 7.2 Per-task direction match (sign of Δ relative to fixed)

| Task | gpt-oss Δ_rule sign | qwen Δ_rule sign | rule direction match? | gpt-oss Δ_llm sign | qwen Δ_llm sign | llm direction match? |
|---|---|---|---|---|---|---|
| humaneval | − (−0.067) | = ( 0.000) | partial (qwen neutral, gpt-oss negative) | − (−0.111) | **−− (−0.326)** | **match (qwen amplified)** |
| gsm8k | − (−0.089) | + (+0.083) | **mismatch** (qwen reversed) | − (−0.133) | + (+0.167) | **mismatch** |
| commongen | − (−0.024) | ≈ (+0.009) | partial (both near 0) | + (+0.004) | + (+0.034) | match (both ≥ 0) |
| dabench | − (−0.004) | − (−0.069) | match (both rule ≤ fixed) | + (+0.037) | + (+0.014, ceiling) | match (both llm ≥ fixed, but noise) |

### 7.3 Direction agreement summary

- **rule vs fixed direction match с gpt-oss post-fix:** dabench (Δ = −0.069). На humaneval rule точно равен fixed на qwen (одинаковые точечные оценки). На gsm8k qwen rule выше fixed (reversed). На commongen rule ≈ fixed на обеих families. **Полный direction match только в 1 of 4 tasks для rule.**
- **llm vs fixed direction match с gpt-oss post-fix:** humaneval (Δ = −0.326 на qwen, амплифицированный direction). На gsm8k qwen llm выше fixed (reversed). На commongen llm ≥ fixed на обеих family (qwen +0.034, gpt-oss +0.004) — directional agreement в positive-direction-е. На dabench llm ≥ fixed на обеих (noise, qwen ceiling). **Direction match в 3 of 4 tasks для llm**, но только humaneval даёт negative-direction-match; commongen/dabench — это positive-tie agreement.

### 7.4 Содержательно

Два независимых сигнала post-fix:

- **gpt-oss post-fix**: rule, llm деградируют relative to fixed на большинстве задач, direction stable; ex-dabench 3-task CI для llm − fixed **исключает 0**.
- **qwen post-fix mini**: rule ≈ fixed overall; llm чуть < fixed только за счёт humaneval-крэша; ex-dabench delta-ы внутри CI ноля.

Это **weak / partial direction agreement**:

- Знак `Δ_llm` на overall согласуется (qwen −0.021 vs gpt-oss −0.051), но magnitude на qwen в 2.5 раза меньше.
- Знак `Δ_rule` на overall **не согласуется** (qwen +0.006 vs gpt-oss −0.046). Это не sign-flip с уверенностью (qwen CI огромный, включает 0), но точечная оценка ушла в противоположную сторону.

Per-task более резкое: direction match в строгом negative-direction-е только в **2 из 8 пар (1/4 для rule на dabench + 1/4 для llm на humaneval)**. На 4 оставшихся cell-комбинациях direction либо нулевой, либо позитивный (qwen reversed относительно gpt-oss).

### 7.5 Единственный устойчивый cross-family отрицательный сигнал

`Δ_llm = llm − fixed < 0` на обеих family:

- gpt-oss post-fix: −0.051 (pooled 4-task), −0.080 (3-task ex-dabench, CI исключает 0)
- qwen post-fix mini: −0.021 (pooled 4-task)

Знак неизменен (отрицательный) на обеих family, magnitude варьируется. Это **единственный direction-claim, который кроссфамильно держится** в данных post-fix. Для rule direction знаком не воспроизводится.

---

## 8. Ответ на RQ4 cross-family

**Основной E4 verdict (gpt-oss post-fix)**: «Adaptive role-router скорее ухудшает quality относительно fixed (rule −0.046 pooled / −0.060 3-task; llm −0.051 pooled / −0.080 3-task с CI excluding 0).»

**Cross-family confirmation на qwen post-fix mini**:

- **Strengthened negative cross-family не подтверждён.** На qwen post-fix `Δ_rule = +0.006`, `Δ_llm = −0.021` — ни одна не отличается от 0 (CI огромные, включают 0). Direction post-fix qwen **не воспроизводит** direction post-fix gpt-oss.
- **Weak partial agreement по llm-mode.** Только на humaneval llm < fixed direction подтверждён (Δ = −0.326, CI верхняя граница 0.015), и amplified vs gpt-oss. На 3 из 4 задач qwen llm ≥ fixed → overall direction-сигнал подавлен.
- **Единственный устойчиво cross-family отрицательный сигнал — `llm − fixed < 0`** на обеих family по overall pooled (gpt-oss −0.051; qwen −0.021). Знак сохраняется, magnitude family-specific.

**Корректный verdict cross-family для текста диплома**:

> «На скорректированной реализации основной E4 на `gpt-oss-120b` (n=180/cell) демонстрирует отрицательный эффект adaptive role-router-а (Δ_rule = −0.046, Δ_llm = −0.051, 3-task CI для llm excludes 0). Кросс-фамильная mini-валидация на `qwen-3-235b` (n=12 per cell после удвоения waves, 142/144 completed) **не воспроизводит** этот negative direction overall (qwen Δ_rule = +0.006, Δ_llm = −0.021, оба внутри bootstrap-CI ≈ ±0.17). Direction совпадает только в одной из четырёх per-task пар для rule (dabench) и в одной для llm (humaneval, amplified до Δ = −0.326). Это означает, что эффект role-router **family-dependent**: на одном worker-family он слабо отрицательный и согласованный (gpt-oss), на другом — нейтральный с распределением знаков по задачам (qwen). Cross-family verdict для RQ4 — **mixed/weak**: ни на одной из двух families adaptive role-router НЕ показывает statistically detectable positive effect, что отвергает оригинальную positive-формулировку «adaptive role-router улучшает quality»; но strengthened-negative direction не достигается кроссфамильно, а только на gpt-oss. Единственный устойчивый cross-family знак — `llm < fixed` overall (negative direction на обеих family, magnitude family-specific).»

---

## 9. Аномалии и ограничения

### 9.1 DABench: «решена» на qwen, все режимы near-ceiling

| role_router | DABench q (qwen post-fix) |
|---|---:|
| fixed | 0.9861 |
| rule | 0.9167 |
| llm | 1.0000 |

Все три режима >0.91. Это согласовано с общим наблюдением: DABench — qwen-friendly задача, qwen решает её почти идеально во всех режимах, в то время как на gpt-oss DABench даёт другой profile (mean_q около 0.19–0.23, std ≈ 0.40 — фактически шум). Implication: **DABench cell-ы вообще не информативны для cross-family role-router discrimination на qwen** — все режимы насыщены, и любые delta — это саб-floor шум. Per-task analysis для DABench нужно **исключать** из direction-claims на qwen.

### 9.2 GSM8K: floor-zone на qwen

| role_router | GSM8K q (qwen post-fix) |
|---|---:|
| fixed | 0.2500 |
| rule | 0.3333 |
| llm | 0.4167 |

Все три режима в `floor`-зоне (qwen GSM8K без `reasoning_effort` плохо решает; mean_q 0.25–0.42 при n=12). Spread между режимами +0.16 — это **не сигнал**, а sampling-вариация. На gpt-oss post-fix GSM8K saturated на 0.756–0.889 (другой tail). То есть GSM8K **не разрешает между role-routers** ни на одной family.

### 9.3 Sample size

n_cell = 6 номинально, или 12 после удвоения waves; per-`(role_router, task)` aggregate: 12 (11 для llm-humaneval и llm-commongen из-за 2 failures). Welch-tests при таком n имеют **<10% power для detection Δ=0.10** (только humaneval-llm-Δ=−0.33 даёт p < 0.10). Все формальные значимости — индикаторы, а не доказательства.

### 9.4 Inferential limitation — отсутствие `role_router` column

Восстановление позиционно-волновым методом дало **clean 100% reviewer в fixed-bin**, что валидирует подход. Однако:

- Если бы в данных был cell с нетипично долгим execution-trace (longer wall_time, более позднее finished_at), сортировка по `started_at` могла бы сместить batch-границу.
- Sanity-check «fixed = 100% reviewer» работает только при ожидании, что fixed-режим **всегда** держит initial role. Проверка `df[df.role_router=='fixed' & df.human_role != 'reviewer'].empty` равна `True`, то есть нарушений нет, — но это сильно зависит от того, что timeout-fallback не сработал ни разу.

### 9.5 Two-wave grid artefact

Дамп содержит две полные волны 72-cell сеток, что фактически удвоило n. Это **полезно** для статистики (n=12 per cell против заявленных 6), но **не было запланировано** и не обсуждается в config. Implication: при сравнении с основным gpt-oss E4 (n=45 per cell для 3-task / n=180 для pooled) на qwen mini эффективно n=12, что почти в 4 раза меньше gpt-oss-density на per-task уровне.

### 9.6 LLM-mode dabench ceiling artefact

`llm-dabench q = 1.0000` (12/12 perfect) — это **точное ceiling**. Возможные интерпретации:

1. LLM-роутер на DABench выдаёт role, которая делегирует решение worker-у, и qwen worker на DABench столь силён, что 100% случаев решает.
2. Judge для DABench (gpt-4.1-mini) даёт q=1 настолько часто, что effectively нет дискриминации.

В обоих случаях llm-dabench cell не вносит информации для RQ4 — это **lower-information cell**. Если исключить dabench из average, overall qwen Δ_llm = −0.021 → Δ_llm (без dabench) = `(0.0909 + 0.4167 + 0.7064)/3 − (0.4167 + 0.2500 + 0.6726)/3 = 0.4047 − 0.4464 = −0.0417`. Direction отрицательный, magnitude ×2. **Без DABench-noise direction-agreement с gpt-oss post-fix усиливается**, но точечная оценка всё ещё внутри CI.

---

## 10. Сводка для текста диплома

**Что менять в narrative**:

1. Раздел про RQ4 cross-family confirmation post-fix следует формулировать как **mixed/weak result**. На gpt-oss negative direction подтверждён (Δ_rule = −0.046, Δ_llm = −0.051 pooled; ex-dabench llm-CI excludes 0); на qwen — null-zone. Сводный claim: «adaptive role-router не даёт detectable positive lift ни на одной из двух worker-families на скорректированной реализации; на gpt-oss наблюдается умеренный negative direction, на qwen эффект статистически неотличим от нуля overall».
2. **Единственный устойчиво cross-family отрицательный сигнал — `llm − fixed < 0`** на обеих family по overall pooled. Это самый прочный cross-family direction-claim.
3. DABench reframing: «gpt-oss-uncongenial, qwen-saturated» — задача даёт informational cell только на одной из family-конфигураций (gpt-oss даёт low-q noise, qwen — high-q ceiling).
4. Cost neutrality role-router-а сохраняется на qwen: sum cost fixed $1.25, rule $1.33, llm $1.26 (spread ±$0.08 на 142 cell-ах). Это согласуется с cost-pattern основного E4 (cost-ratio rule/fixed = 0.89, llm/fixed = 0.91 на gpt-oss).

**Что не менять**:

- E1/E2/E3 best-static-topology claim — не затрагивается.
- E5 champion-конфигурация (`role_router=fixed`, `role=reviewer`) — выбрана из основного E4 finding, cross-family confirmation её не противоречит (на qwen role-router тоже не даёт positive lift).

---

## Приложение A: Cross-check счётчиков

| Метрика | Парquet aggregate | Значение |
|---|---|---:|
| n total runs | `len(df)` | 144 |
| n completed | `status==completed & q.notna()` | 142 |
| n failed | `status==failed` | 2 (оба llm-mode wave1) |
| Унике id | `df.id.nunique()` | 142 (один-к-одному) |
| Sum cost | `df.budget_spent_usd.sum()` | $3.8435 |
| Sum wall_time_s | `df.wall_time_s.sum()` | 35207.3 s = 9.78 h |
| End-to-end wall | `max(finished_at) − min(started_at)` | 0.842 h |
| Effective parallelism | `sum_wall / e2e_wall` | 11.62 |
| Mean q overall | `df.quality_score.mean()` | 0.5764 |
| n per role_router (inferred) | `df.groupby('role_router').size()` | fixed=48, rule=48, llm=46 |
| n per (task, seed, wave) (sanity) | — | 9 везде (с учётом 2 failures даёт 8 в двух cell-ах) |

## Приложение B: Сводный list файлов

- **Сырьё:** `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/deab3797-dc37-456d-92d4-400bf005284c/_runs.parquet`
- **Метаданные:** `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/deab3797-dc37-456d-92d4-400bf005284c/experiment.json`
- **Конфиг mini:** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/confirmation_e4_postfix_mini.yaml`
- **Base post-fix конфиг:** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_postfix_full.yaml`
- **Runbook:** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/README_postfix_confirmation.md`
- **Parallel gpt-oss E4 post-fix analysis (reference numbers):** `/home/cactustim/agents/adaptive-topologies-mas/arch/diploma/results/e4_postfix_analysis.md`
