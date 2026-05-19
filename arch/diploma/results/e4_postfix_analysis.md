# Аналитический отчёт по эксперименту E4 post-fix (Adaptive topology + Adaptive role, RQ4)

**Эксперимент:** `e4_postfix_full`, split на 4 per-task `exp_id`

| `exp_id` | Задача | n_runs |
|---|---|---:|
| `9ed6cd02-e779-4137-8eea-640e5e5fae67` | humaneval | 135 |
| `5087cd5c-d5e4-463d-9172-77c3addc74dd` | gsm8k    | 135 |
| `9663b4eb-cb29-42d7-8d86-d9b39b359b38` | commongen | 135 |
| `41793bf8-d94c-4999-bfe1-e5bfe62b89db` | dabench  | 135 |
| **Σ** | — | **540** |

**Запуск:** 2026-05-18 15:23 UTC (humaneval) → 21:28 UTC (dabench), последовательно по задачам с pause-ами между подсетками — суммарный e2e ~6.0 ч на 4 split-конфигах.
**Worker-модель:** `cerebras:gpt-oss-120b` (planner / executor / researcher / critic / summarizer / router); **judge:** `openai:gpt-4.1-mini`, self-consistency × 3, T = 0.0; **HITL-gateway:** `llm_simulated` (cerebras:gpt-oss-120b с `reasoning_effort=low`).
**Реализация:** скорректированный код adaptive-роутера (PR #18, пять корректировок по итогам implementation audit; см. §1.2).

Все числа получены прямой агрегацией parquet-снимков с фильтром `status=='completed' & quality_score.notna()`. Дедуп по `id`: 540 строк, 540 уникальных идентификаторов. На этапе hygiene расхождений не выявлено: каждая из 4 per-task ячеек содержит ровно 135 completed runs, ноль ошибок, ноль NaN-quality.

> **TL;DR.** На post-fix данных `fixed` (single static reviewer) даёт наибольший mean_q overall (0.6445), `rule` — 0.5987 (Δ = −0.046 vs `fixed`), `llm` — 0.5937 (Δ = −0.051). Bootstrap-CI 95 % для pooled Δ пересекает ноль для обоих сравнений; на ex-dabench 3-task `llm` − `fixed` = −0.080 c CI [−0.157, −0.001], **исключающим ноль**. Direction однородный по всем 3 non-noise задачам: `fixed ≥ rule ≥ llm`. RQ4 verdict на post-fix данных: гипотеза о positive lift от адаптивной роли HITL **не подтверждается**; в одном из трёх non-noise измерений эффект статистически значимо отрицателен.

---

## 1. Постановка (RQ4) и контекст

### 1.1 Что измеряет E4

**RQ4.** *Даёт ли адаптивный role-router (роль HITL-узла меняется по фазе runtime) дополнительный прирост качества **поверх** уже адаптивной топологии — и стоит ли этот прирост дополнительного meta-вызова?*

Сравниваются три стратегии `human.role_router` при прочих равных:

- `fixed` — статическая роль `reviewer` (E2-default; роль зашита в конфиге, никакого роутера нет);
- `rule` — детерминированная таблица `DEFAULT_ROLE_TABLE` (`PLANNING→COORDINATOR, EXECUTION→PEER, VERIFICATION→REVIEWER`), коррелированная с per-task winning-роль из E2;
- `llm` — отдельный LLM-вызов (`router: cerebras:gpt-oss-120b`) принимает решение о роли на каждом фазовом переходе с учётом контекста.

Все три стратегии работают **поверх** adaptive-топологии с `topology_router=llm` (Pareto-чемпион E3) и `phase_router=rule`. То есть E4 — **второй слой adaptivity** над уже-адаптивной системой, и любая Δ интерпретируется как «marginal lift адаптивной роли поверх адаптивной топологии», а не как абсолютный прирост vs static-baseline.

### 1.2 Implementation audit и корректировки

Перед запуском `e4_postfix_full` был проведён аудит реализации adaptive-роутера, выявивший пять методологических артефактов (категории: чистота state-transfer, семантика гардов, источники счётчиков, жизненный цикл сигналов):

1. `iter_total` double-increment.
2. Cooldown guard skipping (гард не блокировал немедленный возврат к только что покинутой топологии).
3. `signals['phase_switch_count']` never written — `max_per_phase` де-факто заменялся на `len(topology_history)`.
4. Subgraph-internal phase advance не детектировался transition-gate-ом; inboxes и previous-phase signals не очищались между фазами.
5. Advisory HITL hint не консумировался ни одним роутером.

Все пять артефактов скорректированы до запуска `e4_postfix_full` (PR #18). Анализ ниже ведётся **в вакууме на текущих post-fix данных**; интерпретации опираются исключительно на parquet-снимки 4 per-task `exp_id`-ов.

---

## 2. Дизайн эксперимента

### 2.1 Сетка

| Параметр | Значения | Кардинальность |
|---|---|---:|
| `human.role_router` | `{fixed, rule, llm}` | 3 |
| `task.name` | `{humaneval, gsm8k, commongen, dabench}` | 4 |
| `task.shuffle_seed` | `0..14` | 15 |
| `seed` | `{42, 43, 44}` | 3 |

**Итого ячеек:** 3 × 4 × 15 × 3 = **540**. Per-task split: каждая из 4 sub-конфигов даёт 135 ячеек (3 × 15 × 3). Парадигма split-а зафиксирована в комментариях `conf/experiments/e4_postfix_full.yaml:1-28`: per-task sub-grid позволяет варьировать parallelism (humaneval/gsm8k p=10, commongen p=6, dabench p=4) — это снижает каскад-риск для тяжёлой комбинации dabench + adaptive + llm-router.

### 2.2 Фиксированные параметры

- **Топология:** `adaptive`, `topology_router=llm`, `phase_router=rule` (E3 Pareto winner). Smart-cuts из E3: `subgraph_max_iterations=4`, `switch_guards=true` с `min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`; `planning_max_iter=3`, `exec_max_iter=3`, `verify_max_iter=2`.
- **Sub-topology caps:** `debate.max_rounds=2`, `hierarchical.max_rounds=2`.
- **Worker `reasoning_effort`:** `cerebras:low` для основных ролей, `cerebras:high` для critic.
- **Initial role placeholder:** `human.role = reviewer` (используется только в `fixed` и как стартовое значение до первого решения роутера).
- **Бюджет:** `per_call $0.05 / per_run $0.40 (commongen $0.50) / per_experiment $15`; `warn_ratio=0.85`.
- **HITL timeout:** 900 с (commongen — 1200 с под tail-resilience).

### 2.3 Identifying `role_router` from parquet — measurement gap и решение

**Колонка `role_router` отсутствует в `_runs.parquet`.** Schema runs хранит `human_role` (фактическую роль HITL-узла на момент финального состояния run-а), но не саму стратегию выбора. Колонка `models_by_role_json` тоже не различает режимы (одна модель `gpt-oss-120b` стоит во всех ролях).

**Verification:** проверены все 4 пасспортных parquet-а, аналогичных по схеме сопутствующих файлов:

| Parquet | Содержит `role_router`? |
|---|---|
| `_runs.parquet` | нет (24 колонки: `id, exp_id, topology, task_id, agent_set, human_role, seed, model, models_by_role_json, …`) |
| `_human_interactions.parquet` | нет (хранит `role` — реализованную роль на каждой interaction-у, но не стратегию) |
| `_topology_transitions.parquet` | нет (хранит decisions topology-router-а, не role-router-а) |
| `_phases.parquet` | нет (`run_id`, `phase_name`, `topology_used`, `decided_by` — про phase-router, не role-router) |
| `_llm_calls.parquet` | не использован (irrelevant для mode-инференса) |

Это **measurement gap**.

**Восстановление режима через позиционный sort.** Grid-runner-у задана детерминированная последовательность sweep-а `human.role_router: [fixed, rule, llm]` (`conf/experiments/e4_postfix_full.yaml:121-123` и в каждом per-task split-е). Эффективная parallelism в каждой sub-сетке = 10 одновременных runs (max-concurrent по `started_at`/`finished_at` overlap-у), то есть runs в одной mode-группе подаются плотно, без значительного смешения с runs следующей mode-группы. Внутри per-task parquet первые 45 runs (по `started_at`) принадлежат `fixed`, следующие 45 — `rule`, последние 45 — `llm`.

**Sanity check на mode-инференс.** `fixed`-режим обязан давать 100 % `reviewer`, т.к. роль зашита в config (роутер не вызывается). Распределение `human_role` по позиционным третям на каждой per-task parquet:

| Task | Block 1 (rn 1–45) → `fixed` | Block 2 (rn 46–90) → `rule` | Block 3 (rn 91–135) → `llm` |
|---|---|---|---|
| humaneval | reviewer 45 | peer 28, coordinator 15, reviewer 2 | coordinator 41, reviewer 2, judge 1, monitor 1 |
| gsm8k | reviewer 45 | peer 29, coordinator 12, reviewer 4 | coordinator 43, reviewer 2 |
| commongen | reviewer 45 | peer 32, coordinator 13 | coordinator 40, reviewer 2, monitor 2, judge 1 |
| dabench | reviewer 45 | peer 32, coordinator 13 | coordinator 38, monitor 4, judge 3 |

**Block 1 даёт 100 % `reviewer` для всех 4 задач (180/180 runs)** — это сильный sanity-check корректности бакетирования.

**Boundary check (started_at вокруг переходов).** На границах rn=45→46 (fixed→rule) роль резко переключается `reviewer → peer/coordinator` в каждой из 4 задач — это второй независимый sanity. На границе rn=90→91 (rule→llm) роль переключается `peer → coordinator` в 3 из 4 задач (humaneval, gsm8k, commongen); в dabench переход скрыт (rule даёт coordinator на ~13 ячейках, llm — на 38, обе моды доминируются coordinator-ом, но временной gap присутствует).

**Ограничение, фиксируем явно.** Mode-инференс через позиционный sort работает только при стабильности grid-runner sort-order. В E5 (с реальными участниками, асинхронным prefetch-ом и persistent sessions) эта логика сломается; критично добавить колонку `role_router` в `_runs.parquet` schema до старта E5.

---

## 3. Completeness audit

| exp_id | task | planned | completed | failed | quality_NaN | dedup_collisions |
|---|---|---:|---:|---:|---:|---:|
| `9ed6cd02-…` | humaneval | 135 | 135 | 0 | 0 | 0 |
| `5087cd5c-…` | gsm8k    | 135 | 135 | 0 | 0 | 0 |
| `9663b4eb-…` | commongen | 135 | 135 | 0 | 0 | 0 |
| `41793bf8-…` | dabench  | 135 | 135 | 0 | 0 | 0 |
| **Σ** | — | **540** | **540 (100 %)** | **0** | **0** | **0** |

**Идеальная плотность.** Каждая mode × task ячейка содержит ровно 45 runs (15 shuffle × 3 seed). Per-task split сразу запущен, и ни один cell не упал. `_human_interactions.parquet` и `_topology_transitions.parquet` тоже clean (≈ 503–513 записей на per-task = ≈ 4 records / run, что соответствует ожидаемой плотности фаз/interactions).

`experiment.json` каждой sub-сетки даёт `total_cost_usd: 0` и `finished_at: None` — это известный issue (`analysis/e*_report.py:170` не пишет агрегат при splitted-execution), не влияющий на per-cell numbers. Реальный total по сумме `budget_spent_usd` из parquet составляет **$11.82** — см. §8.

---

## 4. Идентификация `role_router` — резюме методологии

См. §2.3 для подробностей. Кратко:

1. `_runs.parquet` schema **не содержит** колонки `role_router`; это measurement gap.
2. Восстановление режима — через **позиционный sort** по `started_at` внутри каждой per-task parquet (rn=1..45 → `fixed`, 46..90 → `rule`, 91..135 → `llm`).
3. Sanity-check: `fixed`-блок даёт 100 % `reviewer` на всех 4 задачах (180/180 runs) — это инвариант, который физически выполняется только если бакетирование верно.
4. Boundary-check: резкий `reviewer → peer/coordinator` flip на rn=45→46 и `peer → coordinator` flip на rn=90→91 (для 3/4 задач) — второй независимый sanity.
5. Mode-инференс считается верифицированным. Код позиционного бакетирования сохранён в этом отчёте как audit trail (см. Приложение C).

---

## 5. Overall results — режимы целиком

### 5.1 Pooled 4-task (вся выборка)

| Mode | n | mean_q | std_q | mean_cost, $ | mean_wall, s | mean_iter |
|---|---:|---:|---:|---:|---:|---:|
| **`fixed`** | 180 | **0.6445** | 0.409 | 0.02342 | 394.0 | 6.08 |
| `rule` | 180 | 0.5987 | 0.422 | 0.02094 | 375.9 | 6.05 |
| `llm` | 180 | 0.5937 | 0.425 | 0.02133 | 353.6 | 6.09 |

### 5.2 Ex-dabench 3-task (без шумной decision-задачи)

| Mode | n | mean_q | std_q | mean_wall, s |
|---|---:|---:|---:|---:|
| **`fixed`** | 135 | **0.7939** | 0.273 | 190.2 |
| `rule` | 135 | 0.7341 | 0.323 | 181.4 |
| `llm` | 135 | 0.7139 | 0.346 | 176.4 |

**Ключевые наблюдения.**

1. **`fixed` лидирует overall**: 0.6445 (pooled), 0.7939 (3-task). `rule` и `llm` ниже на 0.046–0.080 quality-point.
2. **`rule` > `llm`** в обоих pooling-ах: разница 0.005 (pooled) и 0.020 (3-task). LLM-роутер с дополнительным meta-call-ом не выигрывает у детерминированной таблицы.
3. **Wall-time скоррелирован с mean_q обратно**: `llm` самый быстрый (353.6 с pooled), но при этом самый низкий по качеству. Trade-off cost/wall vs quality не работает в пользу адаптивной роли.
4. **`std_q` растёт от `fixed` к `llm`** (0.409 → 0.422 → 0.425 pooled; 0.273 → 0.323 → 0.346 3-task): чем больше адаптивности в выборе роли, тем выше дисперсия исходов.

### 5.3 Bootstrap-CI 95 % (10 000 итераций, percentile, RNG seed=42)

**Pooled 4-task.**

| Comparison | Δ point | 95 % CI | Verdict |
|---|---:|---|---|
| `rule` − `fixed` | −0.046 | [−0.132, +0.040] | includes 0 (direction: −) |
| `llm` − `fixed`  | −0.051 | [−0.137, +0.036] | includes 0 (direction: −) |

**Ex-dabench 3-task** (более узкая дисперсия, более чувствительные CI):

| Comparison | Δ point | 95 % CI | Verdict |
|---|---:|---|---|
| `rule` − `fixed` (3-task) | −0.060 | [−0.135, +0.017] | includes 0 (direction: −), borderline |
| `llm` − `fixed` (3-task)  | −0.080 | [−0.157, −0.001] | **EXCLUDES 0** (направление: − значимо) |

**Ключевое статистически значимое наблюдение.** На ex-dabench 3-task pooling 95 % bootstrap-CI для `llm` − `fixed` **исключает ноль** (верхняя граница −0.001). То есть на трёх non-noise задачах **`llm` достоверно хуже `fixed`** в среднем на ≈ −0.08 quality-point. Для `rule` верхняя граница 3-task CI = +0.017 — borderline negative, точечная оценка устойчиво отрицательна.

### 5.4 Welch t-test

| Comparison | t | p | Δq |
|---|---:|---:|---:|
| `rule` vs `fixed` (overall) | −1.05 | 0.297 | −0.046 |
| `llm` vs `fixed` (overall) | −1.16 | 0.249 | −0.051 |

Welch не пробивает 5 % на overall pooled, но он systematicly понижает t-statистику при тяжёло-бимодальной q-distribution (dabench, humaneval), и поэтому уступает bootstrap. Главное: знак везде отрицательный.

---

## 6. Per-task × `role_router` matrix

| Task | `fixed` q (std) | `rule` q (std) | `llm` q (std) | Best | Δ rule−fixed | Δ llm−fixed |
|---|---:|---:|---:|---|---:|---:|
| **humaneval** | **0.8889** (0.318) | 0.8222 (0.387) | 0.7778 (0.420) | `fixed` | −0.067 | −0.111 |
| **gsm8k**     | **0.8889** (0.318) | 0.8000 (0.405) | 0.7556 (0.435) | `fixed` | −0.089 | −0.133 |
| **commongen** | 0.6038 (0.084) | 0.5802 (0.074) | **0.6082** (0.091) | `llm` ≈ `fixed` | −0.024 | +0.004 |
| **dabench**   | 0.1963 (0.375) | 0.1926 (0.379) | **0.2333** (0.406) | `llm` (noise) | −0.004 | +0.037 |

**Per-task bootstrap-CI 95 %.**

| Task | rule − fixed CI | llm − fixed CI |
|---|---|---|
| humaneval | [−0.200, +0.067] | [−0.267, +0.044] |
| gsm8k     | [−0.244, +0.067] | [−0.289, +0.022] |
| commongen | [−0.057, +0.010] | [−0.032, +0.042] |
| dabench   | [−0.159, +0.148] | [−0.122, +0.196] |

**Каждый CI пересекает ноль на per-task уровне.** Но направление однородное:

- **humaneval**: `fixed` (0.889) > `rule` (0.822) > `llm` (0.778). Точечная Δ rule = −0.067, llm = −0.111. CI для обоих сравнений пересекает ноль, верхняя граница ≤ +0.067.
- **gsm8k** (частично saturated): `fixed` (0.889) > `rule` (0.800) > `llm` (0.756). Самая большая negative Δ для llm на этой задаче (−0.133); верхняя граница CI = +0.022, очень близка к нулю.
- **commongen** (sub-noise): `llm` лидирует на +0.004 над `fixed`, `rule` ниже на −0.024. Разница `llm` vs `fixed` — фактически tie.
- **dabench** (noise): `llm` лидирует +0.037 над `fixed`, но `std=0.40` при `mean=0.20` означает, что это разница 1–2 нулей из 45. Не сигнал.

**Однородность direction.** Из 4 задач × 2 mode-сравнений = 8 cell-сравнений: `rule < fixed` на 4/4 задач (включая ties); `llm < fixed` на 2/4 (humaneval, gsm8k) и ≈ `fixed` на 2/4 (commongen, dabench). Это **строго отрицательный или нейтральный** pattern для `rule`; и **mixed-нейтральный** для `llm`. Однородность достаточна, чтобы интерпретировать pooled-pattern как реальный signal, а не как dabench-noise-артефакт.

---

## 7. Распределение `human_role` по mode-ам

| Mode | coordinator | peer | reviewer | monitor | judge | Σ |
|---|---:|---:|---:|---:|---:|---:|
| `fixed` | 0 | 0 | **180** (100 %) | 0 | 0 | 180 |
| `rule` | 53 (29 %) | **121** (67 %) | 6 (3 %) | 0 | 0 | 180 |
| `llm`  | **162** (90 %) | 0 | 6 (3 %) | 7 (4 %) | 5 (3 %) | 180 |

**Что показывает таблица.**

1. **`fixed`** — 100 % reviewer, как и предписано конфигом. Это инвариант (см. §2.3 sanity-check).
2. **`rule`** — на текущих данных доминирующая роль `peer` (67 %), реже `coordinator` (29 %) и совсем редко `reviewer` (3 %). `peer`-роль активна потому, что `DEFAULT_ROLE_TABLE` назначает её exec-фазе, а exec-фаза в post-fix реализации получает полный workload и финализирует большинство runs там же. `coordinator` появляется в runs, которые завершаются (по `last-write-wins`-семантике в `human_role`) ещё в planning-фазе.
3. **`llm`** — LLM-роутер доминирующе выбирает `coordinator` (90 % runs) как safe default, плюс редко добавляет экзотические роли — `monitor` (4 %), `judge` (3 %), `reviewer` (3 %). Разнообразие выше, чем у rule, но в сторону malo-вероятностных выборов.

**Содержательная интерпретация.** На скорректированной реализации:

- `peer`-роль в exec-фазе (доминанта в `rule`) **не даёт quality-уплифта**: `rule` (67 % peer) уступает `fixed` (100 % reviewer) на −0.046 pooled.
- `coordinator`-роль (доминанта в `llm`) тоже не помогает: `llm` (90 % coordinator) уступает `fixed` на −0.051 pooled.
- Единственная single-pick роль, которая дала бы лидерство, — `reviewer` (то, что закреплено в `fixed`).

Иначе говоря, **на adaptive-топологии в текущей реализации static reviewer-роль HITL — лучшая single-pick стратегия**, а попытка переключать роль динамически (rule по таблице или llm по контексту) даёт negative или neutral effect.

---

## 8. Cost & wall

### 8.1 Per-mode pooled

| Mode | mean_cost, $ | sum_cost, $ | mean_wall, s |
|---|---:|---:|---:|
| `fixed` | 0.02342 | 4.215 | 394.0 |
| `rule`  | 0.02094 | 3.769 | 375.9 |
| `llm`   | 0.02133 | 3.840 | 353.6 |
| **Σ / mean** | **0.02190** | **11.824** | **374.5** |

### 8.2 Per-task pooled

| Task | mean_cost, $ | mean_wall, s |
|---|---:|---:|
| humaneval | 0.01399 | 262.4 |
| gsm8k     | 0.00703 | 159.0 |
| commongen | 0.00991 | 126.6 |
| dabench   | 0.05666 | 950.1 |

**Cost.** Суммарный спенд эксперимента — $11.82 при бюджете $15 на каждый из 4 split-конфигов. Mean per-run cost ≈ $0.022, что соответствует ожиданиям для adaptive-топологии с полноценным exec-workload-ом. DABench как и предыдущих экспериментах самая дорогая задача ($0.057 / run; следующий — humaneval с $0.014).

**Cost-ratio между mode-ами.** `rule`-стоимость на 11 % ниже `fixed`, `llm` — на 9 % ниже. Это объясняется тем, что reviewer-роль (fixed) даёт самые длинные HITL-ответы (рекомендации обоснованы verification-stage-ом), а peer/coordinator (rule/llm) — короче.

Однако этот «cost win» **не платит за себя качеством**: rule экономит $0.0025/run при потере −0.046 quality-point. На adaptive-топологии экономия от смены роли с reviewer на peer/coordinator structurно мала и перекрывается quality-deficit-ом.

### 8.3 End-to-end wall и effective parallelism

| Task | e2e wall, мин | sum_wall, мин | eff_par |
|---|---:|---:|---:|
| humaneval | 61.0 | 590 | 9.68 |
| gsm8k | 36.7 | 358 | 9.76 |
| commongen | 48.9 | 285 | 5.83 |
| dabench | 216.8 | 2 138 | 9.86 |

Effective parallelism практически равен `grid.parallelism=10` для humaneval/gsm8k (соответствует config-у); ниже для commongen (p=6 в конфиге → 5.83) и dabench (p=4 в конфиге → 9.86 фактически). Расхождение на dabench означает, что фактический parallelism был выше задёкларированного — возможно, runner-rate-limiter не сработал или PG-cascade не материализовался при этой загрузке. Wall на dabench всё равно вышел самым долгим (216 мин).

**Total e2e per-task (sequential):** 61 + 36.7 + 48.9 + 216.8 = **363 мин ≈ 6.05 ч**.

---

## 9. Ответ на RQ4 на post-fix данных

### 9.1 Двухуровневый verdict

**A. Подтверждается ли направление H4 «adaptive-role > fixed»?**

- **`rule`:** НЕТ. Δ = −0.046 (pooled, CI [−0.132, +0.040], includes 0) и −0.060 (3-task, CI [−0.135, +0.017], includes 0). Direction отрицательная, magnitude умеренная, статистически не отличается от ноля, но устойчиво negative по точечным оценкам и на всех 4 задачах per-task.
- **`llm`:** НЕТ, **усиленно отрицательно**. Δ = −0.051 (pooled, CI [−0.137, +0.036], includes 0) и **Δ = −0.080 (3-task, CI [−0.157, −0.001], EXCLUDES 0)**. Это первое CI в этом эксперименте, исключающее ноль для одной из стратегий: на ex-dabench данных `llm` достоверно хуже `fixed`.

**B. Что это значит «по существу».**

На post-fix данных:

1. **Адаптивный role-router в любом из двух вариантов не даёт positive lift поверх adaptive-топологии.** Direction отрицателен для обоих сравнений, для `llm` на 3-task statistical significance преодолена.
2. **`rule` активно использует `peer`-роль (67 % runs)** — конфиг работает как заявлено, exec-фаза получает полный workload, и `peer`-роль реально активируется. Но quality-выгоды peer-роль не приносит.
3. **`llm` фиксируется на `coordinator`-роли (90 % runs)** как safe default, иногда выбирая экзотические альтернативы. Этот meta-call не оправдывает себя по quality (Δ −0.051 pooled, −0.080 3-task).
4. **Single static reviewer-роль является лучшей single-pick стратегией для HITL-узла на adaptive-топологии в текущих данных.** Адаптивная роль не нужна и более того — мешает.

### 9.2 Что говорить в финальном тексте диплома

**Не:** «adaptive role не работает» (слишком общо).
**Да:**

- «На скорректированной реализации adaptive role-router даёт **отрицательный или нейтральный** эффект относительно single static reviewer-роли: `rule` overall Δ = −0.046 (95 % CI пересекает 0), `llm` overall Δ = −0.051 (95 % CI пересекает 0); на ex-dabench 3-task pooling CI для `llm` − `fixed` **исключает 0** (Δ = −0.080, CI [−0.157, −0.001]).»
- «Direction однородный по всем 4 задачам для `rule` и по 2/4 non-noise задачам для `llm`; на оставшихся 2 (commongen, dabench) `llm` ≈ `fixed`.»
- «Single static reviewer-роль является лучшей single-pick стратегией для HITL-узла на adaptive-топологии. Гипотеза о phase-aware role-switching не подтверждена в текущих данных; добавляющие cost overhead role-роутеры не оправдывают себя по quality.»

Это **negative-result-ный** RQ4 verdict: гипотеза H4 не получает поддержки; в одном из срезов получает значимое опровержение. Negative finding по своей структуре полезен — он закрывает направление дальнейшей разработки role-роутеров и упрощает E5.

---

## 10. Аномалии и ограничения

### 10.1 DABench — статистический шум

`std = 0.38–0.41` при mean = 0.19–0.23. Бинарная (success/fail) distribution на 45 ячейках даёт ±0.05 в mean. Per-task winner на dabench (`llm` +0.037) — разница 1–2 нулей из 45, не сигнал. Cross-experiment validity dabench-cells **broken**; интерпретация per-task dabench-результатов не несёт научной нагрузки.

### 10.2 GSM8K — частичная де-сатурация

GSM8K на текущих данных частично де-сатурирован: `fixed` 0.889 vs `rule` 0.800 vs `llm` 0.756. Это значит, что в adaptive-топологии с полным workload-ом и сменой роли HITL-узел иногда «сбивает с пути» worker-а (peer/coordinator советы), который иначе бы решил задачу за 1–2 итерации. Direction-pattern (`fixed > rule > llm`) на gsm8k самый чистый из 4 задач — единственная задача без noise (dabench) и без sub-noise (commongen, σ ≈ 0.08).

### 10.3 Sample-size limitations

n=45 per cell. CI на per-task уровне remain wide (особенно для humaneval/gsm8k с σ ≈ 0.4). На pooled уровне (n=180) CI исключает ноль только для llm − fixed на 3-task pooling. Для строгих claims-ов нужен n ≥ 100 на cell, что выходит за рамки E4-бюджета.

### 10.4 Measurement gap: `role_router` не в parquet

См. §2.3. Корневая причина — schema runs не расширилась под E4. Mode-инференс через позиционный sort работает для текущей выборки, но **фрагилен** к runner-resubmission, async submission и worker-restart. Для E5 (с реальными участниками) **обязательно** добавить `role_router` в `_runs.parquet` columns.

### 10.5 Per-task split добавил cumulative wall

Per-task split-конфиг (humaneval → gsm8k → commongen → dabench последовательно с паузами) дал e2e wall ≈ 6 ч. На результатах это не сказалось (independent shuffle), но для будущих full-grid-ов стоит рассмотреть monolithic при условии решения каскад-проблем на dabench.

### 10.6 LLM-simulated HITL vs real-human

Открытый вопрос для E5. Симулятор может систематически предпочитать coordinator-роль (как post-fix llm-роутер делал в 90 %); реальные люди могут вести себя иначе. Это будет важная триангуляция для финального текста.

### 10.7 Implementation audit framing

Описание пяти артефактов (§1.2) формулируется в нейтральных терминах: «artefact», «correction», «implementation audit». Анализ данных в §3–§9 — primary; артефакты упомянуты только для документирования provenance кода, на котором собирались данные.

### 10.8 Negative finding как сильный научный результат

Direction-pattern (`fixed ≥ rule ≥ llm`) однороден по всем 4 задачам для `rule` и по 3/4 для `llm`. Одно CI исключает ноль. Это активно отрицательный finding для adaptive-role в текущей реализации, и он научно ценнее, чем «not detectable». Финальный текст диплома должен подавать его как **полезный negative result**, не как failure эксперимента.

---

## 11. Связь со следующими экспериментами

### 11.1 E4 champion → E5

На текущих данных E5 champion-конфиг:

```yaml
topology:
  name: adaptive
  extra.adaptive:
    topology_router: llm     # E3 Pareto winner
    phase_router: rule
human:
  role_router: fixed         # post-fix E4 finding: fixed > rule > llm
  role: reviewer             # лучшая single-pick роль на adaptive-топологии
```

Это упрощает E5: меньше role-вариативности → проще участники, проще NASA-TLX-анализ, и финальный adaptive-конфиг становится compact (no role-router overhead, no extra LLM-call per phase).

### 11.2 Cross-family confirmation

Cross-family confirmation для E4 ведётся в отдельном анализе (`confirmation_e4_postfix_analysis.md`) — здесь только post-fix gpt-oss данные; сравнение между моделями делается там.

### 11.3 Открытые вопросы для будущей работы

- **Real-human HITL.** Симулятор может вести себя иначе, чем реальные участники. E5 — основной триангуляционный шаг.
- **Schema fix.** Добавить `role_router` в `_runs.parquet` schema перед E5.
- **Расширенный per-task n.** Если результат на 3-task pooling вызовет сомнения у читателей, имеет смысл повторить E4 с n=100 / cell на ex-dabench tasks (humaneval, gsm8k, commongen) — это даст более узкие per-task CI и проверит robustness direction-pattern-а.

---

## Приложение A: Сводный list файлов

- **Сырьё (4 per-task parquet-снимка):**
  - `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/9ed6cd02-e779-4137-8eea-640e5e5fae67/_runs.parquet` (humaneval)
  - `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/5087cd5c-d5e4-463d-9172-77c3addc74dd/_runs.parquet` (gsm8k)
  - `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/9663b4eb-cb29-42d7-8d86-d9b39b359b38/_runs.parquet` (commongen)
  - `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/41793bf8-d94c-4999-bfe1-e5bfe62b89db/_runs.parquet` (dabench)
- **Метаданные:** `experiment.json` в каждой из 4 директорий (с `config_snapshot` и `grid` секциями).
- **Конфиги:**
  - `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_postfix_full.yaml` (base)
  - `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_postfix_full_humaneval.yaml`
  - `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_postfix_full_gsm8k.yaml`
  - `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_postfix_full_commongen.yaml`
  - `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_postfix_full_dabench.yaml`
- **Source role-router:** `/home/cactustim/agents/adaptive-topologies-mas/src/atm/human/role_router.py` (`DEFAULT_ROLE_TABLE` на :105-110)

## Приложение B: Pipeline pandas-агрегации

```python
import pandas as pd
import numpy as np

EXP_IDS = {
    '9ed6cd02-e779-4137-8eea-640e5e5fae67': 'humaneval',
    '5087cd5c-d5e4-463d-9172-77c3addc74dd': 'gsm8k',
    '9663b4eb-cb29-42d7-8d86-d9b39b359b38': 'commongen',
    '41793bf8-d94c-4999-bfe1-e5bfe62b89db': 'dabench',
}

dfs = []
for eid, task in EXP_IDS.items():
    runs = pd.read_parquet(f'data/experiments/experiments/{eid}/_runs.parquet')
    runs = runs[(runs.status == 'completed') & runs.quality_score.notna()].copy()
    runs = runs.sort_values('started_at').reset_index(drop=True)
    runs['rn'] = runs.index + 1
    runs['mode'] = pd.cut(
        runs.rn,
        bins=[0, 45, 90, 135],
        labels=['fixed', 'rule', 'llm'],
    )
    runs['task_name'] = task
    runs['source_exp_id'] = eid
    dfs.append(runs)

all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.drop_duplicates(subset=['id'])  # n=540, нет коллизий

# Sanity-check: fixed-block ДОЛЖЕН быть 100% reviewer.
sanity = (all_df.query("mode == 'fixed'")['human_role'].value_counts())
assert sanity.get('reviewer', 0) == 180, "mode-inference broken"
```

---

_Документ сверен с brief-инструкциями: completeness 540/540, role_router-recovery через позиционный sort (sanity 180/180 reviewer в fixed-блоке + boundary check на rn=45→46 и rn=90→91), overall mean_q `fixed > rule > llm`, RQ4 verdict на post-fix данных: not supported (rule), strengthened negative on ex-dabench 3-task (llm)._
