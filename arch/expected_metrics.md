# Expected metrics — E1–E3

Ожидаемые значения метрик для базовых экспериментов. Сетап: workers = `cerebras:llama-3.1-8b`, Critic = `cerebras:llama-3.1-70b`, Judge = `openai:gpt-4o` (self-consistency×3). Task-mix: HumanEval (50), GSM8K (50), CommonGen (50), InfiAgent-DABench (30).

Числа — ожидания, не результаты. Источник: литература по multi-agent на weak models + baseline-скоры llama-3.1-8b на целевых бенчах.

---

## E1 — Baseline static, no human (RQ1)

**Дизайн:** 5 топологий × 4 task_types × 25 задач × 3 seeds = 1500 runs.

### Quality per task_type (top-1 топология)

| Task | Single-agent baseline | Top-1 топология | Top-1 quality | Top-2 топология |
|---|---|---|---|---|
| HumanEval | 0.70 | Chain | 0.78–0.85 | Star |
| GSM8K | 0.76 | Debate | 0.82–0.88 | Star |
| CommonGen | 0.48 | Debate | 0.60–0.72 | Mesh |
| InfiAgent-DABench | 0.45 | Chain | 0.55–0.68 | Star |

### Статистика различимости топологий

| Метрика | Ожидание |
|---|---|
| Cohen's d (best vs worst топология per type) | 0.6–1.2 (medium-to-large) |
| ANOVA p (topology main effect) | <0.01 на всех 4 типах |
| Failure rate (среднее) | 8–18% |
| Failure rate Mesh / Hierarchical | до 20% |

### Cost per run (USD)

| Топология | Workers (Cerebras) | Judge (gpt-4o) | Всего |
|---|---|---|---|
| Chain | $0.00030 | $0.0050 | ~$0.0053 |
| Star | $0.00050 | $0.0050 | ~$0.0055 |
| Debate | $0.00080 | $0.0050 | ~$0.0058 |
| Mesh | $0.00100 | $0.0050 | ~$0.0060 |
| Hierarchical | $0.00150 | $0.0050 | ~$0.0065 |

Judge доминирует в total cost. Workers — шум.

### Wall time per run (s)

| Топология | Время |
|---|---|
| Chain | 2–5 |
| Star | 5–10 |
| Debate | 8–15 |
| Mesh | 12–20 |
| Hierarchical | 15–25 |

### Iterations per run

| Топология | Median iterations |
|---|---|
| Chain | 2–4 |
| Star | 4–8 |
| Debate | 2–4 rounds |
| Mesh | 3–6 rounds |
| Hierarchical | 2 external × 3–5 internal |

---

## E2 — HITL-simulated baseline (RQ3)

**Дизайн:** ~22 осмысленных (topology, role) пар × 4 task_types × 15 задач × 3 seeds ≈ 4000 runs.

### Δquality от роли (среднее по топологиям)

| Роль | Δquality vs E1 | Комментарий |
|---|---|---|
| Reviewer | +5 … +10% | наиболее consistent эффект |
| Judge | +7 … +12% | на Debate/reasoning natural fit |
| Coordinator | +5 … +9% | сильный на Mesh/Hierarchical, слабый на Chain |
| Peer | +2% ± 5% | high variance, иногда negative |
| Monitor | 0 … +2% | не quality, а stability |

### Прочие эффекты ролей

| Метрика | Ожидание |
|---|---|
| Δfailure_rate, Monitor | −15 … −25% |
| Cost overhead от HITL-sim (human_sim_tokens) | +20 … +40% к E1 |
| 2-way ANOVA: role main effect η² | 0.10–0.15 |
| 2-way ANOVA: role × topology interaction η² | 0.03–0.07 |

### Champion (topology, role) per task_type — предварительные фавориты

| Task | Champion config |
|---|---|
| HumanEval | Chain + Reviewer |
| GSM8K | Debate + Judge |
| CommonGen | Debate + Judge / Mesh + Coordinator |
| InfiAgent-DABench | Chain + Reviewer / Star + Monitor |

---

## E3 — Adaptive topology (RQ2)

**Дизайн:** 4 task_types × 25 задач × 3 seeds × 3 router-режима + baseline ≈ 1100 runs. Router-режимы: rule_based, llm_router, oracle_loo, oracle_manual.

### Сводная таблица по router'ам

| Метрика | rule_based | llm_router | oracle_loo | oracle_manual |
|---|---|---|---|---|
| Δquality vs best-static | +4 … +8% | +3 … +7% | +8 … +15% | +5 … +12% |
| Δcost vs best-static | −10 … −20% | 0 … +5% | 0% | 0% |
| hurt_rate | 18–28% | 15–25% | 8–12% | 12–18% |
| router_cost_share | 0% | 8–15% | 0% | 0% |
| oracle_gap_loo (Δ до LOO oracle) | 4–8% | 3–6% | 0% | 2–5% |

### Adaptive-специфичные метрики

| Метрика | Ожидание |
|---|---|
| topology_switch_count (median per run) | 2–4 |
| topology_switch_count (max per run) | 5–7 |
| guard_override_rate | 15–30% |
| Частые switch-и | `stuck` → Mesh, `rejected_count≥3` → Debate |
| Фаза с наибольшей частотой switch | execution |

### Прохождение порога «Adaptive выигрывает» (≥5% quality ИЛИ ≥15% cost)

| Task | rule_based | llm_router | Комментарий |
|---|---|---|---|
| HumanEval | вероятно | 50/50 | Chain уже близок к оптимуму |
| GSM8K | 50/50 | вероятно нет | Debate уже силён как static |
| CommonGen | вероятно по cost | вероятно по cost | ROUGE-дельты малы, Debate дорог |
| InfiAgent-DABench | почти точно | почти точно | сильный разрыв топологий + code sandbox |

**Итого: 3 из 4 task_types** ожидаем проходят порог хотя бы на одном router'е.

---

## Красные флаги (что мониторим на pilot-E1)

| Условие | Действие |
|---|---|
| `failure_rate > 20%` на HumanEval/InfiAgent | tool calling не вывозит на 8b → апгрейд Executor+Critic на 70b |
| `oracle_gap_loo < 3%` на всех типах | даже идеальный router не поможет → менять task-mix или workers на 70b |
| `hurt_rate > 35%` для rule_based | правила слишком агрессивные → ослабить guards |
| `router_cost_share > 20%` у llm_router | перенести router на 70b |
| `Δquality < 3%` между всеми топологиями | floor effect → менять бенчи или workers на 70b |

---

## Итоговый нарратив для презентации (по RQ)

**RQ1 (E1):** Топологии различимы на слабой модели. Chain доминирует на programming/numeric, Debate — на reasoning/creative. Effect size medium-to-large (d=0.6–1.2).

**RQ3 (E2):** Human role даёт заметный вклад, Reviewer и Judge — наиболее полезные роли. Monitor работает не через quality, а через stability (failure_rate ↓).

**RQ2 (E3):** Adaptive выигрывает на 3/4 task_types хотя бы по одному порогу. Oracle_loo показывает потолок +8–15% quality — значит у адаптации есть чем расти. rule_based безопаснее llm_router (ниже router_cost_share, hurt_rate сопоставим).



# Сравнительные таблицы (точечные оценки)

Все числа ниже — **точечные оценки** (середины диапазонов выше), чтобы можно было напрямую положить на слайды. Метрики — по именам полей из `arch.md §3.4` и таблицы `runs`.

## E1 — Topology × Task_type grids

### `quality_score` (среднее по 25 задачам × 3 seeds)

| Topology \ Task | HumanEval | GSM8K | CommonGen | InfiAgent-DABench |
|---|---|---|---|---|
| Single-agent (baseline) | 0.70 | 0.76 | 0.48 | 0.45 |
| **Chain** | **0.82** | 0.78 | 0.52 | **0.62** |
| **Star** | 0.76 | 0.80 | 0.58 | 0.58 |
| **Debate** | 0.72 | **0.85** | **0.66** | 0.50 |
| **Mesh** | 0.68 | 0.78 | 0.64 | 0.54 |
| **Hierarchical** | 0.74 | 0.77 | 0.56 | 0.52 |

Жирным — победитель столбца (top-1 per task_type). `winners_E1` → Chain (HE, IA), Debate (GSM, CG).

### `total_usd` (USD per run, среднее по task_types)

| Topology | Workers (Cerebras) | Judge (gpt-4o) | Total | Rank |
|---|---|---|---|---|
| Chain | $0.00030 | $0.00500 | **$0.0053** | cheapest |
| Star | $0.00050 | $0.00500 | $0.0055 | 2 |
| Debate | $0.00080 | $0.00500 | $0.0058 | 3 |
| Mesh | $0.00100 | $0.00500 | $0.0060 | 4 |
| Hierarchical | $0.00150 | $0.00500 | **$0.0065** | most expensive |

### `wall_time_s` (median per run)

| Topology | Wall time (s) |
|---|---|
| Chain | 3.5 |
| Star | 7.5 |
| Debate | 11.5 |
| Mesh | 16.0 |
| Hierarchical | 20.0 |

### `failure_rate` (доля runs со status ≠ 'completed')

| Topology \ Task | HumanEval | GSM8K | CommonGen | InfiAgent |
|---|---|---|---|---|
| Chain | 0.10 | 0.06 | 0.05 | 0.15 |
| Star | 0.12 | 0.08 | 0.07 | 0.16 |
| Debate | 0.15 | 0.10 | 0.08 | 0.20 |
| Mesh | 0.18 | 0.12 | 0.10 | 0.22 |
| Hierarchical | 0.20 | 0.14 | 0.12 | 0.25 |

### `iterations` (median per run)

| Topology | Median iterations |
|---|---|
| Chain | 3 |
| Star | 6 |
| Debate | 3 rounds |
| Mesh | 4 rounds |
| Hierarchical | 2 ext × 4 int |

---

## E2 — Topology × Role heatmap

### `Δquality_score` vs E1 (absolute delta)

| Topology \ Role | Coordinator | Reviewer | Judge | Peer | Monitor |
|---|---|---|---|---|---|
| Star | +0.08 | +0.08 | +0.07 | +0.03 | +0.01 |
| Chain | +0.05 | **+0.10** | +0.06 | — *excl.* | +0.01 |
| Mesh | +0.09 | +0.06 | +0.08 | +0.04 | +0.02 |
| Debate | — *excl.* | +0.05 | **+0.12** | +0.02 | +0.01 |
| Hierarchical | +0.09 | +0.07 | +0.07 | — *excl.* | +0.01 |

Жирным — топ-ячейки. Excluded — (topology, role), запрещённые в плане (§3).

### `human_interactions_count` (per run, среднее)

| Role | Interactions per run |
|---|---|
| Coordinator | 6–10 |
| Reviewer | 3–5 |
| Judge | 1–2 |
| Peer | 2–4 |
| Monitor | 1–3 |

### Cost overhead от HITL-sim (per role, `human_sim_tokens` → additional USD)

| Role | Δcost vs E1 | Причина |
|---|---|---|
| Coordinator | +35% | активен на каждом шаге |
| Reviewer | +30% | ревью каждого draft |
| Judge | +25% | финальный verdict |
| Peer | +20% | opinion on demand |
| Monitor | +15% | observer, редко вмешивается |

### `Δfailure_rate` (доля failed runs vs E1)

| Role | Δfailure_rate |
|---|---|
| Coordinator | −0.03 |
| Reviewer | −0.04 |
| Judge | −0.02 |
| Peer | +0.01 |
| Monitor | **−0.05** |

Monitor — единственная роль, основной value которой в stability, не в quality.

---

## E3 — Adaptive vs best-static (per task_type)

### Best-static baseline (из E1)

| Task | Best-static topology | `quality_score` | `total_usd` |
|---|---|---|---|
| HumanEval | Chain | 0.82 | $0.0053 |
| GSM8K | Debate | 0.85 | $0.0058 |
| CommonGen | Debate | 0.66 | $0.0058 |
| InfiAgent-DABench | Chain | 0.62 | $0.0053 |

### `quality_score` per (task × router)

| Task | best_static | rule_based | llm_router | oracle_loo | oracle_manual |
|---|---|---|---|---|---|
| HumanEval | 0.820 | 0.850 | 0.840 | 0.910 | 0.880 |
| GSM8K | 0.850 | 0.870 | 0.860 | 0.900 | 0.880 |
| CommonGen | 0.660 | 0.700 | 0.690 | 0.750 | 0.720 |
| InfiAgent | 0.620 | 0.680 | 0.670 | 0.720 | 0.690 |
| **mean** | 0.738 | **0.775** | 0.765 | **0.820** | 0.793 |

### `Δquality_score` vs best-static (абсолютные)

| Task | rule_based | llm_router | oracle_loo | oracle_manual |
|---|---|---|---|---|
| HumanEval | +0.030 (+3.7%) | +0.020 (+2.4%) | +0.090 (+11.0%) | +0.060 (+7.3%) |
| GSM8K | +0.020 (+2.4%) | +0.010 (+1.2%) | +0.050 (+5.9%) | +0.030 (+3.5%) |
| CommonGen | +0.040 (+6.1%) | +0.030 (+4.5%) | +0.090 (+13.6%) | +0.060 (+9.1%) |
| InfiAgent | +0.060 (+9.7%) | +0.050 (+8.1%) | +0.100 (+16.1%) | +0.070 (+11.3%) |
| **mean** | +0.038 (+5.5%) | +0.028 (+4.1%) | +0.083 (+11.6%) | +0.055 (+7.8%) |

### `total_usd` per run (task × router)

| Task | best_static | rule_based | Δ rule | llm_router | Δ llm |
|---|---|---|---|---|---|
| HumanEval | $0.0053 | $0.0043 | −18% | $0.0058 | +9% |
| GSM8K | $0.0058 | $0.0051 | −12% | $0.0062 | +7% |
| CommonGen | $0.0058 | $0.0050 | −14% | $0.0062 | +7% |
| InfiAgent | $0.0053 | $0.0042 | −20% | $0.0057 | +8% |
| **mean** | $0.0055 | $0.0046 | **−16%** | $0.0060 | **+8%** |

### `hurt_rate` per (task × router)

| Task | rule_based | llm_router | oracle_loo | oracle_manual |
|---|---|---|---|---|
| HumanEval | 0.22 | 0.18 | 0.10 | 0.14 |
| GSM8K | 0.26 | 0.22 | 0.12 | 0.18 |
| CommonGen | 0.18 | 0.16 | 0.08 | 0.12 |
| InfiAgent | 0.20 | 0.18 | 0.10 | 0.14 |

### Adaptive-специфичные метрики (среднее по всем тасcам)

| Metric | rule_based | llm_router | oracle_loo |
|---|---|---|---|
| `topology_switch_count` (median) | 2 | 3 | 1 |
| `topology_switch_count` (max) | 5 | 7 | 3 |
| `topology_switch_attempts` (median) | 4 | 6 | 1 |
| `guard_override_rate` | 0.20 | 0.25 | 0.03 |
| `router_cost_share` | 0.00 | 0.11 | 0.00 |
| `oracle_gap_loo` (Δ to LOO) | +5.8 pp | +6.7 pp | 0 |
| `oracle_gap_manual` | +2.3 pp | +3.2 pp | — |

### Прохождение порога «Adaptive выигрывает» (≥5% quality ИЛИ ≥15% cost savings)

| Task | rule_based (q/c) | rule passes? | llm_router (q/c) | llm passes? |
|---|---|---|---|---|
| HumanEval | +3.7% / −18% | ✓ (cost) | +2.4% / +9% | ✗ |
| GSM8K | +2.4% / −12% | ✗ | +1.2% / +7% | ✗ |
| CommonGen | +6.1% / −14% | ✓ (quality) | +4.5% / +7% | ✗ |
| InfiAgent | +9.7% / −20% | ✓ (both) | +8.1% / +8% | ✓ (quality) |
| **Итого** | | **3 / 4** | | **1 / 4** |

**Union (rule_based OR llm_router): 3/4 task_types** проходят порог хотя бы на одном router'е. GSM8K — единственный type, где adaptive не даёт значимого выигрыша (Debate как best-static уже близок к потолку для 8b-модели).

---

## Сводный слайд (one-liner per RQ)

| RQ | Главная цифра | Интерпретация |
|---|---|---|
| RQ1 (E1) | d=0.6–1.2 между лучшей/худшей топологией | Топологии различимы, Chain/Debate доминируют |
| RQ3 (E2) | Reviewer +10% на Chain, Judge +12% на Debate | HITL-роль сильно зависит от топологии; Monitor снижает failure_rate на 5pp |
| RQ2 (E3) | Adaptive (rule_based) +5.5% quality / −16% cost в среднем | Adaptive обгоняет best-static на 3/4 типах; oracle_gap_loo ~6pp — есть куда расти |
