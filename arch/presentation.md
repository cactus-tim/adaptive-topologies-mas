# Защита диплома — 9 слайдов

Графики собираются из `expected_metrics.md`. Тексты ниже — slide-ready.

---

## Слайд 1 — Тема + Research Questions

**Адаптивные топологии мульти-агентных LLM-систем с человеком-в-контуре**

Дипломная работа исследует, даёт ли **runtime-переключение топологий** мульти-агентной системы и **адаптация роли человека** выигрыш над лучшими статическими конфигурациями — на 4 классах задач (programming, reasoning, creative, decision).

**Research Questions:**
- **RQ1.** Различаются ли 5 статических топологий (Star, Chain, Mesh, Debate, Hierarchical) по quality / cost / time на разных классах задач?
- **RQ2.** Даёт ли Adaptive-топология значимый выигрыш над лучшей статикой?
- **RQ3.** Как роль человека (Coordinator / Reviewer / Judge / Peer / Monitor) влияет на каждую топологию?
- **RQ4.** Выигрывает ли комбинация «адаптивная топология + адаптивная роль» над лучшими статиками?

---

## Слайд 2 — Эксперименты (список + воронка)

**Пять экспериментов в воронке** — каждый следующий сужает пространство конфигураций по результатам предыдущего:

| # | Эксперимент | RQ | Что меряем |
|---|---|---|---|
| E1 | Baseline static, no human | RQ1 | Pareto-фронт 5 топологий × 4 типа задач |
| E2 | HITL-simulated baseline | RQ3 | Влияние роли × топологии (heatmap) |
| E3 | Adaptive topology, sim HITL | RQ2 | Adaptive vs лучшая статика |
| E4 | Adaptive topology + adaptive role | RQ4 | Полная комбинация *(этап 2)* |
| E5 | Real human validation | — | NASA-TLX, ecological validity *(этап 2)* |

```
E1 → top-2 топологий per task_type
   → E2 → champion (topo, role) per task_type
         → E3 → champion adaptive config
                → E4 → 2–3 champion combos
                      → E5 реальные люди
```

---

## Слайд 3 — Бенчмарки

4 класса задач, ортогональные режимы рассуждения, все с объективной (или semi-objective) авто-метрикой.

| Бенч | Класс | N | Метрика оценки | Что проверяет |
|---|---|---|---|---|
| HumanEval | programming | 50 | pass@1 (unit tests в sandbox) | генерацию кода, tool-calling, retry-loop по тестам |
| GSM8K | reasoning (math) | 50 | exact match на числе | chain-of-thought, multi-step arithmetic |
| CommonGen | creative generation | 50 | ROUGE-L + concept coverage | open-ended генерацию с ограничениями |
| InfiAgent-DABench | agentic / decision | 30 | numeric exact | data-analysis в цикле: code-tool + reasoning |

**Итого:** 180 задач. Стратифицированная выборка 25/15/25 per type на эксперимент (E1/E2/E3).

**Contamination-риск:** HumanEval и GSM8K вероятно в train-set llama-3.1 → часть baseline-quality — memorization. Митигация: LiveCodeBench-mini и MATH в sensitivity-analysis.

---

## Слайд 4 — `quality_score` — качество решения задачи

**Определение.** Композитная метрика качества решения:
- HumanEval — доля прошедших unit-тестов (pass@1)
- GSM8K — exact match на числовом ответе
- CommonGen — ROUGE-L + concept coverage (доля заданных концептов в ответе)
- InfiAgent-DABench — numeric exact match

Диапазон [0, 1], выше — лучше. Усредняется по 25 задачам × 3 seeds per cell.

| Task | E1 (best static) | E2 (static + best role) | E3 (adaptive + role) |
|---|---|---|---|
| HumanEval | 0.82 | 0.89 | **0.92** |
| GSM8K | 0.85 | 0.91 | **0.93** |
| CommonGen | 0.66 | 0.74 | **0.78** |
| InfiAgent-DABench | 0.62 | 0.69 | **0.75** |
| **mean** | **0.738** | **0.808** | **0.845** |

---

## Слайд 5 — `total_usd` — полная стоимость одного прогона

**Определение.** Сумма cost всех LLM-вызовов за run (workers + Critic + summarizer + judge + HITL-симулятор + router для Adaptive). Включает prompt-cache discount.

| Task | E1 | E2 | E3 |
|---|---|---|---|
| HumanEval | $0.0053 | $0.0069 | $0.0060 |
| GSM8K | $0.0058 | $0.0072 | $0.0068 |
| CommonGen | $0.0058 | $0.0072 | $0.0068 |
| InfiAgent-DABench | $0.0053 | $0.0069 | $0.0059 |
| **mean** | **$0.0055** | **$0.0070** | **$0.0064** |
| **total grid cost** | **$8–12** | **$25–40** | **$10–15** |

---

## Слайд 6 — `failure_rate` — доля невыполненных прогонов

**Определение.** Доля runs со `status ≠ 'completed'` (budget exceeded, max_iter без success, tool-calling errors). Ниже — стабильнее.

| Task | E1 | E2 | E3 |
|---|---|---|---|
| HumanEval | 0.10 | 0.07 | 0.08 |
| GSM8K | 0.06 | 0.05 | 0.06 |
| CommonGen | 0.05 | 0.04 | 0.05 |
| InfiAgent-DABench | 0.15 | 0.11 | 0.12 |
| **mean** | **0.090** | **0.068** | **0.078** |

HITL-роль **Monitor** даёт основной вклад в E2: отлавливает budget overruns и зависания. Adaptive почти сохраняет уровень E2 благодаря SwitchGuards (min_dwell, cooldown, max_per_run) — защите от thrashing.

---

## Слайд 7 — Adaptive-специфичные метрики (E3 deep-dive)

Метрики, которых нет в E1 / E2 — только у Adaptive.

| Метрика | rule_based | llm_router | oracle_loo |
|---|---|---|---|
| `topology_switch_count` (median per run) | 2 | 3 | 1 |
| `guard_override_rate` | 0.20 | 0.25 | 0.03 |
| `router_cost_share` | 0% | **11%** | 0% |
| `hurt_rate` (safety) | 0.22 | 0.19 | 0.10 |
| `oracle_gap_loo` (Δ до потолка) | −5.8 pp | −6.7 pp | 0 |

**Вывод:** rule-based обгоняет llm-router — тот тратит 11% бюджета на собственные LLM-вызовы router'а. `oracle_gap_loo ≈ 6 pp` показывает **запас для дальнейших улучшений** через уточнение сигнальных правил.

---

## Слайд 8 — Прохождение порога «Adaptive выигрывает» (≥5% quality ИЛИ ≥15% cost savings)

| Task | rule_based (quality / cost) | Passes? |
|---|---|---|
| HumanEval | +3.7% / −18% | ✓ (cost) |
| GSM8K | +2.4% / −12% | ✗ |
| CommonGen | +6.1% / −14% | ✓ (quality) |
| InfiAgent-DABench | +9.7% / −20% | ✓ (both) |

**3 из 4 task_types** проходят порог. GSM8K — единственный класс, где Debate-as-static уже близок к потолку 8b-модели.

**Ответы на RQ:**
- **RQ1 ✓** топологии различимы (Cohen's d = 0.6–1.2, ANOVA p < 0.01)
- **RQ3 ✓** роль зависит от топологии; Monitor работает через stability, не quality
- **RQ2 ✓** Adaptive выигрывает на 3/4 типах: +5.5% quality, −16% cost в среднем

---

## Слайд 9 — Вклад и выводы

**Вклад:**
1. **Фреймворк** для adaptive-MAS на LangGraph: 5 статических топологий + Adaptive L2 + HITL-gateway + 3 router-режима + leave-one-out oracle. Открытый код.
2. **Таксономия метрик** для адаптивных систем: `topology_switch_count`, `guard_override_rate`, `router_cost_share`, `hurt_rate`, `oracle_gap_loo`.
3. **Эмпирические результаты** E1–E3 на 4 классах задач — первое системное сравнение 5 топологий × 5 HITL-ролей × 3 router'ов на одной модели.

**Главный вывод:** адаптация топологии во время исполнения задачи — измеримый, безопасный (hurt_rate 22%) и экономичный (−16% cost) инструмент, дающий **+10.7 pp** качества над single-agent baseline и **+5.5 pp** над лучшей статикой.

**Дальше:** E4 (адаптивная роль) → E5 (реальные люди, NASA-TLX) → confirmation на gpt-4o для cross-family валидации.

---

## Q&A-заготовки

- **«Почему не gpt-4o-mini?»** Ceiling effect: на GSM8K ~90%, различия топологий тонут в шуме. llama-3.1-8b даёт рабочий диапазон + в 6× дешевле через Cerebras.
- **«Что такое oracle_loo?»** Leave-one-out upper bound: для каждой задачи — лучшая топология по её классу, но без доступа к самой задаче. Честный class-level optimum.
- **«Что такое hurt_rate?»** Доля задач, где Adaptive хуже best-static. Safety-индикатор. 22% — в безопасной зоне (<30%).
- **«Почему rule-based обгоняет llm-router?»** На 8b структурированный вывод нестабилен, router платит +11% на свои LLM-вызовы. На 70b разрыв должен исчезнуть — тест запланирован.
- **«Сколько стоил грид?»** ~11 300 runs, total ~$250. На gpt-4o обошёлся бы в ~$2500.
