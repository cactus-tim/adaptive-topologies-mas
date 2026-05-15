# Experiment plan — Adaptive Topologies MAS

Экспериментальный план под RQ1–RQ4 диплома. Стратегия — **funnel**: каждый следующий этап сужает пространство конфигураций, опираясь на результаты предыдущего, чтобы избежать комбинаторного взрыва. HITL симулируется LLM-вызовом на этапах 1–4; реальный человек — только на этапе 5, на курированном subset.

## 0. Task-mix (фиксирован)

| Тип | Датасет | N | Оценка |
|---|---|---|---|
| Programming | HumanEval (или HumanEval+) | 50 | unit tests |
| Reasoning | GSM8K | 50 | exact match |
| Creative | CommonGen | 50 | ROUGE + concept coverage |
| Decision | InfiAgent-DABench (numeric subset) | 30 | numeric exact |

**Итого: 180 задач.** Для экспериментов ниже обычно берём **стратифицированную выборку 20–30 задач на тип** (т.е. ~80–120 на прогон), чтобы не взрывать бюджет. Полный прогон 180 делаем только на финальных confirmation runs.

Модели (lineup Variant B'.1, ревизия 2026-05-15 после deprecation Cerebras Llama-family):
- **primary** (workers Planner/Researcher/Executor/Debater, Critic, Summarizer, Router, HITL-simulator) = `cerebras:gpt-oss-120b` (~3000 tok/s на Cerebras WSE; иерархия worker vs critic выражается через `reasoning_effort` и system-prompt strictness, а не размер модели — переменная модели изолирована)
- **judge** (LLM-as-judge, self-consistency × 3, temperature 0.0) = `openai:gpt-4.1-mini` (другая семья весов — независимая оценка; ключевой аргумент для защиты)
- **confirmation primary** (cross-family внутри Cerebras) = `cerebras:zai-glm-4.7` (preview, 355B Z.ai family — другая родословная весов от gpt-oss)
- **confirmation cross-family** = `openai:gpt-4o` (frontier OpenAI — финальная валидация)

Seeds: **3 по умолчанию, 5 для финальных confirmation**. Температура primary = 0.7; judge = 0.0.

> **Note (2026-05-15):** Изначальная редакция плана опиралась на `gpt-4o-mini`/`gpt-4o`. После добавления Cerebras-провайдера и deprecation Llama-семьи в каталоге Cerebras (2026-05-27) lineup пересмотрен. Старые модели (`gpt-4o-mini` primary) сохранены в `conf/pricing.yaml` для back-compat тестов, но в `conf/experiments/*.yaml` не используются.

## 1. Этапы и воронка

```
 E1 Baseline no-human (RQ1)
      ↓ выбираем top-2 топологии per task type
 E2 HITL-sim baseline (RQ3)
      ↓ выбираем лучшую (topology, role) per task type + per phase
 E3 Adaptive topology, sim HITL (RQ2)
      ↓ проверяем выигрыш Adaptive над статикой
 E4 Adaptive + adaptive-role (RQ4)
      ↓ выбираем 2–3 champion configs
 E5 Real human validation (ecological validity)
```

---

## 2. E1 — Baseline static, no human (RQ1)

**Цель:** Pareto-фронт "quality × cost × time" по 5 статическим топологиям на 4 типах задач.

**Дизайн:**
- 5 топологий: Star, Chain, Mesh, Debate, Hierarchical
- 4 типа задач × 25 задач выборка × 3 seed
- Human role: `none` (топологии без HITL-узла)
- **Runs: 5 × 4 × 25 × 3 = 1500**

**Метрики:** quality_score (auto), total_tokens, total_usd, wall_time, iterations, failure_rate.

**Анализ:**
- per (topology × task_type): mean quality, mean cost, Pareto frontier
- ANOVA/Kruskal по качеству; bootstrap CI для costs
- выбрать **top-2 топологии per task type** → `winners_E1`

**Sanity checks перед запуском:**
- dry-run на 5 задачах: итерации сходятся, budget-гварды срабатывают, LLM-calls пишутся
- фиксация версий моделей (snapshot), git sha

**Бюджет (оценка):** 1500 runs × ~15 calls × 400 in + 300 out токенов ≈ $20–30.

---

## 3. E2 — HITL-simulated baseline (RQ3)

**Цель:** влияние роли человека на каждую топологию (на LLM-симуляторе).

**Дизайн:**
- 5 топологий × 5 ролей (Coordinator, Reviewer, Judge, Peer, Monitor) × 4 task types × **15 задач (stratified) × 3 seed**
- но **не все комбинации осмысленны** — исключаем:
  - Debate × Coordinator (coordinator ≡ judge в debate)
  - Chain × Peer (нет места для peer в строго-линейном пайплайне)
  - Hierarchical × Peer (peer ломает иерархию)
- итого ~22 (topology, role) пар × 4 × 15 × 3 ≈ **~4000 runs**
- HITL-симулятор: один и тот же LLM-агент с системным промптом роли, фиксированный persona-seed

**Метрики:** всё из E1 + `human_interactions_count`, `human_sim_tokens`, дельты к E1 (Δquality, Δcost, Δtime).

**Анализ:**
- 2-way ANOVA: role × topology на quality
- per task_type: ранжирование ролей, heatmap role×topology
- выделить **champion (topology, role) per task_type**

**Бюджет:** ~4000 × 18 calls × 500×400 ≈ $60–90.

---

## 4. E3 — Adaptive topology, simulated HITL (RQ2)

**Цель:** показывает ли runtime-переключение топологий выигрыш над лучшей статикой. Архитектура Adaptive — L2 (intra-phase + phase-bound switching), см. `arch.md §7.7, §8bis`.

**Дизайн:**
- Adaptive-topology с фиксированной ролью (берём лучшую per task_type из E2 как контекст-зависимый default)
- **3 режима `TopologyRouter`** (ablation):
  - `rule_based` — decision tree по таблице (phase × signals) → topology
  - `llm_router` — LLM-prompt "выбери топологию из {...}"; temperature=0; Pydantic-валидация; fallback на rule при parse-error
  - `oracle` — upper bound; см. «Oracle sources» ниже
- Baseline: лучшая статика per task_type (из E1)
- 4 task_types × 25 задач × 3 seed × 3 режима router + baseline ≈ **~1100 runs**

**Oracle sources (2 варианта, прогоняются оба):**

1. **Leave-one-out pseudo-oracle (основной).** Автоматический, честный.
   - Источник: E1 уже дал таблицу `(task_id, topology) → quality_score` для всех 5 топологий.
   - Для каждой задачи `t` при построении oracle *исключаем её из агрегации* и вычисляем `best_topology_for_type(task_type(t)) = argmax_topology mean_quality(topology, {same task_type} \ {t})`.
   - Oracle-router на задаче `t`: читает её `task_type`, возвращает соответствующую `best_topology`, игнорирует state.
   - **Что это измеряет:** потолок для адаптации на уровне task_type — «насколько помогает знание, какая топология лучшая для *класса* задач» без доступа к ответу на конкретную задачу. Честный upper bound; в paper защищается как «class-level optimal».
   - Pipeline: `analysis/oracle.py::build_leave_one_out_oracle(exp_id)` → `data/oracle/e1_leave_one_out.json`.

2. **Type-level manual oracle (sanity-check).** 4 task_types × 3 phases = 12 клеток ручной разметки.
   - Формат: `conf/oracle/type_level_manual.yaml` с парами `(task_type, phase) → topology` на основе интуиции (30 мин работы).
   - Прогоняется как **отдельный router** параллельно с (1).
   - **Что это измеряет:** расхождение между интуицией и emergent data. Если совпало — подтверждение; если нет — интересный observation для текста диплома.
   - Phase-level granularity здесь допустима: manual-разметка явно указывает «для programming в planning — star, в execution — chain». При runtime при смене фазы oracle переключает. Это **единственное** место, где phase-level oracle осмыслен (для автоматического (1) phase-level labels дороги и шумны — решение оставить только run-level).

**Метрики** (все определения — в `arch.md §3.4` + `topology_transitions`-аналитика):

- Базовые: `quality_score`, `total_tokens`, `total_usd`, `wall_time_s`, `iterations`, `failure_rate`
- Adaptive-специфичные:
  - `topology_switch_count` — per run (фактические switch'и, без no-change)
  - `topology_switch_attempts` — все решения router'а (включая no-change и guard_override)
  - `guard_override_rate` = |guards_applied > 0| / всего decisions
  - `router_cost_share` = sum(router_cost_usd) / total_usd — критично: «Adaptive платит за решения больше, чем экономит?»
  - `time_per_topology`, `iters_per_topology` (из timeline'а transitions)
  - `oracle_gap_loo` = `quality_oracle_loo` − `quality_<router>` — разрыв до потолка leave-one-out
  - `oracle_gap_manual` = `quality_oracle_manual` − `quality_<router>`
  - `hurt_rate` = доля задач, где Adaptive хуже best-static — ключевой safety-индикатор

**Анализ:**
- paired t-test: каждый router Adaptive vs best-static per task_type
- эффект-сайз (Cohen's d) — адаптация даёт **≥ 5%** прирост качества или **≥ 15%** экономию стоимости → считаем преимуществом
- анализ timeline `topology_transitions`: где switch'и реально помогают (коррелируют с ростом quality), а где шумят
- сравнение `oracle_gap_loo` vs `oracle_gap_manual` — насколько manual-интуиция совпадает с data-driven потолком
- hurt rate — на какой доле задач Adaptive хуже статики
- router-cost Pareto: для llm_router отдельно показать, где gain > router_cost (чистый выигрыш) vs где gain ≤ router_cost (адаптация съедает сама себя)

**Бюджет:** ~$35–55 (+ чуть-чуть на LLM-router overhead, ~10% к базовому E3).

---

## 5. E4 — Adaptive topology + adaptive role (RQ4)

**Цель:** оптимальная комбо topology+role+phase.

**Дизайн:**
- Adaptive-topology + Adaptive-role (роль человека меняется по фазе: e.g. planning=Coordinator, execution=Monitor, verification=Reviewer)
- **3 стратегии смены роли** (ablation):
  - `role_fixed_best` (роль из E2, не меняется)
  - `role_by_phase_rule` (таблица "фаза → роль" из E2 heatmap)
  - `role_by_phase_llm` (LLM решает)
- 4 task_types × 25 задач × 3 seed × 3 стратегии = ~900 runs
- Сравнение с baseline'ами из E2 и E3

**Метрики:** + `human_sim_cognitive_load_proxy` (прокси TLX по симулятору — кол-во interrupt'ов, длина контекста, latency ответа).

**Анализ:**
- 3-way сравнение: static+static-role vs adaptive-topo+static-role vs adaptive-both
- выбрать **2–3 champion configs** (один "quality-max", один "cost-min", один "balanced") для E5

**Бюджет:** ~$30–40.

---

## 6. Между-этапные confirmation runs

После E3 и E4 — повтор champion configs на полном task-mix (180 задач) с n=5 seeds, на двух confirmation-моделях параллельно (cross-family валидация):
1. **`cerebras:zai-glm-4.7`** — другая семья весов на том же провайдере (изолирует "модель" от "провайдера/инфраструктуры").
2. **`openai:gpt-4o`** — другой провайдер + другая семья — финальная frontier-валидация.

Цель: убедиться, что выводы не артефакт ни конкретной семьи весов (gpt-oss), ни конкретной инфраструктуры (Cerebras).

~2 configs × 180 задач × 5 seeds × 2 confirmation-моделей = **~3600 confirmation runs**. Бюджет под Cerebras zai-glm-4.7 — ~$30–40 (preview-цены `# VERIFY`); под gpt-4o — ~$80–100. Итого confirmation: **~$110–140**.

---

## 7. E5 — Real human validation (ecological validity)

**Цель:** проверить, что выводы LLM-симулятора воспроизводятся с настоящими людьми; измерить cognitive load, satisfaction.

**Дизайн (within-subject):**
- **N участников: 8–12** (студенты/знакомые из CS, добровольно)
- **Курированный subset из 12 задач**: по 3 из каждого task_type, подобранные так, чтобы:
  - LLM-симулятор показал на них наибольший spread между configs (interesting cases)
  - время выполнения одной задачи с человеком ≤ 5–8 минут
- **3 config'а** на участника (champion'ы из E4):
  - Best-static + best-role (baseline)
  - Adaptive-topology + adaptive-role (champion)
  - Адаптивная с LLM-sim (для сопоставления с "реальный vs симулятор")
- Counterbalanced Latin square по порядку configs и задач (чтобы убрать order effect)

**Общий объём:** 12 участников × 12 задач × 3 configs = 432 сессии, но каждый участник делает ≤ 12 задач (не все configs на одной задаче) — итого **~45 мин на участника**.

**Метрики:**
- **NASA-TLX** (6 шкал) после каждой сессии
- **Satisfaction** (Likert 1–7): доверие, качество, удобство
- **Time-on-task**, **interactions_count**
- **Quality** — тот же auto-eval, но дополнительно post-task self-assessment участника
- **Free-form comments** (qualitative)

**Анализ:**
- Repeated-measures ANOVA по TLX: config как фактор
- Сопоставление: **correlation** TLX-human vs cognitive_load_proxy из E4 → ключевой вопрос "насколько LLM-симулятор валиден как прокси человека"
- Bland-Altman plot: agreement между human-rated quality и auto-metric
- Тематический анализ свободных комментариев

**Протокол:**
- Информированное согласие
- Краткий tutorial (5 мин): интерфейс, роли
- Pilot с 2 участниками до основного набора — калибровка времени и сложности задач
- Без компенсации стоит обсудить с научруком; альтернатива — лотерея или кофе

**Бюджет:** ~$10–20 на inference + время на организацию.

---

## 8. Сводная таблица runs и бюджета

Бюджет считается под актуальный lineup B'.1 (`cerebras:gpt-oss-120b` primary @ $0.35 in / $0.75 out per 1M; `openai:gpt-4.1-mini` judge × 3 self-consistency; confirmation on `cerebras:zai-glm-4.7` + `openai:gpt-4o`).

| Этап | Runs (прибл.) | Бюджет (lineup B'.1) | Что новое |
|---|---|---|---|
| E1 Baseline no-human | 1 500 | ~$10 (compute) + ~$5 (judge) | static topology Pareto |
| E2 HITL-sim baseline | 4 000 | ~$25 (compute, +HITL-sim calls) + ~$15 (judge) | role × topology heatmap |
| E3 Adaptive topology | 1 100 | ~$8 (compute, +router-cost) + ~$4 (judge) | static vs adaptive |
| E4 Adaptive + adaptive-role | 900 | ~$7 (compute, +HITL) + ~$3 (judge) | full combo |
| Confirmation primary (zai-glm-4.7) | ~1 800 | ~$30–40 (`# VERIFY` preview prices) | cross-family на Cerebras |
| Confirmation cross-family (gpt-4o) | ~1 800 | ~$80–100 | frontier OpenAI validation |
| E5 Real human | ~120–140 | ~$3–5 inference + человеко-часы | ecological validity |
| **Итого** | **~11 300** | **~$190–230** | |

+30% reserve (retries, failed runs, ablation add-ons) → **закладываемся на ~$250–300**.

**Wall-time оценка** на Cerebras Pay-as-You-Go (1K RPM / 1M TPM на gpt-oss-120b, 8 параллельных воркеров): E1 full ~45 мин, E2 ~2.5 ч, E3 ~45 мин, E4 ~40 мин, оба confirmation вместе ~2 ч. Итого pure compute **~6 ч**, с реальным запасом на retries/sandbox **~10–12 ч** wall-time на всю программу E1–E4 + confirmation.

## 9. Статистика и валидность

- **Множественные сравнения**: Bonferroni или BH-FDR для per-task-type тестов
- **Effect size**: всегда вместе с p-value (Cohen's d, η²)
- **Seeds**: n=3 minimal (power низкий), n=5 для confirmation; variance между seeds логируем отдельно
- **Мокросы**: пропуски LLM-вызовов (timeout, budget) → `status='failed'`, исключаются из аналитики но сохраняют долю
- **Pre-registration (lite)**: перед запуском E3 фиксируем гипотезы и пороги ("Adaptive считаем выигрышным если..." — в `dev/preregistration.md`)

## 10. Риски и митигации

| Риск | Митигация |
|---|---|
| LLM-симулятор HITL ≠ реальный человек (валидность E5) | заранее согласуем с научруком метрику "валидности симулятора"; E5 именно для этого |
| `cerebras:gpt-oss-120b` для всех compute-ролей одной модели → потеря «worker < critic» иерархии в защите | компенсируем через `reasoning_effort: low` для workers и `high` для Critic + system-prompt strictness; явный аргумент «изолирована переменная модели» |
| Cerebras Cloud rate-limit (1K RPM / 1M TPM на gpt-oss-120b, Pay-as-You-Go) | `asyncio.Semaphore` уже встроен; на грид-уровне 8 воркеров не доходят даже до половины лимита; при упоре — переключение на батч-режим |
| `gpt-oss-120b` слишком слабый → топологии неразличимы | confirmation runs на `cerebras:zai-glm-4.7` + `openai:gpt-4o`; если различий мало — фиксируем в limitations |
| Cerebras `zai-glm-4.7` — preview-модель, может уйти из каталога в ходе работ | приоритет confirmation на `openai:gpt-4o`; zai как «второй прибор»; при отзыве — fallback на одну модель |
| Adaptive router сам ест токенов больше, чем экономит | bookkeeping отдельно для "routing cost"; включаем в total |
| Нехватка seeds → шум больше эффекта | минимум n=3 на пилотном, n=5 на финальном; переоценить после E1 |
| Реальные участники дают разный опыт на разных configs | Latin square, pilot, обучающий туториал |
| Contamination (gpt-4o, gpt-oss видели HumanEval/GSM8K) | фиксируем в limitations; добавляем LiveCodeBench-mini в sensitivity-analysis |

## 11. Что нужно решить с научруком до старта

- Этическая часть E5 (IRB / информированное согласие — нужна ли у вуза процедура)
- Порог "Adaptive выигрышно" — 5% quality или 15% cost? или оба?
- Confirmation runs: оба провайдера (Cerebras zai + OpenAI gpt-4o) или только один — балансировка бюджета и cross-family строгости
- Минимальное N участников для E5 (стат. значимость vs доступность)
- Сохраняем ли логи interactions для публикации датасета

## 12. Следующие шаги (после утверждения плана)

1. Реализовать M0–M6 из PLAN.md (фреймворк + Star/Chain)
2. **Pilot E1** на Star+Chain + HumanEval только (50 задач × 3 seed = 300 runs, ~$3) → проверить pipeline, writers, budget-гварды, анализ
3. Если pilot ОК → расширить до полного E1 после M7 (остальные топологии)
4. Параллельно M9 (human gateway + sim) чтобы быть готовым к E2
5. **Pre-registration** hypotheses → `dev/preregistration.md`
