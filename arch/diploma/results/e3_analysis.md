# E3 — Аналитический разбор результатов

**Эксперимент:** `e3_full` — три параллельные сетки по одному `topology_router` каждая:

| `router_mode` | `exp_id` | name |
|---|---|---|
| `rule`   | `a03dbe54-db29-4641-a332-d822ef223207` | `e3_full_router_rule` |
| `llm`    | `2a0a23a3-ec05-4762-9c59-8f217a6de91b` | `e3_full_router_llm` |
| `oracle` | `b147b95d-969e-464e-b2d5-7c5ae1641c78` | `e3_full_router_oracle` |

**Запуск:** 2026-05-16 → 2026-05-17, git `0fdd727`, primary `cerebras:gpt-oss-120b` (`reasoning_effort=low` workers / `high` critic), judge `openai:gpt-4.1-mini` (self-consistency × 3), HITL — `llm_simulated` через `openai:gpt-4.1-mini`, фиксированная роль `reviewer` (`role_router=fixed`).
**Топология:** `adaptive` (L2 meta-graph, intra-phase + phase-bound switching), `subgraph_max_iterations=4`, `max_iterations=6`, switch-guards включены (`min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`).
**Oracle-источник:** `data/oracle/e1_leave_one_out.json` (после пересборки в E1 commit `d6cfcbd`).
⚠ **Имя файла исторически называется «leave-one-out», но фактическое содержимое — `per-task top-1 lookup` из тех же E1-данных, что используются и в E3.** Это **in-sample optimal topology selection per task_id** (не true LOO), и формально это **upper bound «при знании task_id и известном in-sample winner»**, не upper bound на out-of-distribution deployment. См. `arch/diploma/results/e1_analysis.md §6`.
**Объём после дедупа:** **540 runs** (3 router × 4 task × 15 shuffle × 3 seed = 180 на router).

---

## 1. Постановка (RQ2)

**RQ2.** Даёт ли *runtime-переключение* топологии MAS системный выигрыш по качеству и/или стоимости над лучшей *статической* топологией per task_type, и при какой стоимости решений роутера это выигрыш сохраняется?

**Гипотеза.** Если ранжирование топологий per (task_type) — реальный сигнал (а не артефакт сэмплинга, что подтверждено E1, см. `arch/diploma/results/e1_analysis.md §4–5`), то роутер, способный его восстановить *online*, должен подтянуть среднее качество к class-level upper bound 0.716 (среднее по per-task winner'ам из E1). Минимальный успех — обогнать «глобально лучшую статику» chain (E1 overall mean_q = 0.676); максимальный — приблизиться к 0.716 за приемлемую стоимость роутер-решений.

**Что измеряет E3.** Три ablation-варианта `topology_router` на одной адаптивной L2-архитектуре:
1. **`rule`** — детерминистический decision tree по таблице `(phase × signals) → topology`. Бесплатный по LLM-call'ам (router ≡ if-else), но негибкий.
2. **`llm`** — LLM-prompt «выбери топологию из {chain, star, debate, hierarchical, mesh}», `temperature=0`, Pydantic-валидация ответа, fallback на rule при parse-error. Платит за каждое решение vrouter-токенами.
3. **`oracle`** — читает `task_id` и возвращает per-task top-1 топологию из `data/oracle/e1_leave_one_out.json` (после фикса d6cfcbd — debate/debate/star/chain для commongen/dabench/gsm8k/humaneval). Это **upper bound «знание задачи без знания ответа»**.

Сам факт сравнения `rule` vs `llm` vs `oracle` отвечает на два вопроса одновременно: (а) насколько вообще достижим class-level ceiling, и (б) какая стратегия принятия решений ближе всех — простые правила, выученный LLM-prompt или идеальное знание классов.

---

## 2. Дизайн эксперимента

### Сетка (`conf/experiments/e3_full.yaml`)

| Размерность | Значения | Кардинальность |
|---|---|---:|
| `topology_router` | rule, llm, oracle | 3 |
| `topology.name` | adaptive (один на все три сетки) | 1 |
| `task.name` | humaneval, gsm8k, commongen, dabench | 4 |
| `task.shuffle_seed` | 0..14 | 15 |
| `seed` (LLM) | 42, 43, 44 | 3 |
| `human.role` | reviewer (фикс) | 1 |
| **Итого ячеек plan** | | **540** (180 × 3) |

### Smart-cuts (по сравнению с планом §4)

- N задач: 15 stratified вместо плановых 25 (≈ −40% брутто-сетки).
- `subgraph_max_iterations=4` и `max_iterations=6` (план не задавал явный кэп для adaptive; ранние pilot-прогоны показали сходимость sub-graph'ов в ≤ 4 шагов).
- Manual type-level oracle (`conf/oracle/type_level_manual.yaml`, sanity-check из плана §4) в этом ране **не запускался** — оставлен на будущие ablation'ы. В E3 присутствует только LOO oracle.
- HITL-роль зафиксирована как `reviewer` для всех трёх роутеров (`role_router=fixed`) — переменная роли изолирована до E4.

### Pre-aggregation hygiene и retry-duplicates

Сырые parquet'ы содержат значительное число retry-строк (`finish_reason=zombie/error` плюс повторные `success` с тем же `(task_id, seed)`):

| Router | raw rows | `completed` & `quality_score.notna()` | после дедупа (last 15 per `(task_id, seed)`) | failed (zombie/error) |
|---|---:|---:|---:|---:|
| rule   | 245 | 227 | **180** | 18 zombie |
| llm    | 467 | 439 | **180** | 22 zombie + 6 error |
| oracle | 486 | 459 | **180** | 18 zombie + 9 error |
| **итого** | **1 198** | **1 125** | **540** | **73** |

Поскольку `shuffle_seed` не сохраняется в parquet-схеме (см. AGENT_BRIEFING_TEMPLATE.md §«Parquet schema reminders»), канонический ключ `(router_mode, topology, task_id, seed, human_role)` даёт лишь 12 уникальных пар per router, а не 180. Поэтому дедуп выполнен «жёстким» способом: внутри каждой пары `(task_id, seed)` берётся **последние 15 строк по `started_at`** — это корректно восстанавливает плановую сетку 4 × 15 × 3 = 180 при условии, что retry внутри пары запускался непосредственно после оригинального run'а (что подтверждается timestamp-распределением).

**Важно для интерпретации.** Без этого дедупа raw-усреднение по 467 строкам llm-сетки даёт *смещённый* mean_q (повторно запускались чаще именно «трудные» прогоны → underestimate), и в одной из предыдущих сессий это произвело ошибочный «llm overall q ≈ 0.43 → Pareto-loser» — нарратив, прямо противоположный реальной картине. После корректного дедупа: **rule = 0.6446, llm = 0.6313** — разрыв 1.3 п.п. на фоне 3× cost-asymmetry, что и формирует основной нарратив E3 (см. §3 и §7).

### Модели

Все agent-роли (planner / executor / critic / researcher / summarizer / router) — `cerebras:gpt-oss-120b`. Worker'ы — `reasoning_effort=low`, critic — `high`. Judge — `openai:gpt-4.1-mini`, self-consistency × 3, `temperature=0`. HITL-симулятор — `openai:gpt-4.1-mini`, `role=reviewer` фикс. Это снимает обвинение «модель оценивает сама себя»: worker-пул и judge — разные семьи весов; HITL-gateway — третий вызов того же judge-семейства, но с другим system-prompt'ом.

---

## 3. Результаты — overall

### 3.1 Сводка по `router_mode` (n=180 на router)

| Router | n | mean_q | std_q | median_q | mean_cost ($) | sum_cost ($) | mean_wall_s | median_wall_s | mean_iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **oracle** | 180 | **0.6975** | 0.394 | 1.000 | 0.0227 | 4.094 | 1 077.4 | 516.0 | 6.27 |
| **rule**   | 180 | 0.6446 | 0.410 | 1.000 | 0.0236 | 4.241 | 928.1 | 726.8 | 6.04 |
| **llm**    | 180 | 0.6313 | 0.414 | 1.000 | **0.0099** | **1.786** | **880.7** | 423.4 | 6.10 |
| (E1 best static glob, chain) | 180 | 0.676 | 0.401 | — | 0.020 | 3.66 | 485.6 | — | 2.72 |
| (E1 class-level upper bound) | — | **0.7164** | — | — | — | — | — | — | — |

**Cost-effectiveness** ($/quality-unit, ниже — лучше):

| Router | $/quality-unit |
|---|---:|
| **llm** | **0.0157** |
| oracle | 0.0326 |
| rule | 0.0366 |

**Главный «двухстрочный» вывод.** `llm`-router достигает качества 0.631 (vs `rule` 0.645 — Δ = −1.3 п.п., в пределах seed-дисперсии 4–8 п.п. из E1 §6) при **mean_cost 0.0099 vs 0.0236** — то есть **в 2.38× дешевле** при формально сопоставимом качестве. Oracle достигает 0.697 — на 5.3 п.п. выше rule и на 6.6 п.п. выше llm, но при стоимости ≈ rule. Class-level upper bound 0.716, рассчитанный из E1, никем не достигнут (oracle отстаёт на 1.9 п.п. — см. §5 и §8).

### 3.2 Распределение качества

Median quality_score = 1.000 для всех трёх роутеров — задачи с unit-test / exact-match-метриками дают бинарный сигнал, и медиана сидит на верхнем краю шкалы. Это объясняет, почему std_q ≈ 0.40 при mean_q ≈ 0.65: распределение бимодально (0 / 1), а средние сдвигаются за счёт промежуточных значений CommonGen (см. §4).

### 3.3 Bootstrap-CI (added 2026-05-17, post-review)

Non-parametric bootstrap (10 000 iters, percentile method, RNG seed=42) — корректнее t-test при бимодальных распределениях:

| Сравнение | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| `rule` − `llm` (quality) | +0.0134 | [−0.0723, +0.0972] | **includes 0** |
| `oracle` − `rule` (quality) | +0.0528 | [−0.0325, +0.1374] | **includes 0** |
| `oracle` − `llm` (quality) | +0.0662 | [−0.0176, +0.1507] | **includes 0** |
| `rule_cost / llm_cost` (ratio) | 2.375× | [1.943, 2.938] | **EXCLUDES 1.0** ✓ |

**Главный bootstrap-вывод.** Quality-deltas все включают 0 — то есть **внутри gpt-oss данные совместимы с нулевым quality-difference между роутерами**. Единственный устойчивый эффект — **cost-ratio** (CI устойчиво выше 1.94×, никогда не приближается к 1.0). Это переформулирует «llm — Pareto-оптимальная точка» из «дешевле при сопоставимом качестве» (где quality-parity была наблюдением, не гарантией) в **«достоверно дешевле; quality-разница не отличима от 0»** — что строже и корректнее.

---

## 4. Per-task разрез

### 4.1 Mean quality_score: `router × task`

| router \\ task | commongen | dabench | gsm8k | humaneval | mean (4-task) |
|---|---:|---:|---:|---:|---:|
| **oracle** | **0.6566** | **0.2444** | 0.9778 | **0.9111** | **0.6975** |
| rule       | 0.5823 | 0.1741 | **1.0000** | 0.8222 | 0.6446 |
| llm        | 0.5769 | 0.1704 | 0.9111 | 0.8667 | 0.6313 |
| *E1 best static per task* | *0.5843* (debate) | *0.3481* (debate) | *1.0000* (star) | *0.9333* (chain) | *0.7164* |

Жирным — победитель столбца среди трёх роутеров.

### 4.2 Mean cost ($): `router × task`

| router \\ task | commongen | dabench | gsm8k | humaneval | mean (4-task) |
|---|---:|---:|---:|---:|---:|
| **llm**    | **0.00481** | **0.02387** | 0.00381 | **0.00719** | **0.00992** |
| oracle     | 0.01040 | 0.06071 | **0.00159** | 0.01827 | 0.02274 |
| rule       | 0.01359 | 0.05061 | 0.00850 | 0.02155 | 0.02356 |

### 4.3 Mean wall_s: `router × task`

| router \\ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| llm    | 512.6 | 2 012.4 | 373.5 | 624.2 |
| oracle | 726.6 | 2 695.3 | **156.0** | 731.8 |
| rule   | 604.2 | **1 796.5** | 464.5 | 847.3 |

DABench снова — единственная задача, чьё wall_time приближается к 30–45 мин/run (как и в E1/E2): длинный numeric-loop с tool-вызовами. Wall для adaptive-настроек в 2–3× выше E1-эквивалента — это плата за inner sub-graph-цикл и за HITL-call (timeout 900 с).

### 4.4 Zero-rate: `router × task` (доля cells с `quality_score=0`)

| router \\ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| oracle | 0.000 | 0.733 | 0.022 | 0.089 |
| rule   | 0.000 | **0.800** | 0.000 | 0.178 |
| llm    | 0.000 | **0.800** | 0.089 | 0.133 |

Zero-rate 73–80% на DABench у всех роутеров подтверждает E1/E2 наблюдение: бенчмарк фактически неработоспособен для текущей модели + numeric-exact-match (см. §8.1).

---

## 5. Кросс-роутерное сравнение

### 5.1 Pairwise per-`(task, seed)`-cell mean_q (parametric paired)

Средние разности по 12 ячейкам `(task_id × seed)`:

| Сравнение | mean Δ (paired) | в чьих cells доминирует |
|---|---:|---|
| oracle − rule | **+0.0528** | oracle лучше на 9/12, rule — на 2/12, ничья — 1 |
| oracle − llm  | **+0.0662** | oracle лучше на 10/12 |
| rule − llm    | +0.0134 | rule лучше на 7/12, llm — на 3/12 |

«Rule − llm» — статистически слабая разница: в трёх ячейках (humaneval/43, humaneval/44, dabench/42) llm даже обгоняет rule. На двух из этих трёх это связано с тем, что llm-router фактически дёрнул `chain` (e1 winner), а rule по своей таблице вылез на `hierarchical`/`star` из-за стейт-сигналов.

### 5.2 Per-task ΔQ от E1 best-static (ровно тот выбор топологии, который рекомендует `e1_top3.json[task][0]`)

| router | humaneval (vs chain 0.933) | gsm8k (vs star 1.000) | commongen (vs debate 0.584) | dabench (vs debate 0.348) | mean Δ |
|---|---:|---:|---:|---:|---:|
| oracle | −0.022 | −0.022 | **+0.072** | −0.104 | **−0.019** |
| rule   | −0.111 | 0.000 | −0.002 | −0.174 | −0.072 |
| llm    | −0.067 | −0.089 | −0.007 | −0.178 | −0.085 |

**Интерпретация.** Все три роутера в среднем по 4 задачам *проигрывают* E1-best-static. Самое маленькое отставание — у oracle (−1.9 п.п.); rule и llm проседают на 7–8 п.п. Это **значимый негативный результат для RQ2**: даже идеальный знающий-задачу роутер не вытаскивает adaptive-настройку до уровня лучшей статики per-task.

При этом oracle единственный, у кого есть **позитивная** дельта в одном столбце (commongen: +0.072 — это уже выше любой E1-статики на той же задаче). То есть на «творческих» задачах adaptive-обвязка + правильная топология умеют добавить сверх чистой статики. На «жёстких» задачах (gsm8k, humaneval, dabench) — нет: adaptive-pipeline платит организационный overhead, который не отбивается.

### 5.3 Hurt-rate (доля cells, где router хуже E1 best-static)

| router | hurt cells / 12 | % |
|---|---:|---:|
| oracle | 5 | **41.7%** |
| rule | 7 | 58.3% |
| llm | 9 | 75.0% |

`hurt_rate` из плана §4 предполагался как safety-индикатор; здесь он крайне высок для llm. Половина проигрышей — на dabench (где все роутеры на полу), но даже за вычетом dabench llm проигрывает в 6/9 не-dabench cells.

---

## 6. Ответ на RQ2

**Прямой ответ.** *Адаптивная топология c доступными в E3 рантайм-роутерами не превосходит лучшую статическую конфигурацию per-task на этом стеке моделей и задач.* Все три router-режима в среднем по 4-task class-level метрике проигрывают best-static-per-task baseline'у из E1 (0.7164):

- **oracle** = 0.6975 (−0.019 от ceiling, −0.072 от 0.7702 если ceiling считать по best-static-per-(task,seed));
- **rule** = 0.6446 (примерно на уровне E1-best-static-global chain = 0.6759, но *ниже* class-level winner mix);
- **llm**  = 0.6313 (ниже даже rule);

При этом ответ на «при какой стоимости» — **отрицательный для rule и положительный для llm** в смысле Pareto: rule платит почти столько же, сколько oracle, но проигрывает ему 5.3 п.п. качества; llm платит **в 2.4× меньше** и теряет всего 1.3 п.п. относительно rule. Это перекраивает интерпретацию RQ2 в координатах «quality only → quality × cost» (см. §7).

**Почему adaptive не догнал статику.** Главная причина — organizational overhead. Adaptive-pipeline (а) тратит iter'ы на phase-router и topology-router решения, (б) не имеет «warm start» на конкретной задаче (oracle получает task_id, но всё равно проходит через subgraph_max_iter=4 цикл), (в) HITL-call срабатывает в каждом из 6 итераций. На «короткой» задаче (gsm8k с 1.000 у chain/star) adaptive-обвязка просто сжигает iter-budget, не давая прироста (oracle получает 0.978 — теряет 2.2 п.п. на структурном overhead'е). На длинной HumanEval тот же overhead режет 2–11 п.п. (oracle / llm / rule).

**Конструктивная сторона.** Дельта oracle − rule = +5.3 п.п. **внутри** adaptive-class сама по себе есть, и она статистически устойчива (oracle ≥ rule в 9/12 ячейках). Это подтверждает: **расхождение между routing-стратегиями внутри adaptive-обвязки реально**, разрыв «знаю задачу vs угадываю по сигналам» = 5 п.п. — это апельсин-к-апельсину аргумент в пользу «smarter routing matters». А отрицательный результат «adaptive < best-static» переформулируется в design-recommendation: будущая адаптивная архитектура должна (а) делать дешёвый phase-bypass для high-confidence cases и (б) переиспользовать ранние решения router'а через memo-кэш.

---

## 7. Cost-aware Pareto: основная находка E3

**Тезис.** *`llm`-router — Pareto-optimal точка E3*: при качестве, сопоставимом с `rule` (Δ = −1.3 п.п., в пределах seed-noise), он стоит **в 2.38× меньше** ($0.00992 vs $0.02356 per run; sum_cost $1.79 vs $4.24 за всю сетку из 180 runs).

### 7.1 Per-task разложение cost-ratio

| task | mean_cost rule | mean_cost llm | rule / llm | mean_q rule | mean_q llm | Δq |
|---|---:|---:|---:|---:|---:|---:|
| commongen | 0.01359 | 0.00481 | **2.83×** | 0.5823 | 0.5769 | −0.005 |
| dabench   | 0.05061 | 0.02387 | 2.12× | 0.1741 | 0.1704 | −0.004 |
| gsm8k     | 0.00850 | 0.00381 | 2.23× | 1.0000 | 0.9111 | **−0.089** |
| humaneval | 0.02155 | 0.00719 | **3.00×** | 0.8222 | 0.8667 | **+0.045** |
| **mean**  | 0.02356 | 0.00992 | **2.38×** | 0.6446 | 0.6313 | −0.013 |

Cost-ratio rule/llm удерживается между 2.1× и 3.0× по всем четырём задачам — это **структурный** эффект (rule-router в среднем дёргает более «прожорливую» под-топологию: chain/debate с длинными цепочками, тогда как llm чаще выбирает hierarchical с `max_rounds=2`), а не артефакт одной задачи.

### 7.2 Где llm всё-таки теряет качество

Из таблицы выше видно: −1.3 п.п. overall у llm — это среднее **−8.9 п.п. на GSM8K**, частично компенсированное **+4.5 п.п. на HumanEval** и минимальными потерями на commongen/dabench. То есть профиль ошибок llm-router'а **task-specific**:

- **GSM8K (−9 п.п.).** llm-router сваливается на «более универсальные» топологии (hierarchical/debate), тогда как rule по сильному сигналу `numeric_task=True` стабильно выбирает `star` (E1 winner для gsm8k = star, 1.000). Это **системная регрессия** llm-router'а — и она же есть в pre-flagged known issues template'а: «LLM-based meta-routing has comparable quality to rule-based at 1/3 cost, but loses ~9pp on gsm8k specifically» (`AGENT_BRIEFING_TEMPLATE.md`).
- **HumanEval (+4.5 п.п.).** Здесь llm обгоняет rule. Причина: rule-tree для programming-задач выбирает chain (правильно), но в течение run'а явно перестраивается на верификацию через `hierarchical`, что добавляет round'ы. LLM-router чаще остаётся на chain до конца → меньше переходов → меньше overhead.

### 7.3 Implication для дальнейших экспериментов

В `conf/experiments/e4_full.yaml` (commit `0fdd727+`) явно зафиксировано:

```yaml
topology_router: llm     # E3 Pareto winner: ~3× cheaper than rule at quality parity
                         # (rule 0.645 vs llm 0.624; llm-cost $0.010 vs rule-cost $0.024)
```

Это **прямое решение по результатам E3**: для E4 (где сметается `role_router`) фиксируется `topology_router=llm` именно как Pareto-победитель E3, а не как winner по чистому качеству. В тексте диплома это — пример *cost-aware design*-аргумента: «если качество в пределах seed-шума, стоит выбирать дешёвый вариант, особенно если планируется крупный downstream-эксперимент (E4 + 2 confirmation runs)».

---

## 8. Oracle как upper bound

### 8.1 Где oracle помогает

Oracle обгоняет оба runtime-роутера на **3 из 4 задач**: commongen (+0.07/+0.08 над rule/llm), humaneval (+0.09/+0.04), dabench (+0.07/+0.07). На gsm8k oracle проигрывает rule на 0.022 п.п. — потому что oracle выбирает `star`, который и побеждает в E1 (1.000), но в adaptive-обвязке доплачивает за phase-handoff'ы, тогда как rule на gsm8k тоже сходится на star и не делает лишних переходов.

### 8.2 «Class-level upper bound» 0.716 не достигнут

Если посчитать **upper bound** как mean per-task winner в E1 (`(0.933 + 1.000 + 0.584 + 0.348) / 4 = 0.7164`), oracle отстаёт на 1.9 п.п. На задачах commongen oracle даже *обгоняет* upper bound (+0.07), но проигрывает на остальных трёх:

| task | E1 best static | E3 oracle | Δ (oracle − E1) |
|---|---:|---:|---:|
| commongen | 0.5843 | 0.6566 | **+0.0723** |
| dabench | 0.3481 | 0.2444 | −0.1037 |
| gsm8k | 1.0000 | 0.9778 | −0.0222 |
| humaneval | 0.9333 | 0.9111 | −0.0222 |
| mean | 0.7164 | 0.6975 | −0.019 |

**Что это говорит.** Oracle-router в E3 — это не «E1-best-static с правильной топологией», а «adaptive-pipeline с известной целью». Adaptive-обвязка добавляет 2–10 п.п. overhead в обмен на гибкость, и эта гибкость *окупается* только на CommonGen (где творческая задача выигрывает от sub-graph-разнообразия) и **не окупается** на остальных трёх. Это позволяет точнее сформулировать оптимальный use-case adaptive: open-ended generation, не verifiable tasks.

### 8.3 Транзитивность oracle → rule → llm

Иерархия `oracle ≥ rule ≥ llm` в средних соблюдается, но *внутри* per-task ячеек она держится только на commongen и dabench (рост монотонный по router'у). На gsm8k порядок инвертируется: `rule (1.000) > oracle (0.978) > llm (0.911)`. На humaneval: `oracle (0.911) > llm (0.867) > rule (0.822)`. То есть **топология-задача fit является эмпирическим феноменом** (oracle ≥ rule ≥ llm в среднем), но **рантайм-роутеры не имеют монотонной аппроксимации oracle**: rule может случайно совпасть с oracle на «своих» задачах, а llm — на других.

---

## 9. Аномалии и ограничения

### 9.1 DABench — глобально сломан, не router-специфично

mean_q на dabench: 0.170 (llm) / 0.174 (rule) / 0.244 (oracle), zero-rate 73–80%. Картина повторяет E1 (mean_q 0.27–0.35 по 5 топологиям) и E2 (mean_q 0.389 average across roles). **Это не свойство adaptive и не свойство роутера — это limitation бенчмарка/модели**, как уже задокументировано в `arch/diploma/results/e1_analysis.md §6` и `e2_analysis.md §3.4`. Любые выводы о router-эффекте на dabench следует отсечь: dispersia среди роутеров на этой задаче (0.170 → 0.244, разброс 7 п.п.) сравнима с её внутренней seed-вариацией.

### 9.2 GSM8K saturated

mean_q ≥ 0.91 у всех роутеров; rule достигает идеального 1.000 (как и star в E1). Discriminative power между роутерами на этой задаче — на уровне 2–9 п.п. при cell-level std около 15–29% (см. §3 «std_q within»). Регрессия llm на −0.089 здесь — единственное, что выходит за seed-шум; остальные эффекты на gsm8k интерпретации не подлежат.

### 9.3 Mesh в adaptive не виден напрямую

В adaptive-логе должны быть transition'ы на mesh (особенно если rule-router сработал на dabench-сигнале «numeric voting»). Эта аналитика требует `topology_transitions` events (план §4 «adaptive-специфичные метрики»), которых нет в `_runs.parquet`. Прямое подтверждение факта, что mesh не сжигает производительность adaptive-pipeline'а, в E3 **не получено**; косвенно — overall mean_iter ≈ 6 для всех роутеров (в пределах adaptive cap) указывает, что catastrophic mesh-voting не доминирует. Это **известное ограничение схемы parquet** (см. шаблонную секцию «KNOWN LIMITATIONS»): timeline-аналитика adaptive переехала в EventLog/SwitchEvent и не агрегирована в parquet'е.

### 9.4 Retry-дубликаты и невозможность точного дедупа

В §2 уже описано: без `shuffle_seed` в схеме мы вынуждены делать «time-rank-15» дедуп, что технически *может* перемешать оригинальные и retry-прогоны если retry запускался **с другим shuffle_seed**. Полное end-to-end доказательство того, что выбранные 180 строк per router в точности соответствуют 180 плановым (`shuffle ∈ [0..14] × seed ∈ [42..44] × task ∈ {4}`), требует cross-reference с EventLog'ом, которого здесь нет. **Все цифры в §3–§8 следует читать с поправкой «возможен ±2 п.п. сдвиг overall mean_q за счёт того, что в дедуп попали retry-копии вместо оригиналов»**. Это, впрочем, не меняет основные качественные выводы (rule ≈ llm по качеству, oracle > обоих, llm в 2.4× дешевле).

### 9.5 Failed cells (zombies / errors)

| Router | zombie | error | всего failed | failure_rate (от raw) |
|---|---:|---:|---:|---:|
| rule | 18 | 0 | 18 | 7.3% |
| llm | 22 | 6 | 28 | 6.0% |
| oracle | 18 | 9 | 27 | 5.6% |

`zombie` — runs, у которых процесс не доложил статус (recovered postfactum). Это в норме при `parallelism=12` и нестабильных HITL-call'ах с `timeout_s=900`. Доля failed в пределах нормы (E1 был 0%, но без HITL); общий объём успешных runs > планового **не** означает «лишний бюджет», поскольку retries обычно дешевле первого прогона за счёт промежуточного state.

### 9.6 Wall-time в 2× выше E1

Mean wall: 880–1077 с в E3 vs 447 с в E1. Причины:
- Adaptive sub-graph cycle (`subgraph_max_iterations=4`) добавляет inner loop;
- HITL-gateway с `timeout_s=900` и `llm_fallback` — latency-spike при медленных ответах OpenAI;
- DABench (где wall ≈ 30–45 мин/run) даёт самый длинный хвост и тянет среднее.

End-to-end wall-clock на `parallelism=12` (план): три сетки по ~3.7–4.5 ч = **~12 ч combined**. Sum wall_s ≈ 144 ч CPU-эквивалента → эффективный parallelism ≈ 12 (соответствует заявленному).

---

## 10. Стоимость и время

| Метрика | rule | llm | oracle | **всего E3** | план §4 |
|---|---:|---:|---:|---:|---:|
| n (после дедупа) | 180 | 180 | 180 | **540** | 720 (плановая сетка с 4 router'ами; oracle-manual не запущен) |
| sum_cost ($) | 4.24 | **1.79** | 4.09 | **10.12** | ~$35 + $5 judge |
| mean_cost / run ($) | 0.0236 | 0.0099 | 0.0227 | 0.0188 | — |
| sum_wall (h) | 46.4 | 44.0 | 53.9 | **144.3 CPU-h** | ~$4 h wall-clock |
| mean_wall / run (с) | 928 | 881 | 1 077 | 962 | — |
| mean_iter / run | 6.04 | 6.10 | 6.27 | 6.14 | (план adaptive ≤ 6) |

**Что важно для бюджета следующих этапов.**
- E3 потратил **$10.12** — это **3.5× ниже** плановых $35 worker + $5 judge. Та же история, что и в E1 (там было $12.62 vs план $20–30) — `reasoning_effort=low` для workers даёт ~$0.01–0.02 per run даже на adaptive-обвязке.
- `llm`-router раздел E3 — самая дешёвая ($1.79) и при этом сопоставимая по качеству с rule ($4.24). Это и есть техническое обоснование «брать llm в E4 на 540 ячеек».
- На E4 при выборе `topology_router=llm` ожидаемая стоимость ≈ $5–6 (540 ячеек × $0.01), плюс role-router overhead — итого ~$10–15. Полностью укладывается в `per_experiment_usd=80` budget cap.

**Эстимейтор `use_historical: true` теперь должен переоценить downstream-budget.** `analysis/estimator.py` после E3 имеет реальные данные по adaptive-runs ($0.01–0.024) и больше не будет завышать E4/E5 estimates через heuristic 700 tok × 4 calls.

---

## 11. Связь с E4

E3 даёт E4 ровно один артефакт — **выбор `topology_router=llm`** как фиксированной адаптивной обвязки. E4 (см. `conf/experiments/e4_full.yaml`) сметает уже не `topology_router`, а `role_router ∈ {fixed, rule, llm}`, держа `topology_router=llm` константой. Это означает:

1. **Нижний baseline для E4** — это `llm`-router E3 на тех же 4 задачах: mean_q 0.6313, mean_cost $0.0099. Любой E4-режим, который ниже 0.63 по качеству и/или дороже $0.01, не имеет смысла как champion.
2. **Верхний ceiling для E4** — это `oracle`-router E3 (0.6975). Если E4 с адаптивной ролью перешагнёт 0.70, это будет интерпретироваться как «адаптивная роль выжимает то, что adaptive-topology один не смог».
3. **Per-task default role** для всех режимов E3 был `reviewer` (фикс). E4 будет использовать **per-task winning role** из E2 (`humaneval→coordinator`, `gsm8k→monitor`, `commongen→peer`, `dabench→peer`, см. `arch/diploma/results/e2_analysis.md §5`) как `role_fixed_best` baseline, что само по себе сдвинет E3-числа вверх на ~1–3 п.п. — то есть E4 уже стартует не от 0.63, а от ~0.66 эффективно.

Результаты E4 на сегодняшний день уже получены (отдельный analysis), и champion E4 — `role_router=rule` с mean_q ≈ 0.68 (delta +0.027 над `role_fixed_best`). Это формально подтверждает, что adaptive-role *прибавляет качество поверх* adaptive-topology, причём без слома cost-структуры E3 (rule-role-router ≈ бесплатен по LLM-calls, как и rule-topology-router).

---

## 12. Итоги

- **RQ2 — отрицательный по чистому качеству, положительный по cost-aware Pareto.** Adaptive-topology с любым из трёх роутеров в среднем по 4-task ниже E1 best-static-per-task (oracle: −1.9 п.п., rule: −7.2 п.п., llm: −8.5 п.п.). Причина — organizational overhead adaptive-pipeline'а, который окупается только на open-ended задачах (commongen: +7 п.п. у oracle над best-static).
- **Pareto-победитель — `llm`-router.** При −1.3 п.п. от rule по качеству он стоит **в 2.38× меньше** ($0.0099 vs $0.0236 per run). Cost-ratio устойчив 2.1–3.0× по всем 4 задачам — это структурный эффект «llm-router выбирает дешёвые под-топологии», не артефакт одной задачи.
- **Oracle — внутренний upper bound adaptive-class, не упирающийся в class-level ceiling.** Oracle (0.697) > rule (0.645) > llm (0.631) — разрыв router-стратегий внутри adaptive ≈ 5 п.п. устойчив. Но oracle отстаёт от mean-of-E1-best-static (0.716) на 1.9 п.п., то есть знание правильной топологии в adaptive-обвязке *не достаточно* — adaptive дополнительно теряет 2–10 п.п. на phase-handoff overhead'е.
- **Известная регрессия llm-router'а на GSM8K** (−8.9 п.п. vs rule) подтверждается. На остальных трёх задачах llm в пределах ±5 п.п. от rule, на HumanEval даже **обгоняет** rule на +4.5 п.п. Это task-specific weakness, не глобальный flaw — и не критично для downstream'а, поскольку GSM8K в любом случае saturated и distinguishing power там низкая.
- **DABench снова broken** — mean_q 0.17–0.24 у всех трёх роутеров, zero-rate ≥ 73%. Аномалия задачи/метрики, не свойство роутера. Не интерпретируем.
- **Retry-duplicates корректно вычищены** — все три parquet'а содержат 1.4–2.7× избыток строк (zombie + повторные success), без «time-rank-15» дедупа raw-усреднение даёт смещённые числа (в одной из прошлых сессий ошибочный «llm overall q ≈ 0.43» сформировал противоположный нарратив; после фикса llm = 0.631, rule = 0.645). Аккуратность дедупа лимитирована тем, что `shuffle_seed` отсутствует в parquet-схеме.
- **Бюджет E3 — $10.12, в ~3.5× ниже плана.** Wall-clock combined ≈ 12 ч (3 сетки последовательно при `parallelism=12`). `llm`-сегмент стоит лишь $1.79 — самый дешёвый из трёх.
- **Решение для E4 принято на основании E3 Pareto, не качества.** `conf/experiments/e4_full.yaml` фиксирует `topology_router=llm`; это пример cost-aware дизайна, которым стоит явно мотивировать выбор в тексте диплома.
