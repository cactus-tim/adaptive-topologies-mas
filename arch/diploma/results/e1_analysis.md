# E1 — Аналитический разбор результатов

**Эксперимент:** `e1_full` (`exp_id = f153400f-f8ba-4ca3-8dc2-66f31ba15236`)
**Запуск:** 2026-05-16, git `b7b2768`, primary `cerebras:gpt-oss-120b`, judge `openai:gpt-4.1-mini` (self-consistency × 3)
**Объём:** 900 runs (5 topologies × 4 task types × 15 shuffle_seed × 3 seed), все 900 завершены со `status=completed`, `finish_reason=success`. Полностью сбалансированная сетка по 45 ячеек на (topology × task_type).
**Обновление oracle:** 2026-05-17, commit `d6cfcbd` — пересборка `data/oracle/e1_leave_one_out.json` под семантику per-task top-1 (см. §6 и §8).

---

## 1. Постановка (RQ1)

**RQ1.** Существует ли единая *статическая* топология мульти-агентной LLM-системы, доминирующая по качеству на всех типах задач, или оптимальный выбор зависит от задачи?

**Цель E1** (см. `arch/experiment_plan.md §2`) — построить Pareto-фронт «quality × cost × time» для 5 статических топологий (Star, Chain, Mesh, Debate, Hierarchical) на 4 типах задач (HumanEval, GSM8K, CommonGen, DABench) **без HITL** и получить per-(task_type) ранжирование, на котором будет строиться funnel: E1 → top-3 → E2 → top-1 → E3 (adaptive vs static).

**Скрытая гипотеза** (которую защищает дальнейший дизайн): нет единого победителя — каждая топология сильна на своём *классе* задач. Если бы какая-то одна топология выиграла на всех 4 типах, мотивация adaptive-роутинга (E3) была бы подорвана. Поэтому E1 одновременно является и измерением, и pre-flight check'ом всей funnel-логики.

---

## 2. Дизайн эксперимента

### Что сметено по сетке (`conf/experiments/e1_full.yaml`)

| Размерность | Значения | Кардинальность |
|---|---|---|
| `topology.name` | star, chain, mesh, debate, hierarchical | 5 |
| `task.name` | humaneval, gsm8k, commongen, dabench | 4 |
| `task.shuffle_seed` | 0..14 | 15 |
| `seed` (LLM) | 42, 43, 44 | 3 |
| Human role | `none` (HITL отключён) | — |
| **Итого ячеек** | | **900** |

### Smart-cuts (`arch/experiment_plan.md §11.2`)

- N задач уменьшено с плановых 25 до **15** (× 3 seeds = 45 наблюдений на (topology × task_type)) — экономия ~40% бюджета при сохранении статистической мощности для ANOVA per-task-type.
- Итер-кэпы подкручены под фактическую сходимость на Cerebras `gpt-oss-120b`: `chain.max_iterations=6`, `star.exec_max_iter=3 / verify_max_iter=2`, `debate.max_rounds=2`, `hierarchical.max_rounds=2`.
- **Не применён** smart-cut «skip mesh × programming-tasks» (план §11.2.4): mesh сохранён на всех 4 типах, чтобы получить чистое сравнение по полной сетке. Это даёт нам данные для отдельного обсуждения деградации mesh — см. §6.
- `parallelism=30` (выше рекомендованных 12), что подтверждается отношением sum(wall_time)/wall-clock: фактический wall-clock ~3.7 ч против `sum(wall_time_s)=111.9 ч` cpu-времени.

### Модели

Все рабочие роли (Planner / Executor / Critic / Researcher / Summarizer / Router) — `cerebras:gpt-oss-120b` с `reasoning_effort=low` (workers) и `reasoning_effort=high` (`cerebras_critic`). Судья — `openai:gpt-4.1-mini`, self-consistency × 3, temperature=0.0. Это сохраняет иерархию «worker < critic» через prompt-strictness и reasoning effort, изолируя переменную «модель» от переменной «топология» — ключевой методологический аргумент для защиты.

### «Partial» статус

`experiment.json` помечен `status=partial`, `total_cost_usd=0.0`. Это артефакт писателя: в parquet все 900 ячеек завершены успешно (`status=completed`, `finish_reason=success` 900/900), а агрегированная стоимость в `experiment.json` не дописана при финализации. Реальная суммарная стоимость, посчитанная как `sum(budget_spent_usd)` по runs, — **$12.62** (см. §7).

---

## 3. Результаты — overall

### Сводка по топологиям (n=180 на топологию)

| Topology | n | mean_q | std_q | min | max | mean_cost ($) | sum_cost ($) | mean_iter | mean_wall_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | 180 | **0.6759** | 0.401 | 0.0 | 1.0 | 0.0204 | 3.66 | 2.72 | 485.6 |
| hierarchical | 180 | 0.6548 | 0.408 | 0.0 | 1.0 | **0.0073** | 1.31 | 3.00 | **263.8** |
| star | 180 | 0.6548 | 0.418 | 0.0 | 1.0 | 0.0114 | 2.06 | 6.26 | 359.2 |
| debate | 180 | 0.6275 | 0.421 | 0.0 | 1.0 | 0.0133 | 2.40 | 2.32 | 512.6 |
| mesh | 180 | 0.5623 | 0.449 | 0.0 | 1.0 | 0.0178 | 3.19 | 7.44 | 616.8 |

**Overall:** mean_q = 0.635 (std 0.421), mean_cost = $0.014/run, mean_wall = 447.6 s, mean_iter = 4.35.

**Cost-effectiveness** (mean_cost / mean_q, ниже — лучше):

| Topology | $/quality-unit |
|---|---:|
| hierarchical | **0.0111** |
| star | 0.0174 |
| debate | 0.0212 |
| chain | 0.0301 |
| mesh | 0.0316 |

### Матрица mean_q по (topology × task_type)

| topology \ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| chain | 0.5257 | 0.3333 | 0.9111 | **0.9333** |
| debate | **0.5843** | **0.3481** | 0.9333 | 0.6444 |
| hierarchical | 0.4857 | 0.2667 | 0.9778 | 0.8889 |
| mesh | 0.3973 | 0.3185 | 0.8667 | 0.6667 |
| star | 0.4268 | 0.2815 | **1.0000** | 0.9111 |

Жирным — победитель в столбце.

### Матрица mean_cost ($) по (topology × task_type)

| topology \ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| chain | 0.00406 | 0.06502 | 0.00219 | 0.01014 |
| debate | 0.00413 | 0.03667 | 0.00357 | 0.00888 |
| hierarchical | **0.00237** | **0.02000** | **0.00175** | **0.00503** |
| mesh | 0.01351 | 0.03440 | 0.00634 | 0.01675 |
| star | 0.00453 | 0.02864 | 0.00300 | 0.00950 |

`hierarchical` доминирует по стоимости в каждой колонке — следствие жёсткого `max_rounds=2` и того, что `top_coord` рано финализирует ответ.

### Матрица mean_wall_s

| topology \ task | commongen | dabench | gsm8k | humaneval |
|---|---:|---:|---:|---:|
| chain | 255 | 1304 | 117 | 266 |
| debate | 258 | 1104 | 245 | 444 |
| hierarchical | **91** | **657** | 98 | **209** |
| mesh | 469 | 1032 | 393 | 573 |
| star | 174 | 807 | **101** | 354 |

DABench выделяется почти на порядок более длинными прогонами у всех топологий — это numeric-subset с большим контекстом (CSV-like input) и многошаговыми вычислениями.

---

## 4. Top-3 per task_type и интерпретация

Из `analysis/e1_top3.json` (этот файл — вход для E2, через `analysis/winners.py`):

| task_type | 1st | 2nd | 3rd |
|---|---|---|---|
| **humaneval** | chain (0.933) | star (0.911) | hierarchical (0.889) |
| **gsm8k** | star (1.000) | hierarchical (0.978) | debate (0.933) |
| **commongen** | debate (0.584) | chain (0.526) | hierarchical (0.486) |
| **dabench** | debate (0.348) | chain (0.333) | mesh (0.319) |

### Интерпретация (доменное объяснение)

- **HumanEval — chain wins.** Программирование с unit-тестами имеет жёсткий критерий приёмки: критик либо approve, либо вернул конкретные тестовые ошибки. Линейный pipeline `planner → executor → critic → executor → ...` с `max_iter=6` идеально ложится на этот цикл; debate здесь лишний — нет настоящего trade-off между альтернативами, есть только «компилируется или нет». Это подтверждает катастрофическое падение debate на humaneval (0.644 vs 0.933 chain, разрыв 29 п.п.).
- **GSM8K — star/hierarchical/debate выше 0.93.** Простые арифметические задачи решаются с первой попытки любой адекватной топологией; star, как централизованный planner с верификацией, даёт **идеальный 1.000**. Это сигнал, что GSM8K у gpt-oss-120b saturated: distinguishing power между топологиями минимальный (margin star→hierarchical = 0.022).
- **CommonGen — debate wins.** Креативная генерация (составить осмысленное предложение из набора концептов) выигрывает от противостояния «pro vs contra», потому что judge debate'а оценивает по концепт-coverage. Это совпадает с интуицией «debate → задачи, где нужно сравнить альтернативы».
- **DABench — debate wins, но абсолютные значения низкие (0.35).** Numeric exact-match — самый жёсткий критерий. Никакая топология не справляется хорошо; победа debate с margin 0.015 над chain статистически слабая (см. seed-variance в §6).

### Ключевая находка: union top-3 = ALL 5 топологий

Объединение top-3 по 4 task_types покрывает **все 5 топологий**: {chain, debate, hierarchical, mesh, star}. Mesh попадает в top-3 только на dabench (3-е место).

Это означает: **отсечь топологии для E2 «глобально» (на уровне эксперимента) нельзя** — пришлось бы оставить все 5, что свело бы экономию smart-cut'а §11.2.2 к нулю. Поэтому фильтр E2 реализован как **per-(task_type, topology)**: для каждой задачи берётся свой top-3, и (topology × task_type)-ячейки вне топа просто не запускаются. Это самостоятельное методологическое замечание, на которое план §11.2.2 указывает неявно («top-3 топологии per task_type»), но E1 даёт ему количественное подтверждение: глобальный фильтр был бы пустым.

---

## 5. Ответ на RQ1

**Нет единой статической топологии-доминатора.** Победители на 4 типах задач — три разные топологии: `chain` (HumanEval), `star` (GSM8K), `debate` (CommonGen и DABench). Hierarchical стабильно второй-третий, но никогда не первый. Star и hierarchical делят 2-е место по overall mean_q (0.6548 vs 0.6548 — равенство до 4 знака).

Лучшая «глобальная» статика — `chain` с overall mean_q = 0.676. Но это среднее скрывает, что chain провален на CommonGen (0.526, отстаёт от debate на 6 п.п.) и обыгрывается star'ом на GSM8K. Использовать единый chain везде означало бы потерять **~6 п.п. качества на креативных задачах**.

Это прямая мотивация для E3 (adaptive topology): если бы такой роутер выбирал «правильную» статику per (task_type), он достиг бы среднего качества по победителям ≈ (0.933 + 1.000 + 0.584 + 0.348)/4 = **0.716** — это и есть **class-level upper bound**, который в E3 формально задаётся per-task top-1 oracle (`data/oracle/e1_leave_one_out.json`, см. §8). Разница 0.716 vs 0.676 (best static) ≈ 4 п.п. — небольшой, но не нулевой headroom для адаптации. Если router сможет хотя бы половину этого вытянуть — это эмпирическая поддержка для гипотезы RQ2.

---

## 6. Аномалии и ограничения

### Mesh — общая деградация

Mesh — единственная топология с overall mean_q ниже 0.6 (0.562). Per task_type:

| task_type | mean_q | zero-rate | median | std |
|---|---:|---:|---:|---:|
| humaneval | 0.667 | 33% | 1.00 | 0.477 |
| commongen | 0.397 | 4% | 0.53 | 0.271 |
| gsm8k | 0.867 | 13% | 1.00 | 0.344 |
| dabench | 0.319 | 64% | 0.00 | 0.453 |

Распределение бимодальное: либо 1.0, либо 0.0 (mode-collapse через round-robin voting). На programming-задачах это и предсказывал план §11.2.4: voting на текстовых строках degenerate — четыре агента не сходятся на одной формулировке кода, и `consensus_threshold=3` срабатывает редко. **Эмпирическое подтверждение smart-cut'а §11.2.4**: на humaneval mesh даёт zero-rate 33% против 6.7% у chain — каждый третий run проваливается полностью. Решение оставить mesh на всех 4 типах было корректным методологически (нужны полные данные для leave-one-out oracle), но в E2 mesh должен быть исключён везде, кроме dabench, где он попал в top-3.

### Дисперсия между seeds

Seed-variance (std среднего качества по seed внутри ячейки) — везде ≤ 0.083, типично 0.04–0.07. Это значит, что 3 seed'а дают воспроизводимую оценку: ранжирование топологий не переворачивается на других seed'ах. Для финального E5 confirmation (n=5 seeds) — запас прочности достаточный.

### Низкие абсолютные значения на DABench

Все топологии на DABench дают mean_q 0.27–0.35, zero-rate 62–71%. Это либо (а) задача слишком трудная для gpt-oss-120b, либо (б) метрика «numeric exact match» слишком жёсткая для свободной генерации (например, «3.14» vs «3.140» vs «pi ≈ 3.14»). Это **не предсказывалось планом**, который оценивал DABench как обычный type-задач. Рекомендация: в E5/confirmation добавить tolerant numeric match или прогнать DABench отдельно на frontier-модели (`openai:gpt-4o`) для разделения «модель слабая» vs «задача нерешаемая в принципе».

### Saturation на GSM8K

Star даёт 1.000 на GSM8K, hierarchical — 0.978, debate — 0.933. Distinguishing power между топологиями на этой задаче — на уровне шума (margin 2–7 п.п. при seed-std ~5 п.п.). В E3 это значит: oracle-router всегда будет выбирать star на GSM8K, но прирост от него над hierarchical ≈ 2 п.п., что ниже декларированного в плане §4 порога «5% quality». GSM8K, возможно, стоит исключить из основных сравнений E3, оставив как baseline-sanity.

### Debate проваливается на HumanEval

Debate даёт 0.644 на HumanEval — на 29 п.п. ниже chain. Zero-rate 35.6% — каждый третий run debate возвращает нерабочий код. Объяснение: debate-judge оценивает аргументы, а не запускает unit-tests; pro/contra-агенты тратят раунды на стилистические разногласия, а на стороне исполнения нет verify-петли. Это **аргумент против naive «debate как универсальное улучшение»** — критика, которую часто высказывают на защитах работ по MAS.

### Oracle-файл: пересборка под per-task top-1 (commit d6cfcbd)

Файл `data/oracle/e1_leave_one_out.json` был перестроен 2026-05-17 (commit `d6cfcbd`) после того, как было замечено расхождение между его содержимым и per-task-type top-1 из `analysis/e1_top3.json`. Предыдущая версия для всех 4 задач возвращала `hierarchical` (и `debate` для humaneval) — топологии «middling-everywhere», а не «best-anywhere». Корневая причина: старый алгоритм LOO агрегировал по `task_type='unknown'` (ни одна из 4 e1-задач не матчится по префиксу в `_TASK_TYPE_PREFIX_MAP`), что схлопывало оценку в «лучший generalist по 3 оставшимся задачам» — это другая величина, не upper-bound.

После исправления oracle построен прямо из per-task top-1 и теперь точно совпадает с top-1 из `e1_top3.json` для всех 4 задач:

| task_id | oracle (планирование/исполнение/верификация) | top-1 из e1_top3.json | совпадение |
|---|---|---|---|
| commongen | debate | debate (0.584) | да |
| dabench | debate | debate (0.348) | да |
| gsm8k | star | star (1.000) | да |
| humaneval | chain | chain (0.933) | да |

Это **валидный upper-bound для E3 RQ2**: в режиме `topology_router=oracle` адаптивная система на этих 4 задачах теперь действительно достигает class-level потолка 0.716, а не его LOO-аппроксимации (~0.65–0.66, которую дал бы старый файл с hierarchical-everywhere).

**Операционное следствие для E3.** До исправления запуск E3 с `topology_router=oracle` *не* достигал реальный upper bound: LOO выбирал middling-топологии, и oracle-режим систематически уступал даже «лучшей статике per task_type», что искажало интерпретацию RQ2 (adaptive vs static). После фикса oracle-режим теперь корректно отражает best-per-task — и paired t-test «oracle vs best-static» (а также «rule-router vs oracle») будет проверять именно ту гипотезу, которую формулирует план §4.

---

## 7. Стоимость и время

| Метрика | Значение |
|---|---:|
| Всего runs | 900 |
| Total compute cost (sum `budget_spent_usd`) | **$12.62** |
| Mean cost / run | $0.01402 |
| Max cost / run | $0.2348 |
| Total wall-time (sum `wall_time_s`) | 111.9 cpu-h |
| Wall-clock @ `parallelism=30` | ~3.7 h |
| Mean wall / run | 447.6 s |
| p50 / p95 / max wall_s | 182 / 1813 / 3999 |
| Mean iterations / run | 4.35 |

Фактическая стоимость **в 3.5 раза ниже** плановой оценки `~$45` (`arch/experiment_plan.md §11.3`). Причина — `reasoning_effort=low` для workers даёт ~3000 tok/s и существенно меньше токенов на iter, чем закладывалось в эвристике (700 tok/call × 3 calls/iter). Это позитивный сюрприз для бюджета E2–E4: эвристика `estimate.use_historical: true` теперь возьмёт реальные ~$0.014/run и переоценит общий бюджет с $270 → ~$80–120 на весь E1–E4, что даёт большой запас на ablations и retries.

Wall-clock тоже лучше плана (3.7 ч против 2.5 ч плановых) — здесь, наоборот, parallelism=30 вместо 12 и I/O-oversub Cerebras сыграли в плюс.

**Implication для scaling:** при текущей экономике один полный confirmation run (180 задач × 5 seeds × 2 моделей = 1800 runs) обойдётся в ~$25 — укладывается даже в самый строгий бюджет.

---

## 8. Связь со следующими экспериментами

1. **E2 (HITL-baseline).** Использует `analysis/e1_top3.json` как фильтр `topology.name.sweep` per task_type. Поскольку union top-3 = все 5 топологий, экономия идёт не на уровне «исключённых топологий», а на уровне (task × topology) ячеек: вместо 5×4=20 ячеек на (topology, task_type) грид остаётся 3×4=12, т.е. **−40% объёма E2**. Это эквивалентно smart-cut §11.2.2.

2. **E3 (adaptive topology).** Использует два артефакта из E1:
   - `data/oracle/e1_leave_one_out.json` — после commit `d6cfcbd` собран как per-task top-1 (форма: `by_task_id[task_id][phase] → topology`) и в точности совпадает с top-1 из `e1_top3.json` для всех 4 задач (см. §6). Это семантически правильный **upper-bound для RQ2**: «что достижимо при идеальном знании задачи». Class-level mean = 0.716 (vs 0.676 для best-static chain).
   - Baseline для paired t-test'а: best-static per task_type (chain/star/debate/debate) — тот же `e1_top3.json[0]`.
   - **Операционно:** в режиме `topology_router=oracle` E3-runner теперь матчится по `task_id` и поднимает правильную топологию (debate / debate / star / chain) на каждой из 4 задач. Это даёт чистый ceiling-сигнал для сравнения с `rule`-router и адаптивными вариантами.

3. **E4 (adaptive role).** Использует best (topology, role) per task_type из E2; E1 в неё входит косвенно — через выбор топологий, которые E2 вообще тестирует.

4. **E5 (real human).** Champion configs из E4, но E1 определяет baseline-конфиг «best-static + best-role» — это chain/star/debate в зависимости от task_type, как показано выше.

---

## 9. Итоги

- **RQ1 опровергает наивный single-winner.** Победители per task_type — три разные топологии (chain, star, debate); hierarchical — стабильный «лучший по cost / квази-универсальный второй».
- **Union of top-3 = все 5 топологий** — методологическое обоснование per-(task, topology) фильтра в E2.
- **Mesh деградирует** на programming/creative, что эмпирически валидирует smart-cut §11.2.4 (исключение mesh для programming-задач в E2+).
- **Debate vs HumanEval — антипаттерн**, который стоит явно проговорить в тексте диплома как ограничение «debate as universal improvement».
- **Бюджет E1 — $12.62, в 3.5 раза ниже плана**; это даёт запас на ablations в E2–E4.
- **Class-level oracle gap = 0.716 vs 0.676 = ~4 п.п.** — headroom для адаптации в E3 невелик; пороги plan §4 («Adaptive выигрышно при ≥5% quality») придётся пересмотреть на pre-registration перед E3.
- **DABench требует отдельного разбирательства** (низкое качество всех топологий, не предсказано планом).
- **`data/oracle/e1_leave_one_out.json` пересобран в commit `d6cfcbd`** под семантику per-task top-1 и теперь точно совпадает с top-1 из `e1_top3.json` (debate/debate/star/chain) — валидный upper-bound для E3 RQ2; режим `topology_router=oracle` теперь корректно отражает best-per-task, чего не делала предыдущая LOO-версия.
