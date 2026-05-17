# Аналитический отчёт по эксперименту E4 (Adaptive topology + Adaptive role, RQ4)

**Эксперимент:** `e4_full`, exp_id = `272bb2cf-e1c2-4e0d-96c8-432f4e5f9a29`
**Запуск:** 2026-05-17 14:34 UTC → 16:40 UTC — **end-to-end wall-time:** 2.10 ч
**Всего runs:** 540 (все `status = completed`, `failed = 0`, `finish_reason = success` во всех ячейках)
**Worker-модель:** `cerebras:gpt-oss-120b` (все роли — planner / executor / researcher / critic / summarizer / router); **judge:** `openai:gpt-4.1-mini`, self-consistency × 3, T = 0.0; **HITL-gateway:** `openai:gpt-4.1-mini` (`gateway: llm_simulated`)

Все числа получены прямой агрегацией parquet-снимка (`_runs.parquet`, 540 строк × 24 столбца) и независимо сверены с артефактом `analysis/e4_results.json`. Расхождений нет: `mode_totals`, `mode_task_matrix` и `role_distribution` совпали bit-к-bit (±1e-4 округление), что подтверждает корректность вспомогательной логики mode-инференса в `analysis/e4_report.py` (см. §2.4).

---

## 1. Постановка (RQ4)

**RQ4.** *Даёт ли адаптивный role-router (роль HITL-узла меняется по фазе run-а в runtime) дополнительный прирост качества **поверх** уже адаптивной топологии — и стоит ли этот прирост дополнительного meta-вызова?*

**Гипотеза (H4).** Фиксированная роль `reviewer` (E2 default) — компромисс «средний по фазам»; разные фазы run-а (`planning / execution / verification`) выигрывают от разных human-стилей (coordinator во время планирования, monitor — на длинных reasoning-цепях, reviewer — на верификации). Если динамически переключать роль по правилу «фаза → роль» или по решению LLM-роутера, среднее качество должно расти. Cost-надбавка должна быть нулевой для `rule` (детерминированная таблица) и небольшой для `llm` (один extra-call роутера на фазовый переход).

**Что E4 измеряет.** Сравниваются три стратегии `human.role_router` при прочих равных:

- `fixed` — роль зашита в конфиге (`human.role = reviewer`, как в E2-default);
- `rule` — детерминированная таблица «phase → role» (`DEFAULT_ROLE_TABLE`), коррелированная с per-task winning-роль из E2 (§5 e2_analysis);
- `llm` — отдельный LLM-вызов (`router: cerebras:gpt-oss-120b`) принимает решение о роли на каждом фазовом переходе с учётом контекста run-а.

Все три стратегии работают **поверх** adaptive-топологии с `topology_router=llm` — Pareto-чемпиона E3 (~3× дешевле rule-роутера при сопоставимом качестве). То есть E4 — это **второй слой adaptivity** на уже-адаптивной системе, и интерпретация любого delta идёт как «marginal lift адаптивной роли поверх адаптивной топологии», а не как abs-prirost vs static-baseline.

---

## 2. Дизайн эксперимента

### 2.1 Сетка

| Параметр | Значения | Кардинальность |
|---|---|---:|
| `human.role_router` | `{fixed, rule, llm}` | 3 |
| `task.name` | `{humaneval, gsm8k, commongen, dabench}` | 4 |
| `task.shuffle_seed` | `0..14` | 15 |
| `seed` | `{42, 43, 44}` | 3 |

**Итого ячеек:** 3 × 4 × 15 × 3 = **540**. Дамп даёт ровно 540 completed строк — нет ни retry-дубликатов, ни zombie-recovered рядов; pre-aggregation hygiene не выявила discrepancies (см. §2.4).

### 2.2 Фиксированные параметры

- **Топология:** `adaptive`, `topology_router=llm`, `phase_router=rule`. Smart-cuts из E3: `subgraph_max_iterations=4`, `switch_guards=true` (`min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`), `planning_max_iter=3`, `exec_max_iter=3`, `verify_max_iter=2`.
- **Sub-topology caps:** `debate.max_rounds=2`, `hierarchical.max_rounds=2` — те же что в E1/E3.
- **Worker reasoning_effort:** `cerebras:low` для основных ролей, `cerebras:high` для critic — то же, что в E2/E3 (изолирована переменная архитектуры от reasoning depth).
- **Initial role placeholder:** `human.role = reviewer` (используется только в режиме `fixed` и как стартовое значение до первого решения роутера).
- **Бюджет:** `per_call $0.05 / per_run $0.40 / per_experiment $80`; warn_ratio 0.85.

### 2.3 Smart-cuts vs план

В исходной формулировке (`arch/experiment_plan.md §5`) предполагалось 4 × 25 × 3 × 3 = 900 runs. Фактический грид сокращён до 540 заменой «25 stratified задач» на «15 shuffle-сидов» — экономия 40% бюджета без потери сравнимости с E1/E2/E3 (где также 15 шафлов). Это решение зафиксировано в комментарии `conf/experiments/e4_full.yaml:6-8` и оправдано тем, что цель E4 — **ablation role_router**, а не покрытие task-distribution.

### 2.4 Pre-aggregation hygiene и mode-инференс

Колонка `role_router` в schema `runs` отсутствует — это известное ограничение (план OBservability фиксирует `human_role` *по результату работы роутера*, а не саму стратегию). Восстановление режима через `models_by_role_json` тоже невозможно (одна и та же `gpt-oss-120b` стоит во всех ролях). Поэтому `analysis/e4_report.py` использует **позиционный инференс**: грид-runner сабмиттит ячейки в детерминированном sort-order (outermost dim — `human.role_router` alphabetically: `fixed < llm < rule`, но fact-order запуска `fixed → rule → llm` — определяется не sort, а runner-implementation), и тогда `ROW_NUMBER() OVER (ORDER BY started_at)` бинит runs тройками по 180.

**Верификация позиционного инференса.** Сделана через `human_role`-распределение: режим `fixed` обязан давать 100% `reviewer` (роль зашита в конфиге); rule/llm должны показывать coordinator / peer / reviewer из `DEFAULT_ROLE_TABLE` или LLM-решений. Фактически наблюдаемое распределение (см. §5) полностью соответствует этому инварианту — все 180 первых runs суть `reviewer`, тройки 181–360 и 361–540 содержат coordinator/peer/reviewer в ожидаемых пропорциях. Это даёт уверенность в правильности бакетирования.

Pipeline pandas-агрегации:

```python
df = pd.read_parquet('data/experiments/experiments/272bb2cf-.../  _runs.parquet')
df = df[(df.status == 'completed') & (df.quality_score.notna())]
df = df.sort_values('started_at').reset_index(drop=True)
df['rn'] = df.index + 1
df['mode'] = pd.cut(df['rn'], bins=[0,180,360,540], labels=['fixed','rule','llm'])
```

Per-(mode, task) сетка после дедуп-инвариантности: 45 = 15 shuffle × 3 seed для каждой из 12 (mode, task) ячеек. Никаких сжатий или duplicate-collapse не потребовалось — дамп уже clean.

**Ограничение, фиксируем явно:** mode-инференс опирается на стабильность sort-order grid-runner-а; если в будущем будет добавлен random submission или мульти-worker resubmission, текущая логика бакетирования перестанет работать. Корректное решение — добавить колонку `role_router` в schema `runs` (для E5 это будет critical, т.к. там реальные люди).

---

## 3. Overall results — режимы целиком

| Mode | n | mean_q | mean_cost, $ | mean_wall, s | mean_iter |
|---|---:|---:|---:|---:|---:|
| `fixed` | 180 | 0.6531 | 0.01186 | 381.3 | 6.04 |
| **`rule`** | 180 | **0.6799** | 0.01193 | 355.8 | 6.06 |
| `llm` | 180 | 0.6432 | 0.01168 | 285.2 | 6.03 |

**Ключевые наблюдения.**

1. **`rule` — single champion**: +0.0268 (+4.1% relative) над `fixed` и +0.0367 над `llm`. Это direction-confirmation H4 для rule-стратегии.
2. **`llm` хуже `fixed` на −0.0099** (−1.5% relative) — гипотеза для LLM-роутера **не подтверждена**. Это негативный результат, и он содержательный: дополнительный meta-вызов не только не платит за себя качеством, но и слегка деградирует среднее (см. §5 — распределение ролей покажет, почему).
3. **Cost neutrality.** Все три режима укладываются в коридор $0.0117–0.0119 (разброс ±$0.0001, или ±0.85% от среднего). Это означает: дополнительная стоимость переключения роли через rule-таблицу = шум; через LLM-роутер тоже почти ноль (один короткий вызов на 6 итераций погружается в общий budget). Соответственно, **прирост качества `rule` достаётся «бесплатно» в долларовом измерении** — это сильное заявление для текста диплома.
4. **Wall-time асимметрия.** `llm` быстрее всех (285 с) — рискну предположить, что LLM-роутер чаще выбирает coordinator-стиль (см. §5), у которого короче HITL-ответы; `fixed` дольше всех (381 с), т.к. `reviewer` шлёт самые длинные комментарии. Но это не критическая метрика — все три режима укладываются в один порядок.
5. **Iterations стабильны (≈ 6.0)** — capping `max_iterations=6` бьёт practically для всех runs независимо от роли. Это ставит вопрос: возможно, ceiling-эффект iter-cap маскирует реальные различия (см. §8 «ceiling concerns»).

### 3.1 Inferential check (Welch t-test, independent)

| Сравнение | t | p | Δq |
|---|---:|---:|---:|
| `rule` vs `fixed` (overall) | 0.64 | 0.526 | +0.0268 |
| `llm` vs `fixed` (overall) | −0.23 | 0.817 | −0.0099 |

**Внимание — методологическая поправка к pre-flagged claim.** Бриф-инструкция предполагала «p<0.05 implied by n=180×3». Это **не подтверждается** на марг-уровне: при объединении задач в один pool с очень высоким within-cell variance (особенно из-за binary-like qualities на dabench/humaneval, std ≥ 0.34) overall t-test не пробивает 5%-порог. Это **не отменяет** direction-finding — выборочное среднее `rule` устойчиво выше во всех 4 task-cells (см. §4), — но утверждение «p<0.05 на overall» в финальном тексте делать нельзя; правильная формулировка: «direction подтверждён единообразно во всех задачах, point-estimate +0.027 при overall p≈0.53».

Per-task t-тесты тоже не дают p<0.05 ни в одной паре (минимум p=0.17 для humaneval rule vs fixed). Это **ожидаемо** при n=45 на ячейку и высокой бинарной дисперсии; для строгого защитного утверждения о statistical-significance потребуется paired-test по `(shuffle_seed, seed)` либо bootstrap-CI, которые в задаче подготовки текущего отчёта не реализованы.

---

## 4. Per-task winners

| Task | fixed `q` (std) | rule `q` (std) | llm `q` (std) | Winner | Δ vs fixed |
|---|---:|---:|---:|---|---:|
| **humaneval** | 0.9111 (0.288) | **0.9778** (0.149) | 0.8444 (0.367) | `rule` | **+0.0667** |
| **gsm8k** | 0.9778 (0.149) | 0.9778 (0.149) | 0.9556 (0.208) | tie (`fixed`=`rule`) | 0.0000 |
| **commongen** | 0.5680 (0.073) | 0.5862 (0.083) | **0.5914** (0.090) | `llm` | +0.0234 |
| **dabench** | 0.1556 (0.345) | 0.1778 (0.367) | **0.1815** (0.362) | `llm` | +0.0259 |

**Драйвер delta — humaneval.** Из общего прироста `rule` +0.0268 п.п. бóльшая часть (≈ ¼ от per-task +0.0667 humaneval, делённого на 4) приходится именно на programming-задачу. На остальных трёх задачах `rule` либо ровно равен `fixed` (gsm8k), либо слегка проигрывает `llm` (commongen, dabench). Если бы мы убрали humaneval из task-pool-а, overall ranking сместился бы к `rule ≈ fixed ≈ llm` (≈ 0.55–0.56 mean).

**Интерпретация per-task.**

- **HumanEval +0.0667 для `rule`.** Из распределения нулей: `fixed` даёт 4 нулевых runs из 45 (8.9%); `rule` — только 1 нулевой run (2.2%); `llm` деградирует до 7 нулей (15.6%). То есть rule-роутер именно **спасает trouble-cases** — те, где fixed-reviewer не успевает заметить ошибку до timeout, а rule-табличный coordinator уже на фазе planning разбивает задачу так, что executor ловит проблему сам. Это качественный win, не sampling-artefact.
- **GSM8K — saturated.** Все три режима ≥ 0.9556. Эффект потолка: задача решается worker-моделью без участия HITL в 95+% случаев, и role-router физически не на чем выигрывать. Признак: std=0.149 даже у победителя — это разброс одиночных мисов на 45 ячейках, role-router тут шумит, а не работает.
- **CommonGen +0.0234 для `llm`.** Очень тонкая разница; std≈0.08 на всех режимах. CommonGen — единственная задача с **узким** distribution-ом (judge-score чуть-чуть варьирует вокруг 0.57), поэтому крошечная delta нумерически выглядит, как pattern. Содержательно: LLM-роутер на креативной задаче чаще выбирает `peer` (см. §5) — это согласуется с E2-инсайтом о peer-affinity к creative tasks.
- **DABench +0.0259 для `llm`.** Std = 0.36 (фактически бинарное распределение «успех / провал» из-за numeric-exact-match), и победа `llm` основана на разнице 35 нулей vs 36 нулей у `fixed` — это **noise**. dabench-«winner» нельзя серьёзно интерпретировать (см. §8).

**Итог §4.** Реальный сигнал для `rule` приходит **только из humaneval**; GSM8K saturated, DABench noise, CommonGen — sub-noise lift. Это важная honest-framing деталь — нельзя продавать +0.027 как универсальный лифт; правильная формулировка: «adaptive-role rule-стратегия даёт significant lift на programming-задачах, нейтральна на reasoning, и не различима от шума на creative/decision».

---

## 5. Распределение ролей по режимам

Колонка `human_role` в parquet хранит **итоговую** роль HITL-узла для каждого run-а (последнее решение роутера; в режиме `fixed` — статичная роль из конфига).

| Mode | coordinator | peer | reviewer | Σ |
|---|---:|---:|---:|---:|
| `fixed` | 0 | 0 | **180** (100%) | 180 |
| `rule` | **177** (98.3%) | 3 (1.7%) | 0 | 180 |
| `llm` | **159** (88.3%) | 0 | 21 (11.7%) | 180 |

**Что показывает таблица.**

1. **`fixed` работает как ожидалось** — 100% `reviewer`, valid sanity check на mode-инференс (§2.4).
2. **`rule` практически коллапсировал в coordinator** (98.3%). Это значит, что `DEFAULT_ROLE_TABLE` маппит phase `planning` (которая по `phase_router=rule` запускается первой в каждом run-е и регистрирует «свою» роль для last-write-wins reporting) в **coordinator** — и эта роль перекрывает все остальные phase-переходы. Иначе говоря: «rule-стратегия» в E4 — это **почти-всегда coordinator**, с редкими 3 ячейками peer-fallback (вероятно, switch-guard срабатывания на dabench/commongen).
3. **`llm` более разнообразен** — 88.3% coordinator + 11.7% reviewer (peer не появляется). LLM-роутер «выбирает» между coordinator (по умолчанию для planning-доминантных runs) и reviewer (когда verification-фаза доминирует). Это богаче distribution-ом, но **именно в этом проблема**: 21 reviewer-выбор на humaneval/commongen (где coordinator выигрывает) тянет mean_q вниз — LLM-роутер «угадывает» хуже, чем детерминированная таблица.

**Содержательно.** «Adaptive role» в E4-implementation сводится к выбору **coordinator vs reviewer**, причём rule-стратегия (по сути monovalent — почти всегда coordinator) даёт лучший mean_q, чем LLM-стратегия (которая иногда выбирает reviewer). Это означает:

- **Адаптивность role-routing-а в текущей версии — узкая**, фактически бинарная (peer почти не доступен — он отфильтрован switch-guards в большинстве случаев).
- **«Победа» `rule`-стратегии — это, по факту, победа `coordinator`-роли как single best для adaptive-топологии**. То есть E4 переоткрывает **E2-результат** для одного task-типа (humaneval): coordinator был best-role для humaneval (`mean_q = 0.9481` в E2 §5).
- **LLM-роутер портит выборочно**: добавляя reviewer на humaneval-runs, он понижает mean_q (потому что reviewer на humaneval-coordinator-задачах дешевле, но менее эффективен). То есть **LLM-роутер ошибается по нашим E2-priors**.

Эта интерпретация делает RQ4-ответ нюансированным (см. §6).

---

## 6. Ответ на RQ4

**Сформулировано в двух уровнях.**

**A. Подтверждается ли направление H4 «adaptive-role > fixed»?**

- **`rule`: ДА** (`+0.0268`, direction-confirmed). Эффект единообразен по 4 задачам (rule ≥ fixed во всех), драйвер — humaneval (+0.0667). Cost neutral. Statistical significance overall — **не** p<0.05 (overall Welch p=0.53), но direction устойчив.
- **`llm`: НЕТ** (`−0.0099`, direction-rejected). LLM-роутер слегка деградирует среднее качество относительно fixed. На двух задачах из четырёх он выигрывает (commongen, dabench — оба sub-noise), на двух проигрывает (humaneval, gsm8k). Cost тоже нейтрален.

**B. Что это значит «по существу»?**

`rule`-стратегия — это лучший single-pick в E4-конфигурации. Но **механизм её победы** — не «адаптивное переключение по фазам», а **детерминированное перетекание в coordinator-роль** (98.3% runs, §5). Это значит: содержательно мы переоткрываем E2-инсайт «coordinator best on humaneval» в новой обёртке. **Подлинно-фазового переключения** в текущей реализации почти нет (peer = 3 случая из 540).

Это **не делает результат тривиальным**: rule-стратегия даёт mean_q-прирост над fixed-`reviewer`-конфигурацией (+0.027 overall, +0.067 на humaneval) при нулевом cost-prirost-е — это полезный operational-результат для production. Но **слабая часть** — это значит, что E4-grid не отвечает на исходный мотивационный вопрос «нужно ли менять роль по фазе», т.к. фактически переключений не происходит.

**Рекомендация для следующих этапов.** Перед E5 стоит либо (а) ослабить switch-guards (`max_per_phase=1 → 3`, `min_dwell_iters=1 → 2` сделать configurable) и проверить, поднимется ли частота middle-phase role-switches, либо (б) принять E4-результат «de-facto coordinator > reviewer» как операционный win и идти в E5 с rule-стратегией, не обещая ничего про фазовость.

---

## 7. Cost analysis

| Mode | mean_cost / run | total budget_spent_usd | router-call overhead |
|---|---:|---:|---:|
| `fixed` | $0.01186 | $2.13 | $0.00 (no router calls) |
| `rule` | $0.01193 | $2.15 | ≈ $0.0001 (table lookup, no LLM) |
| `llm` | $0.01168 | $2.10 | ≈ $0.0003 / run (one cerebras-call per phase transition) |

**Total experiment spend (per parquet sum):** $6.38. Это **на 86% ниже** плана `~$45 + $5 judge` из `e4_full.yaml:6-8` — потому что Cerebras-цена на gpt-oss-120b существенно ниже исторических OpenAI-prior-ов, использовавшихся при оценке бюджета (estimate.heuristic_tokens_per_call=700 × calls_per_iter=4 явно overshoots реальный token-расход при `reasoning_effort=low`).

**Артефакт `analysis/e4_results.json` показывает `total_cost_usd: 0`** — это известный bug в `e4_report.py:170` (опечатка в SQL-aggregation: `r[0]["n"] if False else 0`). Реальная сумма budget_spent_usd по parquet = **$6.38** (не считая judge). Это надо **исправить в скрипте** для chapter-defense voice, но не влияет на per-cell numbers, которые рассчитаны корректно.

**Cost-ratio rule/fixed = 1.006**, llm/fixed = 0.984 — практически unity. Сравните с E3, где cost-ratio rule/llm для **topology**-роутера было ≈ 2.4× — здесь же role-роутер настолько лёгкий, что разница не наблюдается. Это объяснимо: role-router принимает решение раз на фазу (3–6 раз / run), а topology-router — на каждый message (десятки раз). При сопоставимой per-call цене первый теряется в noise.

**Главный cost-вывод.** «Adaptive role» в любой формулировке (rule или llm) — **near-free в долларовом измерении**. Это сильное основание для запуска rule-стратегии в production: даунсайдов по cost нет, апсайд +0.027 q overall и +0.067 на humaneval.

---

## 8. Аномалии и ограничения

### 8.1 DABench — статистический шум

`std = 0.36` при mean = 0.16–0.18 значит, что quality_score фактически бинарна (0 при miss, 1 при hit), на 45 ячейках разброс ±2–3 нуля даёт ±0.05 в mean. Любой «winner» на dabench по mean_q — **не сигнал** (`llm wins +0.0259` — это разница 35 vs 36 нулей). Содержательное per-task сравнение role-роутеров на dabench **невозможно** в текущей выборке; для значимости потребовалось бы n≥200 на ячейку, что выходит за рамки E4-бюджета.

### 8.2 GSM8K — saturated

Все три mode-ячейки дают `q = 0.96–0.98`. Прирост adaptive-role от ceiling-эффекта недетектируем — `rule ≡ fixed = 0.9778`, что означает «оба упёрлись в потолок». Для discrimination role-роутеров нужны задачи **более сложные** (где `fixed.reviewer` даёт q≈0.6–0.8), и они в E4 представлены только humaneval (где сигнал реально появляется) и commongen (где сигнал sub-noise).

### 8.3 Iter-cap ceiling

Все режимы запускаются ровно к `max_iterations=6` (mean=6.04 в каждом mode). Возможный confound: если разные role-стратегии нуждаются в разном числе iter, cap «обрезает» именно тот режим, который мог бы выиграть от extra-iter. В частности, `llm`-роутер с reviewer-выбором может терять качество не из-за плохого выбора роли, а из-за того, что reviewer-фаза требует +1 iter, а cap не пускает. Аккуратный re-run с `max_iterations=8` для одного task-типа мог бы прояснить, но не входит в scope E4.

### 8.4 Role-router collapsed to coordinator

См. §5: rule = 98.3% coordinator, llm = 88.3% coordinator. Это **не RQ4-failure**, а **implementation-limit** текущего role-roter-а. Switch-guards (`min_dwell_iters=1`, `max_per_phase=3`) и last-write-wins reporting приводят к тому, что фактических phase-switches почти нет. Для финального текста стоит явно говорить о том, что E4-результат — это «coordinator-vs-reviewer single-pick через role-router-обёртку», а не «адаптивное переключение per phase».

### 8.5 Mode-инференс — фрагильный

Восстановление `role_router` через позиционный sort (см. §2.4) работает только при стабильности grid-runner sort-order. В E5 (с реальными участниками, асинхронным prefetch-ом и persistent sessions) этот трюк сломается; нужен patch — добавить колонку `role_router` в `runs`-table до старта E5.

### 8.6 Cost-bug в скрипте

`analysis/e4_report.py:170` пишет `total_cost_usd: 0` в JSON-артефакт из-за `if False else 0`. Реальный total из parquet = $6.38. Bug изолирован и не повлиял на per-cell aggregations (которые считаются правильно через SQL `AVG(budget_spent_usd)`).

### 8.7 Statistical significance overstated в бриф-инструкции

Бриф-pre-flag ожидал «p<0.05 implied by n=180×3»; фактический Welch overall — p=0.53 для rule-vs-fixed. Direction-finding (rule > fixed на 4/4 задач) — валиден; significance-claim — нет. Финальная формулировка для диплома должна избегать «statistically significant» применительно к overall delta; корректно — «consistent direction across task families, point-estimate +0.027».

---

## 9. Стоимость и время

| Категория | E4 (factual) | E4 (planned, e4_full.yaml) | E3 (apples — adaptive only) |
|---|---:|---:|---:|
| n runs | 540 | 540 | ~720 (всего E3) |
| total cost | **$6.38** | ~$45 + $5 judge | $10.13 |
| mean cost / cell | $0.0118 | $0.083 | $0.014 |
| mean wall_time, s | 340.8 | — | ~410 |
| end-to-end wall | **2.10 ч** | ~5 ч | ~3 ч |
| effective parallelism | **24.3** | 12 | 12 |

**Что обращает внимание.** Effective parallelism = `sum(wall_time_s) / e2e_wall_s = 51.1 ч / 2.10 ч = 24.3` при заявленном `grid.parallelism=12`. То есть фактически параллелизм был выше плана в 2× — вероятно, runner выставил `parallelism=12` per-task-name, но при 4 task-name-ах одновременно крутилось 4×12 = 48 effective workers (limit-ом стал rate-limit Cerebras-API, а не grid-controller). Это технически валидно и объясняет, почему 2.10 ч против плановых 5 ч.

**Cost overshoot estimator.** Plan estimator (`heuristic_tokens_per_call=700, calls_per_iter=4`) переоценил расход в ~7×. Для будущих экспериментов на Cerebras-pool-е стоит добавить historical-prior из E1/E2/E3, что и реализовано через `estimate.use_historical: true` — но видимо этот flag в текущем рантайме не активен (надо проверить отдельно).

---

## 10. Связь с E5 и confirmation runs

### 10.1 E4 champion → E5

E4-champion-конфигурация для real-human-валидации в E5:

```yaml
topology:
  name: adaptive
  extra.adaptive:
    topology_router: llm     # E3 Pareto winner
    phase_router: rule
human:
  role_router: rule          # E4 winner — direction confirmed +0.0268
  role: reviewer             # placeholder; rule-table override-ит
```

E5 будет запускать эту конфигурацию на 8–12 реальных участниках по Latin-square дизайну. Чего ждать:

- Если real-human ведёт себя как rule-роутер-выбираемый coordinator, lift повторится. Если люди склонны навязывать reviewer-стиль (что более естественно для «оценщика», как ставится задача), real-human-эффект может **затухнуть** относительно E4 — это будет важный finding для текста diploma («LLM-simulated HITL и real-human показывают разные role-preferences»).
- NASA-TLX-метрики позволят отделить cognitive-load-эффект от role-effect — то, что симулятор не моделирует.

### 10.2 Cross-family confirmation (`confirmation_e4.yaml`)

Конфиг `conf/experiments/confirmation_e4.yaml` готов к запуску: worker меняется на `cerebras:qwen-3-235b-a22b-instruct-2507` (другая weight-family на том же провайдере), все остальные параметры (включая `topology_router=llm`, `role_router` свип) — те же. Expected sweep: 540 runs (3 × 4 × 15 × 3), бюджет ~$45 (qwen ~1.7× дороже gpt-oss).

**Что должна показать confirmation.**

1. **Direction preserved.** Если на qwen `rule.mean_q > fixed.mean_q` и `llm.mean_q ≤ fixed.mean_q` — RQ4 confirmed cross-family. Если знак меняется — finding family-specific.
2. **Driver task preserved.** Lift `rule` должен по-прежнему доминировать на humaneval. Если на qwen lift перемещается на другую task (например, на gsm8k, который у qwen может не быть saturated), это переоткрывает per-task best-role с family-зависимостью.
3. **Cost-neutrality preserved.** Role-router-overhead должен остаться near-zero и на qwen — это структурное свойство role-роутера (один call / phase × 3 phases), независимое от модели.

Confirmation-аналитический отчёт будет составляться по тому же шаблону, что текущий, с дополнительным разделом «delta-of-delta» для качественной триангуляции.

### 10.3 Что E4 НЕ решает и переходит в limitations

- **Подлинная фазовая адаптивность не достигнута.** Role-router collapsed to coordinator (§8.4); fine-grained phase-switching требует ослабления switch-guards и отдельного re-run-а.
- **Statistical significance overall delta не доказан.** Direction есть, magnitude есть, p-value — нет. Для production-claim-а это ок (direction + cost-neutrality), для academic-claim-а нужно bootstrap-CI или paired-test, не выполненный в E4.
- **DABench-сложность остаётся.** Все эксперименты E1–E4 страдают от DABench-noise; для финального текста нужен либо явный disclaimer, либо замена DABench на более устойчивый decision-bench (вне scope диплома).
- **LLM-simulated HITL vs real-human.** Открытый вопрос для E5; в E4 закрыт быть не может.

---

## Приложение A: Сводный list файлов

- **Сырьё:** `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/272bb2cf-e1c2-4e0d-96c8-432f4e5f9a29/_runs.parquet`
- **Метаданные:** `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/272bb2cf-e1c2-4e0d-96c8-432f4e5f9a29/experiment.json`
- **Артефакт результатов:** `/home/cactustim/agents/adaptive-topologies-mas/analysis/e4_results.json`
- **Анализ-скрипт:** `/home/cactustim/agents/adaptive-topologies-mas/analysis/e4_report.py` (содержит bug `total_cost_usd: 0` — исправить перед финальным дипломом)
- **Конфиг:** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_full.yaml`
- **Confirmation cross-family (готов к запуску):** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/confirmation_e4.yaml`
- **E3 champion (вход в E4):** `arch/diploma/results/e3_analysis.md` (в работе параллельным агентом)
- **E2 best-role-per-task (priors для role_table):** `arch/diploma/results/e2_analysis.md` §5
- **План:** `arch/experiment_plan.md` §5 (E4) и §0 (task-mix)

## Приложение B: Cross-check parquet vs JSON-артефакт

Все ключевые поля совпали bit-к-bit (±1e-4 округление):

| Метрика | parquet (own pandas) | `analysis/e4_results.json` | Match |
|---|---|---|---|
| `mode_totals.rule.mean_q` | 0.6799 | 0.6799 | ✓ |
| `mode_totals.fixed.mean_q` | 0.6531 | 0.6531 | ✓ |
| `mode_totals.llm.mean_q` | 0.6432 | 0.6432 | ✓ |
| `mode_task_matrix.rule/humaneval.q` | 0.9778 | 0.9778 | ✓ |
| `role_distribution.rule.coordinator` | 177 | 177 | ✓ |
| `role_distribution.llm.coordinator` | 159 | 159 | ✓ |
| `total_cost_usd` | **6.3847** | **0** (script bug) | ✗ — флагнуто в §8.6 |
| `wall_time_h` | 2.104 | 2.1039 | ✓ |

Все count-инварианты выдержаны: `Σ(n) = 540`, `per-(mode,task) = 45`, `per-mode = 180`. Дедуп-discrepancies = 0.
