# Аналитический отчёт по confirmation-эксперименту E4 (cross-family validation на Cerebras Qwen3-235B)

**Эксперимент:** `confirmation_e4`, exp_id = `d7f7d1eb-32c0-4315-a8dc-49b65a4d6894`
**Дамп:** `_runs.parquet`, 540 строк × 24 столбца, все `status = completed`, `finish_reason = success`, `quality_score.notna() == True`.
**End-to-end wall-time:** 1.345 ч (max(`finished_at`) − min(`started_at`)); сумма wall_time_s = 33.0 ч; **effective parallelism = 24.5** при заявленном `grid.parallelism=8` (см. §9).
**Worker-модель:** `cerebras:qwen-3-235b-a22b-instruct-2507` (все роли — planner / executor / critic / researcher / summarizer / router); **judge:** `openai:gpt-4.1-mini` (КОНСТАНТА во всех экспериментах диплома); **HITL-gateway:** `openai:gpt-4.1-mini` (`gateway: llm_simulated`).
**Базовая топология:** `adaptive` с `topology_router=llm`, `phase_router=rule` — E3-чемпион, тот же, что и в оригинальном E4.

Этот отчёт — парный к `e4_analysis.md` (gpt-oss-120b). Цель — не «найти победителя на qwen», а ответить на вопрос **переносится ли E4-вывод между weight-families**. Числовая агрегация прямо из parquet полностью совпала с `analysis/confirmation_e4_results.json` (per-cell mean_q, mean_cost, role_distribution — bit-к-bit ±1e-4); единственное расхождение — `total_cost_usd = 0` в JSON-артефакте (см. §9, тот же баг `e4_report.py:170` `r[0]["n"] if False else 0`). Фактический total из parquet = **$9.40**.

---

## 1. Постановка confirmation-вопроса

Оригинальный E4 на `cerebras:gpt-oss-120b` дал **direction-confirmed** результат для RQ4 в части `rule`-стратегии (`+0.0268` mean_q над `fixed`, overall p≈0.53 — direction without significance) и **rejected** для `llm`-стратегии (`−0.0099`). Драйвером прироста был **humaneval** (`+0.0667`); GSM8K saturated, CommonGen и DABench — sub-noise.

Confirmation-эксперимент **не повторяет дизайн** — он **зеркалит** его строго apples-to-apples (тот же sweep, та же топология, те же switch-guards, тот же judge), меняя **только worker-family**: `gpt-oss-120b` (OpenAI gpt-oss release weights) → `qwen-3-235b-a22b-instruct-2507` (Alibaba Qwen, 235B-parameter MoE, cross-family). Это структурно сильнейший вариант cross-family тест: меняется и архитектура, и pre-training corpus, и instruction-tuning происхождение, и токенизатор — провайдер (Cerebras) и весь остальной стек остаются неизменными.

**Вопросы, на которые отвечает confirmation:**

1. **Сохраняется ли direction delta `rule − fixed`?** На gpt-oss было `+0.0268`. Если на qwen знак сохраняется (`> 0`) — RQ4 confirmed cross-family. Если меняется (`< 0`) — finding family-specific.
2. **Сохраняется ли direction delta `llm − fixed`?** На gpt-oss было `−0.0099` (rejection direction). Если на qwen тоже `< 0` — direction-of-rejection confirmed.
3. **Сохраняется ли role-distribution выбора rule-роутера?** На gpt-oss `rule` коллапсировал в **coordinator** (177/180 = 98.3%). Если на qwen паттерн тот же — rule-таблица family-инвариантна. Если другой — `DEFAULT_ROLE_TABLE` зависит от поведения worker-а через свои phase-classifier-ы, и «победа rule» в E4 не переносится механически.
4. **Сохраняется ли per-task rank-order драйверов?** На gpt-oss драйвером rule-прироста была humaneval; gsm8k saturated; dabench noisy. На qwen task-difficulty profile может быть совсем другим, и переоткрытие «какая задача драйвит» — само по себе finding.

Заявка такого confirmation в текст диплома — это не «retest with bigger N», а **тест на генерализуемость claim-а**. Negative result здесь содержательнее positive: он переопределяет границы основного утверждения работы.

---

## 2. Дизайн и pre-aggregation hygiene

### 2.1 Сетка

| Параметр | Значения | Кардинальность |
|---|---|---:|
| `human.role_router` | `{fixed, rule, llm}` | 3 |
| `task.name` | `{humaneval, gsm8k, commongen, dabench}` | 4 |
| `task.shuffle_seed` | `0..14` | 15 |
| `seed` | `{42, 43, 44}` | 3 |

**Итого ячеек:** 3 × 4 × 15 × 3 = **540**. Дамп содержит ровно 540 completed runs, дедуп-discrepancies = 0 (никаких retry или zombie-recovered рядов).

### 2.2 Фиксированные параметры

Идентичны оригинальному E4 — конфиг `confirmation_e4.yaml` использует `include: e4_full.yaml`, оверрайдя только `model.*` (qwen-3-235b во все роли, judge остаётся `openai:gpt-4.1-mini`) и `budget` (per_call_usd 0.05 → 0.08, per_run_usd 0.40 → 0.65, per_experiment_usd 80 → 100 — qwen ~1.7× дороже по pricing). Smart-cuts из E3 (subgraph_max_iterations=4, switch_guards со `min_dwell_iters=1`, `cooldown_iters=2`, `max_per_run=6`, `max_per_phase=3`) сохранены 1:1. **Reasoning_effort у qwen отключён явно** (`provider_opts.cerebras: null`, `provider_opts.cerebras_critic: null`) — qwen-3-235b возвращает 400 BadRequestError при попытке передать `reasoning_effort`, что превращало бы run в q=0 silent failure. Это техническая особенность qwen API, зафиксирована в комментариях конфига.

### 2.3 Mode-инференс (role_router в schema отсутствует)

Колонка `role_router` отсутствует в `_runs.parquet` (известное ограничение, см. e4_analysis.md §2.4). Восстановление режима — позиционным sort-ом по `started_at` с биннингом тройками по 180. Pipeline:

```python
df = pd.read_parquet('data/experiments/experiments/d7f7d1eb-.../_runs.parquet')
df = df[(df.status == 'completed') & (df.quality_score.notna())]
df = df.sort_values('started_at').reset_index(drop=True)
df['rn'] = df.index + 1
df['mode'] = pd.cut(df['rn'], bins=[0, 180, 360, 540], labels=['fixed', 'rule', 'llm'])
```

**Валидация инференса.** Режим `fixed` обязан давать 100% `human_role == 'reviewer'` (роль зашита в конфиге). Фактически наблюдаем ровно 180 reviewer в первой трети — sanity-check проходит. Остальные две трети должны показывать distribution роутер-выборов; они показывают (см. §5) и пропорции структурно отличаются от gpt-oss — что подтверждает, что бакеты разделены корректно, а **именно поведение** роутеров cross-family отличается.

Per-(mode, task) сетка после инференса: ровно 45 = 15 shuffle × 3 seed для каждой из 12 ячеек.

---

## 3. Side-by-side overall — gpt-oss vs qwen

| Mode | gpt-oss mean_q | qwen mean_q | Δq (qwen − gpt-oss) | gpt-oss mean_cost | qwen mean_cost | cost ratio qwen/gpt-oss |
|---|---:|---:|---:|---:|---:|---:|
| `fixed` | 0.6531 | 0.5820 | **−0.0711** | $0.01186 | $0.01858 | 1.57× |
| `rule` | **0.6799** | 0.5013 | **−0.1786** | $0.01193 | $0.01716 | 1.44× |
| `llm` | 0.6432 | 0.5277 | **−0.1155** | $0.01168 | $0.01647 | 1.41× |

**Главное наблюдение overall.** Qwen-235B **проигрывает gpt-oss-120b абсолютно** во всех трёх режимах — это ожидаемо для diploma framing (qwen ≠ gpt-oss; preview-версия, без reasoning_effort), и не имеет отношения к RQ4. Существенно — **относительный порядок режимов меняется**:

- На gpt-oss: `rule (0.680) > fixed (0.653) > llm (0.643)`.
- На qwen: `fixed (0.582) > llm (0.528) > rule (0.501)`.

То есть **rule из chempion-а превращается в loser-а** — это **смена sign-а delta**, не просто attenuation. На gpt-oss rule выигрывал у fixed на +0.0268; на qwen rule проигрывает fixed на **−0.0807**.

### 3.1 Inferential check (Welch t-test, independent)

| Сравнение | gpt-oss t | gpt-oss p | qwen t | qwen p | Δq на gpt-oss | Δq на qwen |
|---|---:|---:|---:|---:|---:|---:|
| `rule` vs `fixed` | 0.64 | 0.526 | **−1.75** | **0.081** | +0.0268 | **−0.0807** |
| `llm` vs `fixed` | −0.23 | 0.817 | −1.18 | 0.240 | −0.0099 | −0.0543 |

На qwen `rule vs fixed` приближается к p<0.05 (p=0.081) — это **сильнее, чем p=0.53 на gpt-oss**, но в **противоположную сторону**. По формальному критерию обе hypothesis-direction-ы остаются statistically non-significant, но **направление и magnitude rule-эффекта обратились**.

### 3.1.1 Bootstrap-CI (added 2026-05-17, post-review)

Non-parametric bootstrap (10 000 iters, percentile, RNG=42):

| Сравнение (qwen) | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| `rule` − `fixed` | −0.0807 | [−0.1731, +0.0098] | **borderline** (upper edge почти на 0) |
| `llm` − `fixed` | −0.0543 | [−0.1438, +0.0353] | includes 0 |

`rule` reversal — **borderline**: верхняя граница CI = +0.0098, то есть формально включает 0, но почти исключает. Это сильнее, чем gpt-oss-original (где CI [−0.055, +0.110] симметричен относительно 0). Direction-claim «role-rule на qwen НЕ улучшает quality, и возможно ухудшает» — **слабо устойчив**, но абсолютной уверенности bootstrap не даёт.

### 3.1.2 Mode-инференс — boundary verification (added post-review)

Дополнительная проверка mode-инференса через `human_role` transition в точках rn=180→181 и rn=360→361:

| rn | started_at | human_role |
|---:|---|---|
| 179–180 | 17:27:59 / 17:28:02 | reviewer |
| **181** | **17:28:11** | **peer** ← fixed→rule, чёткий flip |
| 182–360 | 17:28:26 ... 17:55:25 | peer (rule dominates peer 94%) |
| **361** | **17:55:34** | **judge** ← rule→llm, flip (llm даёт judge 18%) |
| 362 | 17:55:58 | reviewer (llm mix) |

Обе границы **детектируются** по role-flip: reviewer→peer на 180→181 и появление **judge** на 360→361 (judge не встречается ни в fixed=reviewer, ни в rule=peer-доминантной — это уникальный маркер llm-mode). Это **более сильная** sanity-проверка, чем для gpt-oss E4, и независимо подтверждает корректность биннинга.

### 3.2 Cost-ratio

Cost-ratio qwen/gpt-oss на overall ≈ 1.4–1.6× (теоретическая верхняя оценка из pricing — 1.7×, на практике немного меньше из-за более коротких qwen-completion-ов на gsm8k и dabench). Внутри qwen cost-spread между режимами `0.01647 ($llm) – 0.01858 ($fixed)` — те же ±5% порядка, что и на gpt-oss; никакого role-router-overhead-а не появилось.

**Total experiment spend на qwen (per parquet sum):** **$9.40** vs $6.38 на gpt-oss → ratio 1.47×, в коридоре ожиданий по pricing. JSON-артефакт `analysis/confirmation_e4_results.json` показывает `total_cost_usd: 0` — тот же баг `e4_report.py:170`, отдельный fix-flagged-issue (§9).

---

## 4. Per-task — side-by-side

### 4.1 mean_q × (mode, task), gpt-oss vs qwen

| Task | Mode | gpt-oss q | qwen q | Δq (qwen − gpt-oss) | gpt-oss winner? | qwen winner? |
|---|---|---:|---:|---:|---|---|
| **humaneval** | fixed | 0.9111 | 0.4667 | −0.4444 | | **★** |
| | rule | **0.9778** | 0.3556 | −0.6222 | **★** | |
| | llm | 0.8444 | 0.3556 | −0.4888 | | |
| **gsm8k** | fixed | 0.9778 | **0.2667** | −0.7111 | **★** (tie) | **★** |
| | rule | **0.9778** | 0.1333 | −0.8445 | **★** (tie) | |
| | llm | 0.9556 | 0.1778 | −0.7778 | | |
| **commongen** | fixed | 0.5680 | 0.6613 | +0.0933 | | |
| | rule | 0.5862 | 0.6755 | +0.0893 | | |
| | llm | **0.5914** | **0.6774** | +0.0860 | **★** | **★** (marginal) |
| **dabench** | fixed | 0.1556 | **0.9333** | **+0.7777** | | **★** |
| | rule | 0.1778 | 0.8407 | +0.6629 | | |
| | llm | **0.1815** | 0.9000 | +0.7185 | **★** (noise) | |

**Per-task Welch на qwen (для строгости):**

| Task | rule−fixed Δ | p | llm−fixed Δ | p |
|---|---:|---:|---:|---:|
| humaneval | −0.1111 | 0.289 | −0.1111 | 0.289 |
| gsm8k | −0.1333 | 0.117 | −0.0889 | 0.316 |
| commongen | +0.0142 | 0.386 | +0.0161 | 0.379 |
| dabench | −0.0926 | 0.150 | −0.0333 | 0.552 |

Ни одна per-task разница на qwen не проходит p<0.05, но **направления** разительно отличаются от gpt-oss: rule **проигрывает** fixed на 3 из 4 задач (humaneval, gsm8k, dabench) и формально-марginally выигрывает только на commongen (+0.014, p=0.39). На gpt-oss rule **выигрывал или равнялся** fixed на всех 4 задачах.

### 4.2 Cross-family task-difficulty surprises

Кросс-family сравнение `gpt-oss q` vs `qwen q` (по `fixed`-режиму как baseline-у внутри каждой family) обнаруживает **четыре качественно разных pattern-а по задачам**:

| Task | gpt-oss q (fixed) | qwen q (fixed) | Δ | Интерпретация |
|---|---:|---:|---:|---|
| **humaneval** | 0.911 | 0.467 | **−0.444** | Qwen существенно хуже на code-generation. Drop в 2 раза — не приращение шума, а реальное family-различие в pythoneval-tuning-е. |
| **gsm8k** | 0.978 | 0.267 | **−0.711** | **Crash.** Saturated на gpt-oss, у qwen-235B (preview, no reasoning_effort) практически unsolvable в текущем сетапе. Возможно reasoning_effort disable играет ключевую роль; нужно проверить отдельно. |
| **commongen** | 0.568 | 0.661 | **+0.093** | Stable / slight win — text-generation transfers cleanly cross-family. Это единственная задача с relatively narrow distribution, и она показывает что qwen на «нормальных» NLG-задачах не хуже. |
| **dabench** | 0.156 | 0.933 | **+0.777** | **Reversal.** На gpt-oss DABench был «broken» (q ≈ 0.16 во всех экспериментах E1–E4, использовался как known limitation). На qwen — q = 0.93 в fixed-режиме. **DABench не сломан как benchmark — это была gpt-oss-specific weakness**. Это retroactively важная correction для текста диплома: ранее DABench у нас был disclaim-ом «broken для current LLM stack», теперь становится «broken for gpt-oss-family, solved by qwen-family». Для §10 это центральный finding cross-family валидации. |

Эти reversal-ы делают **простое сравнение** «role-router-эффекта» через mean over tasks методически рискованным: оригинальный E4-сигнал был доминирован humaneval-приростом, qwen-эксперимент имеет совершенно другое распределение task-difficulty, и любая arithmetic mean смешивает heterogeneous эффекты.

---

## 5. Role distribution shift — sole-most-striking finding

Один и тот же код `DEFAULT_ROLE_TABLE` и тот же `phase_router=rule` дают **разные роли** на разных worker-families. Сравнение:

| Mode | gpt-oss distribution | qwen distribution |
|---|---|---|
| `fixed` | reviewer: 180 (100%) | reviewer: 180 (100%) |
| `rule` | **coordinator: 177 (98.3%)**, peer: 3 (1.7%), reviewer: 0 | **peer: 169 (93.9%)**, reviewer: 11 (6.1%), coordinator: 0 |
| `llm` | **coordinator: 159 (88.3%)**, reviewer: 21 (11.7%), peer: 0, judge: 0 | **reviewer: 131 (72.8%)**, judge: 32 (17.8%), peer: 11 (6.1%), coordinator: 6 (3.3%) |

`fixed` валидирует mode-инференс (100% reviewer в обеих family).

**Что фактически меняется.**

1. **Rule на gpt-oss → coordinator-dominant; rule на qwen → peer-dominant.** Это **тот же rule-table**, тот же `phase_router=rule`, тот же кодовый путь. Различие — в тех самых **phase-classifier-ах**, которые внутри adaptive-stack-а определяют «текущая фаза = planning/execution/verification» по сигналам run-а (длина reasoning-trace, число tool-calls, presence of error-tokens). Эти классификаторы получают разные сигналы от qwen vs gpt-oss → классифицируют фазы по-разному → rule-table мапит phase→role по-разному → итоговая роль другая. **Rule-policy не является «model-independent decision rule»**, как могла подразумеваться, — она model-dependent через свои входы.
2. **LLM-роутер на gpt-oss → бинарный (coordinator/reviewer); на qwen → ternary+ (reviewer/judge/peer/coordinator).** Qwen-роутер существенно более «consultative» — он впервые использует роль `judge` (32 случая = 17.8% всех llm-runs; на gpt-oss judge не появлялся ни разу). Это поведенческое различие самих **LLM-роутер-LLM-ов**: qwen больше склонен делегировать judging как отдельную роль, gpt-oss «съедает» judging внутри reviewer. Это не bug — это валидное cross-family behavior-divergence, и оно показывает, что **policy LLM-meta-роутера наследует biases base-модели**, даже если архитектура промптов идентична.
3. **Peer резко чаще на qwen.** На gpt-oss peer был «исчезающим случаем» (3/180 rule = 1.7%); на qwen peer **доминирует** в rule-режиме (169/180 = 93.9%). Это значит: phase-classifier-ы у qwen значительно чаще классифицируют фазу как «execution/iteration» (которая, по `DEFAULT_ROLE_TABLE`, мапится в peer-стиль), а не как «planning» (→ coordinator) — что согласуется с тем, что qwen генерирует более длинные reasoning-trace-ы в общем (другой токенайзер, другая verbosity), и эти trace-ы пересекают thresholds-ы для execution-классификации раньше.

**Интерпретация.** Содержательно это означает, что **«adaptive role-routing» как класс политик чувствителен к base-model behavior на двух уровнях**: (а) rule-policy через phase-classifier-входы, (б) llm-policy через base-LLM-priors самого роутера. Оба механизма cross-family **не инвариантны**. Это первый сильный contra-pattern для предположения «role-router работает как кросс-модельный wrapper» — он не работает; он суть **per-model emergent behavior**.

---

## 6. Replication verdict — по каждому E4-findings

| Original E4 finding (gpt-oss) | Confirmation result (qwen) | Verdict |
|---|---|---|
| **rule beats fixed (+0.0268)** | rule loses to fixed (−0.0807), Welch p=0.081, sign FLIP | **REJECTED.** Magnitude обратной direction-ы (≈3× больше исходной, в противоположную сторону) — это не attenuation, это reversal. |
| **llm worse than fixed (−0.0099)** | llm worse than fixed (−0.0543), direction preserved, magnitude 5.5× | **Direction confirmed; magnitude amplified.** «LLM-роутер не платит за себя» — cross-family прочно подтверждено, даже усилено. |
| **rule is essentially «coordinator-only»** | rule is essentially «peer-only» (169/180 = 94%) | **REJECTED.** Same rule-table, different chosen-role distribution → rule-policy не invariant. |
| **Driver task = humaneval (rule wins +0.067)** | Driver task = humaneval (rule LOSES −0.111) | **REJECTED with sign flip.** Humaneval остаётся «самой spread-ed» задачей, но direction обратный. |
| **GSM8K saturated, role-router indifferent** | GSM8K collapsed to q=0.13–0.27, role-router indifferent in opposite tail | Direction (indifference) preserved, but ceiling → floor. Saturation finding family-specific. |
| **DABench noise (std=0.34, mean ~0.17, все sub-noise)** | DABench solved (q=0.84–0.93), все режимы high | **REJECTED.** DABench не broken — gpt-oss-specific weakness. |
| **CommonGen sub-noise, llm marginal +0.023** | CommonGen sub-noise, llm marginal +0.016 | **Direction confirmed.** Единственная задача где confirmation echo-ит оригинал. |
| **Cost neutrality of role-router (±0.85%)** | Cost neutrality preserved (±5% в коридоре, по-прежнему near-zero overhead) | **Confirmed.** Это структурное свойство role-роутер (1 call/phase), действительно invariant. |

**Сводно:** из ~8 distinct claim-ов оригинального E4 на qwen воспроизводятся **2** (llm-direction, cost-neutrality) и **1 echo-ом** (commongen-marginal-llm-win). **4 claim-а отбиваются с reversal direction-ы.** Это **сильнее, чем «не подтверждено» — это активное опровержение** основных positive findings.

---

## 7. Cross-family novelties

### 7.1 DABench reframing

Самый важный cross-family finding. В тексте e1/e2/e3/e4-analysis DABench фигурировал как «broken benchmark — все топологии score ~0.2, не интерпретировать как topology-specific». Qwen-результат `0.84–0.93` показывает: задача решаема, **gpt-oss-120b просто плох на ней** (numeric-exact-match reasoning + data-table parsing — area где qwen-235B значительно сильнее). Implication для диплома: ранние §-disclaim-ы «DABench broken» нужно переписать в «DABench gpt-oss-uncongenial, использовать с учётом family-bias».

### 7.2 GSM8K collapse

Обратная история. Qwen-235B в текущем сетапе (preview-версия, **без** `reasoning_effort`) **резко хуже** gpt-oss-120b на GSM8K (0.27 vs 0.98). Возможно две причины: (а) преимущество reasoning_effort у gpt-oss работало именно на GSM8K, и его отсутствие у qwen — главный фактор; (б) qwen-235B-instruct ориентирован больше на general/conversational, gpt-oss-120b ориентирован сильнее на math/code. Без отдельного ablation reasoning_effort это два конкурирующих объяснения; первая правдоподобнее (E2 показывал ~0.05 GSM8K-прирост от high-reasoning_effort на gpt-oss). Это **процедурный artefact**, не fundamental qwen weakness — но факт в том, что в **текущем cross-family setup-е** GSM8K функционально не работает.

### 7.3 CommonGen invariance

Единственная задача где (а) absolute mean_q практически одинаков (0.57 vs 0.66, оба mid-range), (б) variance стабильно низкая (std≈0.07–0.09), (в) per-mode rank-order сохраняется (llm marginal winner в обеих family). Это означает что **text-generation creative tasks transfer cross-family чище, чем code/math/decision** — потенциально полезный pattern для будущих обобщающих выводов: для NLG нам нужны разные benchmark-обоснования, чем для STEM.

### 7.4 HumanEval halving

Code-generation сильно деградирует cross-family (gpt-oss 0.91 → qwen 0.47). Это согласуется с известным narrative «Qwen-235B-Instruct менее tuned на pythoneval, чем gpt-oss-120b». В отличие от GSM8K-collapse, здесь reasoning_effort менее правдоподобен как объяснение (он не сильно повышал humaneval даже на gpt-oss в E2). Это **fundamental family-difference в code-tuning**.

---

## 8. Анализ — почему RQ4 не реплицировалось

Три **не взаимоисключающие** гипотезы.

### 8.1 Rule-table implicitly overfit на gpt-oss patterns

`DEFAULT_ROLE_TABLE` была сконструирована (см. `arch/experiment_plan.md §5`) на base-priors из E2-best-role-per-task, которые сами были измерены на gpt-oss-120b. То есть mapping `phase → role` фактически закодировал «как gpt-oss-фазы (по его trace-pattern-ам) лучше всего обслуживаются ролями» — но эта policy была применена к qwen, у которого phase-classifier-ы (см. §5) дают **другие phase-labels** на тех же task-ах. Следствие: то, что таблица называет «coordinator-phase для gpt-oss», на qwen чаще классифицируется как «execution-phase → peer» — а peer оказывается не оптимальной ролью для qwen-humaneval-execution. **Победа на gpt-oss была не general-purpose adaptive insight, а instance of policy-overfitting в условиях noise (p=0.53).**

### 8.2 Qwen имеет разный competence profile

Qwen-235B-Instruct: text-strong (commongen ok), data-strong (dabench solved), math-poor (gsm8k crash), code-medium-poor (humaneval halved). Rule-table предполагает «humaneval = code = coordinator helps with planning». На qwen humaneval — это **fundamentally harder task than for gpt-oss**, и role-tweak уже не имеет значения когда worker сам не справляется — fixed-reviewer (просто проверяющий output) делает ту же работу с меньшим overhead-ом. Иначе говоря: **role-router-эффект zero когда worker bottleneck доминирует над role-style-bottleneck-ом**.

### 8.3 Original E4 «win» — point estimate в noise

Critically важная honest framing. Оригинальный E4 показал `+0.0268` rule-vs-fixed с p=0.53 (Welch overall, см. e4_analysis.md §3.1). Это **point-estimate в zone-of-no-significance**. Если interpret это как «есть direction, но он fragile», confirmation-результат — это просто independent sample, который показал противоположное direction (тоже не significant, p=0.08). При null-hypothesis (нет effect-а) ожидается, что независимые samples будут давать sign-flips случайно. Это **stronger evidence что rule-effect не существует**, чем казалось из одного gpt-oss-run-а. Бoolean «confirmed/not» неуместен; правильное обновление belief-а — **prior после двух samples: effect близок к нулю, или family-dependent чисто структурно**.

Эти три гипотезы не конкурируют — они дополняют друг друга и образуют согласованную интерпретацию: оригинальный E4-win был композицией (a) policy-overfit, (b) gpt-oss-specific competence-task fit, (c) noise. На qwen все три фактора отрабатывают в обратную сторону, и direction reversal — естественное следствие.

---

## 9. Стоимость, время, технические артефакты

### 9.1 Cost

| Категория | qwen (factual) | qwen (planned, `confirmation_e4.yaml`) | gpt-oss E4 |
|---|---:|---:|---:|
| n runs | 540 | 540 | 540 |
| total cost | **$9.40** | ~$45 + judge | $6.38 |
| mean cost / cell | $0.0174 | $0.083 | $0.0118 |
| mean wall_time, s | 220.0 | — | 340.8 |
| end-to-end wall | **1.345 ч** | ~9–10 ч | 2.10 ч |
| effective parallelism | **24.5** | 8 | 24.3 |

**Cost overshoot:** plan estimator опять переоценил расход в ~5× (этот раз менее зверски, чем на gpt-oss — но всё ещё значительно). Cerebras-pricing на qwen-235B (0.60/1.20 per 1M) даёт фактический per-run cost ~$0.017, против заявленных в плане ~$0.083 на cell.

**Per-mode cost** (sum budget_spent_usd внутри 180 ячеек каждого режима):

| Mode | qwen $-spend | gpt-oss $-spend |
|---|---:|---:|
| `fixed` | 3.35 | 2.13 |
| `rule` | 3.09 | 2.15 |
| `llm` | 2.96 | 2.10 |

Cost-neutrality role-роутера сохраняется (разброс ±$0.20 на 180 ячейках). Заявление из e4_analysis.md «adaptive role в любой формулировке near-free» — переносится cross-family.

### 9.2 Wall-time

End-to-end 1.345 ч (vs 2.10 ч на gpt-oss) — короче, потому что qwen runs в среднем **короче** (220 с vs 341 с), что компенсирует более низкий Cerebras-throughput на qwen. Effective parallelism 24.5 при заявленном `grid.parallelism=8` — runner снова идёт выше плана из-за per-task-name parallelism (4 tasks × 8 = 32 effective workers max, limit-ом снова rate-limit).

### 9.3 Технические артефакты и bugs

- **`analysis/confirmation_e4_results.json` показывает `total_cost_usd: 0`** — тот же баг `analysis/e4_report.py:170` (`r[0]["n"] if False else 0`), что и в gpt-oss-отчёте. Реальный total из parquet — **$9.40**. Фикс изолирован; per-cell numbers корректны.
- **role_router columns не в schema** — те же ограничения, что и в E4 (§2.4 e4_analysis.md). Для E5 (с реальными участниками) нужно добавить колонку до старта.
- **reasoning_effort disabled на qwen** — не сравнимость per-call-reasoning между qwen и gpt-oss confounded. Для строгого ablation нужен либо отдельный run с qwen + reasoning_effort (если/когда qwen API это поддержит), либо явный disclaimer в тексте.
- **Mode-инференс sort-order** работает идентично; верификация через `fixed → 100% reviewer` проходит.

### 9.4 JSON `per_task_winner` ловушка

В `confirmation_e4_results.json.per_task_winner` для трёх из четырёх задач (`humaneval`, `gsm8k`, `dabench`) поле `mode = "fixed"` и `delta_vs_fixed = 0.0`. Это **artefact** алгоритма выбора winner-а в скрипте: когда выигрывает baseline (`fixed`), delta естественно равна 0. Не следует интерпретировать как «нет различия» — реальные deltas из `mode_task_matrix`: humaneval rule−fixed = −0.111, gsm8k rule−fixed = −0.133, dabench rule−fixed = −0.093. Все три отрицательные и существенные по magnitude. Это flag-issue для будущего фикса report-скрипта: per_task_winner должен включать **все** deltas, не только vs winning mode.

---

## 10. Связь с diploma narrative

Confirmation **меняет** core claim диплома относительно adaptive role-routing.

### 10.1 До confirmation

Из e4_analysis.md можно было защищать formulation:
> «Adaptive role-routing (rule-based) даёт нумерический lift mean_q (+0.027) над fixed-стратегией при cost neutrality. Direction подтверждён единообразно на 4 задачах. Производственно жизнеспособно.»

### 10.2 После confirmation

Эта formulation **не может быть сохранена** в текущей форме. Корректное обновление:

> «Adaptive role-routing показывает **family-dependent эффект**. На gpt-oss-120b rule-стратегия даёт марgin direction-improvement +0.027 (non-significant); на qwen-3-235b — direction-reversal −0.081 (близко к significance, p=0.08). Знак delta зависит от base worker-модели через два механизма: (а) phase-classifier-ы дают разные phase-labels на разных families, что переводит rule-table в разные effective-role-distribution-ы (gpt-oss → coordinator-dominant; qwen → peer-dominant); (б) LLM-meta-роутер наследует priors base-модели. Заявление "adaptive role помогает универсально" не подтверждается; правильное заявление — "policy-design адаптивных role-roter-ов критически зависит от base-model behavior, и валидация на единственной family не достаточна".»

### 10.3 Что это значит для глав диплома

1. **Глава про RQ4** должна быть переписана: исходный E4 → confirmation → family-dependence как **сам по себе central finding**. Это негативный результат, но он содержательно сильнее, чем point-estimate-positive результат.
2. **Глава про DABench**: removal disclaim-а «broken benchmark» в пользу «benchmark gpt-oss-uncongenial». Тщательнее в §-related-work — это касается всех ссылок на DABench-numbers в E1/E2/E3.
3. **Глава про GSM8K**: добавление «cross-family caveat — saturation на gpt-oss с reasoning_effort, crash на qwen-preview-без-reasoning_effort. Saturation finding inherits from family+reasoning-flag config.»
4. **Новая sub-глава «Generalizability across worker families»** — DABench-reversal как driver insight, role-distribution-shift как methodological flag, оба confirmation-эксперимента (E3 + E4) как evidence base.
5. **Limitations §** должен явно перечислить: «policy-overfit risk», «two-family validation недостаточна для general claim», «judge constancy = sole isolation point — confirmation isolates worker effect, не топологии».

### 10.4 Что **не** меняется

- E1/E2/E3 best-static-topology claims на gpt-oss остаются валидными как family-specific findings.
- E3 cost-finding (LLM-роутер 3× дешевле rule-роутера) — это структурное свойство (call count), оно cross-family-invariant.
- Cost-neutrality role-роутера cross-family invariant.
- LLM-role-роутер «hurts vs fixed» — direction подтверждён, magnitude усилен. Это устойчивый negative finding.

### 10.5 E5 implications

Champion-конфигурация для E5 (`role_router=rule` + `topology_router=llm` + adaptive base) выбиралась по gpt-oss E4. Перед запуском E5 на реальных людях стоит:

- Либо **зафиксировать gpt-oss-120b как production worker** (где rule-policy валидна), и тогда E5 продолжает прежнюю траекторию.
- Либо **признать family-dependence** и оставить fixed-reviewer как default для cross-family deployments — тогда E5 запускается с fixed-role и измеряет real-human effect отдельно от adaptive-role effect-а.
- Либо **переопределить policy** (rebuild rule-table on qwen-data, или использовать model-aware role-router) — но это новая research direction, выходящая за scope diploma.

Рекомендация: для текста diploma фиксировать gpt-oss-120b как primary worker (исходное design intent), а cross-family confirmation использовать как boundary-condition statement, не как replacement.

---

## Приложение A: Cross-check parquet vs JSON-артефакт

Все ключевые поля совпали bit-к-bit (±1e-4 округление):

| Метрика | parquet (own pandas) | `analysis/confirmation_e4_results.json` | Match |
|---|---|---|---|
| `mode_totals.fixed.mean_q` | 0.5820 | 0.582 | ✓ |
| `mode_totals.rule.mean_q` | 0.5013 | 0.5013 | ✓ |
| `mode_totals.llm.mean_q` | 0.5277 | 0.5277 | ✓ |
| `mode_task_matrix.rule/humaneval.q` | 0.3556 | 0.3556 | ✓ |
| `mode_task_matrix.fixed/dabench.q` | 0.9333 | 0.9333 | ✓ |
| `mode_task_matrix.fixed/gsm8k.q` | 0.2667 | 0.2667 | ✓ |
| `role_distribution.rule.peer` | 169 | 169 | ✓ |
| `role_distribution.llm.judge` | 32 | 32 | ✓ |
| `total_cost_usd` | **9.3988** | **0** (script bug) | ✗ — флагнуто в §9.3 |
| `wall_time_h` | 1.345 | 1.3454 | ✓ |
| `rq4.rule_delta` | −0.0807 | −0.0807 | ✓ |
| `rq4.llm_delta` | −0.0543 | −0.0543 | ✓ |

Все count-инварианты выдержаны: `Σ(n) = 540`, `per-(mode,task) = 45`, `per-mode = 180`. Дедуп-discrepancies = 0.

## Приложение B: Сводный list файлов

- **Сырьё:** `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/d7f7d1eb-32c0-4315-a8dc-49b65a4d6894/_runs.parquet`
- **Метаданные:** `/home/cactustim/agents/adaptive-topologies-mas/data/experiments/experiments/d7f7d1eb-32c0-4315-a8dc-49b65a4d6894/experiment.json`
- **Артефакт результатов:** `/home/cactustim/agents/adaptive-topologies-mas/analysis/confirmation_e4_results.json` (с baked-in `total_cost_usd: 0` bug-ом)
- **Анализ-скрипт:** `/home/cactustim/agents/adaptive-topologies-mas/analysis/e4_report.py` (тот же скрипт, что для оригинального E4; bug на line 170 не исправлен)
- **Конфиг:** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/confirmation_e4.yaml`
- **Оригинальный E4 для cross-reference:** `/home/cactustim/agents/adaptive-topologies-mas/arch/diploma/results/e4_analysis.md`
- **Базовый конфиг (наследуется через include):** `/home/cactustim/agents/adaptive-topologies-mas/conf/experiments/e4_full.yaml`
- **План:** `/home/cactustim/agents/adaptive-topologies-mas/arch/experiment_plan.md` §5 (E4) и §6 (confirmation rationale)
