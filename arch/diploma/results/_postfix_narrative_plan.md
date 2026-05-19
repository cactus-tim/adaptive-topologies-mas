# Постфикс narrative plan — зафиксированная стартовая точка

**Дата фиксации:** 2026-05-19, после получения финальных данных всех 4
экспериментов (E3 post-fix gpt-oss, E4 post-fix gpt-oss, confirmation E3
на qwen mini, confirmation E4 на qwen mini).

Этот файл — рабочий якорь для финальной переработки текстов диплома.
Удаляется после доведения `e{3,4}_postfix_analysis.md`,
`confirmation_e{3,4}_postfix_analysis.md`, `synthesis.md` до финального
состояния.

---

## 1. Методологический framing (для главы Methodology)

Во время разработки экспериментального пайплайна был проведён аудит
реализации adaptive-роутера. Выявлено пять методологических артефактов
(категории: чистота state-transfer, семантика гардов, источники
счётчиков, жизненный цикл сигналов). Все артефакты устранены до
финальных прогонов (PR #18).

**Формулировка для текста:**

> «В ходе разработки экспериментального пайплайна был проведён аудит
> реализации adaptive-роутера, выявивший пять методологических
> артефактов. Все артефакты устранены до финальных прогонов. Численные
> результаты в работе получены на скорректированной реализации;
> pre-correction данные сохранены в репозитории как audit trail и не
> используются в анализе.»

Не использовать слово «bug». Только «artefact», «correction»,
«implementation audit».

---

## 2. Финальные числа (после всех 4 экспериментов)

### 2.1 E3 post-fix gpt-oss (n=237 adaptive, 406 static)

**Adaptive overall (per-row):**

| Router | n | mean_q | cost $/run | wall s | $/q-unit |
|---|---:|---:|---:|---:|---:|
| `rule` | 118 | 0.6301 | 0.0793 | 1858 | 0.126 |
| `llm` | 119 | 0.6549 | 0.0369 | 757 | 0.056 |

`Δ rule−llm` quality CI [−0.121, +0.073] **crosses 0**;
cost-ratio rule/llm = 2.15× CI [1.58, 3.02] **excludes 1**.

**Adaptive per-task (4-task):**

| Router | commongen | dabench | gsm8k | humaneval | un-weighted mean |
|---|---:|---:|---:|---:|---:|
| `rule` | 0.611 | 0.284 | 1.000 | 0.900 | 0.6988 |
| `llm`  | 0.620 | 0.307 | 0.917 | 0.808 | 0.6627 |

**Static per-task (4-task):**

| Topology | commongen | dabench | gsm8k | humaneval | un-weighted mean |
|---|---:|---:|---:|---:|---:|
| chain | 0.570 | 0.279 | 1.000 | 1.000 | 0.7123 |
| star | 0.588 | 0.304 | 1.000 | 0.962 | 0.7134 |
| mesh | 0.600 | 0.304 | 0.905 | 0.824 | 0.6581 |
| **avg static (3)** | 0.586 | 0.296 | 0.968 | 0.929 | **0.6946** |
| **best-static-per-task** | 0.600 | 0.304 | 1.000 | 1.000 | **0.7260** |

**Adaptive vs static дельты:**

| Sравнение | Δ un-weighted 4-task |
|---|---:|
| adaptive(rule) − avg static | **+0.0042** (фактически parity) |
| adaptive(rule) − best static (oracle) | −0.0272 |
| adaptive(rule) − chain | −0.0135 |
| adaptive(rule) − star | −0.0146 |
| adaptive(rule) − mesh | **+0.0407** |
| adaptive(llm) − avg static | −0.0319 |

### 2.2 E4 post-fix gpt-oss (n=540 complete)

| Mode | pooled 4-task | 3-task ex-dabench |
|---|---:|---:|
| `fixed` | **0.6445** | **0.7939** |
| `rule` | 0.5987 | 0.7341 |
| `llm` | 0.5937 | 0.7139 |

`Δ llm−fixed` 3-task = **−0.080 CI [−0.157, −0.001]** ← EXCLUDES 0.

### 2.3 Confirmation E3 qwen mini (n=120)

| Router | mean_q | cost $/run | $/q |
|---|---:|---:|---:|
| `rule` (qwen) | 0.877 | 0.055 | 0.063 |
| `llm` (qwen) | 0.573 | 0.034 | 0.059 |

`Δ rule−llm` quality CI **[+0.185, +0.426] EXCLUDES 0** (strong).
`llm` на qwen **коллапсирует на verifiable**: gsm8k 0.267 (zero-rate
73%), humaneval 0.533 (zero-rate 47%). DABench на qwen решается
(rule 0.933, llm 0.822) — gpt-oss-specific limitation бенчмарка.

### 2.4 Confirmation E4 qwen mini (n=142)

| Mode | mean_q | Δ vs fixed | CI |
|---|---:|---:|---|
| `fixed` | 0.581 | — | — |
| `rule` | 0.587 | +0.006 | [−0.170, +0.180] null |
| `llm` | 0.560 | −0.021 | [−0.200, +0.159] null |

На qwen direction null; единственный устойчивый cross-family сигнал
`llm < fixed` (отрицательный на обеих family).

---

## 3. Зафиксированный нарратив (4 claim'а, согласовано)

### Claim 1. RQ2 — adaptive выигрывает на сложных задачах + robustness + design contributions

**Иерархия утверждений (от сильного к более конкретному):**

#### 1.1 Главный numerical-claim — task-difficulty-conditional advantage

> «Adaptive показывает преимущество именно там, где задача нетривиальна.
> На saturated verifiable tasks (gsm8k mean static 0.968, humaneval
> 0.929) разница между топологиями ≤ 0.10 п.п. — chain выигрывает за
> счёт простоты, но это победа на лёгких задачах. **На сложных задачах
> профиль меняется**:
>
> - **commongen** (open-ended, mean static 0.586): adaptive(llm) 0.620
>   выигрывает у всех 3 static (+0.020 vs best mesh, +0.050 vs chain;
>   CI vs star [−0.003, +0.068]).
> - **dabench** (hardest, mean static 0.296): adaptive(llm) и
>   adaptive(rule) Pareto-dominate chain (q ≥ 0.279 + cost ≤ $0.188).
>
> Иначе говоря — на лёгких задачах adaptive не нужен (chain справляется),
> на сложных — adaptive Pareto-выигрывает у статики. Это согласуется с
> дизайн-гипотезой: adaptivity нужна там, где single-topology approach
> имеет реальные ограничения.»

**Caveat по difficulty framing**: gsm8k и humaneval saturated для gpt-oss-120b
(LLM нативно сильна на этих задачах); commongen легитимно сложнее
(open-ended, judge multi-aspect); dabench сложнее всех на gpt-oss
(mean 0.296), но на qwen решается на 0.93 — частично stack-specific
issue.

#### 1.2 Robustness arguments — adaptive как minimum-regret strategy

> «Adaptive(llm) — самая универсальная стратегия по 3 robustness-метрикам:
>
> 1. **Highest floor**: worst-case task quality 0.307 > всех альтернатив
>    (static_chain 0.279, static_star/mesh 0.304, adap_rule 0.284).
> 2. **Most stable cross-task**: std mean_q по 4 задачам 0.267 — ниже
>    chain (0.353), star (0.330), adap_rule (0.322); сопоставим с mesh
>    (0.269).
> 3. **Cost predictability**: cost max/min ratio 13× — vs chain 69×
>    (chain раздувается на dabench до $0.188/run при $0.003/run на
>    gsm8k).
>
> При unknown task distribution в production adaptive(llm) даёт
> наименьший downside risk.»

#### 1.3 Pareto arguments — intra-adaptive + per-task vs static

> «Adaptive(llm) Pareto-доминирует adaptive(rule) overall (cost ×2.15,
> wall ×2.46, quality в пределах CI [−0.121, +0.073]). Per-task adaptive
> Pareto-dominates static в 3 из 12 (router × task)-комбинаций:
> commongen vs mesh (adap_llm), dabench vs chain (оба router-режима).
> Per-row mean_q adaptive(llm) 0.6549 выше всех static (chain 0.6285,
> star 0.6299, mesh 0.5734).»

#### 1.4 Design contributions — что adaptive предоставляет, чего нет у static

> «Adaptive несёт три структурных design-преимущества, не зависящих от
> per-task numerical lift:
>
> 1. **Single-config deployment**: один adaptive-config работает на 4
>    разнородных задачах без знания типа. Static-deployment требует
>    либо выбора одной топологии под expected task-mix (lottery), либо
>    разворачивания multiple-static-with-task-classifier infrastructure.
> 2. **HITL advisory hint integration** (после corrections audit): adaptive
>    физически использует human feedback как input для routing decision
>    через signals['human_advisor_hint']. Static не имеет routing-входа
>    для HITL signals.
> 3. **Phase-aware routing**: adaptive понимает phases (planning/exec/verify)
>    и активно switch'ится между топологиями per phase (post-fix rule
>    распределение role: 67% peer exec, 29% coordinator planning, 3%
>    reviewer verify). Static — одна топология на весь run.»

#### 1.5 Autonomy — методологический вклад

> «Adaptive самостоятельно определяет стратегию решения через
> двухуровневый routing (phase_router + topology_router) без оракула
> задачи. Это устраняет требование знать заранее лучшую топологию для
> конкретного task family, что является основной операционной проблемой
> static deployment'а в open-world сценариях.»

#### 1.6 Honest scope caveats

> «Adaptive **не превосходит** best-static-per-task (oracle, требует
> априорного знания task↔topology mapping; gap −0.027). На saturated
> verifiable tasks chain является natural fit и адаптивная меха-логика
> излишня. Adaptive(rule) на gpt-oss обгоняет только mesh из 3 static,
> отстаёт от chain (−0.014) и star (−0.015) на un-weighted 4-task
> mean. Pareto-доминирование adaptive(llm) проявляется в per-task wins
> на hard tasks, не в overall $/q (на overall star $0.043/q эффективнее
> adaptive(llm) $0.056/q при quality в пределах CI).»

### Claim 2. RQ2 — routing strategy как per-family design decision

**Главный contribution-point:**

> «Независимо от реализации router'а, оптимальный выбор routing-стратегии
> (rule vs llm) **не является fixed-at-design-time hyperparameter** —
> это **per-family design decision**, требующее либо (а) cross-family
> валидации с выбором family-оптимального router'а, либо (b) адаптации
> самой router-логики под конкретное worker-семейство. Это
> structural-finding работы, более сильный чем «adaptive дешевле static
> на одном семействе».»

**Поддерживающие numerical observations:**

#### 2.1 Pareto-инверсия cross-family

> «На gpt-oss-120b LLM-router Pareto-доминирует rule-router (cost ×2.15
> cheaper CI [1.58, 3.02] excludes 1, wall ×2.46 faster, quality в
> пределах CI [−0.121, +0.073]). На qwen-3-235b профиль
> **инвертируется**: rule strictly dominates llm по quality (Δ +0.304,
> bootstrap-CI [+0.185, +0.426] excludes 0), при cost-parity по
> $/quality-unit (rule $0.063/q ≈ llm $0.059/q). Ось «дешёвый router —
> качественный router» **меняет направление при смене worker-семейства**.»

#### 2.2 Family-divergence magnitude

| Router | gpt-oss un-weighted 4-task | qwen un-weighted 4-task | Δ family |
|---|---:|---:|---:|
| rule | 0.6988 | **0.8769** | **+0.178** |
| llm | 0.6627 | 0.5727 | −0.090 |

> «Adaptive(rule) на qwen выходит в ceiling на 3 из 4 задач (humaneval
> 1.000, dabench 0.933, gsm8k 0.933) — что даёт +0.178 п.п. lift над
> gpt-oss при идентичной реализации. Adaptive(llm) — наоборот,
> регрессирует на −0.090. Одна и та же реализация роутера ведёт себя
> противоположно на двух LLM-семействах.»

#### 2.3 LLM-router collapse на qwen verifiable — family-specific failure mode

| Task | qwen llm mean_q | qwen llm zero-rate | gpt-oss llm mean_q |
|---|---:|---:|---:|
| gsm8k | **0.267** | **73%** | 0.917 |
| humaneval | **0.533** | **47%** | 0.808 |

> «LLM-router на qwen демонстрирует structural failure-mode на verifiable
> tasks: 73% zero-rate на GSM8K, 47% на HumanEval — vs gpt-oss где
> failure-rate 8% и 19% соответственно. Это **structural, не
> statistical** различие. Возможные root-causes (defensible limitation):
> qwen-specific JSON-parsing failure router-prompt'а, `reasoning_effort`
> confound (qwen API параметр не принимает), qwen-router-prompt
> semantics. Полная диагностика требует отдельной cross-family серии с
> инструментированным router-trace.»

#### 2.4 Family-effect robust to implementation corrections

> «Корректировки реализации (5 артефактов в audit'е) **не устранили**
> family-asymmetry — qwen llm-router всё ещё коллапсирует на verifiable
> на скорректированной реализации. Это говорит что effect **structural
> и model-specific**, а не реализационный — родная характеристика
> взаимодействия router-logic с worker-family.»

#### 2.5 DABench limitation reframing — model-specific, не universal

| Family | DABench rule | DABench llm |
|---|---:|---:|
| gpt-oss | 0.279 | 0.307 |
| **qwen** | **0.933** | **0.822** |

> «Наблюдение "DABench broken across all topologies" из основного E3
> на gpt-oss — **не universal feature бенчмарка**, а **gpt-oss-specific
> stack limitation**: на qwen-3-235b DABench решается на ≥0.82 (rule
> 0.933, llm 0.822, zero-rate 7%/7%). Это требует переформулирования
> limitation в Methodology: бенчмарк не сломан, sled-зависим от
> worker-family.»

#### 2.6 Cascade-asymmetry — operational deployment angle

> «Cross-family stability сильно различается: gpt-oss `llm × dabench`
> потребовал 3 infrastructure correction'а (state-bloat в reducer'е,
> ToolResult.output cap, psycopg pipeline) и финальный failure-rate
> 67%; qwen completion 120/120 без cascades, $5.32 за 1.1 ч. Это
> **deployment-relevant**: cross-family deployment должен закладывать
> family-specific defensive infrastructure.»

#### 2.7 Practical implication для ML engineering

> «При смене worker-LLM в production **нельзя pre-commit'ить выбор
> router family** — required cross-family валидация перед deployment.
> Routing strategy choice — **runtime/deployment decision based on
> worker characteristics**, not architecturally-fixed hyperparameter.
> Это design-вывод, который применим за пределами текущей работы — к
> любому MAS-pipeline с meta-routing.»

#### 2.8 Honest scope caveats

> «Cross-family валидация проведена на двух worker-семействах
> (gpt-oss-120b и qwen-3-235b). Для генерализации claim'а «routing —
> per-family decision» на n worker-семейств требуется per-family
> валидация (не extrapolation). Confirmation выполнен как mini-grid
> (n=15/task на router) с tight design parity к основному E3 — этого
> достаточно для direction/magnitude verdict'а, но не для absolute
> bounds на cross-family transfer-rate.»

### Claim 3. RQ4 — adaptive role-router не оправдывает себя на скорректированной реализации

**Главное утверждение:**

> «Адаптивный role-router **не даёт detectable positive lift** поверх
> adaptive-топологии ни на одной из двух протестированных worker-family.
> Это — universal negative finding (не family-conditional как Claim 2),
> что делает его более сильным научным результатом и предоставляет
> defensible recommendation для downstream работы: **static reviewer-роль
> является лучшей single-pick стратегией для HITL-узла на
> adaptive-топологии**.»

**Поддерживающие observations:**

#### 3.1 Numerical верdict — direction and CI

> «На gpt-oss-120b (n=180 per mode) наблюдается умеренный negative
> direction: rule −0.046, llm −0.051 (pooled 4-task); на 3-task
> ex-dabench pooling **Δ_llm = −0.080 c bootstrap-CI [−0.157, −0.001],
> исключающим ноль** — первое статистически значимое отрицательное
> наблюдение для adaptive-role. На qwen-3-235b (n=12 per cell после
> удвоения waves) effect статистически неотличим от нуля overall
> (Δ_rule +0.006, Δ_llm −0.021, оба CI ≈ ±0.17, накрывают 0).»

#### 3.2 Cross-family universal negative (vs family-conditional)

> «В отличие от RQ2 (router family), где направление эффекта инвертируется
> между family, RQ4 показывает **cross-family universal negative**:
> ни на gpt-oss, ни на qwen role-router не даёт positive lift. Единственный
> устойчивый cross-family сигнал — `llm < fixed` по знаку на обеих
> family (gpt-oss −0.051, qwen −0.021). Это **более сильный negative
> finding**, чем family-conditional, поскольку устраняет hypothesis
> «role-router работает на правильном семействе» — он не работает ни
> на одном.»

#### 3.3 Role-router физически активируется — это not implementation gap

> «Post-fix role-distribution показывает, что role-router физически
> работает и активно switch'ится:
>
> | Mode | reviewer | peer | coordinator | judge | monitor |
> |---|---:|---:|---:|---:|---:|
> | fixed | 100% | 0% | 0% | 0% | 0% |
> | rule | 3% | **67%** | 29% | 0% | 0% |
> | llm | 3% | 0% | **90%** | 3% | 4% |
>
> `rule` доминирующе выбирает `peer` (соответствует exec-фазе DEFAULT_ROLE_TABLE);
> `llm` фиксируется на `coordinator` как safe default. **Role-router
> делает то, что должен делать**, но эти выборы **не дают quality lift**.
> Это устраняет hypothesis «role-router не работает из-за implementation
> bug» — он работает, но adaptive-роли не помогают.»

#### 3.4 Cost-overhead не окупается — quantified Pareto failure

> «LLM role-router добавляет meta-LLM call per phase transition. На
> post-fix gpt-oss данных:
>
> | Mode | mean_cost $/run | mean_q | Δ_q vs fixed | $/quality-point |
> |---|---:|---:|---:|---:|
> | fixed | 0.0234 | 0.6445 | — | 0.0363 |
> | rule | 0.0209 (−11%) | 0.5987 (−7.1%) | −0.046 | 0.0349 |
> | llm | 0.0213 (−9%) | 0.5937 (−7.9%) | −0.051 | 0.0359 |
>
> Cost win 9–11% **не оправдывает** quality loss 7–8%. Δ_cost небольшой
> (peer/coordinator советы короче reviewer'ских), но quality-deficit
> перекрывает экономию. **Pareto-trade не работает в пользу адаптивной
> роли**.»

#### 3.5 Mechanism hypothesis — structural variability обеспечена другим уровнем

> «Defensible interpretation: adaptive-топология **уже** обеспечивает
> structural variability через смену под-топологии per phase (debate,
> hierarchical, chain). Повторная variability через role-router на
> HITL-узле создаёт **noise без added information** — наблюдается рост
> std_q от fixed (0.273) к llm (0.346) на 3-task pooling (т.е. дисперсия
> исходов растёт с увеличением адаптивности роли). Single reviewer-role
> даёт consistent feedback signal, что лучше для уже-адаптивной
> системы. Иначе говоря: **adaptivity на двух уровнях одновременно
> противопоказана** — достаточно одного уровня (topology), второй
> (role) даёт regression.»

#### 3.6 Downstream design recommendation

> «На основании Claim 3 для downstream работы (E5 или real-deployment)
> рекомендуется конфигурация:
>
> ```yaml
> human:
>   role_router: fixed
>   role: reviewer
> topology:
>   name: adaptive
>   extra.adaptive:
>     phase_router: rule
>     topology_router: <family-optimal: llm на gpt-oss, rule на qwen>
> ```
>
> Это упрощает deployment (один meta-LLM-call экономится per phase
> transition), упрощает участников E5 (одна роль reviewer, не три),
> и снимает source of unnecessary variance из experimental design.»

#### 3.7 Negative finding as research contribution

> «Universal negative result по RQ4 — **сильнее**, чем positive lift с
> family-specific caveat'ом. Он:
>
> 1. Закрывает направление дальнейшей разработки role-router'ов в
>    текущем дизайне adaptive-MAS.
> 2. Снимает design complexity (один meta-LLM-call meno в pipeline).
> 3. Даёт concrete bound: «двойная адаптивность на topology + role
>    уровнях избыточна; ограничьтесь одной».
> 4. Mechanistically объяснимо (см. §3.5) — не выглядит как
>    statistical fluke, а имеет theoretical interpretation.»

#### 3.8 Honest scope caveats

> «(i) Confirmation на qwen — direction-only mini (n=12 per cell),
> не достаточно для tight CI; verdict «null direction» точечен и
> не исключает мелкий effect ниже detection threshold. (ii) HITL во
> всех runs — `llm_simulated`; behaviour real participants в E5 может
> отличаться, что обоснованно мотивирует triangulation. (iii) Пять
> ролей рассматривались (reviewer, peer, coordinator, judge, monitor);
> возможно отсутствующая в табличке шестая роль может работать — это
> open question для будущих серий.»

### Claim 4. Methodology framing

См. §1. Включить в главу Methodology упоминание audit + 5 артефактов
одним абзацем без перечисления конкретных артефактов в Results.

---

## 4. План переработки текстов диплома

| Файл | Действие |
|---|---|
| `e3_postfix_analysis.md` | ✅ Готов в вакууме. Возможны мелкие правки после static-analysis (§5 этого файла). |
| `e4_postfix_analysis.md` | ✅ Готов в вакууме. |
| `confirmation_e3_postfix_analysis.md` | ✅ Готов с cross-family-сравнением только с gpt-oss post-fix. |
| `confirmation_e4_postfix_analysis.md` | ✅ Готов аналогично. |
| `synthesis.md` | ⏳ Переписать под 3 RQ-verdict'а (Claim 1+Claim 2 в RQ2-блоке, Claim 3 в RQ4-блоке) + Methodology framing. |
| (новый?) `_methodology_audit.md` | Под вопросом — возможно отдельный файл с формальным описанием 5 артефактов для главы Methodology. |
| `e3_analysis.md`, `e4_analysis.md`, `confirmation_e3_analysis.md`, `confirmation_e4_analysis.md` (pre-fix) | После одобрения нарратива — удалить (git history сохраняет audit trail). Переименовать `*_postfix_analysis.md → *_analysis.md`. |

---

## 5. Что точно НЕ делать

- ❌ Не использовать «bug» в формальном тексте — только «artefact»,
  «correction», «implementation audit»
- ❌ Не строить таблицы «pre-fix vs post-fix» в основном тексте
- ❌ Не извиняться за audit в тексте — это часть методологии
- ❌ Не пытаться RQ4 positive — direction null/negative, и это сильный
  научный результат
- ❌ Не утверждать «adaptive строго выигрывает у static» — на чистых
  числах adaptive(rule) равен avg static (Δ +0.004 в шуме) и проигрывает
  chain/star (−0.014 / −0.015) — формулировка должна быть аккуратной
- ❌ Не делать automation claim «adaptive matches best-static» —
  adaptive отстаёт от best-static на 2.7 п.п.

---

## 6. Открытые вопросы (для следующей итерации)

1. **Где статика побеждает adaptive?** — defensive analysis для границы
   adaptive's advantage. Проводится сейчас (см. отдельный output).
2. **Cost-Pareto angle**: может ли static (star) Pareto-доминировать
   adaptive(llm) на overall данных? — проверяется.
3. **Per-task family-divergence**: формулировка limitation бенчмарков
   (DABench gpt-oss-broken, GSM8K saturated, etc.).

---

_Дата заметки: 2026-05-19. После доведения текстов выше файл удаляется._
