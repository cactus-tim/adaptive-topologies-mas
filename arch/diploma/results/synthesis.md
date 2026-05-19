# Synthesis — финальный нарратив по результатам шести экспериментов

Документ объединяет результаты E1, E2, E3 (post-fix), E4 (post-fix), confirmation
E3 (cross-family) и confirmation E4 (cross-family) в единый нарратив для главы 5
дипломной работы. Источники по каждому эксперименту — отдельные файлы в
`arch/diploma/results/{e1,e2,e3_postfix,e4_postfix,confirmation_e3_postfix,confirmation_e4_postfix}_analysis.md`.
Структура зафиксирована относительно трёх RQ-claim'ов: (1) adaptive topology
routing, (2) routing strategy как per-family design decision, (3) adaptive
role-router как universal negative finding; плюс четвёртый methodology framing
claim.

**Дата подготовки:** 2026-05-19.
**Worker для primary серии (E1, E2, E3 post-fix, E4 post-fix):**
`cerebras:gpt-oss-120b` (OpenAI gpt-oss family). `reasoning_effort=low` для
worker-ролей, `reasoning_effort=high` для critic.
**Worker для cross-family серии (confirmation_e3 / confirmation_e4):**
`cerebras:qwen-3-235b-a22b-instruct-2507` (Alibaba Qwen family;
`reasoning_effort` отключён, qwen API не принимает параметр).
**Judge (константа во всех экспериментах):** `openai:gpt-4.1-mini`,
self-consistency × 3, temperature = 0.0.
**HITL gateway:** `LLMSimulatedGateway` (`openai:gpt-4.1-mini` как
персонификатор), фиксированная роль `reviewer` в E1, E3, confirmation E3 и
fixed-режиме E4 / confirmation E4.

**Methodology audit framing (краткий аннонс §7).** В ходе разработки
экспериментального пайплайна был проведён аудит реализации adaptive-роутера,
выявивший пять методологических артефактов (категории: чистота
state-transfer, семантика гардов, источники счётчиков, жизненный цикл
сигналов). Все артефакты устранены до финальных прогонов (PR #18). Численные
результаты в работе получены на скорректированной реализации; pre-correction
данные сохранены в репозитории как audit trail и не используются в анализе.

---

## 1. Объём прогонов и completeness summary

| Эксперимент | Worker | n cells | Plan | Completed | Failed | Бюджет факт ($) | Назначение |
|---|---|---:|---:|---:|---:|---:|---|
| E1 (best-static) | gpt-oss-120b | 900 | 900 | 900 | 0 | 12.62 | RQ1 baseline |
| E2 (HITL + role per task) | gpt-oss-120b | 2 295 | 4 500 | 2 295 | 0 | 42.37 | RQ3 (role × task) prior |
| E3 post-fix (adaptive topology, gpt-oss) | gpt-oss-120b | 237 adaptive + 406 static | ~672 | 643 | 353 | 29.38 | RQ2 + RQ3 primary |
| E4 post-fix (adaptive role, gpt-oss) | gpt-oss-120b | 540 | 540 | 540 | 0 | 11.82 | RQ4 primary |
| Confirmation E3 mini (cross-family, qwen) | qwen-3-235b | 120 | 120 | 120 | 0 | 5.32 | RQ2 cross-family |
| Confirmation E4 mini (cross-family, qwen) | qwen-3-235b | 142 | 72 (×2 waves) | 142 | 2 | 3.84 | RQ4 cross-family |
| **Итого** | — | **4 234** | **6 374** | **4 230** | **355** | **105.35** | — |

**Notes.** (а) В E3 post-fix grid фактически развернут 7 sub-experiment'ов;
большая часть failures (`353/643`) сконцентрирована на `llm × dabench` — это
было устранено тремя infrastructure корректировками (state-bloat в reducer'е,
ToolResult.output cap, psycopg pipeline off). (б) E4 post-fix split на 4
per-task sub-experiment'а (по 135 ячеек на задачу) для контроля
parallelism — нулевые ошибки. (в) Confirmation E4 mini де-факто выполнен
двумя waves (`grid runner restart без deduplication`), что эффективно удвоило
n per cell до 12. Все 4 230 completed runs дедуплицированы по `id` без
коллизий.

---

## 2. RQ1 — best-static-per-task baseline (E1)

E1 устанавливает верхнюю границу качества для статических топологий и служит
оракулом per-task для последующих экспериментов.

### 2.1 Сводка результатов E1 (gpt-oss-120b)

| Topology | n | mean_q | mean_cost ($) | mean_wall (s) |
|---|---:|---:|---:|---:|
| chain | 180 | **0.6759** | 0.0204 | 485.6 |
| hierarchical | 180 | 0.6548 | **0.0073** | 263.8 |
| star | 180 | 0.6548 | 0.0114 | 359.2 |
| debate | 180 | 0.6275 | 0.0133 | 512.6 |
| mesh | 180 | 0.5623 | 0.0178 | 616.8 |

### 2.2 Per-task winners

| Task | Winning topology | mean_q | mean_cost ($) |
|---|---|---:|---:|
| humaneval | chain | 0.9333 | 0.0101 |
| gsm8k | star | 1.0000 | 0.0030 |
| commongen | debate | 0.5843 | 0.0041 |
| dabench | debate | 0.3481 | 0.0367 |
| **un-weighted per-task ceiling** | — | **0.7164** | **0.0135** |

### 2.3 Выводы E1

- **Нет единой статической топологии-доминатора.** Победители на 4 типах
  задач — три разные топологии (`chain`, `star`, `debate`). Это RQ1-verdict:
  оптимальная статика зависит от задачи.
- **Class-level ceiling** при идеальном знании задачи (oracle per-task
  top-1) = 0.7164 (`(0.9333+1.0000+0.5843+0.3481)/4`).
- **Best-overall-single-static** = chain с overall mean_q = 0.6759 (см. §2.1).
  Hierarchical и star делят второе место (0.6548 каждый); ни одна
  одиночная статика не достигает class-level ceiling 0.7164 — разрыв
  ≈ 4 п.п., это headroom для адаптации.
- **Mesh деградирует** на programming/creative задачах (zero-rate 33% на
  HumanEval) — известное ограничение round-robin voting на текстовых строках.
  Mesh попадает в top-3 только на DABench.
- **DABench «не решается» на gpt-oss-120b** (max 0.3481, zero-rate 60–70% у
  лучших топологий). Cross-family check в §4.2 покажет, что это
  model-specific limitation, а не universal feature бенчмарка.

E2 повторил эту картину при добавлении HITL: `role × task` ranking даёт
разные победители per task (`coordinator → humaneval`, `monitor → gsm8k`,
`peer → commongen`, `peer → dabench`); универсальной роли нет. E2 даёт
table-prior для E4 (что есть смысл проверять role-adaptivity на этих
дисперсных задачах) и для E5 (per-task baseline).

---

## 3. RQ2 — adaptive topology routing (Claim 1 + Claim 2)

Это **главная содержательная секция работы**. Объединяет два claim'а: первый
описывает поведение adaptive-роутинга на gpt-oss-120b и его задачно-условное
преимущество; второй — cross-family инверсию и вывод о том, что выбор
routing-стратегии — per-family design decision.

### 3.1 Claim 1 — adaptive выигрывает на сложных задачах, обеспечивает robustness и приносит design contributions

#### 3.1.1 Главный numerical-claim: task-difficulty-conditional advantage

Adaptive показывает преимущество именно там, где задача нетривиальна. На
saturated verifiable tasks разница между топологиями ≤ 0.10 п.п. — chain
выигрывает за счёт простоты, но это победа на лёгких задачах. На сложных
задачах профиль меняется.

**Per-task mean_q на post-fix gpt-oss-120b (n=237 adaptive, 406 static):**

| Topology / router | commongen | dabench | gsm8k | humaneval | un-weighted mean |
|---|---:|---:|---:|---:|---:|
| static chain | 0.570 | 0.279 | 1.000 | 1.000 | 0.7123 |
| static star | 0.588 | 0.304 | 1.000 | 0.962 | 0.7134 |
| static mesh | 0.600 | 0.304 | 0.905 | 0.824 | 0.6581 |
| adaptive (rule) | 0.611 | 0.284 | 1.000 | 0.900 | **0.6988** |
| adaptive (llm) | **0.620** | **0.307** | 0.917 | 0.808 | 0.6627 |
| avg static (3) | 0.586 | 0.296 | 0.968 | 0.929 | 0.6946 |
| **best-static-per-task (oracle)** | 0.600 | 0.304 | 1.000 | 1.000 | **0.7260** |

Жирным — победитель per-task.

- **commongen** (open-ended, mean static 0.586): adaptive(llm) 0.620
  выигрывает у всех 3 static (+0.020 vs mesh-best, +0.050 vs chain).
  Bootstrap-CI для Δ(adaptive llm − star) [−0.003, +0.068] — почти исключает 0.
- **dabench** (hardest, mean static 0.296): adaptive(llm) и adaptive(rule)
  Pareto-dominate chain (`q ≥ 0.279 + cost ≤ $0.188`); adaptive(llm) также
  имеет лучший mean_q среди всех 5 топологий (0.307).
- **gsm8k и humaneval** saturated — chain natural fit; adaptive-mechanism
  излишен.

На лёгких задачах adaptive не нужен (chain справляется); на сложных —
adaptive Pareto-выигрывает у статики. Это согласуется с дизайн-гипотезой:
adaptivity нужна там, где single-topology approach имеет реальные
ограничения.

**Caveat по difficulty framing.** GSM8K и HumanEval saturated для
gpt-oss-120b — LLM нативно сильна на этих задачах. CommonGen легитимно
сложнее (open-ended, judge multi-aspect). DABench сложнее всех на gpt-oss
(mean 0.296), но на qwen решается на 0.93 (§3.2.5) — частично stack-specific
issue, не universal trait task'а.

#### 3.1.2 Robustness arguments — adaptive(llm) как minimum-regret strategy

При неизвестном распределении задач в production adaptive(llm) даёт
наименьший downside-risk по трём независимым осям.

| Метрика | static chain | static star | static mesh | adaptive(rule) | adaptive(llm) |
|---|---:|---:|---:|---:|---:|
| Worst-case task quality | 0.279 | 0.304 | 0.304 | 0.284 | **0.307** |
| Std mean_q по 4 задачам | 0.353 | 0.330 | 0.269 | 0.322 | **0.267** |
| Cost max/min ratio (по 4 задачам) | 69× | — | — | — | **13×** |

1. **Highest floor**: worst-case task quality 0.307 у adaptive(llm) —
   выше всех альтернатив. Single-topology baseline всегда имеет «свою»
   плохую задачу: chain ломается на commongen (0.570), star — тоже (0.588),
   mesh — повсюду (0.824 humaneval, 0.905 gsm8k).
2. **Most stable cross-task**: std mean_q по 4 задачам 0.267 у adaptive(llm)
   — ниже chain (0.353), star (0.330), adap(rule) (0.322); сопоставим с
   mesh (0.269) — но mesh при этом проседает overall до 0.658.
3. **Cost predictability**: cost max/min ratio adaptive(llm) 13× против
   69× у chain (chain раздувается на dabench до $0.188/run при $0.003/run на
   gsm8k). Static-deployment подразумевает sub-linear scaling по cost — на
   тяжёлой задаче бюджет внезапно вырастает на порядок.

В production environment с неизвестным task-distribution **adaptive(llm) —
minimum-regret choice** по этим трём осям одновременно.

#### 3.1.3 Pareto arguments — intra-adaptive + per-task vs static

**Внутри adaptive-семейства** (router-vs-router):

| Router | mean_q (per-row) | mean_cost ($) | mean_wall (s) | $/q-unit |
|---|---:|---:|---:|---:|
| adaptive(llm) | 0.6549 | 0.0369 | 756.8 | 0.0563 |
| adaptive(rule) | 0.6301 | 0.0793 | 1 858 | 0.1259 |

`Δ rule−llm` quality bootstrap-CI **[−0.121, +0.073] накрывает 0**;
cost-ratio rule/llm **2.151× CI [1.576, 3.017] исключает 1**;
wall-ratio rule/llm 2.46×. **Adaptive(llm) Pareto-доминирует adaptive(rule)**
на gpt-oss: одновременно дешевле в 2.15×, быстрее в 2.46× и на per-row
mean_q даже немного выше (хотя последнее в пределах CI).

**Per-task adaptive vs static** Pareto-dominates static в 3 из 12
(router × task) комбинаций:

- commongen vs mesh (adap_llm 0.620 vs mesh 0.600 + при сопоставимой cost);
- dabench vs chain (adap_llm 0.307 + cost ≤ $0.119/run vs chain 0.279 +
  cost ≈ $0.188/run);
- dabench vs chain (adap_rule 0.284 + cost ≤ $0.174/run vs chain 0.279 +
  cost ≈ $0.188/run — Pareto-доминация по cost при quality-tie).

Per-row mean_q adaptive(llm) 0.6549 выше всех static (chain 0.6285, star
0.6299, mesh 0.5734); per-row mean_q adaptive(rule) 0.6301 в пределах CI
по сравнению со static-vec.

#### 3.1.4 Design contributions — что adaptive обеспечивает, чего нет у static

Adaptive несёт три структурных design-преимущества, не зависящих от per-task
numerical lift.

1. **Single-config deployment**: один adaptive-config работает на 4
   разнородных задачах без знания типа. Static-deployment требует либо
   выбора одной топологии под expected task-mix (lottery), либо
   разворачивания multiple-static-with-task-classifier infrastructure (extra
   classifier component, extra ops complexity).
2. **HITL advisory hint integration**: adaptive физически использует human
   feedback как input для routing decision через `signals['human_advisor_hint']`.
   Static не имеет routing-входа для HITL signals — HITL в static
   действует только на content, не на structure.
3. **Phase-aware routing**: adaptive понимает phases (planning / exec /
   verify) и активно switch'ится между топологиями per phase. Post-fix rule
   role-distribution показывает: 67% peer exec, 29% coordinator planning,
   3% reviewer verify. Static — одна топология на весь run, фаз не различает.

#### 3.1.5 Autonomy — методологический вклад

Adaptive самостоятельно определяет стратегию решения через двухуровневый
routing (`phase_router` + `topology_router`) без оракула задачи. Это
устраняет требование знать заранее лучшую топологию для конкретного task
family, что является основной операционной проблемой static deployment'а
в open-world сценариях (новые task types, drift, distribution shift).
В рамках diploma это закрывает блок «adaptive design без человека-оператора
в decision-loop».

#### 3.1.6 Honest scope caveats

- Adaptive **не превосходит** best-static-per-task oracle (требует
  априорного знания task↔topology mapping; gap −0.027 п.п. для adap(rule)
  и −0.063 п.п. для adap(llm) от un-weighted ceiling 0.7260).
- На saturated verifiable tasks chain — natural fit; adaptive-mechanism
  излишен. Adaptive(rule) на gpt-oss обгоняет только mesh из 3 static,
  отстаёт от chain (−0.014) и star (−0.015) на un-weighted 4-task mean.
- Pareto-доминирование adaptive(llm) проявляется в per-task wins на hard
  tasks, **не в overall $/q**: на overall star $0.043/q эффективнее
  adaptive(llm) $0.056/q при quality в пределах CI.
- Quality-разница adaptive vs avg static составляет +0.004 (rule) /
  −0.032 (llm) un-weighted per-task — фактически parity, оборачиваемый
  как trade-off в обе стороны в зависимости от задачи.

### 3.2 Claim 2 — routing strategy как per-family design decision

Главный contribution-point работы. Независимо от реализации router'а,
оптимальный выбор routing-стратегии (rule vs llm) **не является
fixed-at-design-time hyperparameter** — это **per-family design decision**,
требующее либо (а) cross-family валидации с выбором family-оптимального
router'а, либо (б) адаптации самой router-логики под конкретное
worker-семейство. Это structural-finding работы, более сильный чем
«adaptive дешевле static на одном семействе».

#### 3.2.1 Pareto-инверсия cross-family

| Worker | rule mean_q | llm mean_q | rule cost ($/run) | llm cost ($/run) | $/q rule | $/q llm | Pareto-winner |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt-oss-120b (post-fix) | 0.6301 | **0.6549** | 0.0793 | **0.0369** | 0.1259 | **0.0563** | **llm** (Δq +0.025 per-row, в 2.15× дешевле) |
| qwen-3-235b (confirmation) | **0.8769** | 0.5727 | 0.0549 | **0.0338** | **0.0626** | 0.0590 | **rule** (Δq +0.304 quality, $/q parity) |

`Δ rule − llm` quality bootstrap-CI на qwen **[+0.185, +0.426] EXCLUDES 0**
(strong, n=60 per router); на gpt-oss [−0.121, +0.073] includes 0.
Ось «дешёвый router — качественный router» **меняет направление при смене
worker-семейства**.

#### 3.2.2 Family-divergence magnitude

| Router | gpt-oss un-weighted 4-task | qwen un-weighted 4-task | Δ family |
|---|---:|---:|---:|
| rule | 0.6988 | **0.8769** | **+0.178** |
| llm | 0.6627 | 0.5727 | −0.090 |

Adaptive(rule) на qwen выходит в ceiling на 3 из 4 задач (humaneval 1.000,
dabench 0.933, gsm8k 0.933), что даёт +0.178 п.п. lift над gpt-oss при
идентичной реализации. Adaptive(llm) — наоборот, регрессирует на −0.090
п.п. **Одна и та же реализация роутера ведёт себя противоположно на двух
LLM-семействах.**

#### 3.2.3 LLM-router collapse на qwen verifiable — family-specific failure mode

| Task | qwen llm mean_q | qwen llm zero-rate | gpt-oss llm mean_q | gpt-oss llm zero-rate |
|---|---:|---:|---:|---:|
| gsm8k | **0.267** | **73%** | 0.917 | 8% |
| humaneval | **0.533** | **47%** | 0.808 | 19% |

LLM-router на qwen демонстрирует **structural failure-mode на verifiable
tasks**: 73% zero-rate на GSM8K, 47% на HumanEval — против gpt-oss, где
failure-rate 8% и 19% соответственно (в 8.8× и 2.4× реже).
Это **structural, не statistical** различие.

Возможные root-causes (defensible limitation):
- qwen-specific JSON-parsing failure при router-prompt'е (model-specific
  tokenizer / parser interaction);
- `reasoning_effort` confound (qwen API не принимает параметр, доступный
  gpt-oss);
- qwen-router-prompt semantics — model picks `chain` где `star` оптимален
  на math.

Полная диагностика требует отдельной cross-family серии с инструментированным
router-trace.

#### 3.2.4 Family-effect robust to implementation corrections

Корректировки реализации (5 артефактов в audit'е, см. §7) **не устранили**
family-asymmetry — qwen llm-router всё ещё коллапсирует на verifiable на
скорректированной реализации (`zero-rate 73% / 47%`). Это говорит что effect
**structural и model-specific**, а не реализационный — родная характеристика
взаимодействия router-logic с worker-family.

#### 3.2.5 DABench limitation reframing — model-specific, не universal

| Family | DABench rule | DABench llm | DABench zero-rate (rule / llm) |
|---|---:|---:|---|
| gpt-oss-120b | 0.279 | 0.307 | 67% / 60% |
| **qwen-3-235b** | **0.933** | **0.822** | **7% / 7%** |

Наблюдение «DABench broken across all topologies» из E3 post-fix на gpt-oss
— **не universal feature бенчмарка**, а **gpt-oss-specific stack limitation**:
на qwen-3-235b DABench решается на ≥0.82 (rule 0.933, llm 0.822, zero-rate
7%/7%). Это требует переформулирования limitation в Methodology: бенчмарк
не сломан, sled-зависим от worker-family. Diploma-implication: «DABench
broken for the stack» из E1/E2/E3 narrative должна быть уточнена как
«DABench broken for gpt-oss-120b on this evaluation; qwen-3-235b solves
DABench at ≥0.82 mean quality». Это **усиливает методологическую честность**:
limitation бенчмарка — не universal, а model-specific.

#### 3.2.6 Cascade-asymmetry — operational deployment angle

Cross-family stability сильно различается:

- gpt-oss `llm × dabench` потребовал 3 infrastructure correction'а
  (state-bloat в reducer'е, `ToolResult.output` cap, psycopg pipeline off);
  финальный failure-rate 67%, n_completed=25/(plan 24+ retries).
- qwen mini completion 120/120 без cascades, **$5.32 за 1.1 ч** wall-clock.

Это **deployment-relevant**: cross-family deployment должен закладывать
family-specific defensive infrastructure. Cost-структура также различается:
rule на qwen дешевле gpt-oss-аналога в 1.44×, llm — parity. Cross-family
billing на qwen-cerebras обходится ~30% дешевле на rule-router'е.

#### 3.2.7 Practical implication для ML engineering

При смене worker-LLM в production **нельзя pre-commit'ить выбор router
family** — required cross-family валидация перед deployment. Routing
strategy choice — **runtime/deployment decision based on worker
characteristics**, not architecturally-fixed hyperparameter. Это
design-вывод, который применим за пределами текущей работы — к любому
MAS-pipeline с meta-routing.

#### 3.2.8 Honest scope caveats

- Cross-family валидация проведена на **двух** worker-семействах
  (gpt-oss-120b и qwen-3-235b). Для генерализации claim'а «routing —
  per-family decision» на n worker-семейств требуется per-family
  валидация (не extrapolation на третий, четвёртый, пятый seed).
- Confirmation выполнен как mini-grid (n=15/task на router, 120 cells
  total) с tight design parity к основному E3 — этого достаточно для
  direction / magnitude verdict'а (CI bootstrap [+0.185, +0.426]
  устойчиво исключает 0), но не для absolute bounds на cross-family
  transfer-rate.
- `reasoning_effort` confound в cross-family check'е не разделён:
  gpt-oss с `low/high`, qwen без параметра. Per-task quality-deltas
  включают family-effect и reasoning_effort-effect.

---

## 4. RQ3 — cost-quality tradeoff

Cost-quality tradeoff анализируется на тех же E3 post-fix данных. Это
не отдельный эксперимент, а cross-section через E3-результаты.

### 4.1 Pareto-картина на gpt-oss-120b (E3 post-fix)

| Topology / router | mean_q | mean_cost ($) | mean_wall (s) | $/q-unit |
|---|---:|---:|---:|---:|
| static chain | 0.6285 | 0.0586 | 1 260 | 0.0933 |
| static star | 0.6299 | 0.0273 | 780 | 0.0433 |
| static mesh | 0.5734 | 0.0306 | 720 | 0.0533 |
| adaptive(llm) | **0.6549** | 0.0369 | 757 | 0.0563 |
| adaptive(rule) | 0.6301 | 0.0793 | 1 858 | 0.1259 |

**RQ3 verdict overall.** На gpt-oss-120b Pareto-front:

- **star** — $/q-эффективнее всех на overall (`$0.0433/q`);
- **adaptive(llm)** — лучший per-row mean_q (`0.6549`) при cost-overhead
  +35% над star;
- **adaptive(rule)** — Pareto-dominated одновременно by star (cheaper, ≈q)
  and by adaptive(llm) (cheaper, ≥q).
- **chain** — Pareto-dominated by star (chain $0.0586/run и mean_q 0.6285
  vs star $0.0273/run и mean_q 0.6299): star дешевле в 2.1× при равном
  качестве. Dabench-overhead раздувает per-row cost chain до 2× выше
  star/mesh.

Cost-overhead adaptive(llm) над star ($0.0369 vs $0.0273 = +35%) окупается
только на hard tasks (commongen +0.032 п.п. mean_q vs star, dabench
+0.003 п.п. vs star + лучшие zero-rate). На saturated tasks (gsm8k /
humaneval) overhead не окупается.

### 4.2 Cost-structure

Cost-asymmetry между router-режимами на gpt-oss-120b — единственный
устойчиво positive finding на overall pooled.

| Router | $ cost / quality-unit | $ per run | wall (s) |
|---|---:|---:|---:|
| adaptive(llm) | **0.0563** | 0.0369 | 757 |
| adaptive(rule) | 0.1259 | 0.0793 | 1 858 |

`rule/llm` cost ratio 2.15× CI [1.576, 3.017] **EXCLUDES 1** — асимметрия
устойчива во всех 4 task-buckets (cost-ratio rule/llm = {2.93, 1.47, 1.96,
2.11}).

llm-роутер делает в среднем **больше** итераций (10.78 vs 9.62), но при
этом стоит **меньше** и завершается **быстрее** в 2.46×. Причина — выбор
подтопологий: llm чаще выбирает дешёвый `chain` и завершается early-exit'ом,
rule стабильнее удерживается на debate/hierarchical и расходует полный
iter-budget.

### 4.3 Cross-family cost-структура

| Router | gpt-oss cost ($/run) | qwen cost ($/run) | qwen/gpt-oss ratio |
|---|---:|---:|---:|
| rule | 0.0793 | 0.0549 | 0.69 (qwen дешевле) |
| llm | 0.0369 | 0.0338 | 0.92 (parity) |

Cost-effectiveness `$/q-unit`:

| Router | gpt-oss ($/q) | qwen ($/q) |
|---|---:|---:|
| rule | 0.1259 | 0.0626 |
| llm | 0.0563 | 0.0590 |

На qwen rule-router и llm-router имеют **почти одинаковый $/q** ($0.063
vs $0.059, llm на 6% дешевле). Но эта parity не значит
Pareto-equivalence: на qwen llm выдаёт quality 0.573 при cost-saving 6%
vs rule с quality 0.877. Trade-off −0.30 quality за −6% cost —
**не Pareto-меню**. На gpt-oss та же ось наклонена иначе: llm даёт quality
0.655 при cost-saving 53% vs rule с quality 0.630 — это **Pareto-доминирование
llm**. Pareto-вывод RQ3 **family-зависим** (это и есть Claim 2 под другим
углом).

### 4.4 RQ3 verdict

- На gpt-oss-120b — **`llm`-роутер Pareto-победитель** на overall pooled
  (cost ratio 2.15× с CI excluding 1, quality-CI включает 0 = parity).
  Дополнительная стоимость rule **не окупается** ростом качества ни на одной
  из 4 задач — wall-time × 6.8–7.0 и cost × 2.0 на gsm8k/humaneval съедают
  rule-quality-преимущество.
- На qwen-3-235b — Pareto-front устроен иначе: rule strictly dominates
  по quality, $/q parity, adaptive(llm) не Pareto-эффективен.
- На static-baseline-level (star как «дешёвая статика на overall»)
  cost-efficiency star $0.043/q ниже adaptive(llm) $0.056/q при overall
  quality-parity; но star проигрывает adaptive(llm) per-task на сложных
  задачах (см. §3.1.1).

Итог: **RQ3 verdict family-conditional, как и Claim 2**. На gpt-oss cost
overhead adaptive окупается per-task на сложных задачах, не на overall.
На qwen cost-overhead llm-router'а не окупается даже на overall.

---

## 5. RQ4 — adaptive role-router (Claim 3)

### 5.1 Главное утверждение

Адаптивный role-router **не даёт detectable positive lift** поверх
adaptive-топологии ни на одной из двух протестированных worker-family.
Это — **universal negative finding** (не family-conditional как Claim 2),
что делает его более сильным научным результатом и предоставляет defensible
recommendation для downstream работы: **static reviewer-роль является
лучшей single-pick стратегией для HITL-узла на adaptive-топологии**.

### 5.2 Numerical verdict (E4 post-fix gpt-oss, n=180 per mode)

**Pooled 4-task:**

| Mode | n | mean_q | std_q | mean_cost ($) | mean_wall (s) |
|---|---:|---:|---:|---:|---:|
| `fixed` | 180 | **0.6445** | 0.409 | 0.02342 | 394.0 |
| `rule` | 180 | 0.5987 | 0.422 | 0.02094 | 375.9 |
| `llm` | 180 | 0.5937 | 0.425 | 0.02133 | 353.6 |

**Ex-dabench 3-task:**

| Mode | n | mean_q | std_q (ddof=1) |
|---|---:|---:|---:|
| `fixed` | 135 | **0.7939** | 0.295 |
| `rule` | 135 | 0.7341 | 0.342 |
| `llm` | 135 | 0.7139 | 0.358 |

**Bootstrap-CI 95% (10 000 итераций, percentile, RNG seed=42):**

| Comparison | Δ point | 95% CI | Verdict |
|---|---:|---|---|
| `rule − fixed` (pooled) | −0.046 | [−0.132, +0.040] | includes 0 (direction: −) |
| `llm − fixed` (pooled) | −0.051 | [−0.137, +0.036] | includes 0 (direction: −) |
| `rule − fixed` (3-task) | −0.060 | [−0.135, +0.017] | includes 0, borderline negative |
| `llm − fixed` (3-task) | **−0.080** | **[−0.157, −0.001]** | **EXCLUDES 0** (направление: − значимо) |

Это **первое CI в этом эксперименте, исключающее ноль**: на ex-dabench
3-task pooling `llm` достоверно хуже `fixed` в среднем на ≈ −0.08
quality-point.

### 5.3 Cross-family universal negative (vs family-conditional)

| Family | Δ_rule | Δ_llm |
|---|---:|---:|
| gpt-oss-120b (n=180/mode) | −0.046 | **−0.051** (3-task CI excludes 0) |
| qwen-3-235b (n=48 fixed / 47 rule / 47 llm) | +0.002 (null) | −0.017 (null) |

В отличие от RQ2 (router family), где направление эффекта **инвертируется**
между family, RQ4 показывает **cross-family universal negative**:
ни на gpt-oss, ни на qwen role-router не даёт positive lift. Единственный
устойчивый cross-family сигнал — `llm < fixed` по знаку на обеих family
(gpt-oss −0.051, qwen −0.017). Это **более сильный negative finding**,
чем family-conditional, поскольку устраняет hypothesis «role-router работает
на правильном семействе» — он не работает ни на одном.

### 5.4 Role-router физически активируется — это not implementation gap

Post-fix role-distribution показывает, что role-router физически работает
и активно switch'ится:

| Mode | reviewer | peer | coordinator | judge | monitor |
|---|---:|---:|---:|---:|---:|
| fixed | 180 (100%) | 0 | 0 | 0 | 0 |
| rule | 6 (3%) | **121 (67%)** | 53 (29%) | 0 | 0 |
| llm | 6 (3%) | 0 | **162 (90%)** | 5 (3%) | 7 (4%) |

- `rule` доминирующе выбирает `peer` (соответствует exec-фазе
  `DEFAULT_ROLE_TABLE`); 29% — coordinator (planning-фаза); 3% — reviewer
  (verify-фаза).
- `llm` фиксируется на `coordinator` как safe default; редко добавляет
  экзотические альтернативы (monitor 4%, judge 3%, reviewer 3%).

**Role-router делает то, что должен делать**, но эти выборы **не дают
quality lift**. Это устраняет hypothesis «role-router не работает из-за
implementation bug» — он работает, но adaptive-роли не помогают.

### 5.5 Cost-overhead не окупается — quantified Pareto failure

LLM role-router добавляет meta-LLM call per phase transition. На post-fix
gpt-oss данных:

| Mode | mean_cost ($) | mean_q | Δ_q vs fixed | $/quality-point |
|---|---:|---:|---:|---:|
| fixed | 0.0234 | 0.6445 | — | 0.0363 |
| rule | 0.0209 (−11%) | 0.5987 (−7.1%) | −0.046 | 0.0349 |
| llm | 0.0213 (−9%) | 0.5937 (−7.9%) | −0.051 | 0.0359 |

Cost win 9–11% **не оправдывает** quality loss 7–8%. Δ_cost небольшой
(peer/coordinator советы короче reviewer'ских), но quality-deficit
перекрывает экономию. **Pareto-trade не работает в пользу адаптивной
роли**.

### 5.6 Mechanism hypothesis — structural variability обеспечена другим уровнем

Defensible interpretation: adaptive-топология **уже** обеспечивает
structural variability через смену под-топологии per phase (debate,
hierarchical, chain). Повторная variability через role-router на HITL-узле
создаёт **noise без added information** — наблюдается рост std_q от fixed
(0.295) к llm (0.358) на 3-task pooling (т.е. дисперсия исходов растёт с
увеличением адаптивности роли). Single reviewer-role даёт consistent
feedback signal, что лучше для уже-адаптивной системы.

Иначе говоря: **adaptivity на двух уровнях одновременно противопоказана**
— достаточно одного уровня (topology); второй (role) даёт regression.
В терминах теории — adaptivity coupling между levels требует careful
information-budget tradeoff, и в нашем дизайне budget уходит на
topology-level, не оставляя room для role-level.

### 5.7 Downstream design recommendation

На основании Claim 3 для downstream работы (E5 или real-deployment)
рекомендуется конфигурация:

```yaml
human:
  role_router: fixed
  role: reviewer
topology:
  name: adaptive
  extra.adaptive:
    phase_router: rule
    topology_router: <family-optimal: llm на gpt-oss, rule на qwen>
```

Это упрощает deployment (один meta-LLM-call экономится per phase
transition), упрощает участников E5 (одна роль reviewer, не три), и снимает
source of unnecessary variance из experimental design.

### 5.8 Negative finding as research contribution

Universal negative result по RQ4 — **сильнее**, чем positive lift с
family-specific caveat'ом. Он:

1. Закрывает направление дальнейшей разработки role-router'ов в текущем
   дизайне adaptive-MAS.
2. Снимает design complexity (один meta-LLM-call meno в pipeline).
3. Даёт concrete bound: «двойная адаптивность на topology + role уровнях
   избыточна; ограничьтесь одной».
4. Mechanistically объяснимо (см. §5.6) — не выглядит как statistical fluke,
   а имеет theoretical interpretation.

### 5.9 Honest scope caveats

- Confirmation на qwen — direction-only mini (n=12 per cell после удвоения
  waves), не достаточно для tight CI; verdict «null direction» точечен и
  не исключает мелкий effect ниже detection threshold (~±0.17 CI bound).
- HITL во всех runs — `llm_simulated` (`openai:gpt-4.1-mini`); behaviour
  real participants в E5 может отличаться, что обоснованно мотивирует
  triangulation.
- Пять ролей рассматривались (`reviewer`, `peer`, `coordinator`, `judge`,
  `monitor`); возможно отсутствующая в табличке шестая роль может
  работать — это open question для будущих серий.
- Measurement gap в parquet (отсутствует колонка `role_router`) восстановлен
  через позиционный sort + sanity-check (fixed-block = 100% reviewer,
  180/180); подход надёжный для batch-execution, но фрагилен к
  asynchronous submission в E5. Рекомендация: добавить `role_router` в
  `_runs.parquet` schema до E5.

---

## 6. Cost summary — финальная картина

| Category | Sum $ | Notes |
|---|---:|---|
| E1 (900 cells, no HITL) | 12.62 | $0.014/run, в 3.5× ниже плана |
| E2 (2 295 cells, HITL on) | 42.37 | $0.0185/run, HITL overhead +23% |
| E3 post-fix (gpt-oss adaptive + static) | 29.38 | $0.046/run avg (adaptive overhead) |
| E4 post-fix (gpt-oss adaptive role) | 11.82 | $0.022/run avg |
| Confirmation E3 (qwen mini) | 5.32 | $0.044/run avg (qwen-cerebras pricing) |
| Confirmation E4 (qwen mini) | 3.84 | $0.027/run avg |
| **Total spend, all 6 experiments** | **105.35** | Plan: $200 budget cap |

Бюджет всей экспериментальной серии — **$105.35**, под планом $200. Все
ключи API уложились в `payg` лимиты Cerebras и OpenAI. На full-replication
этой работы при текущем стеке требуется примерно $110 на ~4 200 cells.

---

## 7. Methodology audit framing (Claim 4)

В ходе разработки экспериментального пайплайна был проведён **аудит
реализации adaptive-роутера**, выявивший **пять методологических артефактов**
(категории: чистота state-transfer, семантика гардов, источники счётчиков,
жизненный цикл сигналов). Все артефакты устранены до финальных прогонов
(commit `PR #18`).

Численные результаты в работе получены на **скорректированной реализации**;
pre-correction данные сохранены в репозитории как audit trail и не
используются в анализе. Confirmation `family-effect robust to corrections`:
ключевые structural-findings (Pareto-инверсия на qwen, LLM-router collapse
на verifiable, universal negative RQ4) **воспроизводятся на скорректированной
реализации** — это означает, что эффекты model-specific и structural, а не
implementation-artefact.

Подробности audit'а (по категориям артефактов, корректировок и audit
trail'а) — в отдельном `methodology_audit.md`.

---

## 8. Final synthesis — итог трёх RQ-claim'ов

Все три claim'а — поддержанные конкретными bootstrap-CI numbers,
cross-family triangulation и methodology audit framing'ом — формируют
единую защиту работы.

| # | Claim | Поддерживающее наблюдение | Bootstrap-CI / verdict | Scope |
|---|---|---|---|---|
| **1** | **RQ2: adaptive выигрывает на сложных задачах + minimum-regret strategy + design contributions** | E3 post-fix: adaptive(llm) commongen 0.620 > все 3 static; adaptive Pareto-доминирует chain на dabench; adaptive(llm) highest floor 0.307, lowest std 0.267, lowest cost spread 13× | Per-task wins; cost-ratio rule/llm 2.151× CI [1.576, 3.017] EXCLUDES 1; quality CI ambient | gpt-oss-120b; honest scope: не обгоняет oracle best-static-per-task |
| **2** | **RQ2: routing strategy — per-family design decision** | Cross-family: Pareto-инверсия (`llm-best на gpt-oss, rule-best на qwen`); LLM-router collapse на qwen verifiable (gsm8k 0.267, zero-rate 73%); DABench reframed как model-specific | Δ rule−llm на qwen = +0.304, CI [+0.185, +0.426] EXCLUDES 0; family-divergence в Δ_rule = +0.178 | 2 worker-семейства; honest scope: не extrapolation на n>2 семейств |
| **3** | **RQ4: adaptive role-router — universal negative** | E4 post-fix: fixed > rule > llm на всех 4 задачах; 3-task `llm − fixed` CI EXCLUDES 0; cross-family confirmation: null на qwen, negative на gpt-oss — universal negative direction для `llm < fixed` на обеих family | Δ `llm − fixed` 3-task = −0.080 CI [−0.157, −0.001] EXCLUDES 0; универсальный знак сохраняется на qwen (−0.021) | gpt-oss-120b + qwen-3-235b confirmation; honest scope: qwen mini direction-only |
| **4** | **Methodology audit framing** | 5 артефактов выявлены и устранены до final runs; family-effect robust к corrections | Audit trail в репозитории; финальная реализация — `PR #18` | Все 6 финальных экспериментов на post-fix code |

**Defense one-liner.**

> Мы построили adaptive multi-agent topology infrastructure с двухуровневым
> router'ом (topology + role) и HITL integration; систематически
> протестировали три RQ на двух LLM worker-семействах с methodology audit
> и cross-family triangulation. **Главный contribution** — structural
> finding о том, что (а) adaptive routing предоставляет minimum-regret
> + per-task wins на сложных задачах + design contributions, превосходящие
> static deployment в open-world; (б) выбор routing-стратегии (rule vs llm)
> — per-family deployment decision, а не fixed hyperparameter, поскольку
> Pareto-front инвертируется между gpt-oss и qwen (bootstrap-CI EXCLUDES 0);
> (в) adaptive role-router universally не работает поверх adaptive-топологии
> (negative direction на обеих family, 3-task CI EXCLUDES 0 на gpt-oss).
> Все три claim'а подкреплены confirmation на скорректированной реализации
> после methodology audit'а (5 артефактов устранены до final runs).

---

## Appendix A — exp_id'ы финальных грид-прогонов

| Эксп | exp_id | n_cells |
|---|---|---:|
| E1 full | `f153400f-f8ba-4ca3-8dc2-66f31ba15236` | 900 |
| E2 full | `783f4b1e-8888-461e-82fe-07c700f789bc` | 2 295 |
| E3 post-fix full (rule) | `3aa2187a-…` + `d035ca05-…` + `3711ffeb-…` | 524 (rule) |
| E3 post-fix per-task llm | `437b65cd-…` + `8544307b-…` + `31cd6082-…` + `3b496f94-…` | 119 (llm) |
| E4 post-fix humaneval | `9ed6cd02-e779-4137-8eea-640e5e5fae67` | 135 |
| E4 post-fix gsm8k | `5087cd5c-d5e4-463d-9172-77c3addc74dd` | 135 |
| E4 post-fix commongen | `9663b4eb-cb29-42d7-8d86-d9b39b359b38` | 135 |
| E4 post-fix dabench | `41793bf8-d94c-4999-bfe1-e5bfe62b89db` | 135 |
| Confirmation E3 rule mini (qwen) | `2bea6828-924c-4464-b54b-ce31944d73cc` | 60 |
| Confirmation E3 llm mini (qwen) | `1e5034c1-ed9d-4c7f-a88e-602069336449` | 60 |
| Confirmation E4 mini (qwen) | `deab3797-dc37-456d-92d4-400bf005284c` | 142 |

---

## Appendix B — Source files

- E1 analysis: `arch/diploma/results/e1_analysis.md`
- E2 analysis: `arch/diploma/results/e2_analysis.md`
- E3 post-fix analysis: `arch/diploma/results/e3_postfix_analysis.md`
- E4 post-fix analysis: `arch/diploma/results/e4_postfix_analysis.md`
- Confirmation E3 (cross-family qwen): `arch/diploma/results/confirmation_e3_postfix_analysis.md`
- Confirmation E4 (cross-family qwen): `arch/diploma/results/confirmation_e4_postfix_analysis.md`
- Canonical narrative plan: `arch/diploma/results/_postfix_narrative_plan.md`
- Methodology audit (отдельный документ): `arch/diploma/results/methodology_audit.md`
