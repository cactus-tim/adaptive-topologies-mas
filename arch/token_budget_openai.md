# Token budget — OpenAI сценарии (floor / baseline / ceiling)

Актуализировано 2026-04-23. Цены подтверждены со страниц OpenAI:

| Модель | Input | Cached Input | Output |
|---|---|---|---|
| gpt-4o-mini | $0.15 | $0.075 (−50%) | $0.60 |
| gpt-4o | $2.50 | $1.25 | $10.00 |
| gpt-4.1-mini | $0.20 | $0.10 (−50%) | $0.80 |
| gpt-4.1 | $2.00 | $0.50 (−75%) | $8.00 |
| gpt-5-mini | $0.25 | $0.025 (−90%) | $2.00 |
| gpt-5 | $0.625 | $0.125 | $5.00 |
| o4-mini | $0.55 | $0.275 | $2.20 |

Batch API −50% на всех. Не подходит для synchronous multi-agent loops, применимо на offline-judge и analysis-runs.

## Базовые допущения per-call

| Параметр | Floor | Base | Ceiling |
|---|---|---|---|
| avg input tokens/call | 600 | 1 000 | 1 800 |
| avg output tokens/call | 300 | 400 | 600 |
| avg LLM-calls/run | 10 | 15 | 25 |
| cache hit ratio | 40% | 40% | 50% |
| retry overhead | +10% | +20% | +30% |

Примечание: на ceiling берём верхнюю оценку потому что (а) scratchpad полноценно растёт на сложных задачах, (б) больше итераций debate/mesh rounds, (в) summarizer вызывается при превышении context.

## Run count по этапам

| Этап | Floor | Base | Ceiling |
|---|---|---|---|
| E1 baseline no-human | 1 000 | 1 500 | 3 000 (seeds=5, 30 tasks) |
| E2 HITL-sim | 3 000 | 4 000 | 11 000 (все role×topology, seeds=5) |
| E3 adaptive topology | 600 | 1 100 | 2 400 (+oracle router) |
| E4 adaptive both | 600 | 900 | 3 600 (3 strategies × 2 baselines × 5 seeds) |
| Confirmation на premium | 300 | 1 800 | 2 500 |
| Sensitivity cross-model | 0 | 0 | 400 (gpt-4.1 + gpt-4o subset) |
| Ablations (router, role, iter-budget) | 0 | 200 | 2 000 |
| E5 real human | 100 | 140 | 150 |
| **Total runs** | **~5 600** | **~9 640** | **~25 050** |
| + retry buffer | +10% → 6 200 | +20% → 11 600 | +30% → 32 600 |

## Token volume

| Scenario | Runs (w/ retries) | Calls | Input M | Output M | Total M |
|---|---|---|---|---|---|
| Floor | 6 200 | 62 000 | **~37** | **~19** | **~56** |
| Baseline | 11 600 | 174 000 | **~174** | **~70** | **~244** |
| Ceiling | 32 600 | 815 000 | **~1 467** | **~489** | **~1 956** |

## Стоимость при разных моделях (с учётом кэша)

### Floor (максимальная экономия)

Всё на gpt-4o-mini:
- Input 37M × (0.6×$0.15 + 0.4×$0.075) = 37 × $0.12 = **$4.44**
- Output 19M × $0.60 = **$11.40**
- **Floor Total: ~$16** (без confirmation)

С минимальной confirmation на gpt-4o (300 runs, ~3M in / 1.2M out):
- +3×$2.50 + 1.2×$10 = **+$19.50**
- **Floor + confirmation: ~$36**

### Baseline (как в experiment_plan.md)

Primary gpt-4o-mini:
- Input 174M × $0.12 = **$20.88**
- Output 70M × $0.60 = **$42.00**
- Main: **~$63**

Confirmation на gpt-4o (1 800 runs × 15 calls × 1000/400 = 27M/11M):
- 27×$2.50 + 11×$10 = **~$178**

**Baseline Total: ~$241**

Альтернатива baseline (primary = gpt-4.1-mini):
- 174 × $0.16 + 70 × $0.80 = **$28 + $56 = $84** main
- + Confirmation на gpt-4.1 (1 800 runs, те же 27M/11M): 27×$1.25 + 11×$8 = **$122**
- **Baseline Premium Total: ~$206**

### Ceiling (щедрый — бесплатные токены)

Primary gpt-5-mini (лучший cache discount 90%):
- Input 1 467M × (0.5×$0.25 + 0.5×$0.025) = 1 467 × $0.1375 = **$202**
- Output 489M × $2.00 = **$978**
- Main: **~$1 180**

Confirmation gpt-5 (~15% объёма = 122K calls, 220M in / 73M out):
- 220 × (0.5×$0.625 + 0.5×$0.125) + 73 × $5 = 220 × $0.375 + $365 = **$83 + $365 = $448**

Sensitivity gpt-4.1 (5% = 40K calls, 72M in / 24M out):
- 72 × $1.25 + 24 × $8 = **$90 + $192 = $282**

**Ceiling Total: $1 180 + $448 + $282 ≈ $1 910**
**+ 20% reserve → ~$2 300**

### Сводная таблица

| Scenario | Input M | Output M | gpt-4o-mini only | + gpt-4o confirm | Full premium mix |
|---|---|---|---|---|---|
| Floor | 37 | 19 | **$16** | $36 | — |
| Baseline | 174 | 70 | $63 | **$241** | $206 (4.1-mini+4.1) |
| Ceiling | 1 467 | 489 | $253 | ~$430 | **$1 910–2 300** |

## Что давать в запас на "ceiling", если токены бесплатны

1. **Seeds до 7–10** вместо 3 — statistical power важнее, variance seeds часто > эффект топологии.
2. **Task-sample до 50 на тип** — 200 total вместо 100. Позволяет per-task-type per-topology CI без дополнительных допущений.
3. **Cross-model validation**: параллельный прогон champion configs на (gpt-4o-mini + gpt-4.1 + gpt-5-mini) на одном subset → защита от "эффект модели, а не топологии".
4. **Router ablations (4 варианта)**: rule-based / LLM-router / oracle / no-router.
5. **Role ablations**: полный 5×6 grid вместо сужения top-2 после E1.
6. **Budget-curves**: тот же E1 на max_iter ∈ {5, 10, 20} → отдельный раздел "scaling behavior".
7. **Human-sim robustness**: 3 persona-prompt варианта для каждой роли → устойчивость к формулированию симулятора.
8. **Tooling stress-test**: run на gpt-5 с force_reasoning=true для "upper-ceiling quality" на 200 задачах.
9. **Contamination check**: LiveCodeBench-mini subset (30 задач) как sensitivity для HumanEval-загрязнения.

## Прикладной дизайн решения

| Если токены бесплатно? | Рекомендуемая конфигурация |
|---|---|
| **Да, флагман не ограничен** | Primary = gpt-5-mini (cache 90% → дёшево даже в ceiling), confirmation gpt-5, sensitivity gpt-4.1. Реализовать **все 9 ceiling-расширений**. Target ~2 000M tokens / ~$2 300 если пересчитывать, но при free — **не считаем $, считаем runs**. |
| **Да, но ограничены до ~$500** | Primary = gpt-4.1-mini (base + cache dicount 50%), confirmation gpt-4.1, без cross-model sensitivity. Расширить seeds до 5 и sample до 30/тип. |
| **Платишь сам, бюджет $200–300** | Baseline как в experiment_plan.md на gpt-4o-mini + confirmation gpt-4o. ~$240. |
| **Платишь сам, жёсткий бюджет <$50** | Floor: все этапы на gpt-4o-mini, seeds=3, sample=20/тип, без confirmation premium. ~$16–20. |

## Формат запроса токенов (если просить у поставщика)

Если тот, кто может дать бесплатные токены, спросит "сколько?":

- **Минимум-достаточно для диплома**: **250M input + 100M output** на mini-tier (gpt-4o-mini / gpt-4.1-mini) + **50M input + 20M output** на premium (gpt-4o / gpt-4.1)
- **Комфортно с запасом**: **500M input + 200M output** на mini + **100M input + 40M output** на premium
- **Щедро (все ablations)**: **1.5B input + 500M output** на mini-tier (gpt-5-mini предпочтительно) + **250M input + 100M output** на premium (gpt-5)

В терминах $ при обычном прайсе:
- Минимум: эквивалент ~$180
- Комфортно: ~$360
- Щедро: ~$1 900–2 300

## Кэш — почему важно не считать его как bonus

Структура промптов multi-agent системы:
- System prompt (role persona + instructions): ~400–800 tokens, **одинаковый во всех вызовах этого агента** → 100% cache hit после первого
- Task description: ~100–300 tokens, одинаковый во всех вызовах одного run
- Scratchpad history: растёт, первые N итераций уже в кэше
- Inbox/messages: динамично

Реалистичный cache hit ratio для наших сценариев: **30–50% input tokens**. Это уже заложено в расчёты выше. У gpt-5-mini (−90%) и gpt-4.1 (−75%) эффект сильнее, что делает их ceiling-сценарии особенно экономными.

## Источники (актуально 2026-04-23)

- openai.com/api/pricing
- developers.openai.com/api/docs/pricing
- pricepertoken.com/pricing-page/provider/openai (обновлён 2026-04-23 08:37 AM)
