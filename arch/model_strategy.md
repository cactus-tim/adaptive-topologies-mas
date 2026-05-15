# Model strategy — сравнение вариантов (исторический анализ, 2026-04-23)

> **⚠ STATUS — SUPERSEDED 2026-05-15.**
> Этот документ — снимок ранних рассуждений о выборе LLM-инфраструктуры (OpenAI vs Z.ai vs DeepInfra-Qwen vs Yandex self-host). Финальное решение принято позже, после добавления Cerebras-провайдера (см. `dev/done/add-cerebras-provider/`) и анонса deprecation Llama-семьи в Cerebras Cloud (2026-05-27).
>
> **Актуальный lineup** см. в:
> - `arch/experiment_plan.md` §0 («Модели») — список ролей и моделей
> - `conf/pricing.yaml` — header c комментарием "Active experimental lineup"
>
> Кратко (Variant B'.1):
> - workers / critic / summarizer / router / HITL-sim → `cerebras:gpt-oss-120b`
> - judge (LLM-as-judge) → `openai:gpt-4.1-mini` (cross-family независимость)
> - confirmation: `cerebras:zai-glm-4.7` + `openai:gpt-4o`
>
> Документ ниже сохранён как контекст для текстовой части диплома (раздел «обоснование выбора провайдера»). Числовые оценки бюджета и TCO под OpenAI/DeepInfra **устарели**, не использовать.

---

Сравнение трёх стратегий выбора LLM-инфраструктуры для грида ~9 500 runs × 15 LLM-calls ≈ **142 500 вызовов**. Вход ≈ 114M токенов, выход ≈ 57M токенов. Все цифры ниже — USD.

## 0. Нагрузка (baseline для расчётов)

| Параметр | Значение |
|---|---|
| Всего LLM-calls | 142 500 |
| Input tokens | ~114M |
| Output tokens | ~57M |
| Повторяемость промптов (persona, system) | ~30–50% → кэш-discount реален |
| Параллелизм | 4–16 concurrent runs (ProcessPool × asyncio) |
| Latency важна | Умеренно (не realtime, но runs по ~минуте) |
| Tool-calling + JSON mode | Обязательно |
| Контекст | ≥64K желательно (scratchpad + inbox) |

---

## 1. Стратегия A — OpenAI (актуализировано)

### Цены (апрель 2026, $/1M tokens)

| Модель | Input | Cached | Output | Batch input/output | Контекст |
|---|---|---|---|---|---|
| gpt-4o-mini | 0.15 | 0.075 (−50%) | 0.60 | 0.075 / 0.30 | 128K |
| gpt-4o | 2.50 | 1.25 | 10.00 | 1.25 / 5.00 | 128K |
| gpt-4.1-mini | 0.40 | 0.10 (−75%) | 1.60 | 0.20 / 0.80 | 1M |
| gpt-4.1 | 2.00 | 0.50 | 8.00 | 1.00 / 4.00 | 1M |
| gpt-5-mini | 0.25 | 0.025 (−90%) | 2.00 | 0.125 / 1.00 | 400K |
| gpt-5 | 1.25 | 0.125 | 10.00 | 0.625 / 5.00 | 400K |
| o4-mini (reasoning) | 1.10 | 0.275 | 4.40 | 0.55 / 2.20 | 200K |

**Замечания:**
- gpt-4o и gpt-4o-mini в API ещё работают, но в ChatGPT убраны → возможен sunset в API на горизонте 6–12 мес. Предупреждение OpenAI минимум 6 мес до отключения.
- **gpt-4.1-mini** — прямой апгрейд gpt-4o-mini: 1M контекст, −75% cache discount. Выше качество tool-calling.
- **gpt-5-mini** — лучший cache discount (90%) и output качество; при высоком caching может быть дешевле gpt-4.1-mini.
- Batch API −50%, асинхронно до 24ч — **не подходит** для multi-agent loops (агенты синхронно обмениваются сообщениями). Применимо только для judge-прогонов offline.

### TCO варианты (без кэша / с кэшем 40%)

| Модель | Без кэша | С кэшем 40% |
|---|---|---|
| gpt-4o-mini | **$51** | **$41** |
| gpt-4.1-mini | $137 | $100 |
| gpt-5-mini | $143 | $108 (−90% cache делает экономию заметной при cache-ratio ≥60%) |
| gpt-4.1 | $684 | $524 |
| gpt-4o (confirmation) | $855 | $681 |

### Рекомендация
- **Рабочая лошадка**: gpt-4o-mini (~$51 базово, ~$41 с кэшем) — проверенная, сопоставима с G-Designer/DyLAN
- **Апгрейд путь**: gpt-4.1-mini (~$100) если нужно лучшее tool-calling / 1M контекст
- **Confirmation runs**: gpt-4.1 (~$100 за 1800 runs на нём); либо оставить gpt-4o-mini + подписать в limitations

**Плюсы:** надёжность, качество tool-calling, литературная сопоставимость, Stripe-оплата (иностранные карты).
**Минусы:** дороже всех альтернатив; contamination HumanEval/MMLU высокая; из РФ нужна иностранная карта.

---

## 2. Стратегия B — Z.ai GLM (альтернатива, умеренная экономия)

### Цены ($/1M tokens, официальный endpoint api.z.ai)

| Модель | Input | Cached | Output | Контекст | Назначение |
|---|---|---|---|---|---|
| GLM-5.1 | 1.40 | 0.26 | 4.40 | 200K | flagship agentic (SWE-bench 58.4%) |
| GLM-5-Turbo | 1.20 | 0.24 | 4.00 | 200K | **best tool-calling (~0.67% error)** |
| GLM-5 | 1.00 | 0.20 | 3.20 | 200K | flagship базовый |
| GLM-4.7 | 0.60 | 0.11 | 2.20 | 200K | mid-tier, τ²-Bench > Claude 4.5 |
| GLM-4.5-Air | **0.20** | **0.03** | **1.10** | 128K | **child agents** |
| GLM-4.7-FlashX | 0.07 | 0.01 | 0.40 | 200K | самый дешёвый платный |
| GLM-4.7-Flash | free | free | free | 200K | бесплатный tier |

### Подписки (GLM Coding Plan) — **НЕ подходит для нашего кейса**

| Plan | ₽/квартал | Промпты/5ч | Ограничения |
|---|---|---|---|
| Lite | ~$30/Q | ~80 | coding-only |
| Pro | ~$90/Q | ~400 | coding-only |
| Max | ~$240/Q | ~1600 | coding-only |

**Критично**: ToS запрещает non-coding use → multi-agent research с разными ролями = риск бана (throttling 1302/1303 → suspension). Используем только pay-per-token.

### TCO варианты

**B1 — Mixed: GLM-4.5-Air (80% child) + GLM-4.7 (20% coordinator)**
- Child: 91M×0.8×$0.20 + 46M×0.8×$1.10 = $14.56 + $40.13 = **$54.69**
- Coord: 91M×0.2×$0.60 + 46M×0.2×$2.20 = $10.92 + $20.06 = **$30.98**
- **Total: ~$86** (без кэша) / **~$55** (с кэшем 40%)

**B2 — Aggressive: GLM-4.7-FlashX (80%) + GLM-5 (20%)**
- **Total: ~$67** / **~$45** (с кэшем)

### OpenAI-compatible endpoint
```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:glm-4.5-air",
    base_url="https://api.z.ai/api/paas/v4/",
    api_key=os.environ["ZAI_API_KEY"])
```
Работает с LangGraph из коробки, поддерживает tools + JSON mode + thinking.

### Подводные камни
- **Доступность из РФ**: z.ai принимает Visa/MC через Stripe; reliability из Европы бывает прерывистая (compute shortage Q1 2026).
- **Tool-calling quality**: GLM-5 base — проблемы с malformed JSON; **используй GLM-5-Turbo или GLM-4.7** для агентных loops.
- **Rate limits непрозрачны** — concurrency-based, новые аккаунты с консервативными лимитами. Перед стартом проверить на pilot.
- **Language bias**: system prompts держать на английском; нейтральных research-промптов это не касается.
- **Bench-уровень**: GLM-5.1 frontier по coding; по reasoning (AIME) уступает GPT-5 — для GSM8K может быть чуть слабее, но для нашей задачи (сравнение топологий) важны относительные различия.

### Когда выбирать B
- Нужна экономия ~40–50% vs GPT-4.1-mini при сохранении уровня "mid-frontier"
- Готов к дополнительной reliability-страховке (fallback на OpenRouter)

---

## 3. Стратегия C — Open-source (Qwen / DeepSeek через API или self-host)

### C.1 — API open-source (дешевле всего, proven path)

**Цены ($/1M tokens, проверено апрель 2026)**

| Модель | Провайдер | Input | Cached | Output | Контекст | Коммент |
|---|---|---|---|---|---|---|
| **Qwen3.5-397B-A17B (MoE)** | DeepInfra | **0.071** | — | **0.10** | 256K | **лучший quality/price** |
| Qwen3-Coder-480B Turbo | DeepInfra | 0.22 | 0.022 (−90%) | 1.00 | 256K | coding специалист |
| Qwen2.5-72B | OpenRouter / DeepInfra | 0.12 | auto (0.25×) | 0.39 | 128K | проверенная, устойчивая |
| **DeepSeek-V4** | DeepSeek direct | 0.30 | 0.03 (−90%) | 0.50 | 128K | **best tool-calling open-source** |
| DeepSeek-V3.2 | DeepInfra | 0.26 | 0.13 (−50%) | 0.38 | 160K | production-grade |
| Llama-3.3-70B | DeepInfra | 0.10 | — | 0.32 | 128K | стандарт для сравнения |
| Qwen3.5-Flash | DashScope | 0.10 | есть | 0.40 | 256K | легковес |
| Qwen3-32B | Groq | 0.29 | −50% | 0.59 | 131K | **662 TPS, ~мгновенно** |

**Важно из РФ:**
- DashScope direct (Alibaba) — **карты РФ не работают**, только иностранные
- **OpenRouter / DeepInfra / Together / Groq** — Stripe, иностранные карты (АРМ/GEO) обычно проходят
- DeepSeek direct — принимает Stripe

### TCO варианты C.1

| Вариант | Конфигурация | Total (без кэша) | С кэшем 40% |
|---|---|---|---|
| C1a. Qwen3.5-397B (DeepInfra) — все calls | 114M×$0.071 + 57M×$0.10 | **$13.80** | **$11.50** |
| C1b. DeepSeek-V4 direct — все calls | 114M×$0.30 + 57M×$0.50 | $62.70 | **$43** (кэш −90% на input) |
| C1c. Qwen2.5-72B (OpenRouter) | 114M×$0.12 + 57M×$0.39 | $36 | $28 |
| C1d. Llama-3.3-70B (DeepInfra) | 114M×$0.10 + 57M×$0.32 | $29.60 | $22 |

**Лидер по цене**: C1a ≈ **$12–14**, в 4× дешевле стратегии A (gpt-4o-mini).

### C.2 — Self-host на Yandex.Cloud (если нужен контроль / russian compliance)

**Доступные GPU (апрель 2026, c НДС 22%):**

| GPU | VRAM | On-demand ₽/ч | Preemptible ₽/ч | ≈ $/ч on-demand |
|---|---|---|---|---|
| 1× A100 80GB | 80 GB | ~4 600–5 000 | **~1 017** | ~$51 / $11 |
| 2× A100 80GB | 160 GB | ~9 200 | ~2 034 | ~$102 / $23 |
| 1× V100 32GB | 32 GB | ~1 500 | ~400 | ~$17 / $5 |
| 1× T4 16GB | 16 GB | небольшая | — | дёшево, но слабо для 14B+ |

**H100, L40S, L4, A10 — у Yandex.Cloud в публичном облаке НЕ доступны** (или только по Enterprise-договору).

**Propmtible**: прерывается через 24ч, нужна checkpoint/resume логика.

### TCO варианты C.2

**C2a — Qwen2.5-72B AWQ (int4, ~40GB) на 1×A100 80GB**
- Throughput: ~70 tok/s output (conservative для AWQ, batch 10–20)
- Prefill: ~500 tok/s
- GPU-часы: 114M/500 + 57M/70 ≈ **290 часов (~12 суток)**

| Провайдер | Цена | Итого |
|---|---|---|
| Yandex.Cloud on-demand | 5 000 ₽/ч × 290 | **~1 450 000 ₽ ($16 000)** — нерентабельно |
| Yandex.Cloud preemptible | 1 017 ₽/ч × 290 | **~295 000 ₽ ($3 300)** |
| RunPod spot | $0.79/ч × 290 | **~$229** |
| Vast.ai | ~$0.80/ч × 290 | **~$230** |
| immers.cloud (RU) | ~1 000 ₽/ч × 290 | **~290 000 ₽ ($3 200)** |

**C2b — Qwen3-14B (bf16, ~28GB) на 1×A100 80GB**
- Throughput: ~400 tok/s output, ~1000 tok/s prefill
- GPU-часы: ~72 часа

| Провайдер | Итого |
|---|---|
| Yandex.Cloud preemptible | **~79 000 ₽ ($880)** |
| RunPod spot | **~$57** |
| immers.cloud | ~72 000 ₽ ($800) |

### Setup overhead для self-host
- VM + CUDA + vLLM: 2–4 часа
- Загрузка модели 40–80 GB: 1–2 часа
- Отладка tool-calling / JSON / concurrency: 1–3 дня
- Интеграция LangGraph (base_url swap): 2–4 часа
- **Итого: 5–10 дней человеко-времени**

### Когда выбирать C
- **C1a (Qwen3.5-397B через DeepInfra)** — если цель максимальная экономия и готов к possible reliability-рискам
- **C2b (Qwen3-14B self-host)** — если есть требование "полный контроль" / российская инфраструктура / нельзя платить иностранным провайдерам
- **C2a (72B self-host)** — почти никогда не рентабельно на Yandex.Cloud; RunPod spot может окупить

### Подводные камни open-source
- **Tool-calling**: Qwen3 и DeepSeek-V3.2+ хорошо работают; Qwen2.5 — чуть хуже; Llama-3.3-70B — приемлемо, но хуже GPT. Нужен retry + JSON schema validation в каждом LangGraph node.
- **Contamination** меньше (свежее обновление), но HumanEval всё равно включён в pretrain.
- **Консистентность**: на 142k вызовах отказов/пропусков больше, чем у OpenAI → +10–20% retry-бюджет.

---

## 4. Сводная таблица — итоговый TCO (142k вызовов)

| # | Стратегия | Модель | Total inference | Overhead | Плюсы | Минусы |
|---|---|---|---|---|---|---|
| A1 | OpenAI cheap | gpt-4o-mini | **$41–51** | — | стандарт, качество | дороже всех opensource |
| A2 | OpenAI mid | gpt-4.1-mini | $100–137 | — | 1M context | 2× A1 |
| A3 | OpenAI premium | gpt-4.1 (confirmation 1800) | $100 (subset) | — | frontier | дорого |
| B1 | GLM mixed | Air + 4.7 | **$55–86** | reliability check | 40% экономия | tool-call квалификация |
| B2 | GLM aggressive | FlashX + GLM-5 | $45–67 | reliability | ещё дешевле | slabeе качество |
| C1a | **Qwen3.5-397B API** | DeepInfra | **$12–14** | карта-fallback | **4× дешевле A1** | possible throttling |
| C1b | DeepSeek-V4 API | direct | $43 (cached) | — | best opensource tool-call | чуть дороже C1a |
| C2a | 72B self-host (YC preempt) | A100 80GB | **~$3 300** | 5–10 дней setup | контроль | не рентабельно |
| C2a′ | 72B self-host (RunPod spot) | A100 80GB | **~$230** | 5–10 дней setup | дешёвый self-host | прерывания |
| C2b | 14B self-host (RunPod spot) | A100 80GB | **~$57** | 5–10 дней setup | дёшево + контроль | слабее 70B по качеству |

+ confirmation runs на большей модели = +$50–150 для всех вариантов.
+ 30% reserve на retries.

---

## 5. Рекомендация для диплома

**Главный кандидат: Стратегия A1 (gpt-4o-mini)** — baseline.
Причины: литературная сопоставимость (G-Designer, DyLAN), надёжный tool-calling, не требует self-host, $41–51 влезает в любой бюджет.

**Альтернатива для экономии: C1a (Qwen3.5-397B через DeepInfra)** — $12–14.
Причины: frontier-class качество, 3–4× дешевле, OpenAI-compatible endpoint (замена `base_url` в одной строке), 256K контекст. Подводные камни: tool-calling reliability на 142k вызовах нужно проверять pilot'ом; русские карты в DeepInfra не работают напрямую — нужен посредник.

**Не рекомендую как основную: C2 (self-host)**.
Причины: 5–10 дней setup + отладка отнимут критический ресурс (время до защиты). Self-host имеет смысл только если (1) есть требование по инфраструктуре, (2) есть свободный H100/A100 в вузе, (3) будущие эксперименты требуют контроля. Yandex.Cloud on-demand — **не рентабельно** в 5–10× по сравнению с RunPod/Vast.ai.

**Гибридная стратегия (если позволяет время):**
1. Разработка + pilot на **gpt-4o-mini** (стандарт, быстрый цикл отладки)
2. **E1–E2** основной грид на **Qwen3.5-397B (DeepInfra)** — экономия 80%
3. **Confirmation runs** (~1800) на **gpt-4o-mini** — подтверждение, что выводы не зависят от выбора модели
4. Sensitivity check: 100 задач параллельно прогнать на обеих моделях, проверить ранжирование топологий

Итог: **$60–80** на весь эксперимент + усиление научной аргументации за счёт cross-model validation.

## 6. Решения, которые нужно принять

- [ ] Основной провайдер: OpenAI / Z.ai / DeepInfra / self-host?
- [ ] Готов ли делать cross-model validation (gpt-4o-mini + Qwen3.5-397B)?
- [ ] Есть ли у вуза бесплатный доступ к OpenAI / компьютерным кластерам?
- [ ] Оплата из РФ: какая иностранная карта доступна?
- [ ] Если self-host — RunPod или Yandex.Cloud (compliance vs цена)?

## 7. Источники

**OpenAI:** openai.com/api/pricing, developers.openai.com/api/docs/deprecations, retiring-gpt-4o page
**Z.ai:** docs.z.ai/guides/overview/pricing, z.ai/subscribe, pricepertoken.com/provider/z-ai
**Qwen/DeepSeek/OSS:** deepinfra.com/pricing, openrouter.ai, together.ai/pricing, groq.com/pricing, api-docs.deepseek.com, alibabacloud.com/help/en/model-studio
**Yandex.Cloud:** cloud.yandex.ru/docs/compute/pricing, cloud.yandex.ru/docs/compute/concepts/gpus, yandex.cloud/ru/all-offers, opensource.yandex/grants
**Self-host альтернативы:** runpod.io/pricing, vast.ai, immers.cloud/prices, lambda.ai
**Бенчмарки:** artificialanalysis.ai, livebench.ai, databasemart.com/blog (vLLM GPU benchmarks)
