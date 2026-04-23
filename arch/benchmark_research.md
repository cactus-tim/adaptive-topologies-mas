# Benchmark research — adaptive topologies MAS

Краткая сводка веб-ресёрча по выбору бенчмарка для эксперимента (6 топологий × 5 ролей × 4 типа задач × 3 seed ≈ 360+ runs, GPT-4o-mini primary, малый бюджет, автоматическая оценка).

## 1. Что берут в литературе

| Работа | Программирование | Reasoning | Decision | Creative |
|---|---|---|---|---|
| G-Designer (2410.11782) | HumanEval | MMLU, GSM8K, MultiArith, SVAMP, AQuA | — | — |
| DyLAN (2310.02170) | HumanEval | MMLU, GSM8K, MATH | WebShop (50) | — |
| MASS / Multi-Agent Design (2502.02533) | MBPP, HumanEval, LiveCodeBench | MATH, DROP, HotpotQA, MuSiQue | — | — |
| MARBLE / MultiAgentBench (2503.01935) | — | — | Research/Poker/Negotiate (milestone KPI) | — |
| AgentBench (2308.03688) | — | — | OS/DB/KG/WebShop/WebArena (heavy, Docker) | — |
| MINT (ICLR'24) | HumanEval/MBPP | MATH/MMLU subsets | AlfWorld, WebShop | — |

Вывод: стандарт де-факто для MAS-топологий = **HumanEval + MMLU + GSM8K**. Creative в MAS-литературе почти не встречается — берут либо LLM-judge, либо структурированные NLG.

## 2. Сравнение по стоимости (GPT-4o-mini)

### Programming
| Датасет | N | ~tokens/prompt | Оценка | Contamination | Коммент |
|---|---|---|---|---|---|
| MBPP (sanitized) | 500 | 150–250 | unit tests | умеренная | самый дешёвый |
| HumanEval | 164 | 300–400 | unit tests | **высокая (8–18%)** | стандарт литературы |
| HumanEval+ (EvalPlus) | 164 | 300–400 | unit tests ×80 | снижена | против inflate |
| LiveCodeBench | 400+ rolling | 600–1500 | unit tests | минимальная | contamination-free |
| APPS | 10k | 1000–5000 | unit tests | умеренная | слишком длинно |

### Reasoning / Analysis
| Датасет | N | ~tokens | Оценка | Коммент |
|---|---|---|---|---|
| GSM8K | 1319 test | 150–200 | exact match | cheapest reasoning |
| MMLU | 14042 | 200–400 | MCQ exact | стандарт, но GPT-4o saturated |
| MMLU-Pro (NeurIPS'24) | 12032 | 300–600 | 10-MCQ exact | замена MMLU |
| BBH | 6511 | 400–800 | exact/partial | нужен CoT → дороже |
| GPQA Diamond | 198 | 400–600 | MCQ exact | PhD-level, anti-contamination |
| ARC-C | 1172 | 150–250 | MCQ exact | saturated |

### Creative / NLG
| Датасет | N | Оценка | Коммент |
|---|---|---|---|
| CommonGen | 35k | ROUGE + concept coverage | 100% auto, но ROUGE шум |
| WritingBench (2503.05244) | 1239 | fine-tuned critic | свежий, 6 доменов |
| EQ-Bench creative | 171 | LLM-judge rubric | community, не academic |
| Кастомные 10–15 prompts | — | LLM-judge | гибко, но дорого при judge |

### Decision / Planning / Data analysis
| Датасет | N | Оценка | Коммент |
|---|---|---|---|
| InfiAgent-DABench (ICLR'25) | 311 | numeric exact match | **единственный lightweight data-analysis**; нужен Python sandbox |
| WebShop | 1000+ | success rate | как в DyLAN, 50 эпизодов; нужен локальный сервер |
| TravelPlanner | 1225 | multi-constraint | GPT-4 solves 0.6% → нецелесообразен |
| GAIA | 465 | exact | требует web-browsing, дорогой |
| AgentBench subtasks | — | success | Docker, тяжёлый |

## 3. Четыре варианта task-mix

### A. Минимальный бюджет (~$30–80 на 360 runs) — полный автомат
- MBPP (30) + GSM8K (30) + CommonGen (30) + InfiAgent-DABench numeric (20)
- все 4 типа, 100% auto-eval, нет LLM-judge
- минус: HumanEval отсутствует → частичная сопоставимость с G-Designer/DyLAN

### B. Академически признанный (~$80–200) — прямо сопоставим с литературой
- HumanEval+ (50) + MMLU-Pro (50) + GSM8K (30) + CommonGen (30) + 10 curated prompts + WebShop (50)
- минус: WebShop multi-turn → дороже; нужен сервер

### C. Contamination-free (~$120–300)
- LiveCodeBench (30) + GPQA Diamond (40) + WritingBench subset (20) + InfiAgent-DABench (30)
- минус: длинные промпты у LiveCodeBench/DABench

### D. Topology-focused (рекомендация) — ~$50–60 total
| Тип | Датасет | N | Оценка |
|---|---|---|---|
| Programming | HumanEval (или +) | 50 | unit tests |
| Reasoning | GSM8K | 50 | exact match |
| Creative/NLG | CommonGen | 50 | ROUGE + concept coverage |
| Decision | InfiAgent-DABench (numeric subset) | 30 | numeric exact |

Оценка: 360 runs × 30 задач × 15 LLM-calls × ~$0.00024 ≈ $39 + overhead ≈ **~$50–60**.

## 4. Подводные камни

- **Contamination**: HumanEval и MMLU высокое — inflate реален у GPT-4o; для сравнения *топологий* это OK (относительные различия), но указать в limitations. Или взять HumanEval+ / MMLU-Pro / LiveCodeBench.
- **LLM-judge**: position bias + self-enhancement; на 360 runs при judge-on-every-task = $200–1000 extra. Правило — judge не более чем на 20% задач, остальное — автоматика.
- **Контейнеризация**: AgentBench, WebArena, TravelPlanner требуют Docker/серверов. InfiAgent-DABench — Python sandbox встроен. HumanEval/MBPP/GSM8K/MMLU/CommonGen — нулевая инфра.
- **Saturation**: ARC-C, MMLU, GSM8K у GPT-4o уже ≈90%+; low ceiling, но разница топологий всё равно различима. MMLU-Pro, GPQA дают больше разрешения.
- **Creative — нет стандарта**: любая оценка субъективна. Безопаснее свести creative к structural (coverage концептов, наличие N required elements) → exact match.

## 5. Итоговая рекомендация для диплома

**Вариант D** как baseline с возможностью апгрейда до B. Обоснование:
1. Покрытие 4 типов задач из методологии (program / reason / creative-proxy / decision).
2. Сопоставимость с G-Designer и DyLAN (HumanEval, GSM8K пересекаются).
3. 100% автоматическая оценка → бюджет на inference, не на judge.
4. Нулевая инфраструктура за исключением Python sandbox для DABench (и он уже нужен для code execution в HumanEval).
5. Укладывается в ~$60 на полный грид. Ablation с бОльшей моделью (GPT-4o вместо mini) — опционально в конце.

**Risks & mitigations**: contamination → указать в limitations; saturation → дополнить 5–10 GPQA/MMLU-Pro задач как hard subset; creative proxy — приложить 10 open-ended prompts с LLM-judge как качественный supplement только на финальном confirmatory run.

## 6. Ссылки

- G-Designer: https://arxiv.org/abs/2410.11782
- DyLAN: https://arxiv.org/abs/2310.02170
- MASS: https://arxiv.org/html/2502.02533v1
- MultiAgentBench / MARBLE: https://arxiv.org/abs/2503.01935 · https://github.com/ulab-uiuc/MARBLE
- AgentBench: https://arxiv.org/abs/2308.03688 · https://github.com/THUDM/AgentBench
- MINT: https://github.com/xingyaoww/mint-bench
- Scaling Agent Systems (Dec'25): https://arxiv.org/html/2512.08296v1
- InfiAgent-DABench: https://arxiv.org/html/2401.05507v1
- WritingBench: https://arxiv.org/html/2503.05244v1
- TravelPlanner: https://arxiv.org/abs/2402.01622
- Contamination survey: https://arxiv.org/html/2406.04244v1
- HumanEval+/MBPP+ (EvalPlus): https://arxiv.org/abs/2412.21199
- CommonGen: https://inklab.usc.edu/CommonGen/
