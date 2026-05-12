# M10 Resync — Задачи

## Волна 1: Enum + удаление устаревшего кода (атомарная) — В РАБОТЕ

> **Критично:** Шаги 1.1 и 1.2 должны попасть в ОДИН коммит. После 1.1 `mmlu.py`/`creative.py`/`analysis.py` невалидны при импорте. Не запускай suite между ними.

- [x] 1.1 Мигрировать `TaskSpec.type` Literal enum — `src/atm/core/types.py`
  - Тип: simple
  - Зависит от: ничего
  - Параллельно с: ничем (все остальные шаги зависят от этого)
  - Принятие: `TaskSpec(type="reasoning")` OK; `type="qa"` → `ValidationError`
  - Имплементация: заменить строку 269 — `Literal["programming", "qa", "creative", "analysis"]` → `Literal["programming", "reasoning", "creative", "decision"]`; не трогать ничего больше в файле

- [x] 1.2 Удалить creative/analysis/mmlu модули, конфиги, фикстуры, тесты — `src/atm/tasks/__init__.py` + 10 файлов `git rm`
  - Тип: simple
  - Зависит от: 1.1 (атомарно, один коммит)
  - Параллельно с: ничем (редактирует `__init__.py`, с которым конфликтуют шаги 2.1–2.3)
  - Принятие: `python -c "import atm.tasks; print(sorted(atm.tasks.TASKS._registry))"` → `['humaneval']`; pytest собирает только `test_base.py`, `test_cache.py`, `test_humaneval.py`, `test_registry_smoke.py`
  - Имплементация:
    - `git rm src/atm/tasks/{creative,analysis,mmlu}.py`
    - `git rm conf/tasks/{creative,analysis}_prompts.yaml`
    - `git rm tests/unit/tasks/{test_creative,test_analysis,test_mmlu}.py`
    - `git rm tests/fixtures/tasks/mmlu_sample.json`
    - `git rm tests/fixtures/llm/m10_judge_{creative,analysis}.yaml`
    - В `__init__.py`: убрать строки guarded import для `mmlu`, `creative`, `analysis`
    - Перед удалением: grep для проверки отсутствия висящих импортов вне `dev/done/m10/`: `grep -rn "from atm\.tasks\.\(creative\|analysis\|mmlu\)\|CreativeLoader\|AnalysisLoader\|MMLULoader" src/ tests/ conf/ --exclude-dir=done`
    - Оставить `conf/tasks/.gitkeep` и `tests/fixtures/tasks/.gitkeep`

- [ ] 1.3 Переименовать стале enum-ключи в oracle + аудит router — `tests/fixtures/oracle_table.json`, `tests/unit/phases/test_topology_router.py`
  - Тип: simple
  - Зависит от: 1.1
  - Параллельно с: 1.2 (разные файлы — `oracle_table.json` и `test_topology_router.py` не трогаются в 1.2)
  - Принятие: `pytest tests/unit/phases/test_topology_router.py -q` зелёный; `grep -n '"qa"\|"analysis"' tests/` — ноль неожиданных совпадений
  - Имплементация:
    - `oracle_table.json` строки 8 и 18: ключи `"qa"` → `"reasoning"`, `"analysis"` → `"decision"` (значения не менять)
    - Аудит: `grep -n '"qa"\|"analysis"' tests/unit/phases/test_topology_router.py`; если есть совпадения — переименовать in-place
    - Запустить: `pytest tests/unit/phases/test_topology_router.py -q`

---

## Волна 2: GSM8K → CommonGen → DABench (TDD, строго последовательно) — НЕ НАЧАТО

> Шаги 2.1, 2.2, 2.3 нельзя параллелить: все три редактируют `src/atm/tasks/__init__.py`. Следуй порядку.

- [ ] 2.1 Добавить GSM8K загрузчик + numeric-match оценщик + 7 тестов — `src/atm/tasks/gsm8k.py`, `tests/unit/tasks/test_gsm8k.py`, `tests/fixtures/tasks/gsm8k_sample.json`, `__init__.py`
  - Тип: tdd
  - Зависит от: 1.1, 1.2
  - Параллельно с: ничем (редактирует `__init__.py`)
  - Принятие: 7 тестов зелёных; `atm.tasks.TASKS._registry['gsm8k']` существует; `EVALUATORS._registry['gsm8k_numeric']` существует
  - Имплементация (красная фаза TDD сначала):
    - Fixture `gsm8k_sample.json`: 3-5 записей `{"question": "...", "answer": "... #### N"}`
    - `gsm8k.py`: `GSM8KLoader` (dataset `openai/gsm8k`, split `test`, cache key `"gsm8k"`); `_extract_final_number` — regex `r"####\s*(-?[0-9][\d,]*(?:\.\d+)?)"`, strip commas; `GSM8KMatcher` — последнее число в ответе, `math.isclose(abs_tol=1e-6)`, при parse failure: `error="no number found"`
    - `TaskSpec(id=f"gsm8k/{idx}", type="reasoning", evaluator_key="gsm8k_numeric", ...)`
    - 7 тестов: #1 loader-via-dataset, #2 cache-hit, #3 exact-match, #4 comma-formatted (1,234 == 1234), #5 float-tolerance (72.0 == 72), #6 wrong-answer, #7 no-number-in-answer
    - `__init__.py`: добавить `importlib.import_module("atm.tasks.gsm8k")` (guarded)

- [ ] 2.2 Добавить CommonGen загрузчик + in-house ROUGE-L + concept-coverage оценщик + 9 тестов — `src/atm/tasks/commongen.py`, `tests/unit/tasks/test_commongen.py`, `tests/fixtures/tasks/commongen_sample.json`, `__init__.py`
  - Тип: tdd
  - Зависит от: 1.1, 1.2, 2.1 (сериализация `__init__.py`)
  - Параллельно с: ничем
  - Принятие: 9 тестов зелёных; Parquet round-trip для list-typed metadata пройден
  - Имплементация:
    - Fixture `commongen_sample.json`: 5 записей, 2 из них с `concept_set_idx=0` (тест агрегации)
    - `commongen.py`: `CommonGenLoader` (dataset `allenai/common_gen`, split `validation`, cache key `"commongen"`); группировка по `concept_set_idx` → `metadata["references"]` = все `target`, `metadata["concepts"]` = список концептов; `input = f"concepts: {', '.join(concepts)}"`;  `TaskSpec(id=f"commongen/{idx}", type="creative", evaluator_key="commongen_rouge_coverage", ...)`
    - `_rouge_l(pred, ref)`: LCS DP, whitespace tokenization, lowercase + strip punctuation, F1; документировать формулу в docstring
    - `CommonGenEvaluator`: `score_rouge = max ROUGE-L по references`; `score_coverage = # concept как подстрока / len(concepts)` (если concepts пустой → 1.0); `final = 0.5*rouge + 0.5*coverage`; `passed = final >= 0.5`
    - 9 тестов: #1 loader-via-dataset (агрегация references), #2 cache-hit, #3 rouge-exact, #4 rouge-partial (precomputed F1), #5 rouge-disjoint==0, #6 evaluator-full-match, #7 evaluator-partial-coverage, #8 empty-concepts-vacuous, #9 **Parquet round-trip** (`write_cache` → `read_cache`, list equality для `concepts` и `references`)
    - `__init__.py`: добавить guarded import для `commongen`

- [ ] 2.3 Добавить DABench загрузчик + dabench_numeric_exact оценщик + curated fallback + 11 тестов — `src/atm/tasks/dabench.py`, `tests/unit/tasks/test_dabench.py`, `tests/fixtures/tasks/dabench_curated.jsonl`, `__init__.py`
  - Тип: tdd
  - Зависит от: 1.1, 1.2, 2.1, 2.2 (сериализация `__init__.py`)
  - Параллельно с: ничем
  - Принятие: 11 тестов зелёных; `ATM_DABENCH_OFFLINE=1 python -c "import atm.tasks; ..."` возвращает ≥ 8 specs без сети
  - Имплементация:
    - Pinned URLs: questions `https://raw.githubusercontent.com/InfiAgent/InfiAgent/6ad4a487a3968682cdcbb9ae24664e680f8981a6/examples/DA-Agent/data/da-dev-questions.jsonl`, labels аналогично; SHA `6ad4a487a3968682cdcbb9ae24664e680f8981a6`; константы `_DABENCH_QUESTIONS_URL`, `_DABENCH_LABELS_URL`, `_DABENCH_COMMIT_SHA` в module scope
    - Fixture `dabench_curated.jsonl`: 8 строк в post-join форме (`id`, `question`, `expected`, `concepts`, `constraints`, `format`, `level`, `file_name`); **минимум одна запись с двумя `@name[value]` парами в отсортированном порядке** (N1: например `"@a[1.0] @b[2.5]"`)
    - `_serialise_common_answers(pairs)`: `" ".join(f"@{n}[{v}]" for n, v in sorted(pairs, key=lambda p: p[0]))`
    - Loader flow: (1) cache hit → return; (2) `ATM_DABENCH_OFFLINE=1` → curated JSONL; (3) try remote via `datasets.load_dataset("json", data_files=...)`, join на `id`; (4) on `(OSError, ConnectionError, URLError)` → fallback + structlog warning; (5) сериализация + write_cache
    - `TaskSpec(id=f"dabench/{q['id']}", type="decision", evaluator_key="dabench_numeric_exact", ...)`; metadata: `concepts`, `constraints`, `format`, `level`, `file_name`
    - `DABenchEvaluator`: parse expected через `r"@([A-Za-z_][\w]*)\[([^\]]+)\]"` → список пар; для каждой пары — per-pair regex в ответе → numeric compare `math.isclose(abs_tol=1e-2)` → при ValueError — строковый fallback (case-insensitive); `score = correct/total`; `passed = score==1.0`
    - 11 тестов: #1 **offline-curated** (assert ≥8 specs, проверить хотя бы одну запись с двумя парами в отсортированном порядке — N1); #2 remote-serialises-common-answers; #3 remote-multi-pair-sorted; #4 falls-back-on-network-error; #5 cache-hit; #6 full-match-numeric; #7 close-within-abs-tol (34.65/34.66 → pass; 34.65/34.67 → fail); #8 multi-pair-partial (score==0.5); #9 missing-template → score==0; #10 **categorical-fallback** (ожидаемое `"@category[YES]"`, ответ `"@category[yes]"`) + суб-утверждение: `math.isclose(float("YES"), ...)` бросает ValueError (т.е. категориальный путь активирован, а не numeric — N2); #11 vacuous-no-expected-pairs
    - `__init__.py`: добавить guarded import для `dabench`

---

## Волна 3: Аудит + правка test_base.py (параллельно) — НЕ НАЧАТО

- [ ] 3.1 Аудит humaneval.py, финальная консолидация `__init__.py`, расширенный grep — `src/atm/tasks/__init__.py`, `src/atm/tasks/humaneval.py` (только чтение), `tests/`, `dev/`
  - Тип: simple
  - Зависит от: 2.1, 2.2, 2.3
  - Параллельно с: 3.2 (разные файлы)
  - Принятие: `sorted(TASKS._registry)` == `['commongen', 'dabench', 'gsm8k', 'humaneval']`; grep `'"(qa|analysis)"'` в `src/atm/tasks/`, `tests/`, `dev/` (кроме `dev/done/m10/`) — ноль совпадений
  - Имплементация:
    - Проверить и при необходимости сортировать 4 guarded imports в `__init__.py` в алфавитном порядке: `commongen`, `dabench`, `gsm8k`, `humaneval`
    - `grep -nE '"(qa|analysis)"' src/atm/tasks/` → ноль
    - `grep -nE '"(qa|analysis)"' tests/` → только ожидаемые файлы (уже обработаны в 1.2, 1.3, 3.2)
    - `grep -nrE '"(qa|analysis)"' dev/ --exclude-dir=done` → ноль
    - Обновить docstring модуля `__init__.py` — список четырёх загрузчиков
    - Запустить: `python -c "import atm.tasks; print(sorted(atm.tasks.TASKS._registry))"`

- [ ] 3.2 Исправить `_make_task_spec` в test_base.py: `type="qa"` → `type="reasoning"` — `tests/unit/tasks/test_base.py`
  - Тип: simple
  - Зависит от: 1.1
  - Параллельно с: 3.1 (разные файлы)
  - Принятие: `pytest tests/unit/tasks/test_base.py -q` зелёный
  - Имплементация: строка 37, единственная замена `type="qa"` → `type="reasoning"`; id-строки вида `"qa/..."` НЕ трогать

---

## Волна 4: Smoke test (exit criterion) — НЕ НАЧАТО

- [ ] 4.1 Переписать `test_registry_smoke.py` для нового четырёхзадачного mix — `tests/unit/tasks/test_registry_smoke.py`
  - Тип: tdd
  - Зависит от: 1.1, 1.2, 2.1, 2.2, 2.3, **3.1** (финализированный `__init__.py`)
  - Параллельно с: ничем
  - Принятие: `pytest tests/unit/tasks/test_registry_smoke.py -q` зелёный; `pytest tests/unit/tasks/ -q` — все задачные тесты зелёные
  - Имплементация:
    - `test_all_loaders_registered`: итерация по `("humaneval", "gsm8k", "commongen", "dabench")`, без `"mmlu"`, `"creative"`, `"analysis"`
    - Оставить существующие 4 humaneval-sampling теста verbatim
    - Добавить 3 новых sampling теста для `gsm8k`, `commongen`, `dabench` — каждый monkeypatches `loader.load` → 20 fake specs, затем `TASKS.sample(name, n=10, seed=42)` → 10 specs нужного `type`/`evaluator_key`
    - Helpers: `_make_fake_gsm8k_specs`, `_make_fake_commongen_specs`, `_make_fake_dabench_specs` — 20 specs каждый с монотонными id
    - **N3:** `TASKS._cache.pop("gsm8k", None)`, `TASKS._cache.pop("commongen", None)`, `TASKS._cache.pop("dabench", None)` перед каждым sampling тестом соответствующего загрузчика

---

## Волна 5: Документация — НЕ НАЧАТО

- [ ] 5.1 Обновить секцию Tasks & Evaluation в `dev/codebase-map.md` — `dev/codebase-map.md`
  - Тип: simple
  - Зависит от: 3.1, 3.2, 4.1 (финальное кол-во тестов должно быть стабильным)
  - Параллельно с: ничем (последний шаг)
  - Принятие: в секции Tasks & Evaluation нет строк `creative`, `analysis`, `mmlu`; дата `Last updated` = 2026-05-12
  - Имплементация:
    - Перед правкой запустить `pytest -q` и зафиксировать итоговое кол-во тестов
    - Загрузчики: `humaneval`, `gsm8k`, `commongen`, `dabench`
    - Оценщики: `humaneval_pytest`, `gsm8k_numeric`, `commongen_rouge_coverage`, `dabench_numeric_exact`
    - Submodule map: `humaneval.py`, `gsm8k.py`, `commongen.py`, `dabench.py`, `_cache.py`, `_judge.py` (retained but unused), `base.py`
    - Добавить заметку о DABench: pinned SHA + offline fallback + `@name[value]` numeric-exact evaluator
    - Добавить заметку о ROUGE-L in-house (без `rouge-score` dep)
    - Убрать упоминания `creative.py`, `analysis.py`, `mmlu.py`, YAML-промпт файлов, `m10_judge_{creative,analysis}.yaml`
    - **N4:** если файл упоминает `oracle_table.json`, добавить однострочную заметку о переименовании ключей `"qa"` → `"reasoning"`, `"analysis"` → `"decision"` в `by_task_type`
    - Обновить строку с кол-вом тестов в `## Test Setup`
    - Обновить `Last updated` → 2026-05-12

---

## Статистика

- Всего задач: 11 (1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 5.1)
- Новых тестов: 7 (GSM8K) + 9 (CommonGen) + 11 (DABench) + 3 (smoke sampling) = **30 новых тестов**
- Оценка: ~6h
- Выполнено: 2 / 11

## Как обновлять

После завершения каждого шага:
1. Отметь задачу `[x]` в этом файле
2. Обнови заголовок волны: все задачи done → `ВЫПОЛНЕНО`, часть done → `В РАБОТЕ`
3. Обнови `SESSION PROGRESS` в `m10-resync-context.md`
4. Обнови статистику Done
