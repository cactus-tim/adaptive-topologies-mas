# M10 Resync — Контекст

## ПРОГРЕСС СЕССИИ (2026-05-12)

### ВЫПОЛНЕНО
- Шаг 1.1: Миграция `TaskSpec.type` Literal enum (`programming|qa|creative|analysis` → `programming|reasoning|creative|decision`) в `src/atm/core/types.py`
- Шаг 1.2: Удаление creative/analysis/mmlu модулей, конфигов, фикстур, тестов; очистка `src/atm/tasks/__init__.py` (атомарный коммит совместно с 1.1)
- Шаг 1.3: Переименование enum-ключей `"qa"` → `"reasoning"`, `"analysis"` → `"decision"` в `tests/fixtures/oracle_table.json`
- Шаг 2.1: GSM8K загрузчик + numeric-match оценщик + 7 тестов (7/7 зелёных); создан `src/atm/tasks/gsm8k.py`, `tests/unit/tasks/test_gsm8k.py`, `tests/fixtures/tasks/gsm8k_sample.json`; guarded import добавлен в `__init__.py`
- Шаг 2.2: CommonGen загрузчик + in-house ROUGE-L + concept-coverage оценщик + 9 тестов (9/9 зелёных); создан `src/atm/tasks/commongen.py`, `tests/unit/tasks/test_commongen.py`, `tests/fixtures/tasks/commongen_sample.json`; guarded import `commongen` добавлен в `__init__.py`; mypy --strict + ruff clean
- Шаг 2.3: DABench загрузчик + dabench_numeric_exact оценщик + curated fallback + 11 тестов (11/11 зелёных); создан `src/atm/tasks/dabench.py`, `tests/unit/tasks/test_dabench.py`, `tests/fixtures/tasks/dabench_curated.jsonl` (8 строк, включая multi-pair entries); guarded import `dabench` добавлен в `__init__.py` (алфавитный порядок: commongen, dabench, gsm8k, humaneval); mypy --strict + ruff clean

- Шаг 3.2: `_make_task_spec` в `tests/unit/tasks/test_base.py` — `type="qa"` → `type="reasoning"`; 13/13 тестов зелёных

### В РАБОТЕ
- Шаг 3.1: Аудит + финализация `__init__.py` + grep

### БЛОКЕРЫ
- Нет

---

## Быстрый старт

1. Прочитай этот файл
2. Проверь `m10-resync-tasks.md` — что делать следующим
3. Прочитай `m10-resync-plan.md` Волна 1 для стратегии
4. Начни с: **шаг 1.1** — `src/atm/core/types.py` строка 269, swap Literal enum

**Критическое замечание по Волне 1:** шаги 1.1 и 1.2 должны попасть в **один коммит**. После миграции enum (1.1) `mmlu.py`, `creative.py`, `analysis.py` становятся невалидны при импорте (`type="qa"` → ValidationError) — поэтому их удаление (1.2) нельзя откладывать. Не запускай тестовый suite между этими шагами.

---

## Ключевые файлы

**`src/atm/core/types.py`**
- Роль: Pydantic-модель `TaskSpec`, frozen; `type` — `Literal[...]`
- Плановое изменение: строка 269, заменить `["programming", "qa", "creative", "analysis"]` на `["programming", "reasoning", "creative", "decision"]`
- Статус: НЕ НАЧАТО

**`src/atm/tasks/__init__.py`**
- Роль: защищённые `importlib.import_module` для регистрации загрузчиков
- Плановое изменение: убрать 3 старых импорта (`mmlu`, `creative`, `analysis`), добавить 3 новых (`gsm8k`, `commongen`, `dabench`); финальный вид — 4 guarded import в алфавитном порядке
- Статус: НЕ НАЧАТО

**`src/atm/tasks/gsm8k.py`** (новый)
- Роль: `GSM8KLoader(name="gsm8k")` + `GSM8KMatcher(name="gsm8k_numeric")`
- Плановое изменение: создать; dataset `openai/gsm8k`, split `test`, `_extract_final_number` по `#### N`
- Статус: НЕ НАЧАТО

**`src/atm/tasks/commongen.py`** (новый)
- Роль: `CommonGenLoader(name="commongen")` + `CommonGenEvaluator(name="commongen_rouge_coverage")` + in-house `_rouge_l`
- Плановое изменение: создать; dataset `allenai/common_gen`, split `validation` (у `test` пустые `target`)
- Статус: ВЫПОЛНЕНО

**`src/atm/tasks/dabench.py`** (новый)
- Роль: `DABenchLoader(name="dabench")` + `DABenchEvaluator(name="dabench_numeric_exact")`
- Плановое изменение: создать; pinned SHA `6ad4a487a3968682cdcbb9ae24664e680f8981a6`; offline fallback через `ATM_DABENCH_OFFLINE=1`; без LLM-judge
- Статус: НЕ НАЧАТО

**`src/atm/tasks/{mmlu,creative,analysis}.py`**
- Роль: старые загрузчики (удаляются)
- Плановое изменение: `git rm`
- Статус: НЕ НАЧАТО

**`conf/tasks/{creative,analysis}_prompts.yaml`**
- Роль: YAML prompt-корпуса (удаляются)
- Плановое изменение: `git rm`
- Статус: НЕ НАЧАТО

**`tests/fixtures/oracle_table.json`**
- Роль: фикстура для `topology_router.py:394-401`, ключи `by_task_type`
- Плановое изменение: строки 8 и 18 — переименовать ключи `"qa"` → `"reasoning"`, `"analysis"` → `"decision"`
- Статус: НЕ НАЧАТО

**`tests/fixtures/tasks/dabench_curated.jsonl`** (новый)
- Роль: 8 рукописных записей в post-join форме; offline CI fallback
- Плановое изменение: создать; минимум 1 запись с двумя `@name[value]` парами в отсортированном порядке (N1)
- Статус: НЕ НАЧАТО

**`tests/fixtures/tasks/gsm8k_sample.json`** (новый)
- Роль: 3-5 GSM8K записей для мок-тестов загрузчика
- Статус: НЕ НАЧАТО

**`tests/fixtures/tasks/commongen_sample.json`** (новый)
- Роль: 5 CommonGen записей (2 с одинаковым `concept_set_idx=0`) для теста агрегации references
- Статус: НЕ НАЧАТО

**`tests/unit/tasks/test_gsm8k.py`** (новый)
- Роль: 7 TDD тестов GSM8K
- Статус: НЕ НАЧАТО

**`tests/unit/tasks/test_commongen.py`** (новый)
- Роль: 9 TDD тестов CommonGen (включая Parquet round-trip для list-typed metadata)
- Статус: НЕ НАЧАТО

**`tests/unit/tasks/test_dabench.py`** (новый)
- Роль: 11 TDD тестов DABench
- Статус: НЕ НАЧАТО

**`tests/unit/tasks/test_base.py`**
- Роль: базовые тесты registry; содержит `_make_task_spec` с `type="qa"`
- Плановое изменение: строка 37, `"qa"` → `"reasoning"`
- Статус: НЕ НАЧАТО

**`tests/unit/tasks/test_registry_smoke.py`**
- Роль: exit criterion M10 resync; проверяет регистрацию всех 4 загрузчиков
- Плановое изменение: переписать для `("humaneval", "gsm8k", "commongen", "dabench")`; `TASKS._cache.pop` для `commongen`, `dabench`, `gsm8k` (N3)
- Статус: НЕ НАЧАТО

**`tests/unit/phases/test_topology_router.py`**
- Роль: тесты роутера топологий
- Плановое изменение: аудит grep; переименование только при наличии совпадений
- Статус: НЕ НАЧАТО

**`dev/codebase-map.md`**
- Роль: карта кодовой базы, секция Tasks & Evaluation
- Плановое изменение: переписать секцию; отразить `oracle_table.json` ключи (N4 — если файл упоминается); дата 2026-05-12
- Статус: НЕ НАЧАТО

---

## Решения

### In-house ROUGE-L (CommonGen)
- Решение: реализовать LCS-based F1 (~25 строк) без новой зависимости
- Обоснование: `rouge-score` добавляет ~5 MB транзитивных (`absl-py`, `nltk` punkt); CommonGen задачи короткие, пробельная токенизация достаточна; детерминировано; без внешнего `nltk`
- Альтернатива: `rouge-score>=0.1.2` + один вызов `RougeScorer` — если нужен референсный паритет

### `common_answers` сериализация (DABench)
- Решение: `" ".join(f"@{name}[{value}]" for name, value in sorted(pairs))` — сортировка по `name`
- Обоснование: `TaskSpec.expected: str | None` (frozen Pydantic), список нельзя хранить напрямую; сортировка обеспечивает детерминизм при повторных загрузках

### DABench: pinned URL + curated fallback (не vendoring)
- Решение: закрепить GitHub raw URL по SHA; fallback — `dabench_curated.jsonl` в репозитории
- Обоснование: vendoring (~50 KB) требует лицензионного заголовка; curated fallback уже даёт offline-детерминизм CI; production-запуски тянут канонический upstream
- Если потребуется vendoring: удалить remote-fetch ветку, добавить JSONL под `tests/fixtures/tasks/dabench/`

### Substring-matching для концептов CommonGen
- Решение: `concept.lower() in answer.lower()` (подстрока, не word-boundary)
- Обоснование: принимает морфологические варианты (`"ski"` → `"skiing"`); соответствует CommonGen-Lite baseline; явно задокументировано в module docstring `CommonGenEvaluator`

### MMLU — удалить полностью
- Решение: удалить, не сохранять как opt-in ablation loader
- Обоснование: минимальный sync с архом; реинтродукция — однофайловая задача в M12+; подтверждено директивой пользователя "не спрашивать подтверждения, делаем A+B"

### DABench numeric tolerance
- Решение: `abs_tol=1e-2` (фиксированный)
- Обоснование: соответствует "Round to 2 decimals" в большинстве `constraints` DABench; динамический допуск из `constraints` — M11+

### Интеграционные тесты
- Решение: не добавлять
- Обоснование: внутренний рефакторинг; публичный контракт не меняется; unit тесты + smoke достаточны

---

## Ограничения

- **Волна 2 строго последовательна** — Steps 3→4→5 нельзя параллелить: все три редактируют `src/atm/tasks/__init__.py`
- **Волна 1 атомарна** — 1.1 + 1.2 в одном коммите; иначе CI сломан между ними
- **`filterwarnings = ["error"]`** в pytest конфиге — любое deprecation warning от `datasets` ломает suite; если возникнет, добавить точный `ignore` в `pyproject.toml`, не заглушать в исходниках
- **`dev/done/m10/`** — не трогать; это история
- **`_judge.py`** — остаётся, даже если не используется после ресинха
- **`TaskSpec.expected: str | None`** — строго строка; никаких list/dict в поле `expected`

---

## Поправки при создании (USER_AMENDMENTS)

Рекомендации N1–N4 из `.draft-review.md` встроены в план:

- **N1** → Шаг 2.3 (Test #1 для DABench): в `dabench_curated.jsonl` минимум одна запись должна содержать две `@name[value]` пары в детерминированно отсортированном порядке; тест `test_dabench_loader_uses_curated_when_offline` проверяет это явно.
- **N2** → Шаг 2.3 (Test #10 для DABench): `test_dabench_evaluator_categorical_fallback` дополнен суб-утверждением — числовой mismatch (например, ожидаемое `"34.65"`, ответ `"@category[34.99]"`) **не** попадает в categorical fallback (потому что парсинг float успешен для обоих; сравнение идёт через `math.isclose`, не через строковое равенство).
- **N3** → Шаг 4.1 (`test_registry_smoke.py`): `TASKS._cache.pop` должен покрывать `"commongen"`, `"dabench"`, `"gsm8k"` (все три новых загрузчика) перед sampling-тестами.
- **N4** → Шаг 5.1 (`dev/codebase-map.md`): если файл упоминает `oracle_table.json` как фикстуру топологического роутера, добавить однострочную заметку о переименовании ключей `"qa"` → `"reasoning"` и `"analysis"` → `"decision"`.
