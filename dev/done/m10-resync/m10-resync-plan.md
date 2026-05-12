# M10 Resync: Миграция Task-Mix на programming/reasoning/creative/decision — План

## Краткое резюме

Ресинхронизация уже реализованного M10 (HumanEval / MMLU / Creative / Analysis) с обновлённым `arch/PLAN.md` §M10 (аудит G1, PR#11) и `arch/experiment_plan.md` §0.

Канонический task-mix: **HumanEval (programming) / GSM8K (reasoning) / CommonGen (creative) / InfiAgent-DABench (decision)**.

Работа состоит из двух связанных частей:

**A.** Мигрировать `TaskSpec.type` enum с `programming | qa | creative | analysis` на `programming | reasoning | creative | decision`.

**B.** Заменить четыре загрузчика/оценщика: HumanEval — без изменений; MMLU → GSM8K; Creative (YAML-промпты + LLM-judge) → CommonGen (HF dataset + ROUGE-L + покрытие концептов); Analysis (YAML CSV + гибридный судья) → InfiAgent-DABench (закреплённый удалённый JSONL + numeric-exact regex evaluator). Опциональный MMLU-загрузчик **удаляется** (не сохраняется) — упрощает дерево, уменьшает площадь поверхности; при необходимости возможно вернуть в M12+. Обновляются защищённые импорты в `tasks/__init__.py`, удаляются артефакты `creative`/`analysis`, переименовываются устаревшие значения enum в `tests/fixtures/oracle_table.json`, обновляется `dev/codebase-map.md`.

## Текущее состояние

| Компонент | Текущее (до ресинха) |
|-----------|---------------------|
| `TaskSpec.type` | `Literal["programming", "qa", "creative", "analysis"]` |
| Загрузчики | `humaneval.py`, `mmlu.py`, `creative.py`, `analysis.py` |
| Оценщики | `humaneval_pytest`, `mmlu_accuracy`, `creative_llm_judge`, `analysis_hybrid` |
| Промпт-корпуса | `conf/tasks/creative_prompts.yaml`, `conf/tasks/analysis_prompts.yaml` |
| Фикстуры | `mmlu_sample.json`, `m10_judge_creative.yaml`, `m10_judge_analysis.yaml` |
| `oracle_table.json` | Ключи `by_task_type`: `"qa"`, `"analysis"` |

## Предложенный подход

### Ключевые архитектурные решения

1. **ROUGE-L реализован in-house** — не добавляем зависимость `rouge-score` (~5 MB с транзитивными). Для CommonGen (короткие предложения, пробельная токенизация) достаточно 25-строчной LCS-реализации. Решение детерминировано, без внешнего `nltk`.

2. **MMLU удаляется полностью** — не сохраняется для ablation. Минимальный sync с архом, меньше поверхности. Возврат — однофайловое введение в M12+.

3. **DABench: закреплённый URL + curated fallback** — пиннинг к SHA `6ad4a487a3968682cdcbb9ae24664e680f8981a6`; при недоступности сети (или `ATM_DABENCH_OFFLINE=1`) используется `tests/fixtures/tasks/dabench_curated.jsonl` из репозитория. CI никогда не зависит от внешней сети.

4. **`common_answers` сериализуется в строку** — `TaskSpec.expected: str | None` (frozen Pydantic). Сериализация: `" ".join(f"@{name}[{value}]" for name, value in sorted(pairs, key=lambda p: p[0]))`. Детерминировано по имени.

5. **Substring-matching для покрытия концептов в CommonGen** — принимает морфологические варианты (`"ski"` совпадает с `"skiing"`), соответствует CommonGen-Lite baseline. Строгий word-boundary отклонён (недооценка разумных ответов).

6. **Допуск DABench numeric** — `abs_tol=1e-2` соответствует типичному "Round to 2 decimals" в `constraints`. Динамический допуск из `constraints` — будущий M11+.

7. **Интеграционные тесты не добавляются** — внутренний рефакторинг, публичный контракт (`TaskSpec`, `TaskRegistry.sample`, `Evaluator.evaluate`) не меняется.

## Фазы реализации

### Волна 1: Enum + удаление устаревшего кода (~1.5h)

**Цель:** привести типы и файловое дерево в соответствие с новым task-mix.

- [ ] 1.1 Мигрировать `TaskSpec.type` Literal enum
  - Файл: `src/atm/core/types.py` (строка 269)
  - Принятие: `TaskSpec(type="reasoning")` валидируется; `type="qa"` бросает `ValidationError`
  - **Важно:** шаги 1 и 2 должны попасть в один коммит — до удаления `mmlu.py`/`creative.py`/`analysis.py` они не импортируются

- [ ] 1.2 Удалить creative/analysis/mmlu модули, конфиги, фикстуры, тесты
  - Файлы: `src/atm/tasks/{creative,analysis,mmlu}.py`, `conf/tasks/{creative,analysis}_prompts.yaml`, `tests/unit/tasks/{test_creative,test_analysis,test_mmlu}.py`, `tests/fixtures/tasks/mmlu_sample.json`, `tests/fixtures/llm/m10_judge_{creative,analysis}.yaml`
  - Модификация: `src/atm/tasks/__init__.py` — убрать три guarded import
  - Принятие: `python -c "import atm.tasks; print(sorted(atm.tasks.TASKS._registry))"` → `['humaneval']`

- [ ] 1.3 Переименовать устаревшие enum-ключи в oracle и audit router
  - Файлы: `tests/fixtures/oracle_table.json` (ключи `"qa"` → `"reasoning"`, `"analysis"` → `"decision"`), `tests/unit/phases/test_topology_router.py` (аудит, переименование при наличии совпадений)
  - Принятие: `pytest tests/unit/phases/test_topology_router.py -q` зелёный; grep `'"qa"\|"analysis"'` в `tests/` — ноль неожиданных совпадений

### Волна 2: GSM8K → CommonGen → DABench (TDD, последовательно из-за `__init__.py`) (~3h)

**Цель:** реализовать три новых загрузчика/оценщика по TDD.

- [ ] 2.1 Добавить GSM8K загрузчик + numeric-match оценщик + тесты
  - Файлы: `src/atm/tasks/gsm8k.py` (новый), `tests/unit/tasks/test_gsm8k.py` (новый), `tests/fixtures/tasks/gsm8k_sample.json` (новый), `src/atm/tasks/__init__.py` (одна строка)
  - Принятие: 7 тестов зелёных; `atm.tasks.TASKS._registry['gsm8k']` существует

- [ ] 2.2 Добавить CommonGen загрузчик + in-house ROUGE-L + concept-coverage оценщик + тесты
  - Файлы: `src/atm/tasks/commongen.py` (новый), `tests/unit/tasks/test_commongen.py` (новый), `tests/fixtures/tasks/commongen_sample.json` (новый), `src/atm/tasks/__init__.py`
  - Принятие: 9 тестов зелёных; round-trip Parquet для list-typed metadata пройден

- [ ] 2.3 Добавить DABench загрузчик + dabench_numeric_exact оценщик + curated fallback + тесты
  - Файлы: `src/atm/tasks/dabench.py` (новый), `tests/unit/tasks/test_dabench.py` (новый), `tests/fixtures/tasks/dabench_curated.jsonl` (новый), `src/atm/tasks/__init__.py`
  - Принятие: 11 тестов зелёных; `ATM_DABENCH_OFFLINE=1` работает без сети

### Волна 3: Аудит + правка test_base.py (параллельно, разные файлы) (~0.5h)

**Цель:** устранить все остаточные `"qa"`/`"analysis"` литералы вне `dev/done/m10/`.

- [ ] 3.1 Аудит humaneval.py, финальная консолидация `__init__.py`, расширенный grep
  - Файлы: `src/atm/tasks/__init__.py` (финальный, алфавитный порядок), `src/atm/tasks/humaneval.py` (только аудит), `tests/`, `dev/` (аудит)
  - Принятие: `sorted(TASKS._registry)` == `['commongen', 'dabench', 'gsm8k', 'humaneval']`; нулевые совпадения grep вне `dev/done/m10/`

- [ ] 3.2 Исправить helper `_make_task_spec` в test_base.py: `type="qa"` → `type="reasoning"`
  - Файл: `tests/unit/tasks/test_base.py` (строка 37, одна замена)
  - Принятие: `pytest tests/unit/tasks/test_base.py -q` зелёный

### Волна 4: Smoke test (~0.5h)

**Цель:** smoke-тест как exit criterion M10 resync.

- [ ] 4.1 Переписать `test_registry_smoke.py` для нового четырёхзадачного mix
  - Файл: `tests/unit/tasks/test_registry_smoke.py`
  - Принятие: `pytest tests/unit/tasks/test_registry_smoke.py -q` зелёный; `TASKS._cache.pop` покрывает `commongen`, `dabench`, `gsm8k`
  - Зависит от шага 3.1 (финализированный `__init__.py`)

### Волна 5: Документация (~0.5h)

**Цель:** обновить codebase-map до актуального состояния.

- [ ] 5.1 Обновить секцию Tasks & Evaluation в `dev/codebase-map.md`
  - Файл: `dev/codebase-map.md`
  - Принятие: нет строк `creative`, `analysis`, `mmlu` в секции Tasks & Evaluation; дата обновления 2026-05-12
  - Примечание: если в codebase-map упоминается `oracle_table.json`, отразить переименование ключей `"qa"` → `"reasoning"`, `"analysis"` → `"decision"`

## Ключевые затронутые файлы

| Файл | Изменение | Причина |
|------|-----------|---------|
| `src/atm/core/types.py` | Один literal swap | Схема-истина для enum |
| `src/atm/tasks/__init__.py` | Удалить 3 импорта, добавить 3 | Регистрация модулей |
| `src/atm/tasks/gsm8k.py` | Новый | GSM8K загрузчик + numeric оценщик |
| `src/atm/tasks/commongen.py` | Новый | CommonGen + in-house ROUGE-L |
| `src/atm/tasks/dabench.py` | Новый | DABench + numeric-exact + offline fallback |
| `src/atm/tasks/{mmlu,creative,analysis}.py` | Удалить | Устаревший task-mix |
| `conf/tasks/{creative,analysis}_prompts.yaml` | Удалить | CommonGen/DABench заменяют YAML-корпуса |
| `tests/fixtures/oracle_table.json` | 2 ключа переименовать | Ключи `by_task_type` устарели |
| `tests/fixtures/tasks/gsm8k_sample.json` | Новый | 3-5 GSM8K записей для тестов |
| `tests/fixtures/tasks/commongen_sample.json` | Новый | 5 CommonGen записей (2 с одним concept_set_idx) |
| `tests/fixtures/tasks/dabench_curated.jsonl` | Новый | 8 записей, curated fallback, offline CI |
| `tests/fixtures/tasks/mmlu_sample.json` | Удалить | MMLU удалён |
| `tests/fixtures/llm/m10_judge_{creative,analysis}.yaml` | Удалить | LLM-judge более не нужен |
| `tests/unit/tasks/test_{gsm8k,commongen,dabench}.py` | Новые | TDD для новых загрузчиков |
| `tests/unit/tasks/test_{mmlu,creative,analysis}.py` | Удалить | Устаревшие тесты |
| `tests/unit/tasks/test_base.py` | 1 строка | `"qa"` → `"reasoning"` в helper |
| `tests/unit/tasks/test_registry_smoke.py` | Переписать | Exit criterion для нового mix |
| `tests/unit/phases/test_topology_router.py` | Аудит | Возможные stale literal hits |
| `dev/codebase-map.md` | Секция Tasks & Evaluation | Актуализация |

## Порядок зависимостей

```
Step 1.1 → Step 1.2 (один коммит, атомарно)
Step 1.1 → Step 1.3 (разные файлы, параллельно со Step 1.2)
Steps 1.1 + 1.2 → Step 2.1 → Step 2.2 → Step 2.3 (последовательно: shared __init__.py)
Steps 2.1–2.3 → Steps 3.1 и 3.2 (параллельно, разные файлы)
Step 3.1 → Step 4.1 (smoke зависит от финализированного __init__.py)
Step 4.1 → Step 5.1
```

## Риски

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Шаги 1.1+1.2 не в одном коммите → CI сломан | Средняя | Высокое | Явное указание в плане + wave 1 атомарна |
| `datasets` emits deprecation warning → `filterwarnings=error` | Низкая | Средняя | Точный `ignore` в `pyproject.toml`, только если нужен |
| Скрытый импортёр `MMLULoader`/`CreativeLoader` вне `tasks/` | Низкая | Высокое | Grep перед удалением (шаг 1.2 Implementation Notes) |
| DABench repo force-pushed → URL невалиден | Очень низкая | Низкое | Curated fallback + `ATM_DABENCH_OFFLINE=1` гарантирует CI без сети |
| Stale Parquet от pre-resync запуска | Низкая | Низкое | Cache key `"dabench"` ранее не существовал; инструкция для dev: `rm data/cache/tasks/dabench.parquet` |
| In-house ROUGE-L расходится с `rouge-score` | Неприменимо | Низкое | Scoring внутренний, паритет с бенчмарком не требуется; задокументировано в модульном docstring |

## Вне scope

- `arch/PLAN.md`, `arch/arch.md`, `arch/experiment_plan.md` — не изменяются
- `dev/done/m10/` — история, не изменяется
- `src/atm/tasks/_judge.py` — остаётся (будущая утилита), но после ресинха не используется
- Alembic миграции — не нужны
- `pyproject.toml` зависимости — не меняются
- Интеграционные тесты — не добавляются
- Промпт-инжиниринг для агента (чтобы эмитировал `@name[...]` формат) — M11+
- Динамический `abs_tol` из поля `constraints` DABench — M11+
- Возврат MMLU как ablation loader — M12+

## Временная оценка

- Всего: ~6h
- Волна 1: ~1.5h (enum + удаление + oracle fix)
- Волна 2: ~3h (3 TDD загрузчика последовательно)
- Волна 3: ~0.5h (аудит + test_base.py)
- Волна 4: ~0.5h (smoke test)
- Волна 5: ~0.5h (codebase-map)
- Создан: 2026-05-12
