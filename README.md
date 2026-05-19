# Adaptive Topologies MAS

Техническая часть дипломной работы: фреймворк для экспериментов с адаптивными
топологиями мульти-агентных LLM-систем с human-in-the-loop.

## Возможности

- Пять статических топологий (Star, Chain, Mesh, Debate, Hierarchical) и
  адаптивная топология с переключением внутри и между фазами.
- Канонический набор агентов (Planner, Researcher, Executor, Critic, Debater)
  поверх единой обёртки над LLM с учётом бюджета.
- Human-in-the-loop через сменные шлюзы: LLM-симулятор, CLI, Streamlit-интерфейс.
- Задачи HumanEval, GSM8K, CommonGen, DABench с детерминированными и
  LLM-оценщиками, метрики качества и NASA-TLX.
- Запуск одиночных экспериментов и параллельных grid-sweep'ов с оценкой
  стоимости, чекпоинтами, resume/replay и хранением результатов в Postgres и Parquet.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (менеджер пакетов)
- Docker + Docker Compose (для локального Postgres 16)

## Быстрый старт

```bash
# 1. Склонировать репозиторий и перейти в каталог
git clone <repo-url> adaptive-topologies-mas
cd adaptive-topologies-mas

# 2. Создать локальный .env из примера
cp .env.example .env

# 3. Установить зависимости (uv создаст .venv автоматически)
uv sync --group dev

# 4. Поднять Postgres в Docker
docker compose up -d
# ждём ~15 секунд, пока postgres станет healthy
docker compose ps

# 5. Применить миграции БД
uv run alembic upgrade head

# 6. Запустить тесты
uv run pytest

# 7. Линт и проверка типов
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

## Структура репозитория

```
src/atm/             # основной пакет (atm = adaptive topologies mas)
├── core/            # базовые типы, state, errors
├── llm/             # LLMWrapper, провайдеры, бюджет
├── tools/           # инструменты + Docker sandbox
├── agents/          # Planner, Researcher, Executor, Critic, Debater
├── topology/        # Star, Chain, Mesh, Debate, Hierarchical, Adaptive
├── phases/          # PhaseManager, TopologyRouter, guards, signals
├── human/           # HumanGateway (LLM-симулятор / CLI / Streamlit)
├── storage/         # SQLAlchemy models, Parquet writer, checkpointer
├── observability/   # callbacks, tracer
├── tasks/           # TaskSpec + конкретные задачи
├── evaluation/      # judges, metrics, NASA-TLX
├── experiment/      # runner, grid, CLI
└── analysis/        # loaders, plots
tests/               # unit / integration / fixtures
conf/                # YAML-конфиги (topology, agents, task, model, experiment)
alembic/             # миграции БД (async template)
analysis/            # скрипты и результаты анализа экспериментов
scripts/             # вспомогательные скрипты
notebooks/           # jupyter-анализ
data/                # результаты экспериментов (Parquet-агрегаты)
```

## Полезные команды

| Команда | Назначение |
|---|---|
| `uv sync --group dev` | установить все зависимости |
| `uv run pytest` | прогнать тесты |
| `uv run ruff check .` | линт |
| `uv run ruff format .` | автоформат |
| `uv run mypy src/` | проверка типов |
| `uv run alembic upgrade head` | применить миграции |
| `uv run alembic revision -m "..."` | создать новую миграцию |
| `docker compose up -d` | поднять Postgres |
| `docker compose down -v` | остановить и **удалить данные** Postgres |

## CLI (`atm`)

Все команды доступны через `uv run atm <sub-command>` (точка входа зарегистрирована в `pyproject.toml`).

### Сводка команд

| Команда | Назначение |
|---|---|
| `atm run` | Запустить один эксперимент из YAML-конфига |
| `atm grid` | Запустить параллельный sweep-эксперимент (ProcessPoolExecutor) |
| `atm estimate` | Оценить стоимость конфига / grid без запуска |
| `atm status` | Показать агрегированный статус эксперимента (cells done/failed/running, cost) |
| `atm resume` | Возобновить прерванный run из последнего LangGraph-чекпоинта |
| `atm replay` | Воспроизвести существующий run детерминированно или семантически |
| `atm reconcile` | Найти «зомби»-ранги (running, но процесс мёртв) и опционально пометить их failed |

### Описание и примеры

```
atm run --config conf/experiments/smoke.yaml [+key=val ...]
```
Запускает один эксперимент. Опция `--estimate` выводит оценку стоимости и запрашивает
подтверждение, если она превышает 50% от `budget.per_experiment_usd`. `--yes` пропускает
вопрос. Коды выхода: 0 — completed, 1 — failed, 2 — budget_exceeded, 3 — config error.

```
atm grid --config <path> [+key=val ...]
    [--parallelism N] [--fail-fast] [--yes]
    [--no-reconcile] [--force-resume] [--resume-incomplete] [--no-estimate]
```
Разворачивает sweep из YAML-блока `grid:` и запускает все cells параллельно.
`--parallelism` переопределяет `grid.parallelism` из конфига. `--fail-fast` прерывает sweep
на первом провале. Коды выхода: 0 — все cells завершены, 1 — часть провалилась, 2 — все провалились.

```
atm estimate --config conf/experiments/grid.yaml [+key=val ...]
```
Выводит таблицу оценки стоимости (tokens in/out, USD) по каждой cell и итог.
Использует исторические данные из БД (по парам topology+task) или эвристику.

```
atm status --exp-id <uuid> [--json]
atm status --exp-name my_experiment [--json]
```
Выводит сводку по эксперименту: статус, количество cells, avg quality, суммарные расходы.
`--json` эмитирует JSON вместо таблицы.

```
atm resume --run-id <uuid> [--force]
```
Возобновляет run из последнего чекпоинта LangGraph (`thread_id=str(run_id)`).
`ExperimentConfig` реагрузается из `experiments.config_snapshot` в БД.
`--force` пропускает проверку живости исходного процесса.

```
atm replay <run_id> [--mode deterministic|semantic] [--output-config-only]
```
Создаёт новый run (`replay_of=<original_run_id>`) с той же конфигурацией.
`deterministic` (default) — FakeLLM воспроизводит ответы из parquet; `semantic` — живая LLM.
`--output-config-only` выводит реагрузенный JSON конфига и выходит без запуска.

```
atm reconcile --exp-id <uuid> [--dry-run]
```
Сканирует runs с `status='running'` у которых процесс уже не жив, и помечает их
`status='failed', finish_reason='zombie'`. `--dry-run` только классифицирует, не мутирует БД.
Требует переменную окружения `ATM_PG_DSN`.

### Пример: полный цикл с grid-sweep

**1. Создайте минимальный YAML-конфиг** (например, `conf/experiments/quick_sweep.yaml`):

```yaml
name: quick_sweep
seed: 42

model:
  default: "openai:gpt-4o-mini"

agents:
  set: "canonical_4"

topology:
  name: "star"

task:
  name: "gsm8k"

observability:
  parquet_dir: "data/experiments"
  pg_dsn: "${oc.env:PG_DSN}"

grid:
  parallelism: 2
  seeds: [42, 43]
  sweep:
    topology.name: [star, chain]
```

Этот конфиг порождает 4 cells: 2 топологии × 2 seed.

**2. Оцените стоимость:**

```bash
uv run atm estimate --config conf/experiments/quick_sweep.yaml
```

**3. Запустите sweep:**

```bash
uv run atm grid --config conf/experiments/quick_sweep.yaml --parallelism 2
```

**4. Проверьте статус:**

```bash
uv run atm status --exp-name quick_sweep
# или с JSON-выводом:
uv run atm status --exp-name quick_sweep --json
```

**5. Воспроизведите конкретный run:**

```bash
# <run_id> — UUID из вывода atm grid или atm status
uv run atm replay <run_id> --mode deterministic
```

## Лицензия

[MIT](LICENSE) © 2026 Sosnin Timofei
