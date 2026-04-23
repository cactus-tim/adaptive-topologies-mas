# Adaptive Topologies MAS

Техническая часть дипломной работы: фреймворк для экспериментов с адаптивными
топологиями мульти-агентных LLM-систем с human-in-the-loop.

Подробный план работы — `dev/PLAN.md`. Архитектурные заметки — `dev/arch.md`.

## Статус

M0 — bootstrap: скелет проекта, базовая инфраструктура (Postgres, Alembic,
pytest, ruff, mypy). Реальные компоненты (агенты, топологии, LLM-обёртки)
появятся в M1+.

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

# 5. (Опционально на M0) применить миграции — сейчас noop, но проверка того,
# что alembic корректно подключается к БД
uv run alembic upgrade head

# 6. Запустить smoke-тесты
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
notebooks/           # jupyter-анализ
dev/                 # план, arch-заметки, активные и завершённые задачи
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
| `uv run alembic revision -m "..."` | создать новую миграцию (M3+) |
| `docker compose up -d` | поднять Postgres |
| `docker compose down -v` | остановить и **удалить данные** Postgres |

## Дорожная карта milestone'ов

См. `dev/PLAN.md §8` — подробный план M0...M13 с exit-criteria.

## Лицензия

Внутренний проект (дипломная работа), лицензия будет добавлена позже.
