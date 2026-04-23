# M0 — Bootstrap проекта: контекст сессии

## Метаданные задачи

- **Task-slug**: m0
- **Class**: standard
- **Integration tests**: no
- **Ветка**: feat/m0
- **Создан**: 2026-04-23

---

## SESSION PROGRESS (2026-04-23)

### ЗАВЕРШЕНО
- Step 4 (2026-04-23): Создан `.env.example` (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, PG_PORT, PG_DSN asyncpg-format, DATA_DIR, OPENAI_API_KEY, ANTHROPIC_API_KEY). Создан `.gitignore` (игнорирует `.env`, Python кэши, .venv, data/, IDE dirs, OS files). Все 8 verification checks прошли.
- Step 3: создан `docker-compose.yml` — postgres:16-alpine, container atm-postgres, named volume postgres_data, healthcheck pg_isready, port override via PG_PORT env var
- Step 2 (2026-04-23): Создан скелет каталогов. 16 __init__.py в src/atm/** (корень + 15 подпакетов вкл. llm/providers и tools/sandbox); 4 __init__.py в tests/{unit,integration,fixtures}+root; 6 .gitkeep в conf/**; notebooks/.gitkeep. src/atm/__init__.py содержит __version__ = "0.1.0".
- Step 1 (2026-04-23): Создан `pyproject.toml` (hatchling, src/atm layout, requires-python>=3.11), `.python-version` (3.11), `uv.lock`. Минимальный `README.md` placeholder (hatchling требует наличия файла). `uv sync` exit 0; `uv run python -c "print('ok')"` → ok.

- Step 5 (2026-04-23): Добавлены dependency groups `test` (pytest>=8.3, pytest-asyncio>=0.24) и `dev` ({include-group="test"}, ruff>=0.9, mypy>=1.14, alembic>=1.13, sqlalchemy[asyncio]>=2.0, asyncpg>=0.29) через `uv add --group`. uv.lock обновлён. Все 6 verification checks прошли: ruff 0.15, mypy 1.20, pytest 9.0, alembic 1.18, sqlalchemy asyncio + asyncpg import ok.

- Step 6 (2026-04-23): Добавлены секции [tool.ruff], [tool.ruff.lint], [tool.ruff.format], [tool.mypy] в pyproject.toml. line-length=100, target-version=py311, strict mypy, files=["src/atm"]. Все verification checks пройдены: ruff check exit 0, ruff format --check exit 0, mypy "Success: no issues found in 16 source files".

- Step 7 (2026-04-23): alembic init -t async завершён. Созданы alembic/ с async env.py (PG_DSN override из os.environ), alembic.ini (sqlalchemy.url закомментирован), и initial migration bc5f66dd0897. ruff check exit 0, mypy src/ success, alembic history показывает "initial".

- Step 8 (2026-04-23): Добавлена секция [tool.pytest.ini_options] в pyproject.toml (filterwarnings=error, testpaths=tests, asyncio_mode=auto, addopts=-ra). Создан tests/unit/test_smoke.py. uv run pytest: 2 passed. ruff check exit 0, ruff format --check exit 0, mypy src/ success.
- Step 9 (2026-04-23): README.md полностью перезаписан — русский quickstart с 7-шаговым fast-start (cp .env.example .env → uv sync → docker compose up -d → alembic upgrade head → pytest → ruff check → mypy). Все 8 verification checks прошли. Commit: 1a16a25.

### В ПРОЦЕССЕ
- Все шаги завершены (M0 complete)

### БЛОКЕРЫ
- Нет

---

## Quick Resume

1. Прочитать этот файл
2. Открыть `m0-tasks.md` — найти первый незачеркнутый таск
3. Прочитать `m0-plan.md` Phase 1 (Wave 1) для стратегии
4. Начать с: **Шаг 1** (Wave 1, параллельно с шагами 2, 5, 7)

**Порядок волн**: 1+2+5+7 (параллельно) → 3 → 4 → 6 → 8 → 9

---

## Key Files

**`pyproject.toml`**
- Роль: центральный конфиг (uv build-system, ruff, mypy, pytest, dependency-groups)
- Плановое изменение: создаётся в Step 1; дополняется секциями в Steps 3, 4, 8
- Статус: NOT STARTED
- Важно: шаги 3, 4, 8 пишут в него **последовательно** (не параллельно)

**`src/atm/__init__.py`**
- Роль: корень пакета atm; задаёт `__version__ = "0.1.0"`
- Плановое изменение: создаётся в Step 2
- Статус: DONE

**`src/atm/*/`** (14 подпакетов: core, llm, llm/providers, tools, tools/sandbox, agents, topology, phases, human, storage, observability, tasks, evaluation, experiment, analysis)
- Роль: пустые Python-пакеты с `__init__.py` (docstring-only)
- Плановое изменение: создаются в Step 2
- Статус: DONE
- Важно: mypy strict требует явных `__init__.py` — namespace packages не подойдут

**`tests/unit/test_smoke.py`**
- Роль: smoke-тест importability пакета atm
- Плановое изменение: создаётся в Step 8
- Статус: DONE

**`docker-compose.yml`**
- Роль: Postgres 16-alpine dev-окружение
- Плановое изменение: создаётся в Step 3
- Статус: DONE

**`.env.example`**
- Роль: шаблон env-переменных (PG_DSN, OPENAI_API_KEY, ANTHROPIC_API_KEY, DATA_DIR)
- Плановое изменение: создаётся в Step 4
- Статус: DONE

**`.gitignore`**
- Роль: исключить `.env`, `.venv`, Python кэши, data/, IDE и OS артефакты
- Плановое изменение: создаётся в Step 4
- Статус: DONE

**`alembic/env.py`**
- Роль: async-alembic окружение, читает PG_DSN из `os.environ`
- Плановое изменение: генерируется `alembic init -t async` и редактируется в Step 7
- Статус: DONE

**`alembic/versions/bc5f66dd0897_initial.py`**
- Роль: noop initial-миграция (DDL появится в M3)
- Плановое изменение: генерируется в Step 7
- Статус: DONE

**`conf/**/.gitkeep`** (6 файлов: conf/, topology/, agents/, task/, model/, experiment/)
- Роль: placeholder для будущих OmegaConf YAML-конфигов (M1+)
- Плановое изменение: создаются в Step 2
- Статус: DONE
- Важно: `conf/` — НЕ Python-пакет, только `.gitkeep`

**`README.md`**
- Роль: user-facing quickstart на русском языке
- Плановое изменение: создаётся в Step 9 (последний)
- Статус: NOT STARTED

---

## Decisions

### SQLAlchemy + asyncpg добавлены в dev-group уже на M0

- Решение: `sqlalchemy[asyncio]>=2.0` и `asyncpg>=0.29` входят в `[dependency-groups].dev`
- Обоснование: `alembic init -t async` генерирует env.py с `from sqlalchemy.ext.asyncio import async_engine_from_config`. Без пакетов `uv run alembic upgrade head` падает с `ImportError`. Это минимально необходимый набор, не фичерный.
- НЕ добавляем в M0: `langgraph`, `langchain-*`, `langgraph-checkpoint-postgres`, `pyarrow`, `pandas`, `matplotlib`, `jupyter` — приходят в M1–M8.

### conf/ через .gitkeep, не __init__.py

- Решение: `conf/` и все её подкаталоги содержат только `.gitkeep`
- Обоснование: `conf/` — каталог OmegaConf YAML-конфигов, не Python-пакет. YAML появятся в M1+.

### mypy files = src/atm (alembic/ исключён)

- Решение: `[tool.mypy] files = ["src/atm"]`
- Обоснование: alembic-генерированные файлы не типизированы под mypy strict. Явное ограничение через `files` проще, чем `[[tool.mypy.overrides]] ignore_errors = true`.

### PG_DSN в asyncpg-формате

- Решение: `PG_DSN=postgresql+asyncpg://atm:atm@localhost:5432/atm`
- Обоснование: этот DSN используется Alembic env.py (M0), storage/session.py (M3),
  langgraph-checkpoint-postgres (M3). Для plain psql формат другой: `postgresql://...` — задокументировано в `.env.example` комментарием.

### tests/fixtures/ — Python-пакет

- Решение: `tests/fixtures/__init__.py` (пакет, не просто директория)
- Обоснование: удобный импорт общих фикстур в M2+ как `from tests.fixtures.llm import scripted_response`.

### filterwarnings = "error" в pytest

- Решение: агрессивная настройка на M0
- Обоснование: на M0 нет warnings-source'ов. В M1+ ослабляется по первому инциденту (sqlalchemy 2.x MovedIn20Warning, pytest-asyncio deprecations). Задокументировано inline в `pyproject.toml`.

### Language Policy

- `*.py`, все конфиги, `docker-compose.yml`, `alembic.*`, `.env.example`, `.gitignore`, commit messages → **English**
- `README.md`, dev-документы → **Russian**
- Developer-агенты не переводят конфиги на русский

---

## Constraints

- Хост-порт 5432 должен быть свободен (или переопределить `POSTGRES_PORT` в `.env`)
- `uv` ≥ 0.5, `docker` ≥ 24, docker compose V2 должны быть установлены
- Ветка `feat/m0` в git (см. gitStatus)
- `.env` файл НЕ коммитится никогда — только `.env.example`
- Шаги 3, 4, 8 пишут в `pyproject.toml` → строго последовательно (не параллельно)
- `atm/__init__.py` не импортирует ничего из подпакетов на M0 (иначе mypy/ruff полезут в незаполненные модули)
