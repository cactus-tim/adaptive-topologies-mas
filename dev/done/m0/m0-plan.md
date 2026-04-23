# M0 — Bootstrap проекта: финальный план

## Executive Summary

Поднять рабочий скелет репозитория для дипломного фреймворка **Adaptive Topologies MAS**:
`uv`-проект на Python 3.11+, пустая структура каталогов по §4 PLAN.md, `docker-compose.yml`
с Postgres 16, инициализированный Alembic (async-template) с пустой initial-миграцией,
конфиги ruff/mypy, pytest-скелет с одним smoke-тестом, `.env.example`, `README.md`
с quickstart-ом на русском.

Область M0 — сугубо scaffolding: **никакой бизнес-логики не пишется**. Цель — дать в руки
последующим milestone'ам готовый чистый «стенд»: при `docker-compose up -d && uv run pytest`
всё зелёное; `uv run ruff check .` без ошибок; `uv run mypy src/` без ошибок.

**Final Class**: standard
**Needs Integration Tests**: no (scaffolding без published contracts)

---

## Current State

Репозиторий **greenfield** — в корне сейчас только:

- `dev/PLAN.md` — план milestone'ов (§4 — точная структура каталогов, §5 — модель данных, §8 — M0 чеклист)
- `dev/arch.md` — архитектурный документ (§10.3 observability, §11.3 two-pool Postgres, §12.1 ModelCfg, §14.4 reproducibility bundle)
- `dev/active/m0/` — рабочая папка milestone'а

Ни `pyproject.toml`, ни `src/`, ни `tests/`, ни `docker-compose.yml` не существуют.

---

## Proposed Approach

Девять шагов, реализующих scaffolding волнами:

1. Шаги 1, 2, 5, 7 — полностью независимы, выполняются параллельно (Wave 1).
2. Шаги 3 → 4 → 6 → 8 → 9 — сериализованы (каждый дополняет `pyproject.toml` или
   зависит от предыдущего артефакта).

Принципиальные архитектурные решения, зафиксированные на M0:

- **SQLAlchemy + asyncpg уже в dev-group**: нужны для работоспособного async-alembic env.py.
  Fичерные зависимости (langgraph, langchain-*, pyarrow, pandas) — не добавляем.
- **conf/ через .gitkeep**: каталог не является Python-пакетом, YAML появятся в M1+.
- **mypy files = src/atm**: alembic/ сознательно исключён из type-checking.
- **PG_DSN в asyncpg-формате**: `postgresql+asyncpg://atm:atm@localhost:5432/atm` —
  используется Alembic env.py, storage/session.py (M3), langgraph-checkpoint-postgres.
- **tests/fixtures/ — Python-пакет**: для удобного импорта в M2+ как `from tests.fixtures.llm import ...`.
- **filterwarnings = "error" в pytest**: на M0 нет warnings-source'ов, агрессивная настройка
  безопасна; в M1+ ослабляется по факту первого warning.

---

## Implementation Phases

### Phase 1: Wave 1 — независимые файлы (~0.5h параллельно)

**Goal**: создать все независимые артефакты одновременно в разных worktrees.

- [ ] 1.1 Инициализировать `uv`-проект и базовый `pyproject.toml`
  - Файлы: `pyproject.toml`, `.python-version`, `uv.lock`
  - Acceptance: `uv sync` exit 0; `uv run python -c "print('ok')"` выводит `ok`

- [ ] 1.2 Создать скелет каталогов `src/atm/`, `tests/`, `conf/`, `notebooks/` с placeholder-файлами
  - Файлы: 15 `__init__.py` в `src/atm/**`, 4 `__init__.py` в `tests/**`, `notebooks/.gitkeep`, 6 `conf/**/.gitkeep`
  - Acceptance: `uv run python -c "import atm; print(atm.__version__)"` выводит `0.1.0`

- [ ] 1.3 Создать `docker-compose.yml` с Postgres 16
  - Файл: `docker-compose.yml`
  - Acceptance: `docker compose up -d` → через ~15 с `docker compose ps` показывает `healthy`

- [ ] 1.4 Создать `.env.example` и обновить `.gitignore`
  - Файлы: `.env.example`, `.gitignore`
  - Acceptance: `grep -q "PG_DSN" .env.example` и `grep -q "SQLAlchemy-specific" .env.example` и `git check-ignore -v .env` — OK

### Phase 2: Wave 2 — dev/test зависимости (~0.3h)

**Goal**: установить инструменты и runtime-зависимости для alembic async.

- [ ] 2.1 Добавить dev/test dependency groups через `uv add --group dev/test ...`
  - Файлы: `pyproject.toml` (`[dependency-groups]`), `uv.lock`
  - Acceptance: `uv run ruff --version` ≥ 0.9, `uv run mypy --version` ≥ 1.14, `uv run pytest --version` ≥ 8.3, `uv run alembic --version` ≥ 1.13, `uv run python -c "import sqlalchemy.ext.asyncio, asyncpg; print('ok')`

### Phase 3: Wave 3 — конфиги линтеров (~0.2h)

**Goal**: настроить ruff и mypy под src-layout.

- [ ] 3.1 Настроить конфиги `ruff` и `mypy` в `pyproject.toml`
  - Файл: `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`)
  - Acceptance: `uv run ruff check .` exit 0; `uv run mypy src/` "Success: no issues found"

### Phase 4: Wave 4 — Alembic async (~0.3h)

**Goal**: инициализировать Alembic с async-шаблоном и пустой initial-миграцией.

- [ ] 4.1 `alembic init -t async` + пустая initial migration
  - Файлы: `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`, `alembic/README`, `alembic/versions/<ts>_initial.py`
  - Acceptance: `uv run alembic history` показывает одну ревизию `initial`; `uv run alembic upgrade head` при поднятом Postgres exit 0

### Phase 5: Wave 5 — pytest smoke-тест (~0.2h)

**Goal**: pytest-конфиг и базовый smoke-тест.

- [ ] 5.1 Создать smoke-тест и конфиг pytest
  - Файлы: `pyproject.toml` (`[tool.pytest.ini_options]`), `tests/unit/test_smoke.py`
  - Acceptance: `uv run pytest` — "2 passed"

### Phase 6: Wave 6 — README (~0.2h)

**Goal**: user-facing README на русском с воспроизводимым quickstart.

- [ ] 6.1 Написать `README.md` с quickstart'ом (на русском)
  - Файл: `README.md`
  - Acceptance: README существует, quickstart содержит все 6 команд в правильном порядке,
    команды совпадают с реальными инвокациями из предыдущих шагов

---

## Key Files Affected

| Файл | Изменение | Зачем |
|------|-----------|-------|
| `pyproject.toml` | создаётся, затем дополняется шагами 2, 3, 5 | центральный конфиг проекта (uv, build, ruff, mypy, pytest) |
| `uv.lock` | генерируется | фиксирует граф зависимостей |
| `.python-version` | создаётся | фиксирует Python 3.11 |
| `src/atm/__init__.py` | создаётся | корень пакета, `__version__` |
| `src/atm/*/` __init__.py (×14) | создаются | явные пакеты для mypy strict src-layout |
| `tests/{unit,integration,fixtures}/__init__.py` | создаются | pytest rootdir + импорт fixtures |
| `tests/unit/test_smoke.py` | создаётся | smoke-тест importability |
| `conf/**/.gitkeep` (×6) | создаются | placeholder для OmegaConf-конфигов (M1+) |
| `notebooks/.gitkeep` | создаётся | placeholder для Jupyter-нотбуков |
| `docker-compose.yml` | создаётся | Postgres 16 dev-окружение |
| `.env.example` | создаётся | шаблон env-переменных (PG_DSN, API keys, DATA_DIR) |
| `.gitignore` | создаётся/дополняется | исключить `.env`, `.venv`, `data/`, кэши |
| `alembic.ini` | создаётся | конфиг Alembic (sqlalchemy.url из env) |
| `alembic/env.py` | создаётся + редактируется | async-engine, читает `PG_DSN` из `os.environ` |
| `alembic/versions/<ts>_initial.py` | создаётся | noop initial-миграция |
| `README.md` | создаётся | user-facing quickstart на русском |

---

## Dependencies & Order Constraints

```
Step 1 ──┬──▶ Step 3 ──▶ Step 4 ──▶ Step 6 ──▶ Step 8 ──▶ Step 9
Step 2 ──┤                                                    ▲
Step 5 ──┤                                                    │
Step 7 ──┴───────────────────────────────────────────────────┘
```

- **Wave 1 (параллельно)**: шаги 1, 2, 5, 7 — не трогают общие файлы
- **Wave 2 (serial)**: шаг 3 — дополняет `pyproject.toml [dependency-groups]`
- **Wave 3 (serial)**: шаг 4 — дополняет `pyproject.toml [tool.ruff]/[tool.mypy]`
- **Wave 4 (serial)**: шаг 6 — требует шаги 3 (deps) + 4 (mypy-конфиг) + 5 (postgres UP)
- **Wave 5 (serial)**: шаг 8 — дополняет `pyproject.toml [tool.pytest.ini_options]`
- **Wave 6 (serial)**: шаг 9 — README, финальный, зависит от всех

---

## Risks

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Порт 5432 занят локальным Postgres | средняя | высокое (alembic upgrade head упадёт) | переменная `POSTGRES_PORT=5433` в `.env`; README предупреждает |
| `uv init` конфликтует с существующим `pyproject.toml` | низкая | низкое | проверить наличие до запуска; при наличии — редактировать вручную |
| `filterwarnings = "error"` валит M1+ тесты из-за deprecation warnings | средняя | низкое | задокументировано inline; ослабляется по первому инциденту |
| Забытый `__init__.py` в подпакете — mypy strict ломается | средняя | среднее | Verification Step 2 явно проверяет все 15 каталогов |
| `alembic init -t async` создаёт `alembic/README` (не путать с корневым) | низкая | низкое | шаг 6 явно указывает, что корневой `README.md` не затрагивается |
| asyncpg C-extension — нет prebuilt wheel на экзотической платформе | очень низкая | среднее | стандартные linux/macOS ок; uv выдаёт внятную ошибку |

---

## Out of Scope

- Redis, Prometheus, Jaeger — не в M0 чеклисте
- LLM-провайдеры (langgraph, langchain-*, OpenAI SDK) — M1/M2
- SQLAlchemy-модели (Base, таблицы) — M3
- Два async-engine'а под two-pool pattern (arch.md §11.3) — M3
- OmegaConf YAML-конфиги в `conf/` — M1+
- pre-commit hooks — опционально, не в чеклисте M0
- Jupyter-нотбуки — заполняются с M5+
- CI/CD (GitHub Actions) — отдельный milestone

---

## Timeline

- Total: ~1.7h (Wave 1 параллельно ~0.5h + serial цепочка ~1.2h)
- Версия плана: Revision 1
- Создан: 2026-04-23

---

## Language Policy (обязательно для всех developer-агентов)

| Артефакт | Язык содержимого |
|---|---|
| Все `*.py` (код, docstrings, комментарии) | English |
| `pyproject.toml` (включая description/keywords) | English |
| `docker-compose.yml`, `alembic.ini`, `alembic/env.py`, миграции | English |
| Конфиги ruff/mypy/pytest в `pyproject.toml` | English |
| `.env.example` (ключи, комментарии) | English |
| YAML в `conf/` (placeholder'ы, если создаются) | English |
| `.gitignore` | English-конвенционные шаблоны |
| Commit messages | English |
| `README.md` | **Russian** (user-facing) |
| dev-документы (`dev/active/m0/*.md`) | Russian |

Developer-агенты **не должны** переводить конфиги на русский; docstrings/комментарии — только English.

---

## Testing Strategy / Exit Criteria

| Проверка | Команда | Ожидаемый результат |
|---|---|---|
| uv sync | `uv sync` | exit 0 |
| ruff lint | `uv run ruff check .` | 0 errors, "All checks passed!" |
| ruff format check | `uv run ruff format --check .` | exit 0 |
| mypy | `uv run mypy src/` | "Success: no issues found" |
| pytest | `uv run pytest` | "2 passed" |
| docker-compose | `docker compose up -d && docker compose ps` | postgres status=healthy |
| alembic noop | `PG_DSN=... uv run alembic upgrade head` | exit 0 |
| package import | `uv run python -c "import atm; print(atm.__version__)"` | `0.1.0` |
| full import | `uv run python -c "import atm, atm.core, atm.llm, atm.tools.sandbox, atm.topology, atm.phases, atm.storage, atm.observability, atm.experiment, atm.evaluation, atm.analysis, atm.agents, atm.human, atm.tasks; print('ok')"` | `ok` |

---

## History

- **Revision 1 (2026-04-23)**: закрыты оба блокера предыдущего ревью — добавлены
  `sqlalchemy[asyncio]>=2.0` и `asyncpg>=0.29` в `[dependency-groups].dev` (Step 3);
  явно перечислены все шесть `.gitkeep`-файлов под `conf/` в Files/Components Step 2.
  Применены non-blocking рекомендации: комментарий про SQLAlchemy-specific DSN формат в
  `.env.example`; документация про исключение `alembic/` из mypy в Step 4;
  inline-комментарий про `filterwarnings=["error"]` в Step 8; зафиксированный порядок
  quickstart в Step 9. Per-step `Can-Parallel-With` приведены в соответствие с wave-plan.
  Review вердикт: APPROVED, BLOCKING_ISSUES: 0.

---

_Конец плана M0._
