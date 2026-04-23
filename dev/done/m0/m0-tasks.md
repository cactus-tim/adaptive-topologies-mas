# M0 — Bootstrap проекта: чеклист задач

## Wave 1 (параллельно, 4 worktrees) — независимые файлы

- [x] 1. Инициализировать `uv`-проект и базовый `pyproject.toml` · Type=simple · Depends On=none · Can-Parallel-With=2,5,7
  - Acceptance: `uv sync` exit 0; `uv run python -c "print('ok')"` выводит `ok`

- [x] 2. Создать скелет каталогов `src/atm/`, `tests/`, `conf/`, `notebooks/` с placeholder-файлами · Type=simple · Depends On=none · Can-Parallel-With=1,5,7
  - Acceptance: `uv run python -c "import atm; print(atm.__version__)"` → `0.1.0`; `find src/atm -type d | wc -l` = 15; все 6 `conf/**/.gitkeep` присутствуют

- [x] 3. Создать `docker-compose.yml` с Postgres 16 · Type=simple · Depends On=none · Can-Parallel-With=1,2,7
  - Acceptance: `docker compose up -d` → через ~15 с postgres status=healthy

- [x] 4. Создать `.env.example` и обновить `.gitignore` · Type=simple · Depends On=none · Can-Parallel-With=1,2,3
  - Acceptance: `grep -q "PG_DSN" .env.example` и `grep -q "SQLAlchemy-specific" .env.example` и `git check-ignore -v .env` — все OK

---

## Wave 2 (serial, после Wave 1)

- [x] 5. Добавить dev/test dependency groups через `uv add --group dev/test ...` · Type=simple · Depends On=1 · Can-Parallel-With=none
  - Acceptance: `uv run ruff --version` ≥ 0.9; `uv run mypy --version` ≥ 1.14; `uv run pytest --version` ≥ 8.3; `uv run alembic --version` ≥ 1.13; `uv run python -c "import sqlalchemy.ext.asyncio, asyncpg; print('ok')"` → `ok`

---

## Wave 3 (serial, после Wave 2)

- [x] 6. Настроить конфиги `ruff` и `mypy` в `pyproject.toml` · Type=simple · Depends On=5 · Can-Parallel-With=none
  - Acceptance: `uv run ruff check .` exit 0; `uv run ruff format --check .` exit 0; `uv run mypy src/` — "Success: no issues found"

---

## Wave 4 (serial, после Wave 3)

- [x] 7. `alembic init -t async` + пустая initial migration · Type=simple · Depends On=5,6,3 · Can-Parallel-With=none
  - Acceptance: `uv run alembic history` показывает ревизию `initial`; `uv run alembic upgrade head` (при поднятом Postgres) exit 0; `uv run alembic downgrade base` exit 0

---

## Wave 5 (serial, после Wave 4)

- [x] 8. Создать smoke-тест и конфиг pytest · Type=tdd · Depends On=1,2,5,7 · Can-Parallel-With=none
  - Acceptance: `uv run pytest` — "2 passed"; `uv run pytest -v` показывает `test_atm_package_importable PASSED` и `test_python_arithmetic_sanity PASSED`

---

## Wave 6 (serial, после Wave 5)

- [x] 9. Написать `README.md` с quickstart'ом (на русском) · Type=simple · Depends On=1,2,3,4,5,6,7,8 · Can-Parallel-With=none
  - Acceptance: `README.md` существует; quickstart содержит команды в зафиксированном порядке: `cp .env.example .env` → `uv sync` → `docker compose up -d` → `uv run alembic upgrade head` (опц.) → `uv run pytest` → `uv run ruff check .` + `uv run mypy src/`; текст на русском, код-блоки на английском

---

## Waves — схема параллелизма

```
Wave 1 (параллельно): [1] [2] [3] [4]
                           ↓
Wave 2 (serial):          [5]
                           ↓
Wave 3 (serial):          [6]
                           ↓
Wave 4 (serial):          [7]
                           ↓
Wave 5 (serial):          [8]
                           ↓
Wave 6 (serial):          [9]
```

**Max parallelism**: 4 worktrees в Wave 1.
**Critical path**: 1 → 5 → 6 → 7 → 8 → 9 (~1.2h serial после Wave 1).

---

## Stats

- Итого: 9 задач · ~1.7h
- Выполнено: 0 / 9
- Тип tdd: 1 (задача 8)
- Тип simple: 8 (задачи 1, 2, 3, 4, 5, 6, 7, 9)

## Как обновлять

После завершения каждой задачи:
1. Отметить `[x]` в этом файле
2. Обновить `m0-context.md` секцию SESSION PROGRESS
3. Если задача Wave 1 — отметить соответствующий файл в Key Files как DONE

---

## SESSION PROGRESS

### ЗАВЕРШЕНО
- Step 2: Создан скелет каталогов — 16 __init__.py в src/atm/, 4 в tests/, 6 .gitkeep в conf/, notebooks/.gitkeep

### В ПРОЦЕССЕ
- Не начато (Step 2 завершён; ожидаются шаги Wave 1: 1, 4; Wave 2+: 5, 6, 7, 8, 9)

### БЛОКЕРЫ
- Нет
