# Session Handoff — 2026-05-16

Контекст для продолжения работы на другой машине (16 vCPU / 32 GB RAM).

---

## 1. Где мы сейчас (TL;DR)

**Главная цель сессии:** довести E1-mini до состояния, когда все 5 топологий
дают честный non-zero `quality_score`, без скрытых runner-фолбеков. После —
запустить полный E1 и далее по пайплайну до E4.

**Прогресс:** найдено и пофикшено **8 багов** за сессию. Mesh, hierarchical,
debate, dabench, chain gsm8k — все либо работают, либо понятная причина
оставшихся 0. Дополнительно сделан рефакторинг `topology.extra` в
namespaced-схему (Pydantic), который **сейчас в финальной стадии — code-review
обнаружил 2 critical + 1 major, исправляются background-агентом** (см. §5).

**Текущий HEAD `main`:** `241db19` (push'нуть после фикса от code-review).

---

## 2. Что сделано за сессию (хронология по commit'ам)

### Mesh / debate / hierarchical fixes
- `4780d41` — dabench URL typo (`da-dev-data` → `da-dev-tables`)
- `c5ef5c7` — dabench: разрешить пробелы/скобки в `file_name` + URL-encode
- `7b56e33` — debate: усиленный prompt (PREPEND + `###ANSWER###...###END###` marker) + extractor парсит marker
- `4c4f195` — mesh: default `mesh_max_rounds=12`, decoupled from shared `max_rounds`
- `d585bbb` — debate judge prompt override (для non-code задач) + safety net в `_judge_postprocess` (extract DRAFT даже при REJECT) + chain task-aware extraction (`solution.py` только для humaneval, DRAFT для остального)
- `0d878b7` — critic.yaml: task-aware prompt (убрал hardcoded `solution.py` bias для гpм8к/commongen/dabench)
- `8b579bc` — добавлен `conf/experiments/e1_mini_fixes.yaml` (focused 27-cell sanity на mesh/hier/debate)

### Namespace refactor (5 коммитов)
- `0056447` — wave 1: schema + `get_topology_extras` helper
- `897f95b` — wave 2: tests + runner + 5 builders + 9 production configs migrated
- `384066d` — wave 3: adaptive `_get_subgraph` forwarding fix + 6 validation tests
- `da0e4ea` — wave 4-5: docstrings + final smoke
- `241db19` — chore: ruff autofix + format

### Pending (НЕ закоммичено, agent работает)
- Фикс 2 critical + 1 major от code-review (см. §5)

---

## 3. Что СЕЙЧАС в работе (background agent)

**Agent ID:** `a4c417ca8b7ed32e3` (dispatch'нут в текущей сессии)
**Output file:** `/tmp/claude-1002/.../tasks/a4c417ca8b7ed32e3.output`
**ETA:** ~5-10 мин

**Что фиксит:**
1. **CRITICAL #1** — `MeshExtras` schema names не совпадают с builder reads
   (`max_messages` vs `broadcast_bus_cap`, `dispatch` vs `activation_policy`,
   `round_robin_order` vs `agent_order`) → YAML-значения silently дропались
2. **CRITICAL #2** — `_build_agents` в `runner.py:763` делает `dict(cfg.topology.extra)`
   flat → `sub_teams` / `debater_pro_id` / `debater_contra_id` / `judge_id`
   всегда `None` → user-supplied hierarchical/debate worker configs игнорируются
3. **MAJOR** — `_FLAT_TO_NAMESPACE` омитит `activation_policy` / `agent_order` /
   `broadcast_bus_cap` для mesh → pre-refactor flat mesh YAMLs падают
   `ValidationError`

Agent также должен добавить 2-3 regression-теста.

**Когда продолжишь:**
1. Проверь `git log --oneline -5` — должен быть новый коммит от агента или его changes uncommitted
2. Если uncommitted → `git status`, прогнать `uv run pytest tests/unit/ -q`,
   закоммитить с типичным сообщением `fix: code-review findings...`
3. Push в origin: `git push origin main`

Если агент завис (как security review до того) — diagnose: `tail -c 500
.../tasks/a4c417ca8b7ed32e3.output`. Если crashed — диагноз есть в §5,
можно фиксить вручную (изменения мелкие).

---

## 4. Что осталось до E1 full

1. **Закоммитить + push** фикс code-review (когда агент закончит)
2. **Phase 6 закрыть**: `mv dev/active/namespace-topology-extra dev/done/`,
   спавн `Explore` обновить `dev/codebase-map.md` background
3. **E1-mini v6** прогнать (10-25 мин на новой машине) — финальная sanity
   после всех фиксов. Конфиг: `conf/experiments/e1_mini.yaml`.
   Команда: `uv run atm grid run -c conf/experiments/e1_mini.yaml > logs/e1_mini_v6.log 2>&1`
4. Если v6 зелёный (все 5 топологий ≥ 0.5 mean, dabench non-zero) → **E1 full**
   (`conf/experiments/e1_full.yaml`, 900 cells, ~2.5-4ч на 16 vCPU)
5. Дальше по `dev/experiments_runbook.md` (E2 → E3 → E4)

---

## 5. Объяснение последнего фикса (namespace refactor) — для debug

### Что было сломано

`topology.extra` был `dict[str, Any]` где разные топологии читали одни и те же
ключи с разной семантикой. Самый болезненный пример:

| Topology | `extra.max_rounds` означает |
|---|---|
| debate | debate rounds (pro vs contra exchange) |
| hierarchical | team rounds |
| mesh | dispatch round-robin activations |

В `e1_pilot.yaml` стояло `max_rounds: 2` (для debate). Mesh читал тот же
ключ и активировал только 2 агента из 4 в round-robin'е → executor НИКОГДА не
запускался → no `solution.py` → quality=0 на всех mesh cells.

### Что сделали

Заменили плоский dict на **typed namespaced Pydantic schema**:

```yaml
# Было (flat — конфликт):
topology:
  extra:
    max_rounds: 2

# Стало (namespaced — изоляция):
topology:
  extra:
    debate: {max_rounds: 2}
    hierarchical: {max_rounds: 2}
    mesh: {max_rounds: 12}
    star: {exec_max_iter: 3, verify_max_iter: 2}
    adaptive: {...}
```

### Ключевые места (для debug)

| Файл | Что там |
|---|---|
| `src/atm/experiment/config.py` | `TopologyExtras` composite + `Star/Chain/Debate/Hierarchical/Mesh/AdaptiveExtras` sub-моделi + `_FLAT_TO_NAMESPACE` table + `_remap_flat_extras()` + `@model_validator(mode="before")` на `TopologyCfg` для bw-compat |
| `src/atm/topology/base.py` | `get_topology_extras(cfg, topology_name) -> dict[str, Any]` — единая точка чтения. Handles BOTH typed BaseModel (production) AND legacy flat dict (test fixtures). |
| `src/atm/topology/{star,chain,debate,hierarchical,mesh,adaptive}.py` | Каждая использует `extras = get_topology_extras(cfg, "<own_name>")`. |
| `src/atm/experiment/runner.py:1093, 2049` | `extra=cfg.topology.extra.model_dump(exclude_none=True)` — **критично `exclude_none=True`** (иначе AdaptiveExtras.run_id=None → `str(None)` = `"None"` в run_id). |
| `tests/fixtures/experiment/star_flat_extras_bwcompat.yaml` | Deliberate fixture для тестирования bw-compat пути с `DeprecationWarning`. |

### Bw-compat правила (важно при debug)

Если YAML стартует с плоским `extra.max_rounds: 2`, валидатор делает:
- `max_rounds` → scatter в `debate.max_rounds` AND `hierarchical.max_rounds`
  ONLY (НЕ в mesh — иначе re-вернёт starvation; НЕ в adaptive — там нет
  такого поля)
- `mesh_max_rounds` → `mesh.max_rounds` (legacy alias)
- `exec_max_iter`, `verify_max_iter`, `planning_max_iter` → `star.*`
- `consensus_threshold`, `broadcast_bus_cap`, `activation_policy`,
  `agent_order` → `mesh.*` (последние 3 добавятся в pending фиксе)
- `debater_pro_id`, `debater_contra_id`, `judge_id` → `debate.*`
- `sub_teams`, `final_answer_strategy` → `hierarchical.*`
- `phase_router`, `topology_router`, `subgraph_max_iterations`,
  `switch_guards`, `switch_guards_config`, `run_id` → `adaptive.*`

Эмитится `DeprecationWarning` via `warnings.warn(...)`.

### Что МОЖЕТ сломаться (на что смотреть при ошибке)

1. **`AttributeError: 'dict' object has no attribute 'model_dump'`** — где-то
   код ожидает `TopologyExtras` model но получил dict. Проверь, что
   `cfg.topology.extra` действительно прошёл через `TopologyCfg` валидацию
   (load_config). В тестах часто конструируют TopologyConfig напрямую с dict
   — тогда нужен `get_topology_extras(cfg, name)` (он обрабатывает оба
   случая).

2. **`ValidationError: extra forbidden`** — YAML использует поле которое
   schema не знает. Либо опечатка в YAML, либо схема устарела (добавь поле
   в соответствующий *Extras BaseModel).

3. **`KeyError: 'mesh'` / unexpected None** — где-то в коде делают
   `cfg.extra["mesh"]` напрямую вместо `get_topology_extras(cfg, "mesh")`.
   Найди через grep `cfg.extra\[` или `cfg\.topology\.extra\[`.

4. **`run_id == "None"` (literal string)** в логах adaptive run'ов — забыли
   `exclude_none=True` в `model_dump()`. Гарантировано в `runner.py:1093` и
   `runner.py:2049`, но если добавишь третий call site — не забудь.

5. **Mesh executor не запускается** — это исходный баг. Проверь что
   `extra.mesh.max_rounds >= 8` в твоём конфиге, ИЛИ что default 12
   используется (`MeshExtras.max_rounds: int = 12`).

### Где тесты, которые ловят регрессии

- `tests/unit/experiment/test_config.py::TestTopologyExtras` — 14 тестов на
  schema + bw-compat scatter
- `tests/unit/experiment/test_config.py::test_schema_defaults_match_topology_builder_constants`
  — parity guard, ловит drift между schema и `_DEFAULT_*`
- `tests/unit/topology/test_adaptive_namespace_forwarding.py` — 6 тестов на
  `_get_subgraph` forwarding
- `tests/unit/topology/test_mesh.py::TestMeshNamespacedExtras` — 2 теста на
  чтение из namespace

---

## 6. Как продолжить (resume protocol)

### На новой машине

```bash
git clone https://github.com/cactus-tim/adaptive-topologies-mas.git
cd adaptive-topologies-mas
# pull последние коммиты если они уже запушены:
git pull origin main

# Setup
uv sync                          # install deps
cp .env.example .env             # заполнить CEREBRAS_API_KEY + OPENAI_API_KEY
# Postgres + alembic:
docker compose up -d pg          # или локальный postgres
uv run alembic upgrade head

# Sanity:
uv run pytest tests/unit/ -q     # должно быть 1818+ passed
```

### Если pending фикс от code-review НЕ запушен

Проверь git log: ожидаемый последний commit — `fix: code-review findings`
или похожий. Если его нет → агент не закончил/упал. Восстанови вручную:

1. Открой `dev/done/namespace-topology-extra/.code-review.md` (где-то в done/
   после mv) или `dev/active/...` если ещё не двигался
2. 3 фикса описаны в §5 этого файла + в самом отчёте
3. Закоммить + push

### E1-mini v6 на новой машине

См. §7 (hardware runbook).

---

## 7. Smell-tests перед E1 full

Перед запуском полного E1 ($60+, 2.5-4ч) обязательно прогнать:

1. **Unit tests:** `uv run pytest tests/unit/ -q` → 1818+ passed
2. **Schema bw-compat:** `uv run python -W error::DeprecationWarning -c "import glob; from atm.experiment.config import load_config
for f in sorted(glob.glob('conf/experiments/*.yaml')): load_config(f); print(f, 'ok')"`
3. **E1-mini-fixes** (27 cells, 3 топологии): `uv run atm grid run -c conf/experiments/e1_mini_fixes.yaml > logs/e1_mini_fixes_v2.log 2>&1` — должно дать mesh/hier ≥ 0.8, debate ≥ 0.4
4. **E1-mini full** (100 cells): `uv run atm grid run -c conf/experiments/e1_mini.yaml > logs/e1_mini_v6.log 2>&1` — все 5 топологий non-zero, dabench non-zero (50/52 файлов available)

---

## 8. Известные quirks этой кодовой базы

- **Worktree auto-merge ненадёжен.** Когда run-task пайплайн выдаёт wave с
  worktrees — после завершения агентов **копируй файлы вручную** через
  `cp .claude/worktrees/agent-XXX/<file> <file>` для каждого declared file.
  Авто-merge может тихо потерять часть работы (потеряли 2/3 в одном раннем
  wave).
- **Worktrees branch'атся от старого `main`** — иногда от commit'а ДО недавних.
  Это значит, что мерджить через `git merge worktree-branch` притянет
  негативные изменения. Делай copy-out.
- **Security review agent иногда виснет** на `InputValidationError` (известно
  с этой сессии). Если security review нужен — следи за progress'ом, или
  пропусти если scope изменений минимален (pure schema/refactor).
- **Cerebras PAYG лимит:** 1000 RPM / 1M TPM на gpt-oss-120b — далеко выше
  чем потребляет p=16-30 параллелизм. Bottleneck — CPU/memory локальной
  машины, не Cerebras.

---

## 9. Файлы в репо для контекста

- `arch/session_handoff_2026-05-16.md` — этот файл
- `arch/hardware_16_32_runbook.md` — как настроить parallelism на 16/32 (см.
  следующий файл)
- `dev/experiments_runbook.md` — высокоуровневый playbook E1-E4
- `arch/experiment_plan.md` §11-12 — план экспериментов
- `dev/active/namespace-topology-extra/` — текущий task (после фикса move в
  `dev/done/`)
- `dev/done/fix-dabench-task/` — предыдущий task (для reference)
- `dev/codebase-map.md` — обзор кодовой базы (нужно обновить после refactor)
