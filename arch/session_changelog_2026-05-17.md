# Session changelog — 2026-05-16 / 2026-05-17

Что было сделано за сессию (по веткам main `36fd2b1..712591a`). Все
изменения уже на `origin/main`. Этот файл — навигация, чтобы можно было
быстро восстановить контекст после возврата с дампами E1/E2.

---

## Коммиты сессии (хронологически)

| Hash | Что |
|---|---|
| `983b1b5` | **fix(hitl)**: wire llm_fallback gateway в star/mesh/debate/adaptive (4 файла, унификация с chain/hier) |
| `499de6c` | **fix(finalize)**: trigger Path C для CommonGen meta-comments / missing concepts |
| `24d5529` | **feat(adaptive)**: wire LLM/Oracle topology routers + `atm oracle` CLI |
| `7bfe553` | **chore(conf)**: E2/E3 smoke configs (6 yaml) |
| `f442343` | **docs(e3)**: full handoff for E3 run on 16/32 box |
| `0fdd727` | **docs(e3)**: parallelize Wave 4 — 3 router modes simultaneously (6 ч → 2 ч) |
| `3c87b4e` | **chore(conf)**: E4 smoke configs (e4_pilot_sanity / e4_role_smoke / e4_pilot) |
| `2772424` | **docs(e4)**: full handoff for E4 run |
| `712591a` | **feat(storage)**: auto-flush PG aggregate rows to parquet on grid completion + `atm export-exp` CLI |

---

## Что сделано: 3 фикса + 2 фичи + 9 yaml + 2 handoff

### Фиксы (regression / bug)

1. **HITL `fallback_gateway`** (`983b1b5`)
   - Star/mesh/debate/adaptive падали с `policy='llm_fallback' requires
     llm_fallback_gateway` при таймауте gateway
   - Унифицировал паттерн с chain/hierarchical: при `timeout_policy ==
     "llm_fallback"` локально строится свежий `LLMSimulatedGateway`
   - Поймано через `e2_pilot_sanity` (2 of 6 star cells failed → 6/6 после фикса)

2. **CommonGen finalize trigger** (`499de6c`)
   - Path C не срабатывал когда executor выдавал мета-комментарий
     ("Issues identified: ...") вместо story — текст не пустой, не code,
     trigger пропускался → score~0.5 потолок на всех топологиях
   - Расширил `_needs_finalize`: trigger при отсутствующем concept +
     critic-style meta-markers; добавил explicit format hint в Path C
     prompt
   - `mean_q: 0.43 → 0.58`, min `0.03 → 0.52`, **10/10 passed** на
     `e3_commongen_smoke` (было 6/10)

3. **Topology router wiring** (`24d5529`)
   - `topology_router='llm'/'oracle'` в `e3_full.yaml` parsed но not
     consumed — adaptive всегда RuleBased. E3 sweep по router modes
     дал бы 3× duplicate data
   - Read `extras.topology_router`, build LLMTopologyRouter (через
     `kwargs['topology_router_llm']`) или OracleTopologyRouter (читает
     `data/oracle/e1_leave_one_out.json` или
     `extras['oracle_table_path']`). Soft-fallback на rule если LLM/file
     отсутствует
   - Runner.py: build `topology_router_llm` через `cfg.model.router` для
     `cfg.topology.name == "adaptive"`; передаёт через kwargs (две точки)
   - Поймано / проверено через `e3_router_smoke`: до фикса все 6 cells
     `decided_by='rule'`, после — `4 rule + 2 llm_router` (cost > 0)

### Фичи

4. **`atm oracle` CLI** (`24d5529`)
   - `atm oracle --exp-id <UUID> [--out PATH]`
   - Дёргает `build_leave_one_out_oracle()` + `OracleTable.to_router_dict()`,
     пишет JSON в `data/oracle/e1_leave_one_out.json` (default —
     ровно туда где OracleTopologyRouter ищет)
   - Используется когда дополит E1; cli доступна на всех машинах

5. **Auto-flush + `atm export-exp` CLI** (`712591a`)
   - `src/atm/storage/export.py`: `export_experiment(exp_id, session_factory,
     root)` пишет `experiment.json` + `_runs.parquet` в
     `{root}/experiments/{exp_id}/`
   - В `run_grid` после `_update_experiment_status` авто-вызывает
     `export_experiment` (re-uses session_factory, try/except debug-swallow)
   - CLI `atm export-exp --exp-id <UUID>` для ручного дампа существующих
     exps без рестарта
   - **13 unit-тестов** для export + 1 stub в `test_grid_progress` (чтобы
     inline-worker counter не считал лишний run_in_executor от export)

### Конфиги (9 yaml)

E2:
- `conf/experiments/e2_pilot_sanity.yaml` (6 cells)
- `conf/experiments/e2_pilot.yaml` (30 cells)

E3:
- `conf/experiments/e3_pilot_sanity.yaml` (6 cells)
- `conf/experiments/e3_pilot.yaml` (30 cells)
- `conf/experiments/e3_router_smoke.yaml` (6 cells)
- `conf/experiments/e3_commongen_smoke.yaml` (10 cells, диагностика после finalize фикса)

E4:
- `conf/experiments/e4_pilot_sanity.yaml` (6 cells)
- `conf/experiments/e4_role_smoke.yaml` (6 cells)
- `conf/experiments/e4_pilot.yaml` (45 cells)

Все включают `e1_pilot.yaml` или `e3_full.yaml`, лимиты budget/timeout
понижены для smoke. `e4_*` явно задают cross-family
`role_router_model: openai:gpt-4.1-mini`.

### Handoffs

- `arch/e3_run_handoff.md` (326 строк) — пошаговая инструкция для E3 на
  16/32 box с 4 wave'ами (sanity / router_smoke / pilot / full × 3 modes
  параллельно)
- `arch/e4_run_handoff.md` (388 строк) — параллельно для E4 (4 wave'а,
  single launch full с sweep по role_router внутри)

---

## Тесты

- `tests/unit/`: **1874 passed, 0 failed** (после моих изменений)
- Новые: 13 для `export`, 3 для `_commongen_needs_finalize`, 6 для
  topology_router wiring через adaptive

---

## Состояние экспериментов (на момент 2026-05-17 11:00 UTC+3)

| Эксп | Где | Статус | Файлы |
|---|---|---|---|
| **E1 full** | другая 16/32 машина | ✅ completed, exp `f153400f` | `analysis/e1_top3.json` + oracle |
| **E1 oracle** | committed в репу | ✅ ready | `data/oracle/e1_leave_one_out.json` |
| **E2 full** | другая машина | ✅ completed (2295/2295, $42, 13.92h) | `analysis/e2_results.json` |
| **E3 (rule/llm/oracle)** | TBD | ❌ не запущен | конфиг + handoff готов |
| **E4 (3 role_routers)** | TBD | ❌ не запущен | конфиг + handoff готов |
| **E5 real human** | — | ❌ обсуждается | spec в `experiment_plan.md §7` |

---

## Что нужно от пользователя

1. **Дамп PG для E1 + E2** с других машин — для self-contained dataset.
   На каждой машине:
   ```bash
   git pull --ff-only
   for EXP in $(docker exec atm-postgres psql -U atm -d atm -t -c \
       "SELECT id FROM experiments WHERE status IN ('completed','partial','failed')"); do
       uv run atm export-exp --exp-id "$EXP"
   done
   git add data/experiments/experiments/*/experiment.json data/experiments/experiments/*/_runs.parquet
   git commit -m "data: snapshot PG aggregates from machine-X"
   git push
   ```
   После этого на любой машине `git pull` и аналитика без PG-зависимости.

2. **Champion topology_router** из E3 (когда запустится) — нужен для
   обновления `e4_full.yaml` (фиксируем `topology_router` на winner).

3. **Решение по E5**: spec / mini-infra / lean Google Forms / отложить.

---

## Известные проблемы (не зафикшены, но осознаны)

| # | Что | Где | Severity |
|---|---|---|---|
| 1 | LLM router выбирает плохую топологию → q=0 в edge case | `e3_router_smoke` показал 2/6 cells | low — fallback на rule работает |
| 2 | CommonGen ROUGE-L потолок ~0.78 (lexical) | `tasks/commongen.py` | medium — для diploma можно добавить BERTScore или LLM-judge |
| 3 | `_infer_task_type` ожидает `task_id` с слешем (`humaneval/0`), для наших runs `task_id='humaneval'` без слеша → `by_task_type='unknown'` в oracle | `analysis/oracle.py` | low — `by_task_id` всё равно полный |
| 4 | `atm grid estimate` падает на include-based configs (FieldRequired) | pre-existing, до этой сессии | low — обходится `--no-estimate` |
| 5 | Cerebras 429 TPM при параллельных grid'ах на одном ключе | внешнее | medium — учтено в handoff'ах |
| 6 | DABench CSV upstream 404 на отдельных задачах | внешнее | low — DABench включён обратно (пользователь сообщил что починен) |
