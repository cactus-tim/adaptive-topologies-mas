# E3 Run Handoff — Claude Code on 16/32 box

Полная инструкция для запуска **E3 (adaptive + HITL, RQ2)** на другой
машине. Эта машина (6-core) не справляется параллельно с zalupa2 (5.3
ядра занято), поэтому E3 переезжает на 16/32.

**Дата подготовки:** 2026-05-16
**Контекст:** E1 завершён (exp_id `f153400f-f8ba-4ca3-8dc2-66f31ba15236`),
oracle файл уже в репе (`data/oracle/e1_leave_one_out.json`).

---

## 0. Prerequisites — что должно быть на машине

- **Hardware:** 16 vCPU / 32 GB RAM (Hetzner CCX43 или эквивалент)
- **OS:** Linux (Ubuntu/Debian)
- **Tools:** `git`, `uv`, `docker` (для PostgreSQL), `psql` (опционально)
- **Cerebras API key:** ОТДЕЛЬНЫЙ от того что бежал E1/E2 (TPM коллизии).
  Пользователь даст ключ в чат — положить в `.env` как `CEREBRAS_API_KEY=...`
- **OpenAI API key:** тот же что и везде (для judge и HITL gateway)
- **HF token:** опционально (для DABench fallback на HF mirror)

---

## 1. Setup — единоразово

```bash
# Clone (если ещё не)
git clone https://github.com/cactus-tim/adaptive-topologies-mas.git
cd adaptive-topologies-mas

# Pull last
git pull --ff-only origin main

# Install deps via uv
uv sync

# PostgreSQL via docker (если ещё нет)
docker run -d --name atm-postgres \
  -p 5433:5432 \
  -e POSTGRES_USER=atm \
  -e POSTGRES_PASSWORD=atm \
  -e POSTGRES_DB=atm \
  -c max_connections=500 \
  postgres:16-alpine

# Wait until healthy
until docker exec atm-postgres pg_isready -U atm; do sleep 1; done

# .env (создать если нет)
cat > .env <<'EOF'
CEREBRAS_API_KEY=<PASTE_NEW_KEY_HERE>
OPENAI_API_KEY=sk-<openai_key>
HF_TOKEN=hf_<token>
PG_DSN=postgresql+asyncpg://atm:atm@localhost:5433/atm
EOF

# Apply migrations
set -a; source .env; set +a
uv run alembic upgrade head
```

---

## 2. Sanity checks — ДО запуска экспериментов (5 мин)

### 2.1. Unit tests

```bash
uv run pytest tests/unit/ -q
# Ожидаемо: ~1818 passed, 0 failed (можно 1 flaky в test_oracle async)
```

### 2.2. Все конфиги валидны

```bash
uv run python -c "
from atm.experiment.loader import load_grid_configs
import os
for f in sorted(os.listdir('conf/experiments/')):
    if not (f.startswith('e3_') or f == 'e3_full.yaml') or not f.endswith('.yaml'):
        continue
    cfgs = load_grid_configs('conf/experiments/'+f, overrides=[])
    print(f'{f:30s} cells={len(cfgs):4d}')
"
# Ожидаемо:
#   e3_commongen_smoke.yaml        cells=  10
#   e3_full.yaml                   cells= 180
#   e3_pilot.yaml                  cells=  30
#   e3_pilot_sanity.yaml           cells=   6
#   e3_router_smoke.yaml           cells=   6
```

### 2.3. Oracle файл на месте

```bash
ls -la data/oracle/e1_leave_one_out.json
cat data/oracle/e1_leave_one_out.json | python3 -m json.tool | head -20
# Ожидаемо: 32-строчный JSON с by_task_id для humaneval/gsm8k/commongen/dabench
```

### 2.4. Cerebras + OpenAI ключи работают

```bash
set -a; source .env; set +a
uv run python -c "
import os
from atm.llm.factory import build_llm
from atm.llm.pricing import Pricing
from atm.core.types import Message, MessageKind

p = Pricing.from_file('conf/pricing/prices.yaml')
import asyncio
async def t():
    for model in ['cerebras:gpt-oss-120b', 'openai:gpt-4.1-mini']:
        llm = build_llm(model_id=model, pricing=p, budget=None)
        r = await llm.ainvoke([Message(sender='u', kind=MessageKind.REQUEST, content='reply with: ok')], agent_id='sanity')
        print(f'{model}: {r.text[:60]!r}')
asyncio.run(t())
"
# Ожидаемо: оба возвращают что-то осмысленное.
```

---

## 3. Запуск E3 — поэтапно

### Wave 1: e3_pilot_sanity (~5 мин, 6 cells, $0.10)

Минимальный wiring. Если падает — НЕ продолжай, дебагай.

```bash
mkdir -p logs
uv run atm grid run -c conf/experiments/e3_pilot_sanity.yaml --no-estimate -p 6 > logs/e3_pilot_sanity.log 2>&1
tail -3 logs/e3_pilot_sanity.log
grep "GridResult" logs/e3_pilot_sanity.log
# Acceptance: 6/6 completed, 0 failed, mean q ≥ 0.8 на humaneval+gsm8k
```

Если упало:
- `fallback_gateway` ошибки? — должны быть починены в commit 983b1b5; проверь `git log --oneline -5` что коммит есть
- Cerebras 429? — снижай `-p` до 4

### Wave 2: e3_router_smoke (~5 мин, 6 cells, $0.10)

**КЛЮЧЕВОЙ smoke** — проверяет что oracle файл подцепился.

```bash
uv run atm grid run -c conf/experiments/e3_router_smoke.yaml --no-estimate -p 6 > logs/e3_router_smoke.log 2>&1
# После завершения проверь decided_by в parquet:
EXP=$(grep "GridResult" logs/e3_router_smoke.log | sed 's/.*exp_id=\([^ ]*\).*/\1/')
uv run python -c "
import pandas as pd, glob
for p in sorted(glob.glob(f'data/experiments/experiments/$EXP/runs/*/topology_transitions.parquet')):
    rid=p.split('/')[-2][:8]
    tt=pd.read_parquet(p)
    by=tt.groupby('decided_by').size().to_dict() if 'decided_by' in tt.columns else {}
    print(f'  {rid}  decided_by={by}')
"
```

Acceptance:
- 6/6 completed
- **МИНИМУМ 3 разных values в `decided_by`**: должны быть `rule`, `llm_router`, `oracle`
- Если `oracle` отсутствует — oracle файл не подцепился, проверь путь
- Если 2 LLM router cells получили q=0 — это known degenerate case, не блокер

### Wave 3: e3_pilot battle (~30-60 мин, 30 cells, $1)

Полный smoke с 3 task types на adaptive.

```bash
uv run atm grid run -c conf/experiments/e3_pilot.yaml --no-estimate -p 8 > logs/e3_pilot.log 2>&1 &
PILOT_PID=$!

# Мониторинг прогресса:
watch -n 30 'grep -oE "\[[0-9]+/[0-9]+\] done=[0-9]+ failed=[0-9]+" logs/e3_pilot.log | tail -3'

# Дождаться:
wait $PILOT_PID
grep GridResult logs/e3_pilot.log
```

Acceptance (per task):
- humaneval: mean_q ≥ 0.8, 0 failed
- gsm8k: mean_q ≥ 0.9, 0 failed (Path C finalize работает)
- commongen: mean_q ≥ 0.55, 0 failed (фикс из commit 499de6c)
- Total failed ≤ 2 (transient Cerebras issues допустимы)

### Wave 4: E3 full — 3 router modes (~6 ч, 540 cells, $40)

**КРИТИЧНО:** `e3_full.yaml` НЕ свипает по `topology.extra.adaptive.topology_router`.
Нужно запустить **3 раза** с разным override, либо отредактировать
`e3_full.yaml` чтобы добавить sweep по router_mode (тогда cells = 540, не 180).

#### Вариант A — 3 отдельных запуска (рекомендую, чище exp_id)

```bash
for MODE in rule llm oracle; do
    echo "=== E3 full, topology_router=$MODE ==="
    EXP_NAME="e3_full_router_${MODE}"
    uv run atm grid run \
        -c conf/experiments/e3_full.yaml \
        --no-estimate -p 24 \
        +name=$EXP_NAME \
        +topology.extra.adaptive.topology_router=$MODE \
        > logs/${EXP_NAME}.log 2>&1
    echo "$MODE done — GridResult:"
    grep GridResult logs/${EXP_NAME}.log
done
```

Wall: ~2 ч × 3 = 6 ч на p=24, $40 total.

Acceptance per router mode:
- rule:   mean_q ≥ 0.7 (baseline)
- llm:    mean_q ≥ 0.65 (LLM router может выбрать неоптимально — норм)
- oracle: mean_q ≥ 0.8 (upper bound)
- Все три: failed ≤ 5% от cells

#### Вариант B — единый sweep (если хочется один exp_id)

Отредактируй `conf/experiments/e3_full.yaml` — добавь в `grid.sweep`:
```yaml
    topology.extra.adaptive.topology_router:
      - rule
      - llm
      - oracle
```
Получится 540 cells одним запуском. Менее удобно для split-by-router-mode
анализа, но один прогон.

### Failure modes — что делать

| Симптом | Лечение |
|---|---|
| `429 Tokens per minute limit exceeded` | Снизь `-p` (24 → 16). На отдельном Cerebras ключе обычно ОК на p=24. |
| `oracle table data/oracle/...json missing` | `ls -la data/oracle/` — файл должен быть. Если нет — `git pull`. |
| `topology_router='llm' but no topology_router_llm` kwarg | Баг в коде, не в конфиге. Проверь что commit 24d5529 в HEAD. |
| Postgres `too many connections` | Уже `max_connections=500` в setup. Если всё равно — рестартни docker. |
| Adaptive cell зависает > 10 мин | `recursion_limit` хитро высчитывается. Cell сам убьётся через `wall_time_s` лимит. |
| Mean q << ожидаемого на task type | Дёрни parquet конкретного run, посмотри `topology_transitions`, `llm_calls`, что executor выдал. Гайд — в `arch/session_handoff_2026-05-16.md`. |

---

## 4. После завершения

### 4.1. Проверь итоги
```bash
docker exec atm-postgres psql -U atm -d atm -c "
SELECT name, status, total, completed, failed,
       ROUND(AVG(quality_score)::numeric, 2) as avg_q
FROM experiments e
LEFT JOIN runs r ON r.exp_id=e.id AND r.quality_score IS NOT NULL
WHERE name LIKE 'e3_full_router_%'
GROUP BY name, status, total, completed, failed;"
```

### 4.2. Сохрани результаты + лог
```bash
mkdir -p arch/results
cp logs/e3_full_router_*.log arch/results/
# Краткий summary в markdown
docker exec atm-postgres psql -U atm -d atm -c "
SELECT (SELECT name FROM experiments WHERE id=r.exp_id) as exp,
       r.task_id, COUNT(*),
       ROUND(AVG(r.quality_score)::numeric, 2) as mean_q
FROM runs r WHERE (SELECT name FROM experiments WHERE id=r.exp_id) LIKE 'e3_full_router_%'
GROUP BY exp, r.task_id ORDER BY exp, r.task_id;" > arch/results/e3_full_summary.txt
```

### 4.3. Закоммить и пушни
```bash
git add arch/results/
git commit -m "data(e3): full router-sweep results (rule/llm/oracle)"
git push origin main
# Пользователю сообщи exp_id'ы трёх runs
```

---

## 5. Что НЕ делать
- **НЕ запускай E3 на 6-core машине** — там zalupa2 жрёт 5.3 ядра, wall будет в 10× больше.
- **НЕ пересобирай oracle файл** — он уже в репе (`data/oracle/e1_leave_one_out.json`). Пересобирать нужно ТОЛЬКО если E1 будет перезапущен.
- **НЕ запускай E3 параллельно с E2** на одном Cerebras ключе — TPM 429.
- **НЕ модифицируй analysis/e1_top3.json или data/oracle/** — это frozen artifacts.

---

## 6. Эскалация
Если что-то падает не по списку failure modes выше — собери:
1. Последние 100 строк лога
2. Список изменённых файлов (`git diff --stat HEAD~5`)
3. Содержимое `.env` БЕЗ ключей
4. И спроси пользователя.

---

## Чек-лист TL;DR (для копипаста)

```bash
# Setup (один раз)
git pull --ff-only
uv sync
docker run -d --name atm-postgres -p 5433:5432 -e POSTGRES_USER=atm -e POSTGRES_PASSWORD=atm -e POSTGRES_DB=atm -c max_connections=500 postgres:16-alpine
until docker exec atm-postgres pg_isready -U atm; do sleep 1; done
set -a; source .env; set +a
uv run alembic upgrade head

# Sanity
uv run pytest tests/unit/ -q
ls -la data/oracle/e1_leave_one_out.json

# Wave 1-3 smokes
uv run atm grid run -c conf/experiments/e3_pilot_sanity.yaml --no-estimate -p 6 > logs/e3_pilot_sanity.log 2>&1
uv run atm grid run -c conf/experiments/e3_router_smoke.yaml --no-estimate -p 6 > logs/e3_router_smoke.log 2>&1
uv run atm grid run -c conf/experiments/e3_pilot.yaml --no-estimate -p 8 > logs/e3_pilot.log 2>&1

# Wave 4: full E3
for MODE in rule llm oracle; do
    uv run atm grid run -c conf/experiments/e3_full.yaml --no-estimate -p 24 \
        +name=e3_full_router_${MODE} \
        +topology.extra.adaptive.topology_router=${MODE} \
        > logs/e3_full_router_${MODE}.log 2>&1
done
```
