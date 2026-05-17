# Confirmation E3 Run Handoff — Box A (16/32)

Полная инструкция для запуска **confirmation E3** на одном из двух 16/32 боксов.

**Цель:** проверить что E3 Pareto-нарратив (`topology_router=llm` quasi-парирует
`rule` по quality при ~2.3× меньшей цене) воспроизводится на cross-family
worker'е — `cerebras:qwen-3-235b-a22b-instruct-2507`. Judge остаётся
`openai:gpt-4.1-mini` (постоянный измерительный прибор).

**Дата подготовки:** 2026-05-17
**КРИТИЧНЫЙ ДЕДЛАЙН:** qwen-3-235b deprecates **2026-05-27** (Cerebras). Запас 10 дней.
**Связанный handoff:** confirmation E4 будет в отдельном доке для Box B (после E4_full).

---

## 0. Prerequisites

- **Hardware:** 16 vCPU / 32 GB RAM
- **Tools:** `git`, `uv`, `docker` (PostgreSQL)
- **Cerebras API key:** ЛЮБОЙ из существующих ключей (qwen-3-235b — preview, отдельных квот нет; идёт по общим RPM/TPM аккаунта)
- **OpenAI API key:** тот же что и везде (для judge `gpt-4.1-mini`)

---

## 1. Setup (если бокс уже был использован для E3/E4 — пропускай до §2)

```bash
git clone https://github.com/cactus-tim/adaptive-topologies-mas.git
cd adaptive-topologies-mas
git pull --ff-only origin main

uv sync

# PostgreSQL via docker
docker run -d --name atm-postgres \
  -p 5433:5432 \
  -e POSTGRES_USER=atm -e POSTGRES_PASSWORD=atm -e POSTGRES_DB=atm \
  -c max_connections=500 \
  postgres:16-alpine
until docker exec atm-postgres pg_isready -U atm; do sleep 1; done

cat > .env <<'EOF'
CEREBRAS_API_KEY=<PASTE_KEY>
OPENAI_API_KEY=sk-<openai_key>
PG_DSN=postgresql+asyncpg://atm:atm@localhost:5433/atm
EOF

set -a; source .env; set +a
uv run alembic upgrade head
```

---

## 2. Sanity checks — ДО запуска (5 мин)

### 2.1. Pricing entry для qwen на месте

```bash
grep -A2 "qwen-3-235b" conf/pricing.yaml
# Ожидаемо: input_per_1k: 0.00060, output_per_1k: 0.00120
```

Если grep пустой → `git pull --ff-only` (должно быть в коммите `9c5edb9`).

### 2.2. Qwen endpoint живой + ключ работает

```bash
set -a; source .env; set +a
uv run python -c "
import asyncio
from atm.llm.factory import build_llm
from atm.llm.pricing import Pricing
from atm.llm.budget import BudgetTracker
from langchain_core.messages import HumanMessage

async def main():
    pricing = Pricing.from_yaml('conf/pricing.yaml')
    budget = BudgetTracker(per_call_usd=0.10, per_run_usd=1.0, per_experiment_usd=5.0)
    llm = build_llm(model_id='cerebras:qwen-3-235b-a22b-instruct-2507', pricing=pricing, budget=budget)
    resp = await llm.ainvoke([HumanMessage(content='Reply with just: ok')])
    print('OK:', str(resp.content if hasattr(resp, 'content') else resp)[:100])
asyncio.run(main())
"
# Ожидаемо: 'OK: ok' или вариант. Если 404 на model → qwen уже deprecated.
# Если 401 → CEREBRAS_API_KEY неверный.
```

### 2.3. Конфиги валидны

```bash
uv run python -c "
from atm.experiment.loader import load_grid_configs
for f in ['confirmation_e3_sanity.yaml', 'confirmation_e3.yaml']:
    cfgs = load_grid_configs('conf/experiments/'+f)
    print(f'{f}: cells={len(cfgs)}')
"
# Ожидаемо:
#   confirmation_e3_sanity.yaml: cells=2
#   confirmation_e3.yaml: cells=360
```

---

## 3. Запуск — поэтапно

### Wave 1: sanity smoke (~2 мин, 2 cells, $0.20)

```bash
mkdir -p logs
uv run atm grid run -c conf/experiments/confirmation_e3_sanity.yaml --no-estimate -p 2 > logs/confirmation_e3_sanity.log 2>&1
grep "GridResult" logs/confirmation_e3_sanity.log
```

**Acceptance:**
- 2/2 completed, 0 failed
- Оба cells вернули q > 0 (хоть какой-то ответ)

**Если упало** → НЕ запускай full, дебагай. Чек-лист:
- 401 / API key — проверь `CEREBRAS_API_KEY` в `.env`
- 404 / model — qwen дёрнули раньше дедлайна; fallback на `cerebras:zai-glm-4.7` (поменять `model.default` в обоих yaml)
- HITL fallback_gateway None — должно быть починено в commit `983b1b5`; проверь `git log --oneline -10 | grep fallback`

### Wave 2: full confirmation (~6 ч, 360 cells, ~$25-30)

```bash
nohup uv run atm grid run \
    -c conf/experiments/confirmation_e3.yaml \
    --no-estimate -p 8 \
    > logs/confirmation_e3.log 2>&1 &
CONF_PID=$!

# Мониторинг (раз в минуту):
watch -n 60 'grep -oE "\[[0-9]+/[0-9]+\] done=[0-9]+ failed=[0-9]+" logs/confirmation_e3.log | tail -3'

# Дождаться:
wait $CONF_PID
grep GridResult logs/confirmation_e3.log
```

**Acceptance:**
- 360/360 attempted; failed ≤ 5% (transient Cerebras / qwen rate-limits допустимы)
- mean_q per (router × task) ненулевые
- router sweep дал 2 distinct experiences (rule vs llm) — `decided_by` в `topology_transitions.parquet` различается

**Failure modes:**

| Симптом | Лечение |
|---|---|
| `429 Tokens per minute limit exceeded` | Снизь `-p` до 6 или 4. Qwen preview = более жёсткие квоты чем gpt-oss. |
| `404 model not found` | Qwen deprecated раньше срока. Переключись на zai-glm-4.7: правь `model.*` блок в `confirmation_e3.yaml`. |
| Postgres `too many connections` | Уже `max_connections=500`; если всё равно — рестартни docker. |
| `BudgetExceededError per_experiment` | Увеличь `budget.per_experiment_usd` в yaml до 80; qwen на DABench может tokens жечь больше ожидаемого. |
| Cell зависает >15 мин | `recursion_limit` высчитывается автоматически; cell сам убьётся через wall_time_s. |

---

## 4. После завершения

### 4.1. Проверь итоги в PG

```bash
docker exec atm-postgres psql -U atm -d atm -c "
SELECT name, status, total, completed, failed
FROM experiments WHERE name='confirmation_e3';"
```

### 4.2. Dump → parquet (snapshot для другой машины)

```bash
EXP=$(docker exec atm-postgres psql -U atm -d atm -tA -c "SELECT id FROM experiments WHERE name='confirmation_e3' ORDER BY created_at DESC LIMIT 1;")
echo "exp_id: $EXP"
uv run atm export-exp --exp-id $EXP --root data/experiments
ls -la data/experiments/experiments/$EXP/
# Должны увидеть: experiment.json + _runs.parquet
```

### 4.3. Закоммить parquet snapshot

```bash
git add data/experiments/experiments/$EXP/
git commit -m "data(confirmation-e3): parquet snapshot, exp_id=$EXP"
git push origin main
```

И сообщи юзеру:
- `exp_id`
- `mean_q` per router (rule / llm) из последней строки `GridResult`
- failed count
- wall time

---

## 5. Что НЕ делать

- **НЕ запускай confirmation параллельно с E4 на ОДНОМ ключе Cerebras** — общий TPM/RPM. На разных боксах с разными ключами — ок.
- **НЕ меняй judge model** — `openai:gpt-4.1-mini` остаётся постоянным во всех экспах включая confirmation. Иначе сравнивать с E3 будет невозможно.
- **НЕ свипай `topology_router=oracle`** — oracle файл собран на gpt-oss данных, на qwen дал бы смещение. Только rule + llm.
- **НЕ меняй sweep размер** (15 shuffle × 3 seeds) — он зеркалит E3_full ровно для apples-to-apples replication. Больше N — потеря comparability.
- **НЕ паникуй если delta направление поменялось** — это валидный научный результат для diploma (limitation: "Pareto-нарратив зависит от family workers").

---

## 6. Эскалация

Если падает не по списку — собери:
1. Последние 100 строк `logs/confirmation_e3.log`
2. Содержимое `.env` БЕЗ ключей
3. `git rev-parse HEAD`
4. И спроси пользователя.

---

## Чек-лист TL;DR (копипаст)

```bash
# Setup (один раз — если новый бокс)
git pull --ff-only
uv sync
# (docker postgres см. §1 если не запущен)
set -a; source .env; set +a
uv run alembic upgrade head

# Sanity (5 мин)
uv run python -c "from atm.experiment.loader import load_grid_configs; print(len(load_grid_configs('conf/experiments/confirmation_e3.yaml')))"
# → 360

# Wave 1: smoke (2 мин, $0.20)
uv run atm grid run -c conf/experiments/confirmation_e3_sanity.yaml --no-estimate -p 2 > logs/confirmation_e3_sanity.log 2>&1
grep GridResult logs/confirmation_e3_sanity.log
# acceptance: 2/2 completed, 0 failed

# Wave 2: full (6 ч, $25-30)
nohup uv run atm grid run -c conf/experiments/confirmation_e3.yaml --no-estimate -p 8 > logs/confirmation_e3.log 2>&1 &
wait
grep GridResult logs/confirmation_e3.log

# Post-run: export + push
EXP=$(docker exec atm-postgres psql -U atm -d atm -tA -c "SELECT id FROM experiments WHERE name='confirmation_e3' ORDER BY created_at DESC LIMIT 1;")
uv run atm export-exp --exp-id $EXP --root data/experiments
git add data/experiments/experiments/$EXP/
git commit -m "data(confirmation-e3): parquet snapshot exp_id=$EXP"
git push origin main
```
