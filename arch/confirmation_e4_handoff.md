# Confirmation E4 Run Handoff — Box B (16/32)

Полная инструкция для запуска **confirmation E4** на втором 16/32 боксе
(параллельно с confirmation_e3 на Box A).

**Цель:** проверить что E4 находки воспроизводятся на cross-family worker'е
`cerebras:qwen-3-235b-a22b-instruct-2507`. Конкретно — что role_router=rule
остаётся champion'ом (delta vs fixed > 0), а llm_router не подтверждается.

**Дата подготовки:** 2026-05-17
**КРИТИЧНЫЙ ДЕДЛАЙН:** qwen-3-235b deprecates **2026-05-27** (Cerebras). 10 дней.
**Связанный handoff:** `arch/confirmation_e3_handoff.md` для Box A — запускать ПАРАЛЛЕЛЬНО (разные ключи Cerebras рекомендуются).

---

## Контекст — что мы валидируем

Original E4 на `cerebras:gpt-oss-120b` (exp_id `272bb2cf-e1c2-4e0d-96c8-432f4e5f9a29`):

| role_router | mean_q | mean_cost | delta vs fixed |
|---|---|---|---|
| fixed (baseline) | 0.6531 | $0.01186 | — |
| **rule** (champion) | **0.6799** | $0.01193 | **+0.0268** ✓ RQ4 confirmed |
| llm | 0.6432 | $0.01168 | −0.0099 ✗ RQ4 НЕ confirmed |

Per-task winners:
- humaneval: rule (+0.0667 vs fixed)
- gsm8k: fixed (tied with rule at 0.978 — saturated)
- commongen: llm (+0.0234)
- dabench: llm (+0.0259, но std 0.36 = noise)

**Вопросы к confirmation:**
1. Сохраняется ли rule_q > fixed_q > llm_q порядок на qwen?
2. Сохраняется ли magnitude rule_delta ≈ +0.027 (±50%)?
3. Меняется ли role distribution? (rule на gpt-oss выдал 177 coordinator + 3 peer — почти deterministic; на qwen может быть другая раскладка)

---

## 0. Prerequisites

- **Hardware:** 16 vCPU / 32 GB RAM
- **Cerebras API key:** ЖЕЛАТЕЛЬНО отдельный от Box A (общий TPM/RPM иначе)
- **OpenAI API key:** тот же что везде (judge `gpt-4.1-mini`)

---

## 1. Setup (если бокс уже настроен — пропускай до §2)

```bash
git clone https://github.com/cactus-tim/adaptive-topologies-mas.git
cd adaptive-topologies-mas
git pull --ff-only origin main

uv sync

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

## 2. Sanity checks (5 мин)

### 2.1. Qwen entry в pricing.yaml

```bash
grep -A2 "qwen-3-235b" conf/pricing.yaml
# Ожидаемо: input_per_1k: 0.00060, output_per_1k: 0.00120
```

### 2.2. Qwen endpoint живой

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
```

### 2.3. Конфиги валидны

```bash
uv run python -c "
from atm.experiment.loader import load_grid_configs
for f in ['confirmation_e4_sanity.yaml', 'confirmation_e4.yaml']:
    cfgs = load_grid_configs('conf/experiments/'+f)
    print(f'{f}: cells={len(cfgs)}')
"
# Ожидаемо:
#   confirmation_e4_sanity.yaml: cells=3
#   confirmation_e4.yaml: cells=540
```

---

## 3. Запуск — поэтапно

### Wave 1: sanity smoke (~3 мин, 3 cells, $0.30)

```bash
mkdir -p logs
uv run atm grid run -c conf/experiments/confirmation_e4_sanity.yaml --no-estimate -p 3 > logs/confirmation_e4_sanity.log 2>&1
grep "GridResult" logs/confirmation_e4_sanity.log
```

**Acceptance:**
- 3/3 completed, 0 failed
- 3 разных role_router значения отработали (fixed/rule/llm)

**Если упало** → НЕ запускай full. Типичные проблемы:
- `_fallback_gateway None` на llm role_router — должно быть починено в commit `983b1b5`
- `topology_router_llm kwarg missing` — фикс в commit `24d5529`
- 401 / API key — проверь `CEREBRAS_API_KEY`
- 404 / model — qwen deprecated раньше; fallback на `cerebras:zai-glm-4.7`

### Wave 2: full confirmation (~9-10h, 540 cells, ~$45)

```bash
nohup uv run atm grid run \
    -c conf/experiments/confirmation_e4.yaml \
    --no-estimate -p 8 \
    > logs/confirmation_e4.log 2>&1 &
CONF_PID=$!

# Мониторинг (раз в 2 мин):
watch -n 120 'grep -oE "\[[0-9]+/[0-9]+\] done=[0-9]+ failed=[0-9]+" logs/confirmation_e4.log | tail -3'

# Дождаться:
wait $CONF_PID
grep GridResult logs/confirmation_e4.log
```

**Acceptance:**
- 540/540 attempted, failed ≤ 5%
- mean_q per (role_router × task) ненулевые
- 3 role_router режима дали ≠ распределения active_role в `human_interactions.parquet`

**Failure modes:**

| Симптом | Лечение |
|---|---|
| `429 Tokens per minute limit` | Снизь `-p` до 6 или 4. Qwen preview — более жёсткие квоты. |
| `404 model not found` | Qwen deprecated раньше срока. Fallback: правь `model.*` в `confirmation_e4.yaml` на `cerebras:zai-glm-4.7`. |
| `BudgetExceededError per_experiment` | Бамп `budget.per_experiment_usd` со 100 до 150. |
| HITL timeout — все cells fail | Проверь что `human.gateway: llm_simulated` (не `human_input`); должно быть так из e4_full. |
| Cell зависает >15 мин | Auto-kill через wall_time_s. |

---

## 4. После завершения

### 4.1. PG check

```bash
docker exec atm-postgres psql -U atm -d atm -c "
SELECT name, status, total, completed, failed
FROM experiments WHERE name='confirmation_e4';"
```

### 4.2. Dump → parquet

```bash
EXP=$(docker exec atm-postgres psql -U atm -d atm -tA -c "SELECT id FROM experiments WHERE name='confirmation_e4' ORDER BY created_at DESC LIMIT 1;")
echo "exp_id: $EXP"
uv run atm export-exp --exp-id $EXP --root data/experiments
ls -la data/experiments/experiments/$EXP/
```

### 4.3. Push snapshot

```bash
git add data/experiments/experiments/$EXP/
git commit -m "data(confirmation-e4): parquet snapshot, exp_id=$EXP"
git push origin main
```

Сообщи юзеру:
- `exp_id`
- `mean_q` per role_router (fixed / rule / llm)
- delta rule_vs_fixed (главная метрика confirmation)
- failed count, wall time

---

## 5. Что НЕ делать

- **НЕ запускай с тем же CEREBRAS_API_KEY что Box A** — общий TPM пол. На разных ключах OK.
- **НЕ меняй judge model** — `openai:gpt-4.1-mini` константа.
- **НЕ меняй topology_router** — фиксирован `llm` (E3 champion), как в e4_full. Меняя его смешаешь E3 + E4 эффекты.
- **НЕ меняй sweep размер** — 540 mirrors e4_full apples-to-apples.
- **НЕ паникуй если знак delta поменялся** — валидный научный результат ("E4 finding family-specific").

---

## 6. Эскалация

Падает не по списку → собрать:
1. Последние 100 строк `logs/confirmation_e4.log`
2. `.env` БЕЗ ключей
3. `git rev-parse HEAD`
4. Спросить пользователя.

---

## Чек-лист TL;DR

```bash
git pull --ff-only
uv sync
set -a; source .env; set +a
uv run alembic upgrade head

# Sanity
uv run python -c "from atm.experiment.loader import load_grid_configs; print(len(load_grid_configs('conf/experiments/confirmation_e4.yaml')))"
# → 540

# Wave 1: smoke (3 мин, $0.30)
uv run atm grid run -c conf/experiments/confirmation_e4_sanity.yaml --no-estimate -p 3 > logs/confirmation_e4_sanity.log 2>&1
grep GridResult logs/confirmation_e4_sanity.log

# Wave 2: full (9-10h, $45)
nohup uv run atm grid run -c conf/experiments/confirmation_e4.yaml --no-estimate -p 8 > logs/confirmation_e4.log 2>&1 &
wait
grep GridResult logs/confirmation_e4.log

# Post-run
EXP=$(docker exec atm-postgres psql -U atm -d atm -tA -c "SELECT id FROM experiments WHERE name='confirmation_e4' ORDER BY created_at DESC LIMIT 1;")
uv run atm export-exp --exp-id $EXP --root data/experiments
git add data/experiments/experiments/$EXP/
git commit -m "data(confirmation-e4): parquet snapshot exp_id=$EXP"
git push origin main
```
