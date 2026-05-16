# Hardware Runbook — 16 vCPU / 32 GB RAM

Гайд по подбору параметров под более мощную машину (8-16 ядер + RAM запас).

Текущая dev-машина: 6 vCPU / 15 GB RAM → `parallelism: 12` упирается в CPU
(load avg ~8.4). Новая машина: 16/32 → можно поднять parallelism в 2-3 раза.

---

## TL;DR — параметры для копипасты

Открыть `conf/experiments/e1_pilot.yaml` (и других где `grid.parallelism`) и
изменить:

```yaml
grid:
  parallelism: 24       # было 12 на 6-core; на 16-core безопасно 24
  # parallelism: 30     # агрессивный вариант (75% × 16 cores × 2.5 oversub)
```

Опционально подкрутить max_iterations для скорости (но не безопасности):

```yaml
topology:
  max_iterations: 6     # было 8 — большинство задач сходятся за 3-7 iter
  extra:
    mesh:
      max_rounds: 12    # НЕ снижать — мешает round-robin executor'у
    star:
      exec_max_iter: 3      # уже smart-cut
      verify_max_iter: 2
    debate:
      max_rounds: 2     # уже smart-cut
    hierarchical:
      max_rounds: 2     # уже smart-cut
```

---

## Расчёт parallelism

### Bottleneck analysis

| Resource | На что тратится | Лимит |
|---|---|---|
| **CPU** | LangGraph state machines, JSON parse, sandbox subprocesses (HumanEval), event loop overhead | ~1.5-2 workers/core (I/O-bound на Cerebras await) |
| **RAM** | Python event loop + agents + LLM wrapper + sandbox = ~250-400 MB / worker | 32 GB / 0.4 GB = ≤80 workers (RAM не bottleneck) |
| **Cerebras quota PAYG** | 1000 RPM / 1M TPM на `gpt-oss-120b` | На p=30: ~60 RPM = 6% квоты |
| **Network/socket** | Outbound HTTPS to Cerebras + OpenAI | Лимит ~1024 fd по умолчанию |

### Формула

```
optimal_parallelism ≈ min(
    vCPUs × 2.0,                    # I/O oversub fits 2x for await-heavy workers
    rate_limit_rpm / req_per_worker_per_min,
    avail_ram_gb / 0.4
)
```

Для 16/32:
- CPU bound: `16 × 2.0 = 32` (теоретический максимум)
- Cerebras bound: `1000 / 60 = 16` minimum (если 1 req/min на worker)
   — но в реальности **request latency = 10-30s**, поэтому per worker
   ≈ 2-6 RPM → можно держать 160-500 workers по Cerebras. Не bottleneck.
- RAM bound: `28 / 0.4 = 70`

→ **CPU — реальный потолок. Рекомендую parallelism=24 (safe) или 30 (агрессивно).**

---

## Этапы deployment на новой машине

### 1. Setup (~10 мин)

```bash
git clone https://github.com/cactus-tim/adaptive-topologies-mas.git
cd adaptive-topologies-mas
uv sync                          # installs all deps via uv (fast)

# Постгрес (если ещё нет):
docker run -d --name atm-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=atm_main \
  postgres:16
sleep 5
uv run alembic upgrade head      # apply migrations

# Env
cp .env.example .env             # заполнить ключи (см. ниже)
```

`.env` должен содержать:
```
CEREBRAS_API_KEY=...             # https://cloud.cerebras.ai/
OPENAI_API_KEY=sk-...            # OpenAI (для judge на gpt-4.1-mini)
HF_TOKEN=hf_...                  # опционально (для dabench fallback)
PG_DSN=postgresql+asyncpg://postgres:postgres@localhost/atm_main
```

### 2. Sanity (~5 мин)

```bash
uv run pytest tests/unit/ -q     # должно быть 1818+ passed
uv run python -W error::DeprecationWarning -c "
import glob
from atm.experiment.config import load_config
for f in sorted(glob.glob('conf/experiments/*.yaml')):
    load_config(f); print(f, 'ok')
"
```

### 3. Tune parallelism

Edit `conf/experiments/e1_pilot.yaml`:
```yaml
grid:
  parallelism: 24   # начинаем с 24
```

Это автоматически отнаследуется в `e1_mini.yaml`, `e1_full.yaml`,
`e1_mini_fixes.yaml`, `e1_pilot_sanity.yaml`, `e1_sanity_hier.yaml`
(они `include: e1_pilot.yaml`).

Для E2/E3/E4 — отдельно поднять там же.

### 4. Validate parallelism (~3 мин)

Прогнать e1_mini_fixes как warm-up:
```bash
uv run atm grid run -c conf/experiments/e1_mini_fixes.yaml > logs/warmup.log 2>&1 &
sleep 60
# Параллельно мониторить load:
uptime  # load avg должен быть < 32 (= 2× cores)
free -h  # RAM available > 4 GB
```

Если `load avg > 30` (т.е. больше чем 2× cores) — снизить parallelism до 18.
Если `load avg < 20` и RAM спокоен → можно увеличить до 30.

### 5. E1-mini (~5-15 мин)

```bash
uv run atm grid run -c conf/experiments/e1_mini.yaml > logs/e1_mini_v6.log 2>&1
# 100 cells / 24 = ~4 батча
# Старая wall-time @ p=12 на 6-core: ~25-35 мин
# Новая @ p=24 на 16-core: 5-15 мин
```

Acceptance: все 5 топологий ≥ 0.5 mean (см. `arch/session_handoff_2026-05-16.md` §7).

### 6. E1 full (~1-2 ч)

```bash
uv run atm grid run -c conf/experiments/e1_full.yaml > logs/e1_full.log 2>&1 &
```

900 cells / 24 = 38 batch ≈ **~75 мин на 16/32** (vs 2.5-4ч на 6-core).

---

## Мониторинг прогона

В отдельном терминале:

```bash
# Watch progress
tail -f logs/e1_full.log | grep -oE "\[[0-9]+/[0-9]+\] done=[0-9]+ failed=[0-9]+ in_progress=[0-9]+ eta=[0-9]+s"

# System load
watch -n 5 'uptime && free -h | head -2 && ps -ef | grep "atm grid" | grep -v grep | wc -l'

# Per-topology mean quality (для запущенного run'а)
python3 -c "
import json, re
from collections import defaultdict
starts = {}; agg = defaultdict(list)
with open('logs/e1_full.log') as f:
    for line in f:
        m = re.search(r'\{.*\"event\": \"run started\".*\}', line)
        if m:
            try: o = json.loads(m.group(0)); starts[o['run_id']] = (o.get('topology'), o.get('task'))
            except: pass
        m = re.search(r'\{.*\"event\": \"run completed\".*\}', line)
        if m:
            try:
                o = json.loads(m.group(0))
                t, _ = starts.get(o['run_id'], (None, None))
                agg[t].append(o.get('quality_score'))
            except: pass
for t in sorted(agg):
    qs = [q for q in agg[t] if q is not None]
    if qs: print(f'{t:14s} n={len(qs):3d} q={sum(qs)/len(qs):.2f}')
"
```

---

## Что делать если что-то идёт не так

### Слишком долго / висит

1. Глянуть Cerebras dashboard на rate limits
2. `tail -f logs/e1_full.log | grep -i "error\|timeout\|rate"`
3. Если стабильные timeout'ы — снизить `parallelism` (16 → 12)

### Memory pressure

`free -h` показывает swap usage > 1 GB → снизить `parallelism` или
killnouт зомби-worker'ов: `pkill -f 'atm grid run'` и перезапустить.

### Postgres backpressure

`max_connections` по умолчанию 100. При parallelism=30 каждый worker
открывает 2-3 connection → 60-90, близко к лимиту. Если падает с
`FATAL: sorry, too many clients` — увеличить:
```sql
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```
И перезапустить postgres.

### Mesh снова даёт 0 quality

Проверь после миграции конфигов что `extra.mesh.max_rounds >= 8` —
если нет (например default из-за неправильного namespace), вернётся
старый starvation bug. См. `arch/session_handoff_2026-05-16.md` §5.

---

## Бюджет на новой машине

| Этап | Wall-time @ p=24 16-core | Cerebras | OpenAI judge | Итого |
|------|--------------------------|----------|--------------|-------|
| E1-mini | 5-15 мин | $1.5 | $0.3 | $2 |
| E1 full (900) | 75 мин | $50 | $10 | $60 |
| E2 full (2160) | 5-6 ч | $150 | $20 | $170 |
| E3 (×3 + baseline) | 6 ч | $90 | $15 | $105 |
| E4 full (540) | 2.5 ч | $40 | $5 | $45 |
| **Итого** | **~15 ч wall** | **~$330** | **~$50** | **~$380** |

(Старая оценка на 6-core: ~32 ч wall, $435 — savings от bigger box: 50%
времени, ~$50 за счёт меньшего overhead на retry / ETA waste.)

---

## Альтернативы если эта машина окажется в bottleneck

1. **Hetzner CCX43** (16 vCPU dedicated, 64 GB RAM) — €0.083/час → $2 на
   полный E1-E4 цикл (16 ч).
2. **Hetzner CCX53** (32 vCPU) — €0.166/час → ещё в 2× быстрее, $4-5.
3. **Spot instances AWS/GCP** — дешевле но риск preemption во время
   long-running E2.

Для diploma timeline дешевле остановиться на одной нормальной машине и
прогонять последовательно.
