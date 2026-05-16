# E4 Run Handoff — Claude Code on 16/32 box

Полная инструкция для запуска **E4 (adaptive topology + adaptive role-router, RQ4)**.
Запускать только **ПОСЛЕ** E2 и E3 (TPM коллизии на общем Cerebras
ключе, плюс нужен champion topology_router из E3).

**Дата подготовки:** 2026-05-16
**Контекст:** E1 завершён (oracle файл в репе), E2/E3 запущены. E4 —
финальный adaptive-both эксперимент: runtime topology switching (из E3) +
runtime role switching (новое для E4).

---

## 0. Prerequisites

- **Hardware:** 16 vCPU / 32 GB RAM
- **E2/E3 завершились** (или близки к завершению — TPM освободился)
- **E3 champion известен** — какой `topology_router` дал лучший mean_q.
  Узнать можно из:
  ```bash
  docker exec atm-postgres psql -U atm -d atm -c "
  SELECT REPLACE(name, 'e3_full_router_', '') as router_mode,
         ROUND(AVG(quality_score)::numeric, 3) as mean_q,
         COUNT(*) as n
  FROM runs r
  WHERE (SELECT name FROM experiments WHERE id=r.exp_id) LIKE 'e3_full_router_%'
    AND r.quality_score IS NOT NULL
  GROUP BY name ORDER BY mean_q DESC;"
  ```
  Запиши **winner** (rule | llm | oracle) — нужен для `e4_full.yaml`.

- **Cerebras key:** тот же что для E3 (после E3 finish — квота свободна)
- **OpenAI key:** тот же
- Если запускаешь сразу после E3 на этой же машине — `docker ps` проверь
  что atm-postgres ещё бежит, .env на месте.

---

## 1. Pre-flight updates

```bash
git pull --ff-only origin main
uv sync   # на случай если зависимости поменялись
```

### 1.1. Обновить e4_full.yaml champion

E4 fixes `topology_router` на champion из E3 и **свипает** только
`human.role_router`. Перед запуском E4 full нужно подставить:

```bash
# Замени <CHAMPION> на rule / llm / oracle по результатам E3
CHAMPION=rule   # ← заменить!

# В e4_full.yaml уже есть `topology_router: rule` в extras —
# либо отредактируй вручную, либо передавай overrides на CLI:
#   +topology.extra.adaptive.topology_router=$CHAMPION
```

### 1.2. КРИТИЧНО: cross-family role_router_model

`e4_full.yaml` сейчас НЕ задаёт `human.role_router_model` явно →
наследует `cfg.model.default = cerebras:gpt-oss-120b`. Это даст:
- TPM коллизию с workers (которые тоже на Cerebras)
- Не cross-family (один провайдер для всего HITL pipeline)

**Лечить ОДНИМ из:**
- (a) Отредактировать `e4_full.yaml` — добавить под `human:`:
  ```yaml
    role_router_model: openai:gpt-4.1-mini
  ```
- (b) Передать override на CLI каждый раз:
  ```bash
  +human.role_router_model=openai:gpt-4.1-mini
  ```

Pilot конфиги (`e4_pilot*.yaml`) уже задают это явно — НЕ требуют правки.

---

## 2. Sanity checks (5 мин)

### 2.1. Unit tests
```bash
uv run pytest tests/unit/ -q
# Ожидаемо: ~1852 passed, 0 failed
# (1 flake в tests/unit/analysis/test_oracle.py::test_build_loo_oracle_topology_router_contract — асинк timing, ОК)
```

Если падают `test_role_router*` или `test_runner_human_role*` —
**не запускай E4**, дебагай.

### 2.2. Все E4 конфиги валидны
```bash
uv run python -c "
from atm.experiment.loader import load_grid_configs
for f in ['conf/experiments/e4_pilot_sanity.yaml',
          'conf/experiments/e4_role_smoke.yaml',
          'conf/experiments/e4_pilot.yaml',
          'conf/experiments/e4_full.yaml']:
    cfgs = load_grid_configs(f, overrides=[])
    print(f'{f}: cells={len(cfgs)}')
"
# Ожидаемо:
#   e4_pilot_sanity.yaml: cells=6
#   e4_role_smoke.yaml:   cells=6
#   e4_pilot.yaml:        cells=45
#   e4_full.yaml:         cells=540
```

### 2.3. Ключи живы
```bash
set -a; source .env; set +a
uv run python -c "
import asyncio
from atm.llm.factory import build_llm
from atm.llm.pricing import Pricing
from atm.core.types import Message, MessageKind
p = Pricing.from_file('conf/pricing/prices.yaml')
async def t():
    for m in ['cerebras:gpt-oss-120b', 'openai:gpt-4.1-mini']:
        llm = build_llm(model_id=m, pricing=p, budget=None)
        r = await llm.ainvoke([Message(sender='u', kind=MessageKind.REQUEST, content='reply: ok')], agent_id='sanity')
        print(f'{m}: {r.text[:50]!r}')
asyncio.run(t())
"
```

---

## 3. Запуск E4 — 4 wave'а

### Wave 1: e4_pilot_sanity (~5 мин, 6 cells, $0.30)

Wiring smoke. Если падает — НЕ продолжай.

```bash
mkdir -p logs
uv run atm grid run -c conf/experiments/e4_pilot_sanity.yaml --no-estimate -p 4 > logs/e4_pilot_sanity.log 2>&1
grep GridResult logs/e4_pilot_sanity.log
```

Acceptance:
- 6/6 completed, 0 failed
- Логи не содержат `unknown role_router` / `role_router LLM failed`
- mean_q ≥ 0.7 на humaneval+gsm8k

### Wave 2: e4_role_smoke (~5 мин, 6 cells, $0.30)

**КЛЮЧЕВОЙ smoke** — проверяет реальную работу 3 role_router режимов.

```bash
uv run atm grid run -c conf/experiments/e4_role_smoke.yaml --no-estimate -p 4 > logs/e4_role_smoke.log 2>&1
grep GridResult logs/e4_role_smoke.log
```

**Проверка через parquet** — какие роли реально были использованы:
```bash
EXP=$(grep "GridResult" logs/e4_role_smoke.log | sed 's/.*exp_id=\([^ ]*\).*/\1/')
uv run python -c "
import pandas as pd, glob
for p in sorted(glob.glob(f'data/experiments/experiments/$EXP/runs/*/human_interactions.parquet')):
    rid = p.split('/')[-2][:8]
    try:
        df = pd.read_parquet(p)
        if 'role' in df.columns:
            roles = df['role'].value_counts().to_dict()
            print(f'  {rid}  roles={roles}  n_interactions={len(df)}')
        else:
            print(f'  {rid}  cols={list(df.columns)} (no role col)')
    except Exception as e:
        print(f'  {rid}  ERR {e}')
"
```

Acceptance (то что отличает Wave 2 от обычного wiring smoke):
- **fixed cells**: только 1 role в каждой записи (= cfg.human.role)
- **rule cells**: 2-3 разных role (по DEFAULT_ROLE_TABLE: planning→coordinator,
  execution→peer, verification→reviewer)
- **llm cells**: 1-5 разных role + cost > 0 на role_router LLM calls
  (проверь budget_events.parquet → agent_id='role_router')

Если все cells показывают одинаковую role — `role_router` не подцепился,
дебагай `_build_role_router` или передачу через kwarg.

### Wave 3: e4_pilot (~25-35 мин, 45 cells, $4)

Battle smoke на 3 задачах. Параллельно с E2/E3 НЕ запускать (TPM).

```bash
uv run atm grid run -c conf/experiments/e4_pilot.yaml --no-estimate -p 8 > logs/e4_pilot.log 2>&1 &
PILOT_PID=$!

watch -n 30 'grep -oE "\[[0-9]+/[0-9]+\] done=[0-9]+ failed=[0-9]+" logs/e4_pilot.log | tail -3'

wait $PILOT_PID
grep GridResult logs/e4_pilot.log
```

Per-mode breakdown:
```bash
uv run python -c "
import json, re
from collections import defaultdict
agg = defaultdict(list)
starts = {}
with open('logs/e4_pilot.log') as f:
    for line in f:
        m = re.search(r'\{.*\"event\": \"run started\".*\}', line)
        if m:
            try:
                o = json.loads(m.group(0))
                starts[o['run_id']] = (o.get('topology'), o.get('task'))
            except: pass
        m = re.search(r'\{.*\"event\": \"run completed\".*\}', line)
        if m:
            try:
                o = json.loads(m.group(0))
                agg[starts.get(o['run_id'], ('?','?'))[1]].append(o.get('quality_score'))
            except: pass
for task, qs in sorted(agg.items()):
    qs = [q for q in qs if q is not None]
    if qs:
        print(f'  {task:12s} n={len(qs):3d} mean_q={sum(qs)/len(qs):.2f} min={min(qs):.2f}')
"
```

Acceptance per task:
- humaneval: mean_q ≥ 0.85
- gsm8k: mean_q ≥ 0.9
- commongen: mean_q ≥ 0.55
- failed ≤ 2

### Wave 4: e4 full (~3-4 ч, 540 cells, $45)

`e4_full.yaml` свипает по 3 role_router модам **в одном запуске** (в отличие
от E3 где надо было 3 launch). Зависит от champion topology_router из E3.

#### Запуск (single launch — единый exp_id)

```bash
# Подставь CHAMPION = rule | llm | oracle (см. Wave 0 запрос к PG)
CHAMPION=rule

uv run atm grid run \
    -c conf/experiments/e4_full.yaml \
    --no-estimate -p 24 \
    +topology.extra.adaptive.topology_router=$CHAMPION \
    +human.role_router_model=openai:gpt-4.1-mini \
    > logs/e4_full.log 2>&1 &
E4_PID=$!

watch -n 60 'grep -oE "\[[0-9]+/[0-9]+\] done=[0-9]+ failed=[0-9]+" logs/e4_full.log | tail -3 && uptime && free -h | head -2'

wait $E4_PID
grep GridResult logs/e4_full.log
```

Wall: ~3-4 ч на p=24.

Acceptance:
- 540/540 attempted, failed ≤ 30 (5%)
- Per role_router (180 cells each):
  - fixed:  baseline — mean_q как у E3 best mode
  - rule:   ≥ baseline (если меньше — RQ4 опровергнут на rule)
  - llm:    ≥ baseline (если меньше — LLM router выбирает плохо)
- В parquet `human_interactions.role` распределение ролей **различно** между modes

#### Если хочешь параллельно 3 launch (как в E3 Wave 4)

Можно разбить swap по role_router на 3 отдельных launch на p=8 каждый:
```bash
for MODE in fixed rule llm; do
    nohup uv run atm grid run \
        -c conf/experiments/e4_full.yaml --no-estimate -p 8 \
        +name=e4_full_role_${MODE} \
        +topology.extra.adaptive.topology_router=$CHAMPION \
        +human.role_router=$MODE \
        +human.role_router_model=openai:gpt-4.1-mini \
        > logs/e4_full_role_${MODE}.log 2>&1 &
done
wait
```
Wall: ~2 ч (3× speedup), но 3 exp_id (нужно объединять при анализе).

---

## 4. Failure modes

| Симптом | Лечение |
|---|---|
| `unknown role_router: 'xxx'` | проверь что в sweep строго `fixed`/`rule`/`llm` |
| `LLMRoleRouter: LLM call failed` | gateway fallback на rule — норм, посмотри warning |
| Все cells одинаковая role в parquet | `role_router` kwarg не доходит до topology — проверь commit 24d5529 в HEAD |
| 429 TPM | если параллельно с E2/E3 — стоп. Если один — снизь `-p` до 16 |
| `role_router_model` not provided | `human.role_router_model` отсутствует → default cfg.model.default (Cerebras). Передай override (см. §1.2) |
| Cell зависает > 15 мин | adaptive+adaptive может цикломатить; cell сам умрёт по wall_time_s |
| Adaptive choose `mesh` каждый раз → mesh starvation | проверь что mesh.max_rounds ≥ 8 в YAML (от смокa E1, должно быть OK) |

---

## 5. После завершения

### 5.1. Summary
```bash
docker exec atm-postgres psql -U atm -d atm -c "
SELECT name, status, total, completed, failed,
       ROUND(AVG(r.quality_score)::numeric, 3) as avg_q,
       ROUND(SUM(r.budget_spent_usd)::numeric, 2) as cost
FROM experiments e
LEFT JOIN runs r ON r.exp_id=e.id AND r.quality_score IS NOT NULL
WHERE name LIKE 'e4_full%'
GROUP BY name, status, total, completed, failed;"
```

### 5.2. Per role_router breakdown
```bash
docker exec atm-postgres psql -U atm -d atm -c "
SELECT
  CASE
    WHEN r.human_role IS NULL THEN 'fixed (no human_role recorded)'
    ELSE r.human_role
  END as role_mode_or_dominant,
  r.task_id,
  COUNT(*) as n,
  ROUND(AVG(r.quality_score)::numeric, 3) as mean_q
FROM runs r
WHERE (SELECT name FROM experiments WHERE id=r.exp_id) LIKE 'e4_full%'
  AND r.quality_score IS NOT NULL
GROUP BY role_mode_or_dominant, r.task_id
ORDER BY r.task_id, mean_q DESC;"
```

Этот запрос требует чтобы runs.human_role был установлен — что зависит от
конкретного mode. Лучший анализ — через analysis/loaders.py + parquet.

### 5.3. Закоммитить результаты
```bash
mkdir -p arch/results
cp logs/e4_full*.log arch/results/
git add arch/results/
git commit -m "data(e4): full results — adaptive topology + adaptive role"
git push origin main
```

Сообщи пользователю:
- exp_id E4 full
- mean_q per role_router mode
- RQ4 hypothesis: «adaptive role > fixed» — подтверждена или нет?

---

## 6. Что НЕ делать
- **НЕ запускай E4 параллельно с E2/E3** на одном Cerebras ключе — TPM.
- **НЕ забудь cross-family role_router_model** override — иначе HITL Cerebras vs workers Cerebras = коллизия.
- **НЕ используй `oracle` режим** без файла `data/oracle/e1_leave_one_out.json` (он в репе, но если случайно удалится — graceful fallback на rule).
- **НЕ перезапускай E1/E3** для генерации новых oracle — используй уже committed.

---

## TL;DR (копипаст)

```bash
# Setup
git pull --ff-only
uv sync
set -a; source .env; set +a

# Champion из E3 (заменить по результатам)
CHAMPION=rule

# Sanity
uv run pytest tests/unit/ -q

# Wave 1-3 smoke
uv run atm grid run -c conf/experiments/e4_pilot_sanity.yaml --no-estimate -p 4 > logs/e4_pilot_sanity.log 2>&1
uv run atm grid run -c conf/experiments/e4_role_smoke.yaml   --no-estimate -p 4 > logs/e4_role_smoke.log 2>&1
uv run atm grid run -c conf/experiments/e4_pilot.yaml        --no-estimate -p 8 > logs/e4_pilot.log 2>&1

# Wave 4 — single launch (540 cells, ~3-4 ч на p=24)
uv run atm grid run \
    -c conf/experiments/e4_full.yaml \
    --no-estimate -p 24 \
    +topology.extra.adaptive.topology_router=$CHAMPION \
    +human.role_router_model=openai:gpt-4.1-mini \
    > logs/e4_full.log 2>&1
grep GridResult logs/e4_full.log
```
