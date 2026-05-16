# Experiments Runbook (E1 → E4)

Action-oriented playbook для запуска экспериментов диплома. Methodology / RQ /
hypotheses — в `arch/experiment_plan.md`. Здесь только **что и в каком порядке
запускать**, с командами и точками синхронизации.

Статус артефактов на момент создания (2026-05-16):
- ✓ Все 5 топологий работают (sanity v10: quality_score>0 на всех)
- ✓ `src/atm/analysis/oracle.py::build_leave_one_out_oracle` реализован
- ✓ `src/atm/analysis/metrics.py` — все RQ-метрики
- ✓ Все конфиги: `e1_mini`, `e1_full`, `e2_full`, `e3_full`, `e4_full`
- ☐ `analysis/winners.py` (top-3 топологии per task_type) — НЕ реализован, нужен между E1 и E2
- ☐ `dev/preregistration.md` — НЕ написан, нужен ДО E3
- ☐ E3 static-baseline grid — нужно отдельно прогнать subset E1 на best-static-per-task_type

---

## 0. Pre-flight (~20 мин)

- [ ] **E1-mini**: `atm experiment grid run conf/experiments/e1_mini.yaml`
  - 100 cells (5 topo × 4 task × 5 shuffle × 1 seed), p=12, ~15 мин
  - **Цель**: валидация judge на CommonGen/DABench (sanity покрывал только HumanEval)
  - **Зелёный критерий**: каждая (topology, task_type) пара даёт quality_score>0
    хотя бы в одной cell; нет hard-крашей; total cost ≤ $10
  - Если красный — фиксим bug, перезапуск. НЕ идём дальше.

---

## 1. E1 — статические топологии (~2.5 ч, ~$60)

- [ ] **E1 full**: `atm experiment grid run conf/experiments/e1_full.yaml`
  - 900 cells (5 topo × 4 task × 15 shuffle × 3 seeds), p=12
  - Сохраняет `data/experiments/<exp_id>.parquet`
- [ ] **Анализ E1**: запустить notebook `notebooks/e1_analysis.ipynb`
  (или ad-hoc скрипт): средний quality_score / cost / latency по
  (topology × task_type), CI 95%
- [ ] **TODO: реализовать `src/atm/analysis/winners.py`**:
  - функция `top_k_topologies_per_task_type(parquet_path, k=3) → dict[task_type, list[topology]]`
  - aggregation: mean(quality_score) per (topology, task_type), tie-break по cost
  - сохранить в `data/winners_e1.json`
- [ ] **Подставить winners в `e2_full.yaml::topology.name.sweep`**
  (сейчас плейсхолдер). Если top-3 разные per task_type — придётся либо
  гнать 4 отдельных подгрида (по task_type), либо взять union top-3 across
  всех task_types и фильтровать post-hoc.
- [ ] **Записать best static topology per task_type** → понадобится для
  E3 static-baseline grid (шаг 4)

**Точки синхронизации перед E2**: winners.py готов, `data/winners_e1.json`
существует, `e2_full.yaml` обновлён.

---

## 2. Pre-registration (~30 мин) — БЛОКИРУЕТ E3

- [ ] Создать `dev/preregistration.md` со структурой:
  - **H1** (RQ1): какая статическая топология побеждает per task_type
    (формулируется из E1 результатов, но регистрируется ДО E3)
  - **H2** (RQ2): adaptive topology даёт lift ≥ X% quality vs best static,
    при cost ≤ Y% от best static
  - **H3** (RQ3): oracle router показывает upper bound; gap_loo < Z%
  - **H4** (RQ4): adaptive role даёт дополнительный lift поверх adaptive topology
  - Пороги X, Y, Z — фиксируем здесь, не меняем после прогонов
  - Sample size justification: 15 shuffle × 3 seeds = 45 runs per cell, ≥80% power
    при effect size 5% quality
  - Stopping rules: только pre-defined budget cap; no peeking

---

## 3. E2 — HITL roles (~10 ч, ~$170)

- [ ] **Подготовка `e2_full.yaml`**:
  - top-3 topology подставлены (шаг 1)
  - проверить `human.gateway: llm_simulated` и `human.role` sweep
- [ ] **E2 full**: `atm experiment grid run conf/experiments/e2_full.yaml`
  - ~2160 cells, p=12, ~10 ч
- [ ] **Анализ E2**:
  - best `(topology, role)` per task_type → `data/winners_e2.json`
  - winner-role per task_type → нужен для `e3_full.yaml::human.role`
  - winner-role overall (для E4 baseline) → нужен для `e4_full.yaml::human.role`

---

## 4. Подготовка к E3 (~1 ч)

- [ ] **Build oracle**: запустить
  ```python
  from atm.analysis.oracle import build_leave_one_out_oracle
  await build_leave_one_out_oracle(exp_id="<E1_exp_id>", out_path="data/oracle/e1_loo.json")
  ```
  Сохраняет `data/oracle/e1_loo.json` (потребляется `OracleTopologyRouter` через
  `topology.extra.topology_router=oracle`).
- [ ] **Smoke-тест oracle router**: 1 cell adaptive с `topology_router=oracle`,
  убедиться что lookup работает без падений
- [ ] **Обновить `e3_full.yaml::human.role`** под winner per task_type из E2
  (если winners разные per task_type — гнать 4 подгрида или sweep по `human.role`)

---

## 5. E3 — adaptive topology, 3 router'а (~12 ч, ~$120)

- [ ] **E3 rule**: `e3_full.yaml` с `topology.extra.topology_router=rule`
  - 720 cells, ~4 ч → `runs/e3_rule/`
- [ ] **E3 llm**: тот же конфиг, override на `topology_router=llm`
  - 720 cells, ~4 ч → `runs/e3_llm/`
- [ ] **E3 oracle**: тот же конфиг, override на `topology_router=oracle`
  - 720 cells, ~4 ч → `runs/e3_oracle/`
- [ ] **E3 static-baseline**: subset `e1_full` на best-static-per-task_type
  - **Опция A**: вытащить нужные cells из E1 parquet (бесплатно, рекомендуется)
  - **Опция B**: отдельный grid `e3_static_baseline.yaml` (нужно создать)
  - 4 task_type × 15 shuffle × 3 seed = 180 cells
- [ ] **Анализ E3**:
  - `compute_oracle_gap_loo` (metrics.py:173) — gap adaptive vs oracle
  - lift adaptive vs static baseline
  - switch-counts через SwitchGuards (`compute_topology_switch_counts`)
  - winner `topology_router` → нужен для `e4_full.yaml::topology.extra.topology_router`

---

## 6. E4 — adaptive topology + adaptive role (~5 ч, ~$45)

- [ ] **Подготовка `e4_full.yaml`**:
  - `topology.extra.topology_router` = winner из E3
  - `human.role` = winner из E2
- [ ] **E4 full**: `atm experiment grid run conf/experiments/e4_full.yaml`
  - 540 cells (4 task × 15 shuffle × 3 seed × 3 role_router), ~5 ч
- [ ] **Анализ E4**: lift adaptive role vs fixed role на топ adaptive topology

---

## 7. Confirmation runs (опционально, ~$50)

- [ ] **Cross-provider check** (если бюджет позволяет):
  - top-3 cells на `zai:glm-4.7` ИЛИ `openai:gpt-4o`
  - подтверждает, что результаты не специфичны для Cerebras gpt-oss-120b

---

## 8. Post-processing (несколько дней)

- [ ] Сводный отчёт по RQ1–RQ4 (по критериям успеха `experiment_plan.md §6`)
- [ ] Графики:
  - topology × task_type heatmap (E1)
  - role × topology heatmap (E2)
  - switch-count distribution (E3)
  - cost/quality Pareto (все эксперименты)
  - oracle_gap_loo bar (E3)
- [ ] Главы диплома: methodology / results / discussion
- [ ] `arch/defense_cheatsheet.md` — обновить с реальными числами

---

## Критические зависимости (НЕ нарушать)

```
E1 (statics)
  ├─→ winners.py → e2_full.yaml (top-3 topology)
  ├─→ oracle.py → data/oracle/e1_loo.json → e3 oracle режим
  └─→ best static per task_type → e3 static-baseline

E2 (HITL roles)
  ├─→ winner role per task_type → e3_full.yaml::human.role
  └─→ winner role overall → e4_full.yaml::human.role

pre-registration.md → ДО запуска E3 (не после)

E3 (adaptive topo)
  └─→ winner topology_router → e4_full.yaml::topology.extra.topology_router
```

## Бюджет (суммарно)

| Этап | Wall-time (p=12) | Compute | Judge | Итого |
|------|------------------|---------|-------|-------|
| E1-mini | 15 мин | $8 | $1 | $9 |
| E1 full | 2.5 ч | $50 | $10 | $60 |
| E2 full | 10 ч | $150 | $20 | $170 |
| E3 (×3 + baseline) | 12 ч + 0 | $90 | $15 | $105 |
| E4 full | 5 ч | $40 | $5 | $45 |
| Confirmation | ~2 ч | $40 | $5 | $50 |
| **Итого** | **~32 ч** | **~$378** | **~$56** | **~$435** |

(Из них ~$80 — опциональные confirmation runs.)
