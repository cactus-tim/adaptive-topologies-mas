# m8-foundation — Core types, SharedState extension, reducers, PhaseManager

## Goal
Заложить фундамент для адаптивной топологии L2 (M8.1 + M8.2): доменные типы решений/переходов, расширение `SharedState` сигналами и счётчиками, идемпотентный reducer для merge messages/transitions, persistence-слой для `topology_transitions`, а также монотонный PhaseManager в двух режимах (Rule + LLM с fallback).

Этот блок — единственный starting block для волны M8. Без него m8-routing/m8-signals не могут компилироваться (нет типов и нет slots в state).

## Scope
- `src/atm/core/types.py` — Pydantic frozen-модели:
  - `TopologyTransition` (from_topology, to_topology, phase, at_iter, decided_by, considered_alternatives, router_cost_usd, reason, …)
  - `TopologyDecision` (chosen, considered_alternatives, reason, router_cost_usd, decided_by)
  - `PhaseDecision` (next_phase, reason, by) — со строгим валидатором монотонности (`next_phase >= current_phase`)
- `src/atm/core/state.py` — расширить `SharedState` обратно совместимо:
  - `signals: dict[str, Any]` (default `{}`)
  - `iter_total: int`
  - `phase_started_at_iter: int`
  - `topology_started_at_iter: int`
  - `topology_switch_count: int`
  - `topology_history: list[TopologyTransition]`
  - дефолты подобрать так, чтобы существующие графы M5–M7 не падали при чтении state без этих полей
- `src/atm/core/reducers.py`:
  - `dedup_by_id_reducer(key: str, sort_by: str | None = None)` — фабрика reducer'ов
  - инварианты: idempotent (f(x,x)==x), associative (f(f(a,b),c)==f(a,f(b,c))), empty-neutral (f([], x)==x, f(x, [])==x)
  - unit-тесты на все 3 инварианта (property-based по возможности)
- Alembic migration — новая таблица `topology_transitions` (колонки соответствуют `TopologyTransition` + run_id FK + at TIMESTAMP)
- `src/atm/storage/models.py` — SQLAlchemy ORM `TopologyTransition` (та же схема)
- `src/atm/phases/manager.py`:
  - `RuleBasedPhaseRouter` — guards: `ready_for_execution`, `ready_for_verification`, `critic_approved`, iter caps (per-phase max_iter)
  - `LLMPhaseRouter` — parse JSON из LLM, валидация монотонности, fallback на `RuleBasedPhaseRouter` при любой ошибке (rollback attempt, malformed JSON, schema mismatch)
- Unit-тесты:
  - rollback attempt от LLM → fallback на Rule (логируется как `decided_by='fallback'`)
  - каждый guard покрыт (positive + negative case)
  - reducer'ы — 3 инварианта

## Out-of-scope
- TopologyRouter (Rule/LLM/Oracle) → блок **m8-routing**
- SwitchGuards (min_dwell, cooldown, max_per_run, max_per_phase) → блок **m8-routing**
- SignalBus, `emit_signal`, обновления агентов → блок **m8-signals**
- TransitionGate + adaptive meta-граф → блок **m8-adaptive**
- Oracle pipeline (M8.7) — после E1.

## Inputs / Preconditions
- M5 (scratchpad policy C) и M7 (5 статических топологий) уже в проекте.
- Существующий `SharedState` (TypedDict / Pydantic — посмотреть как реализовано в M5/M6).
- Существующая Alembic-конфигурация (head перед миграцией нужно посмотреть).

## Outputs / Contract
Что следующие блоки могут полагаться:
- Импорты: `from atm.core.types import TopologyTransition, TopologyDecision, PhaseDecision`
- В `SharedState` гарантированы поля: `signals`, `iter_total`, `phase_started_at_iter`, `topology_started_at_iter`, `topology_switch_count`, `topology_history`
- `from atm.core.reducers import dedup_by_id_reducer`
- ORM-модель `TopologyTransition` доступна для insert из adaptive meta-graph
- `RuleBasedPhaseRouter().decide(state) -> PhaseDecision` и `LLMPhaseRouter(llm, fallback=RuleBasedPhaseRouter()).decide(state) -> PhaseDecision`

## References
- `arch/PLAN.md` строки 469–479 (M8.1 + M8.2 чек-лист)
- `arch/arch.md` §7.7 — phase × signals → topology (нужно для семантики `PhaseDecision`)
- `arch/arch.md` §8 — Adaptive L2 общая картина
- `dev/codebase-map.md` — текущие модули `src/atm/core/*`, `src/atm/storage/*`
- Существующая Alembic-конфигурация в репозитории

## Depends On
- (внутри m8): нет — стартовый блок.

## Suggested run-task class
**complex** — затрагивает 5+ файлов, миграция БД, новые типы, два класса PhaseRouter с fallback-логикой, инвариантные тесты reducer'а.

## Notes
- Расширение `SharedState` ДОЛЖНО быть обратно совместимо со scratchpad policy C из M5 — существующие узлы M5/M6/M7 не должны падать при чтении state без новых полей (использовать дефолты, `Field(default_factory=...)`).
- Reducer обязан пройти инварианты idempotent / associative / empty-neutral — это пишется как unit-тест и блокирует merge при провале.
- Migration: проверь head перед добавлением, добавь downgrade.
- LLMPhaseRouter валидирует монотонность фазы — попытка вернуть фазу назад трактуется как ошибка и триггерит fallback (а не raise наружу).
