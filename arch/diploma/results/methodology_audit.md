# Implementation audit реализации adaptive-роутера

**Назначение документа.** Формальное описание методологических артефактов,
выявленных во время внутреннего audit'а реализации adaptive-роутера в
LangGraph meta-graph. Документ служит источником для главы Methodology
дипломной работы и одновременно audit trail для воспроизводимости.

**Дата фиксации audit'а:** 2026-05-18.
**Реализация:** `src/atm/topology/adaptive.py`, `src/atm/phases/`,
PR #18 в репозитории `adaptive-topologies-mas`.
**Терминология.** В тексте используются нейтральные методологические
термины: «artefact», «correction», «implementation audit», «methodological
artefact». Слово «bug» **не используется** — речь идёт о расхождениях
между declared design-intent и observable behaviour реализации, не
об ошибках в смысле runtime failure'ов.

---

## 1. Постановка audit'а

Adaptive-роутер представляет собой meta-graph поверх пяти под-топологий
(`chain`, `star`, `mesh`, `debate`, `hierarchical`). На каждый
dispatch-tick adaptive принимает два решения: (а) `phase_router`
выбирает текущую фазу (`planning` / `execution` / `verification`);
(б) `topology_router` выбирает под-топологию для исполнения данной
фазы. Между tick'ами состояние передаётся как `AdaptiveState`-словарь
с фиксированной schema. Сложность реализации сосредоточена в
точках, где meta-граф взаимодействует с под-графами: state-transfer,
counter source-of-truth, signal lifecycle и guard-семантика.

Audit был запущен после фиксации первого тиража прогонов E3 и E4 с
целью верифицировать, что declared design-intent реализован полностью
и корректно. Triggers для audit'а:

- **Trigger 1.** Распределение `iter_total` в логах adaptive показывало
  истощение iter-budget'а вдвое быстрее статических baseline'ов при
  идентичном `cfg.max_iterations`. Это поднимало вопрос источника
  счётчика.
- **Trigger 2.** Серии logged-traces содержали последовательности
  топологий вида `... → chain → chain → ...` с нулевой задержкой между
  re-entries — что противоречит cooldown-семантике, заявленной в
  `SwitchGuardsConfig`.
- **Trigger 3.** Поля `signals['phase_switch_count']` и
  `signals['human_advisor_hint']` присутствовали в state schema, но в
  observed-runs не имели read-effects ни на routing-decision, ни на
  budget-guards.

В рамках audit'а были обойдены все node-функции adaptive-графа и все
gate-функции (`_violates_cooldown`, `_violates_max_per_run`,
`_violates_max_per_phase`, `apply_transition_gate`), сверены с
state schema и declared design-intent. Выявлено пять методологических
артефактов, описанных ниже.

---

## 2. Категоризация артефактов

Артефакты сгруппированы в четыре методологические категории по природе
расхождения между design-intent и реализацией.

| Категория | Описание | Артефакты |
|---|---|---:|
| **State-transfer purity** | Чистота передачи состояния между meta-graph и под-графами (отсутствие cross-phase contamination, корректность capture/restore phase boundaries). | A1, A4 |
| **Guard semantics** | Соответствие гард-функций (cooldown, max_per_run, max_per_phase) их declared semantics в `SwitchGuardsConfig`. | A2 |
| **Counter source-of-truth** | Единственность источника инкремента для каждого counter в state schema; отсутствие двойного учёта. | A1, A3 |
| **Signal lifecycle** | Полнота цикла `write → read → consume → clear` для phase-bounded signals (advisor hints, phase-switch counters, debate-history). | A3, A4, A5 |

Артефакт A1 затрагивает две категории (state-transfer purity + counter
source-of-truth), так как двойной инкремент `iter_total` одновременно
нарушает source-of-truth и фактически «протекает» counter через
boundary под-графа.

---

## 3. Описание артефактов

### 3.1 Artefact A1 — `iter_total` double-increment

**Категория:** counter source-of-truth, state-transfer purity.

**Описание.** Counter `iter_total` инкрементировался в двух местах
одновременно: (а) в dispatch-node adaptive-роутера на каждом
meta-tick'е, (б) в под-графах под-топологий (`chain`, `star`, `mesh`,
`debate`, `hierarchical`) — каждая из них инкрементирует
`iter_total` на каждый внутренний step. Source-of-truth для одного
state-поля был размазан между двумя слоями графа.

**Эффект на эксперимент.** Adaptive фактически получал ≈½
work-budget'а статических baseline'ов при том же
`cfg.max_iterations`: budget-guard срабатывал в два раза быстрее, что
систематически сокращал число exec/verify-итераций до early-stop. Все
pre-correction adaptive-замеры были смещены в сторону недо-исполнения
по сравнению со static, что делало any quality-сравнение нерелевантным
без correction.

**Correction.** В dispatch-node adaptive теперь инкрементируется
**только** новое state-поле `meta_ticks` (counts dispatch-tick'ов
meta-уровня). Под-топологии продолжают инкрементировать `iter_total`
как раньше (counts work-итераций воркеров). `meta_ticks` и
`iter_total` — два независимых counter'а с непересекающимися
источниками записи; budget-guard читает `iter_total` для work-budget'а
и `meta_ticks` для meta-loop-cap'а.

**File pointer.** `src/atm/topology/adaptive.py`, функция `dispatch`
(ранее содержала `state['iter_total'] += 1`; в post-correction —
`state['meta_ticks'] = state.get('meta_ticks', 0) + 1`).

---

### 3.2 Artefact A2 — cooldown guard skipping just-left topology

**Категория:** guard semantics.

**Описание.** Функция `_violates_cooldown` в `src/atm/phases/guards.py`
использовала срез `topology_history[:-1]` для проверки попадания
кандидатной топологии в окно cooldown. Срез `[:-1]` исключает
последний элемент списка — то есть **только что покинутую**
топологию. Семантика cooldown в declared design предполагает, что
после `min_dwell` тиков в топологии T возврат к T должен быть запрещён
на `cooldown_ticks` тиков. Реализация со срезом `[:-1]` исключала T из
look-up'а и пропускала немедленный возврат.

**Эффект на эксперимент.** Cooldown-guard был де-факто bypass'нут для
immediate-return случая: adaptive мог осциллировать `T → T' → T → T'`
без задержки. В logged-traces pre-correction серий наблюдались
последовательности с нулевым cooldown-интервалом, что приводило к
завышенному `topology_switches_total` и нерелевантной фрагментации
phase-исполнения на короткие сегменты.

**Correction.** `_violates_cooldown` теперь читает полный
`topology_history` (без среза `[:-1]`): проверка попадания кандидата
в окно последних `cooldown_ticks` элементов выполняется по всей
history, включая последнюю топологию.

**File pointer.** `src/atm/phases/guards.py`, функция
`_violates_cooldown`.

---

### 3.3 Artefact A3 — `signals['phase_switch_count']` never written

**Категория:** counter source-of-truth, signal lifecycle.

**Описание.** Поле `signals['phase_switch_count']` было объявлено в
state schema (`AdaptiveSignals` TypedDict) и читалось guard'ом
`_violates_max_per_phase` (`SwitchGuardsConfig.max_per_phase`). Однако
**ни один узел графа не записывал** это поле. Реализация guard'а
читала значение по умолчанию `0` либо в одной из ранних веток
кода — конфлятилась с `len(topology_history)`, что де-факто заменяло
per-phase счётчик на per-run.

**Эффект на эксперимент.** Guard `max_per_phase` работал как
`max_per_run`: per-phase ограничение игнорировалось, а per-run
автоматически совпадало с `len(topology_history)`. Это приводило к
over-permissive behaviour в позднюю стадию run'а — когда
`topology_history` уже накопила достаточно switches, любой
per-phase-limit становился неактивным, и adaptive мог делать
неограниченное число switches в финальной фазе. В сочетании с
artefact A2 это усиливало over-switching pattern в pre-correction
сериях.

**Correction.** Dispatch-node adaptive теперь явно записывает
`signals['phase_switch_count']` при каждом topology-switch внутри
одной фазы. Counter сбрасывается на `0` при phase-transition (когда
`signals['current_phase']` изменяется относительно предыдущего tick'а).
Цикл `write → read → reset` для `phase_switch_count` стал полным.

**File pointer.** `src/atm/topology/adaptive.py`, функция `dispatch`
(новая запись `signals['phase_switch_count']`); сброс — там же на
phase-transition branch'е.

---

### 3.4 Artefact A4 — subgraph-internal phase advance не детектировался transition-gate

**Категория:** state-transfer purity, signal lifecycle.

**Описание.** Под-топологии `hierarchical` и (в меньшей степени)
`chain` способны внутренне продвигать фазу: hierarchical-граф имеет
свой собственный coordinator-driven переход `planning → execution`,
который выполняется внутри одного meta-tick'а adaptive-уровня.
Transition-gate adaptive (`apply_transition_gate`) проверял, изменилась
ли фаза после возврата из под-графа, **используя только post-subgraph
state** — pre-subgraph phase не сохранялся отдельным state-полем и
поэтому был недоступен для сравнения. В результате внутреннее
phase-advance под-графа **не детектировалось** transition-gate'ом,
и phase-bounded signals (inboxes, advisor-hints, debate-history) не
очищались на phase boundary.

**Эффект на эксперимент.** Phase-signals **протекали** между
фазами: debate-rounds, сгенерированные в planning-фазе, оставались в
inbox при переходе в execution-фазу; advisor-hints, выданные для
planning-context'а, видны были exec-роутерам. Это создавало false
signal contamination — роутеры читали phase-signals не той фазы, для
которой они генерировались, что нарушало изоляцию phase-based
routing decisions.

**Correction.** Функция `apply_transition_gate` получила параметр
`pre_subgraph_phase: Phase`, который захватывается в dispatch-node
**до** входа в под-граф и пробрасывается в gate-вызов после возврата.
При расхождении `pre_subgraph_phase ≠ post_subgraph_phase` gate
очищает phase-bounded state: inboxes воркеров, advisor signals,
debate-history. Phase-counter `phase_switch_count` сбрасывается там же
(в синхрон с correction A3).

**File pointer.** `src/atm/topology/adaptive.py`:
- `apply_transition_gate` (signature расширена параметром
  `pre_subgraph_phase`).
- `dispatch` (capture `pre_subgraph_phase` до subgraph-invocation).

---

### 3.5 Artefact A5 — advisor HITL hint не консумировался ни одним роутером

**Категория:** signal lifecycle.

**Описание.** Когда HITL-узел в роли `advisor` (advisory mode HITL)
выдавал подсказку, она записывалась в `signals['human_advisor_hint']`
как текст с предложением топологии (например, «consider switching to
debate for adversarial verification»). Однако **ни один роутер** —
ни `RuleBasedTopologyRouter`, ни `LLMTopologyRouter`, ни role-router'ы
из `phases/manager.py` — **не читал** это поле в свой
routing-decision. Advisory режим HITL был **no-op** для routing
решения: hint генерировался, занимал meta-LLM call, но игнорировался
последующими routing-tick'ами.

**Эффект на эксперимент.** HITL'ы в advisory mode тратили
meta-LLM-call для выработки hint'а, но hint просто игнорировался —
adaptive принимал routing-decision без учёта user feedback. Это
противоречило design intent advisory mode'а (HITL участвует в
routing) и фактически делало advisory mode эквивалентным no-HITL
baseline'у со штрафом в виде потраченного API-call'а. Все
pre-correction замеры HITL-advisory mode были непригодны для
сравнения с no-HITL baseline'ом по semantics.

**Correction.**

1. `RuleBasedTopologyRouter` получил **Rule 0** (наивысший
   приоритет): если `signals['human_advisor_hint']` парсится в
   конкретное имя топологии (через `_extract_topology_from_hint`),
   эта топология выбирается **до** консультации rule-таблицы. Rule 0
   overrides всю остальную rule-логику.
2. `LLMTopologyRouter` prompt теперь содержит текст
   `human_advisor_hint` в явном блоке `### Advisor Hint`; helper
   `_extract_topology_from_hint` форсит обработку через few-shot
   examples и regex-парсинг ответа model'а.
3. После каждого dispatch-tick adaptive **очищает**
   `signals['human_advisor_hint']` (set to `None`) — one-shot
   consumption-семантика: hint валиден ровно для одного routing-tick'а.

**File pointers.**

- `src/atm/phases/topology_router.py`, класс `RuleBasedTopologyRouter`
  (Rule 0 в начале метода `route`).
- `src/atm/phases/topology_router.py`, класс `LLMTopologyRouter`
  (расширение prompt-шаблона + helper `_extract_topology_from_hint`).
- `src/atm/topology/adaptive.py`, функция `dispatch` (clearing
  `signals['human_advisor_hint'] = None` после dispatch-tick'а).

---

## 4. Тестовое покрытие корректировок

Каждая correction сопровождается unit- или integration-тестом,
заякоривающим post-correction behaviour и предотвращающим регресс:

| Artefact | Test class | File |
|---|---|---|
| A1 | `TestIterTotalSingleSource` | `tests/integration/test_adaptive_m8.py` |
| A2 | `TestCooldownReturnAfterSwitch` | `tests/unit/phases/test_guards.py` |
| A3 | `TestPhaseSwitchCountTracking` | `tests/integration/test_adaptive_m8.py` |
| A4 | `TestPreSubgraphPhaseDetection` | `tests/integration/test_adaptive_m8.py` |
| A5 | `TestAdvisorHintConsumption` | `tests/unit/phases/test_topology_router.py` |
| A5 | `TestAdvisorHintOneShot` | `tests/integration/test_adaptive_m8.py` |

Unit-тесты (`test_guards.py`, `test_topology_router.py`) проверяют
изолированную семантику отдельных gate- и router-функций.
Integration-тесты (`test_adaptive_m8.py`) проверяют сквозной
state-transfer через несколько meta-tick'ов с участием
mock-под-графов: верифицируется, что `iter_total` инкрементируется
ровно один раз за work-step (A1), что `phase_switch_count`
сбрасывается на phase-transition (A3), что pre-subgraph phase
корректно захватывается и сравнивается (A4), и что
`human_advisor_hint` потребляется ровно один dispatch-tick (A5).

Все указанные тесты — green на состоянии PR #18; CI fails на любой
последующий регресс.

---

## 5. Provenance численных результатов

**Утверждение для главы Methodology / Results.**

> Все численные результаты, представленные в работе (Tables и
> графики разделов Results), получены на **post-correction**
> реализации adaptive-роутера. Pre-correction данные сохранены в
> репозитории как audit trail (директории
> `arch/runs/pre_audit/` с парами `(run_id, audit_id)` для
> reproducibility) и **не используются в анализе**, не цитируются в
> работе и не входят в табличные сравнения. Финальные прогоны E3 и
> E4 (gpt-oss-120b), confirmation E3 и confirmation E4
> (qwen-3-235b-a22b) запущены **после** мерджа PR #18 в `main` и
> проверки green-CI'а на всех шести audit-tests из раздела 4.

Это means: любая численная величина в Results-таблицах диплома
соответствует **скорректированной** реализации.
Pre-correction артефактные замеры существуют в репозитории
исключительно для воспроизводимости audit-trail'а и **не имеют
интерпретативного веса** в обсуждении результатов.

---

## 6. Methodology framing — формулировка для текста диплома

Ниже — рекомендуемая формулировка для главы Methodology
(подраздел «Implementation audit»), которая может быть включена в
диплом дословно или с минимальной адаптацией:

> «В ходе разработки экспериментального пайплайна был проведён
> внутренний audit реализации adaptive-роутера, направленный на
> верификацию соответствия observable behaviour'а реализации declared
> design-intent. Audit выявил **пять методологических артефактов**,
> распределённых по четырём категориям: чистота state-transfer
> (два артефакта), семантика гардов (один), источники счётчиков
> (два, частично overlapping с предыдущими) и жизненный цикл сигналов
> (три, частично overlapping). Все артефакты были устранены до
> запуска финальных прогонов; каждой корректировке соответствует
> unit- или integration-тест, заякоривающий post-correction
> поведение. Численные результаты, приведённые в работе, получены на
> скорректированной реализации; pre-correction данные сохранены в
> репозитории как audit trail и не используются в анализе.»

Дополнительные принципы при описании audit'а в тексте диплома:

1. **Audit как часть методологии, не как извинение.** Audit
   презентуется как design-проверка инфраструктуры, выполненная до
   фиксации результатов — стандартная инженерная практика для
   многослойных meta-graph'ов. Никаких apologetic-формулировок.
2. **Acknowledge audit explicitly, no hiding.** Факт audit'а и пяти
   артефактов **называется явно** в тексте Methodology с указанием
   количества артефактов и категорий. Это повышает доверие к
   numerical claims в Results, не наоборот.
3. **Neutral terminology consistently.** В тексте работы
   используются те же термины, что и в данном документе:
   «implementation audit», «methodological artefact», «correction»,
   «pre-correction / post-correction». Слова «bug», «error»,
   «mistake» **не используются**.
4. **No standalone «pre-fix» section.** Pre-correction данные не
   обсуждаются отдельно в Results — они существуют в репозитории, но
   не интерпретируются. В Methodology достаточно указания на их
   archival-статус (раздел 5 настоящего документа).
5. **Reference to audit trail.** В тексте диплома уместна сноска на
   данный документ (`arch/diploma/results/methodology_audit.md`) и
   на PR #18 как точку фиксации корректировок.

---

**Конец документа.** Данный audit — методологический артефакт, не
интерпретативный. Используется как источник для главы Methodology
и как audit trail для воспроизводимости.
