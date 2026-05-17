# M14 — Операционный гайд: запуск лабораторной HITL-сессии

## Архитектурный обзор

Runner и Streamlit-UI взаимодействуют исключительно через одну таблицу-очередь в Postgres —
без прямых RPC-вызовов между процессами.

```
┌─────────────────────────┐        Postgres (ATM_PG_DSN)        ┌──────────────────────────┐
│  atm run (runner-процесс)│                                      │  Streamlit UI (:8501)    │
│                          │  INSERT → human_request_queue        │                          │
│  StreamlitHumanGateway  │─────────────────────────────────────►│  views.py (Queue-страница│
│  (enqueue + polling)     │                                      │  polling 1 s)            │
│                          │◄─────────────────────────────────────│                          │
│  wait_for_response()     │  UPDATE response_json, tlx_scores    │  Response + TLX form     │
└─────────────────────────┘                                      └──────────────────────────┘
          │                                                                 │
          └────────────── INSERT human_interactions ────────────────────────┘
                          (callbacks.py: _handle_human_response)
```

Ключевые компоненты (по решениям D1–D7 из m14-plan.md):

- **D1** — polling с интервалом 1 с (LISTEN/NOTIFY отложен до M14.1)
- **D2** — один участник на одну Streamlit-инстанс
- **D3** — NASA-TLX снимается после каждого решения (per-decision), не после задачи
- **D4** — `HumanResponse.source = "human"` (как в CLIGateway)
- **D5** — `study_session_id` передаётся через `HumanResponse.payload`
- **D6** — при `gateway: llm_simulated` (умолчание) поведение идентично M9.1/M9.2
- **D7** — runner передаёт gateway через kwarg `human_gateway`; debate.py читает также
  старый ключ `"gateway"` для обратной совместимости тестов

## Как запустить

### 1. Поднять инфраструктуру

```bash
docker compose --profile ui up -d
```

Запускает:
- `postgres` — основная база на стандартном порту 5432
- `ui` — Streamlit на порту 8501

### 2. Применить миграции

```bash
uv run alembic upgrade head
```

После этого в базе появятся таблицы `study_sessions`, `human_request_queue`;
в `human_interactions` добавятся поля `tlx_scores`, `raw_tlx_score`, `study_session_id`.

### 3. Настроить переменные окружения

В файле `.env` (скопировать из `.env.example`):

```
ATM_PG_DSN=postgresql+asyncpg://atm:atm@localhost:5432/atm
ATM_UI_SECRET=<случайная строка, которую оператор передаёт участнику>
```

`ATM_UI_SECRET` — это общий секрет для лабораторного режима (shared-secret auth, без OAuth).
Каждую сессию можно использовать новый секрет.

### 4. Запустить эксперимент на стороне runner-а

Оператор создаёт конфигурационный файл для сессии (пример: `conf/experiments/streamlit_smoke.yaml`)
со следующими ключами в блоке `human`:

```yaml
human:
  gateway: streamlit
  participant_id: "participant_42"
  study_session_id: "<uuid из таблицы study_sessions>"
  shared_secret: "<совпадает с ATM_UI_SECRET>"
  timeout_s: 120
  fallback_llm_model: "cerebras/llama-3.3-70b"
```

Затем запустить:

```bash
atm run --config conf/experiments/streamlit_smoke.yaml
```

### 5. Подключить участника

Участник открывает браузер:

```
http://<host>:8501
```

На экране Login вводит:
- **participant_id** — идентификатор, выданный оператором
- **shared_secret** — секрет из `ATM_UI_SECRET`

После входа участник видит очередь запросов от MAS и отвечает через форму.
После каждого ответа автоматически открывается NASA-TLX форма (6 шкал).

## SQL-примеры для анализа

### Латентность ответов участника

```sql
SELECT
    request_id,
    status,
    EXTRACT(EPOCH FROM (responded_at - created_at)) AS latency_s
FROM human_request_queue
WHERE run_id = '<run_id>'
ORDER BY created_at;
```

### NASA-TLX за сессию

```sql
SELECT
    raw_tlx_score,
    tlx_scores
FROM human_interactions
WHERE study_session_id = '<study_session_id>'
ORDER BY created_at;
```

`tlx_scores` — JSONB-объект со значениями по 6 шкалам (mental_demand, physical_demand,
temporal_demand, performance, effort, frustration, каждая 0–100).
`raw_tlx_score` — среднее арифметическое по 6 шкалам.

### Сессии участника

```sql
SELECT *
FROM study_sessions
WHERE participant_id = '<participant_id>'
ORDER BY started_at DESC;
```

### Обзор очереди в реальном времени

```sql
SELECT status, count(*)
FROM human_request_queue
GROUP BY status;
```

## Известные ограничения

1. **Polling 1 с** — UI проверяет новые запросы каждую секунду путём SELECT.
   При большой задержке ответа runner ждёт до истечения `timeout_s`, после чего
   активируется `fallback_llm_model`. Переход на LISTEN/NOTIFY запланирован в M14.1.

2. **Один участник на инстанс** — Streamlit-процесс обслуживает одного вошедшего участника.
   Параллельные сессии требуют отдельных инстансов UI на разных портах.

3. **Shared-secret auth** — нет OAuth, нет ролей. Подходит для контролируемого лабораторного
   окружения. Усиленная аутентификация запланирована в M14.x.

## Документация

- [proctor-protocol.md](proctor-protocol.md) — пошаговый чеклист оператора сессии
- [participant-consent.md](participant-consent.md) — форма информированного согласия
