# Протокол прокторинга — M14 HITL User Study

Документ описывает действия оператора (proctor) до, во время и после лабораторной сессии.
Proctor несёт ответственность за техническое сопровождение, соблюдение условий согласия
и корректную запись данных.

---

## До сессии

- [ ] Убедиться, что Docker-окружение поднято:

  ```bash
  docker compose --profile ui ps
  # ожидаемое состояние: postgres и ui — "running"
  ```

- [ ] Убедиться, что база данных достижима и миграции применены:

  ```bash
  uv run alembic current
  # ожидаемый вывод: ...0005_m14_streamlit_hitl (head)
  ```

- [ ] Создать строку `study_sessions` для участника.
  Через прямой SQL:

  ```sql
  INSERT INTO study_sessions (
      id,
      participant_id,
      status,
      consent_given,
      started_at
  ) VALUES (
      gen_random_uuid(),
      '<participant_id>',
      'pending',
      false,
      now()
  )
  RETURNING id;
  ```

  Сохранить полученный `id` — это `study_session_id` для конфигурационного файла runner-а
  и для передачи участнику.

- [ ] Убедиться, что участник подписал форму информированного согласия
  (см. [participant-consent.md](participant-consent.md)).

- [ ] После подписания согласия обновить флаг в базе:

  ```sql
  UPDATE study_sessions
  SET consent_given = true
  WHERE id = '<study_session_id>';
  ```

- [ ] Подготовить конфигурационный файл runner-а
  (`conf/experiments/<session_name>.yaml`) с полями:

  ```yaml
  human:
    gateway: streamlit
    participant_id: "<participant_id>"
    study_session_id: "<study_session_id>"
    shared_secret: "<значение из ATM_UI_SECRET>"
    timeout_s: 120
    fallback_llm_model: "cerebras/llama-3.3-70b"
  ```

- [ ] Передать участнику:
  - URL интерфейса: `http://<host>:8501`
  - `participant_id`
  - `shared_secret` (значение из ATM_UI_SECRET на текущую сессию)

- [ ] Провести краткий инструктаж: объяснить порядок работы с формой ответа
  и NASA-TLX шкалами (см. форму согласия).

---

## Во время сессии

- [ ] Запустить runner на отдельной машине или в отдельном терминале:

  ```bash
  atm run --config conf/experiments/<session_name>.yaml
  ```

- [ ] Следить за очередью запросов в реальном времени:

  ```bash
  watch -n 5 'psql $ATM_PG_DSN -c "SELECT status, count(*) FROM human_request_queue GROUP BY status;"'
  ```

  Нормальная картина: запросы переходят `pending → claimed → completed`.

- [ ] Следить за прогрессом прогона:

  ```sql
  SELECT status, updated_at
  FROM runs
  WHERE id = '<run_id>'
  ORDER BY updated_at DESC
  LIMIT 1;
  ```

- [ ] Если участник не реагирует на запрос дольше, чем `timeout_s` (по умолчанию 120 с):
  - Уточнить у участника причину задержки (технические трудности, усталость).
  - При необходимости прервать прогон: завершить процесс `atm run` (Ctrl+C).
    Runner помечает прогон как `failed`; runner-процесс освобождает ресурсы.

- [ ] Если участник хочет взять паузу:
  - Приостановить runner (`Ctrl+Z` или отдельный сигнал, если поддерживается в конкретной версии).
  - Зафиксировать время паузы в `proctor_notes`.

- [ ] Наблюдать за поведением участника, фиксировать в блокноте технические инциденты,
  вопросы, нестандартные реакции — они войдут в `proctor_notes` после сессии.

---

## После сессии

- [ ] Дождаться завершения прогона или явно прервать runner.

- [ ] Обновить статус сессии:

  ```sql
  UPDATE study_sessions
  SET
      status    = 'completed',
      ended_at  = now()
  WHERE id = '<study_session_id>';
  ```

- [ ] Внести заметки наблюдателя:

  ```sql
  UPDATE study_sessions
  SET proctor_notes = '<текст наблюдений, технических инцидентов>'
  WHERE id = '<study_session_id>';
  ```

- [ ] Проверить, что данные записаны корректно:

  ```sql
  SELECT
      hi.request_id,
      hi.raw_tlx_score,
      hi.tlx_scores,
      hi.study_session_id
  FROM human_interactions hi
  WHERE hi.study_session_id = '<study_session_id>'
  ORDER BY hi.created_at;
  ```

- [ ] Экспортировать срез данных эксперимента:

  ```bash
  atm export-exp <exp_id>
  ```

  Если команда `export-exp` не реализована в текущей версии, использовать SQL-дамп:

  ```bash
  pg_dump --table=human_interactions \
          --table=human_request_queue \
          --table=study_sessions \
          "$ATM_PG_DSN" \
          > export_<study_session_id>.sql
  ```

- [ ] Убедиться, что участник получил подтверждение об окончании сессии и мог задать вопросы.

- [ ] Если планируются последующие сессии с тем же участником — зафиксировать
  `participant_id` и время следующего визита.

---

## Экстренная остановка

Если требуется немедленно прервать сессию (технический сбой, отказ участника продолжать):

1. Остановить runner: Ctrl+C в терминале `atm run`.
2. Обновить статус сессии:

   ```sql
   UPDATE study_sessions
   SET status = 'aborted', ended_at = now()
   WHERE id = '<study_session_id>';
   ```

3. Зафиксировать причину в `proctor_notes`.
4. Данные, уже записанные в `human_interactions`, сохраняются и пригодны для анализа.

---

## Справочные ссылки

- [README.md](README.md) — архитектурный обзор и команды запуска
- [participant-consent.md](participant-consent.md) — форма информированного согласия
