"""Unit tests for atm.evaluation.tlx.persist_tlx — pure in-memory (no PG).

Tests:
  1. persist_tlx calls session.execute with a SQL UPDATE statement.
  2. persist_tlx calls session.commit after execute.
  3. Idempotency: calling twice with different raw_score — each call issues
     its own UPDATE; the second call is independent of the first.
  4. The UPDATE statement targets the correct table and columns
     (human_interactions, raw_tlx_score, id).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from atm.evaluation.tlx import persist_tlx


@pytest.mark.asyncio
async def test_persist_tlx_executes_update() -> None:
    """persist_tlx calls session.execute with a non-None SQL construct."""
    mock_session = AsyncMock()
    interaction_id = uuid.uuid4()

    await persist_tlx(mock_session, interaction_id, raw_score=42.5)

    mock_session.execute.assert_called_once()
    stmt = mock_session.execute.call_args[0][0]
    assert stmt is not None


@pytest.mark.asyncio
async def test_persist_tlx_commits_after_execute() -> None:
    """persist_tlx awaits session.commit() after session.execute()."""
    mock_session = AsyncMock()
    interaction_id = uuid.uuid4()

    await persist_tlx(mock_session, interaction_id, raw_score=42.5)

    mock_session.commit.assert_called_once()
    assert mock_session.execute.call_count == 1
    assert mock_session.commit.call_count == 1
    call_names = [c[0] for c in mock_session.mock_calls]
    assert "execute" in call_names
    assert "commit" in call_names
    assert call_names.index("execute") < call_names.index("commit")


@pytest.mark.asyncio
async def test_persist_tlx_idempotent_two_calls() -> None:
    """Calling persist_tlx twice yields two independent UPDATE + commit cycles."""
    mock_session = AsyncMock()
    interaction_id = uuid.uuid4()

    await persist_tlx(mock_session, interaction_id, raw_score=42.5)
    await persist_tlx(mock_session, interaction_id, raw_score=88.0)

    assert mock_session.execute.call_count == 2
    assert mock_session.commit.call_count == 2


@pytest.mark.asyncio
async def test_persist_tlx_statement_shape() -> None:
    """The compiled SQL UPDATE targets human_interactions with raw_tlx_score."""
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    mock_session = AsyncMock()
    interaction_id = uuid.uuid4()
    raw_score = 55.0

    await persist_tlx(mock_session, interaction_id, raw_score=raw_score)

    stmt = mock_session.execute.call_args[0][0]
    compiled = stmt.compile(dialect=sqlite_dialect.dialect())
    sql_str = str(compiled).lower()
    assert "human_interactions" in sql_str
    assert "raw_tlx_score" in sql_str
    assert "id" in sql_str
