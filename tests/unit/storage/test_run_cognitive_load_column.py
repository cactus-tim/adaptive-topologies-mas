"""Step 3.3 — verify Run.cognitive_load_proxy column exists and is nullable."""

from __future__ import annotations

import sqlalchemy as sa

from atm.storage.models import Run


def test_cognitive_load_proxy_column_exists() -> None:
    assert hasattr(Run, "cognitive_load_proxy")


def test_cognitive_load_proxy_is_nullable() -> None:
    col = Run.__table__.columns["cognitive_load_proxy"]
    assert col.nullable is True


def test_cognitive_load_proxy_is_double_or_float() -> None:
    col = Run.__table__.columns["cognitive_load_proxy"]
    assert isinstance(col.type, (sa.Double, sa.Float))
