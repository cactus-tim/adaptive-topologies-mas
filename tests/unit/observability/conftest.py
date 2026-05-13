"""Conftest for tests/unit/observability.

m10-merge-m8 (Wave 7): test_callbacks_human.py expects the m8-version of
`atm/observability/callbacks.py` which adds idempotent `human_request` and
`human_response` event handlers writing to the `human_interactions` table.
That update arrives in Wave 9 (final merge of feat/m8). Skip-collect until then.
"""

from __future__ import annotations

collect_ignore_glob = ["test_callbacks_human.py"]
