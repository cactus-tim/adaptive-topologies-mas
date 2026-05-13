"""Conftest for tests/unit/human.

m10-merge-m8 (Wave 7): test_role_router_back_compat.py imports m8 topology HITL
helpers (e.g. _build_human_judge_node, _build_human_reviewer_node) that arrive
only in Wave 9 (final merge of feat/m8). Skip-collect until then.
"""

from __future__ import annotations

collect_ignore_glob = ["test_role_router_back_compat.py"]
