"""Storage layer: SQLAlchemy models, session factory, parquet writer, checkpointer.

NOTE: The SQLAlchemy ORM model for phase transitions is exported as ``PhaseRow``
(not ``Phase``) to avoid a name collision with ``atm.core.types.Phase`` (StrEnum).
Import ``PhaseRow`` when you need the ORM model; import ``atm.core.types.Phase``
when you need the enum.
"""
from atm.storage.checkpointer import (
    build_checkpointer,
    checkpointer_scope,
)
from atm.storage.models import (
    Base,
    BudgetEvent,
    Experiment,
    FinishReason,
    HumanInteraction,
    Run,
    TopologyTransition,
)
from atm.storage.models import (
    Phase as PhaseRow,
)
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.schemas import (
    LLM_CALL_SCHEMA,
    MESSAGE_SCHEMA,
    PHASE_SCHEMA,
    SCRATCHPAD_SCHEMA,
    TOOL_CALL_SCHEMA,
    TOPOLOGY_TRANSITION_SCHEMA,
)
from atm.storage.session import (
    create_engine,
    create_session_factory,
    session_scope,
)

__all__ = [
    "LLM_CALL_SCHEMA",
    "MESSAGE_SCHEMA",
    "PHASE_SCHEMA",
    "SCRATCHPAD_SCHEMA",
    "TOOL_CALL_SCHEMA",
    "TOPOLOGY_TRANSITION_SCHEMA",
    "Base",
    "BudgetEvent",
    "Experiment",
    "FinishReason",
    "HumanInteraction",
    "ParquetWriter",
    "PhaseRow",
    "Run",
    "TopologyTransition",
    "build_checkpointer",
    "checkpointer_scope",
    "create_engine",
    "create_session_factory",
    "session_scope",
]
