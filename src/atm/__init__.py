__version__ = "0.1.0"

from atm.tasks import EVALUATORS, TASKS

__all__ = [
    "EVALUATORS",
    "TASKS",
    "__version__",
]

# G2 reproducibility bundle: structlog bootstrap with filter_secrets.
# Opt out via ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1 (e.g. when an embedding
# application owns logging configuration).
import os as _os

if _os.environ.get("ATM_DISABLE_STRUCTLOG_BOOTSTRAP") != "1":
    try:
        from atm.observability.log_processors import configure_structlog as _cfg

        _cfg()
    except Exception:
        # Never let bootstrap failures break import — observability is best-effort.
        pass
del _os
