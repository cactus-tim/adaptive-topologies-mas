__version__ = "0.1.0"

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
except Exception:
    pass

from atm.tasks import EVALUATORS, TASKS

__all__ = [
    "EVALUATORS",
    "TASKS",
    "__version__",
]

import os as _os

if _os.environ.get("ATM_DISABLE_STRUCTLOG_BOOTSTRAP") != "1":
    try:
        from atm.observability.log_processors import configure_structlog as _cfg

        _cfg()
    except Exception:
        pass
del _os
