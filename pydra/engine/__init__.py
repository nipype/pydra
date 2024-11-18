"""The core of the workflow engine."""

from .submitter import Submitter
import __main__
import logging
from ._version import __version__

__all__ = [
    "Submitter",
    "logger",
    "check_latest_version",
]

logger = logging.getLogger("pydra")


def check_latest_version():
    import etelemetry

    return etelemetry.check_available_version("nipype/pydra", __version__, lgr=logger)


# Run telemetry on import for interactive sessions, such as IPython, Jupyter notebooks, Python REPL
if not hasattr(__main__, "__file__"):
    from .engine.core import TaskBase

    if TaskBase._etelemetry_version_data is None:
        TaskBase._etelemetry_version_data = check_latest_version()
