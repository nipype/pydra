"""
The Pydra workflow engine.

Pydra is a rewrite of the Nipype engine with mapping and joining as
first-class operations. It forms the core of the Nipype 2.0 ecosystem.

"""
# This call enables pydra.tasks to be used as a namespace package when installed
# in editable mode. In normal installations it has no effect.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

import logging

import __main__
import attr

from . import mark
from .engine import AuditFlag, DockerTask, ShellCommandTask, Submitter, Workflow, specs

__all__ = (
    "Submitter",
    "Workflow",
    "AuditFlag",
    "ShellCommandTask",
    "DockerTask",
    "specs",
    "mark",
)

try:
    from ._version import __version__
except ImportError:
    pass

logger = logging.getLogger("pydra")


def check_latest_version():
    import etelemetry

    return etelemetry.check_available_version("nipype/pydra", __version__, lgr=logger)


# Run telemetry on import for interactive sessions, such as IPython, Jupyter notebooks, Python REPL
if not hasattr(__main__, "__file__"):
    from .engine.core import TaskBase

    if TaskBase._etelemetry_version_data is None:
        TaskBase._etelemetry_version_data = check_latest_version()
