"""
The Pydra workflow engine.

Pydra is a rewrite of the Nipype engine with mapping and joining as
first-class operations. It forms the core of the Nipype 2.0 ecosystem.

"""
import logging

logger = logging.getLogger("pydra")
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .engine import Submitter, Workflow, AuditFlag, ShellCommandTask, DockerTask, specs
from . import mark


def check_latest_version():
    import etelemetry

    return etelemetry.check_available_version("nipype/pydra", __version__, lgr=logger)


# Run telemetry on import for interactive sessions, such as IPython, Jupyter notebooks, Python REPL
import __main__

if not hasattr(__main__, "__file__"):
    from .engine.core import TaskBase

    if TaskBase._etelemetry_version_data is None:
        TaskBase._etelemetry_version_data = check_latest_version()
