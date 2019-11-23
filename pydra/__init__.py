from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .engine import Submitter, Workflow, AuditFlag, ShellCommandTask, DockerTask, specs
from . import mark


def check_latest_version(raise_exception=False):
    """Check for the latest version of the library

    parameters:
    raise_exception: boolean
        Raise a RuntimeError if a bad version is being used
    """
    import etelemetry
    import logging
    from pkg_resources import parse_version

    logger = logging.getLogger("pydra")

    INIT_MSG = "Running {packname} version {version} (latest: {latest})".format

    latest = {"version": "Unknown", "bad_versions": []}
    result = None
    try:
        result = etelemetry.get_project("nipype/pydra")
    except Exception as e:
        logger.warning("Could not check for version updates: \n%s", e)
    finally:
        if result:
            latest.update(**result)
            if parse_version(__version__) != parse_version(latest["version"]):
                logger.info(
                    INIT_MSG(
                        packname="pydra", version=__version__, latest=latest["version"]
                    )
                )
            if latest["bad_versions"] and any(
                [
                    parse_version(__version__) == parse_version(ver)
                    for ver in latest["bad_versions"]
                ]
            ):
                message = (
                    "You are using a version of Pydra with a critical "
                    "bug. Please use a different version."
                )
                if raise_exception:
                    raise RuntimeError(message)
                else:
                    logger.critical(message)
    return latest


# Run telemetry on import for interactive sessions, such as IPython, Jupyter notebooks, Python REPL
import __main__

if not hasattr(__main__, "__file__"):
    from .engine.core import TaskBase

    if TaskBase._etelemetry_version_data is None:
        TaskBase._etelemetry_version_data = check_latest_version()
