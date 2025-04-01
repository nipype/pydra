import logging
from ._version import __version__

# import __main__

logger = logging.getLogger("pydra")


def check_latest_version():
    import etelemetry

    return etelemetry.check_available_version("nipype/pydra", __version__, lgr=logger)


# Run telemetry on import for interactive sessions, such as IPython, Jupyter notebooks, Python REPL
# if not hasattr(__main__, "__file__"):
#     from pydra.compose.base import Task

#     if Task._etelemetry_version_data is None:
#         Task._etelemetry_version_data = check_latest_version()
