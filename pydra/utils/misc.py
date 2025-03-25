from pathlib import Path
import re
import platformdirs
from pydra._version import __version__

user_cache_dir = Path(
    platformdirs.user_cache_dir(
        appname="pydra",
        appauthor="nipype",
        version=__version__,
    )
)


def add_exc_note(e: Exception, note: str) -> Exception:
    """Adds a note to an exception in a Python <3.11 compatible way

    Parameters
    ----------
    e : Exception
        the exception to add the note to
    note : str
        the note to add

    Returns
    -------
    Exception
        returns the exception again
    """
    if hasattr(e, "add_note"):
        e.add_note(note)
    else:
        e.args = (e.args[0] + "\n" + note,)
    return e


def exc_info_matches(exc_info, match, regex=False):
    if exc_info.value.__cause__ is not None:
        msg = str(exc_info.value.__cause__)
    else:
        msg = str(exc_info.value)
    if regex:
        return re.match(".*" + match, msg)
    else:
        return match in msg
