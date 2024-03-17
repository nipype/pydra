from pathlib import Path
import platformdirs
from pydra._version import __version__

user_cache_dir = Path(
    platformdirs.user_cache_dir(
        appname="pydra",
        appauthor="nipype",
        version=__version__,
    )
)
