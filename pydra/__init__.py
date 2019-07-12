from .__about__ import (
    __version__,
    __author__,
    __license__,
    __maintainer__,
    __email__,
    __status__,
    __url__,
    __packagename__,
    __description__,
    __longdesc__,
)

# expose api
try:
    from .engine.core import Workflow
    from .engine.task import to_task
    from .engine.submitter import Submitter
except ImportError:  # on install
    pass