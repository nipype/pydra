from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .engine import Submitter, Workflow, AuditFlag, ShellCommandTask, DockerTask, specs
from . import mark
