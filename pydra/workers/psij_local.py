from pydra.workers.psij_base import PsijWorker


class Worker(PsijWorker):
    """A worker to execute tasks using PSI/J on the local machine."""

    subtype = "local"
    plugin_name = f"psij-{subtype}"
