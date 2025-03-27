from pydra.workers.psij_base import PsijWorker


class Worker(PsijWorker):
    """A worker to execute tasks using PSI/J using SLURM."""

    subtype = "slurm"
    plugin_name = f"psij-{subtype}"
