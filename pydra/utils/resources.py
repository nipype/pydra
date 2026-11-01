"""Compute resources a task asks of a scheduler, and their Slurm and SGE spellings."""

import re
import typing as ty
import attrs

_MEM_RE = re.compile(r"^\d+[KMGT]$")


def _positive(_, attribute, value):
    if value is not None and value < 1:
        raise ValueError(f"{attribute.name} must be a positive integer, not {value!r}")


def _to_mem(value: "int | str | None") -> "str | None":
    """An integer is megabytes; a string is digits followed by K, M, G or T."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        value = f"{value}M"
    value = str(value).upper()
    if not _MEM_RE.match(value) or int(value[:-1]) < 1:
        raise ValueError(
            f"mem must be a positive integer (megabytes) or digits followed by "
            f"K, M, G or T, not {value!r}"
        )
    return value


def _to_tags(value: ty.Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@attrs.frozen
class ComputeResources:
    """What a task asks of the machine it runs on.

    The resources are scheduler metadata: they never enter a task's checksum.

    Parameters
    ----------
    cpus : int, optional
        CPU cores
    mem : int or str, optional
        Memory: an integer is megabytes, a string is digits followed by K, M, G or T
    gpus : int, optional
        Number of GPUs
    gpu_type : str, optional
        Model of GPU, only meaningful with ``gpus``
    time : int, optional
        Wall-clock limit in seconds
    tags : tuple of str, optional
        Labels for routing, not mapped to scheduler arguments
    """

    cpus: int | None = attrs.field(default=None, validator=_positive)
    mem: str | None = attrs.field(default=None, converter=_to_mem)
    gpus: int | None = attrs.field(default=None, validator=_positive)
    gpu_type: str | None = None
    time: int | None = attrs.field(default=None, validator=_positive)
    tags: tuple[str, ...] = attrs.field(default=(), converter=_to_tags)

    def __attrs_post_init__(self):
        if self.gpu_type is not None and self.gpus is None:
            raise ValueError("gpu_type requires gpus")

    def slurm_args(self) -> list[str]:
        """The ``sbatch`` arguments for these resources."""
        args = []
        if self.cpus is not None:
            args.append(f"--cpus-per-task={self.cpus}")
        if self.mem is not None:
            args.append(f"--mem={self.mem}")
        if self.gpus is not None:
            gres = (
                f"gpu:{self.gpu_type}:{self.gpus}"
                if self.gpu_type
                else f"gpu:{self.gpus}"
            )
            args.append(f"--gres={gres}")
        if self.time is not None:
            args.append(f"--time={_slurm_time(self.time)}")
        return args

    def sge_args(self) -> list[str]:
        """The ``qsub`` arguments for these resources.

        ``-pe smp`` and ``-l gpu``/``gpu_type`` follow the common site convention;
        clusters that name their parallel environment or GPU complexes differently
        should pass ``qsub_args`` instead.
        """
        args = []
        if self.cpus is not None:
            args += ["-pe", "smp", str(self.cpus)]
        if self.mem is not None:
            args += ["-l", f"mem_free={self.mem}"]
        if self.gpus is not None:
            args += ["-l", f"gpu={self.gpus}"]
        if self.gpu_type is not None:
            args += ["-l", f"gpu_type={self.gpu_type}"]
        if self.time is not None:
            args += ["-l", f"h_rt={_hms(self.time)}"]
        return args


def _hms(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    return f"{hours:02d}:{rem // 60:02d}:{rem % 60:02d}"


def _slurm_time(seconds: int) -> str:
    days, rem = divmod(seconds, 86400)
    return f"{days}-{_hms(rem)}"
