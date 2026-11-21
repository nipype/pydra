"""Worker classes: what a substrate can start."""

import json
import re
import tomllib
import typing as ty
from pathlib import Path
import attrs

_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)I?B?$")
_SCALE = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(value: "int | str | None") -> "int | None":
    """Bytes from an integer (bytes) or a string such as ``"40G"`` (powers of 1024)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"not a size: {value!r}")
    if isinstance(value, int):
        return value
    match = _SIZE_RE.match(str(value).strip().upper())
    if not match:
        raise ValueError(f"not a size: {value!r}")
    return int(float(match.group(1)) * _SCALE[match.group(2)])


def _size(value):
    return parse_size(value)


def _to_tags(value) -> frozenset:
    if isinstance(value, str):
        return frozenset((value,))
    return frozenset(value)


def _positive_int(_, attribute, value):
    if value is not None and value < 1:
        raise ValueError(f"{attribute.name} must be positive, not {value!r}")


@attrs.frozen
class WorkerClass:
    """A kind of worker a substrate can start.

    Sizes are bytes; they may be written ``"40G"`` and are converted.
    """

    name: str
    device: ty.Literal["cpu", "cuda", "mps"]
    gpus: int = 0
    gpu_type: str | None = None
    gpu_mem: int | None = attrs.field(default=None, converter=_size)
    cpus: int | None = None
    mem: int | None = attrs.field(default=None, converter=_size)
    time: int | None = None
    max_workers: int = attrs.field(default=1, validator=_positive_int)
    workers_per_device: int = attrs.field(default=1, validator=_positive_int)
    mounts: tuple[Path, ...] = attrs.field(
        default=(), converter=lambda v: tuple(Path(m) for m in v)
    )
    substrate_args: ty.Mapping[str, str] = attrs.field(
        factory=dict, converter=lambda v: dict(v), eq=False
    )
    quantum_jobs: int = attrs.field(default=64, validator=_positive_int)
    quantum_s: float = 300.0
    tags: frozenset[str] = attrs.field(factory=frozenset, converter=_to_tags)

    def __attrs_post_init__(self):
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be cpu, cuda or mps, not {self.device!r}")
        if self.device == "cpu" and self.gpus:
            raise ValueError(f"class {self.name!r}: a cpu class has no gpus")
        if self.gpu_type is not None and not self.gpus:
            raise ValueError(f"class {self.name!r}: gpu_type requires gpus")

    @classmethod
    def load(
        cls, source: "ty.Mapping | ty.Sequence | str | Path"
    ) -> "list[WorkerClass]":
        """Classes from data.

        ``source`` is a mapping of class name to fields, a list of mappings each
        with a ``name``, or the path of a TOML, JSON or YAML file holding either
        (under an optional top-level ``classes`` key). YAML needs PyYAML, which
        pydra does not depend on. Unknown fields are an error.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            text = path.read_text()
            if path.suffix == ".json":
                source = json.loads(text)
            elif path.suffix == ".toml":
                source = tomllib.loads(text)
            else:
                try:
                    import yaml
                except ImportError as e:
                    raise ImportError(
                        f"reading {path.name} needs the 'pyyaml' package; install it "
                        f"or write the classes as TOML or JSON"
                    ) from e
                source = yaml.safe_load(text)
        if isinstance(source, ty.Mapping) and set(source) == {"classes"}:
            source = source["classes"]
        if isinstance(source, ty.Mapping):
            entries = []
            for name, fields in source.items():
                entries.append({**fields, "name": name})
        else:
            entries = [dict(e) for e in source]
        classes = [cls(**e) for e in entries]
        names = [c.name for c in classes]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate worker class names: {dup}")
        return classes
