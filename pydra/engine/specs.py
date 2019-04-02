import dataclasses as dc
from hashlib import sha256
from pathlib import Path
import typing as ty

File = ty.NewType("File", Path)
Directory = ty.NewType("Directory", Path)


@dc.dataclass
class SpecInfo:
    name: str
    fields: ty.List[ty.Tuple] = dc.field(default_factory=list)
    bases: ty.Tuple[dc.dataclass] = dc.field(default_factory=tuple)


@dc.dataclass(order=True)
class BaseSpec:
    """The base dataclass specs for all inputs and outputs"""

    @property
    def hash(self):
        """Compute a basic hash for any given set of fields"""
        return sha256(str(self).encode()).hexdigest()

    def retrieve_values(self, wf, state_index=None):
        temp_values = {}
        for name, _ in self.fields:
            value = getattr(self, name)
            if isinstance(value, LazyField):
                value = value.get_value(wf, state_index=state_index)
                temp_values[name] = value
        for field, value in temp_values.items():
            setattr(self, field, value)


@dc.dataclass
class Runtime:
    rss_peak_gb: ty.Optional[float] = None
    vms_peak_gb: ty.Optional[float] = None
    cpu_peak_percent: ty.Optional[float] = None


@dc.dataclass
class Result:
    output: ty.Optional[ty.Any] = None
    runtime: ty.Optional[Runtime] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        fields = tuple(state["output"].__annotations__.items())
        state["output_spec"] = (state["output"].__class__.__name__, fields)
        state["output"] = dc.asdict(state["output"])
        return state

    def __setstate__(self, state):
        spec = list(state["output_spec"])
        del state["output_spec"]
        klass = dc.make_dataclass(spec[0], list(spec[1]))
        state["output"] = klass(**state["output"])
        self.__dict__.update(state)


@dc.dataclass
class RuntimeSpec:
    outdir: ty.Optional[str] = None
    container: ty.Optional[str] = "shell"
    network: bool = False
    """
    from CWL:
    InlineJavascriptRequirement
    SchemaDefRequirement
    DockerRequirement
    SoftwareRequirement
    InitialWorkDirRequirement
    EnvVarRequirement
    ShellCommandRequirement
    ResourceRequirement

    InlineScriptRequirement
    """


@dc.dataclass
class ShellSpec(BaseSpec):
    executable: ty.Union[str, ty.List[str]]


@dc.dataclass
class ShellOutSpec(BaseSpec):
    return_code: int
    stdout: ty.Union[File, str]
    stderr: ty.Union[File, str]


@dc.dataclass
class ContainerSpec(ShellSpec):
    image: ty.Union[File, str]
    container: ty.Union[File, str, None]
    container_xargs: ty.Optional[ty.List[str]] = None
    bindings: ty.Optional[
        ty.List[
            ty.Tuple[
                Path,  # local path
                Path,  # container path
                ty.Optional[str],  # mount mode
            ]
        ]
    ] = None


@dc.dataclass
class DockerSpec(ContainerSpec):
    container: str = "docker"


@dc.dataclass
class SingularitySpec(ContainerSpec):
    container: str = "singularity"


class LazyField:
    def __init__(self, node, attr_type):
        self.name = node.name
        if attr_type == "input":
            self.fields = [name for name, _ in node.input_spec.fields]
        elif attr_type == "output":
            self.fields = node.output_names
        else:
            raise ValueError("LazyField: Unknown attr_type: {}".format(attr_type))
        self.attr_type = attr_type
        self.field = None

    def __getattr__(self, name):
        if name in self.fields:
            self.field = name
            return self
        if name in dir(self):
            return self.__getattribute__(name)
        raise AttributeError(
            "Task {0} has no {1} attribute {2}".format(
                self.name, self.attr_type, name
            )
        )

    def __repr__(self):
        return "LF('{0}', '{1}')".format(self.name, self.field)

    def get_value(self, wf, state_index=None):
        if self.attr_type == "input":
            return getattr(wf.inputs, self.field)
        elif self.attr_type == "output":
            node = getattr(wf, self.name)
            result = node.result(state_index=state_index)
            return getattr(result.output, self.field)
