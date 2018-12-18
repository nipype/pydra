import dataclasses as dc
from hashlib import sha256
from pathlib import Path
import typing as ty

File = ty.NewType('File', Path)
Directory = ty.NewType('Directory', Path)


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
        fields = tuple(state['output'].__annotations__.items())
        state['output_spec'] = (state['output'].__class__.__name__,
                                fields)
        state['output'] = dc.asdict(state['output'])
        return state

    def __setstate__(self, state):
        spec = list(state['output_spec'])
        del state['output_spec']
        klass = dc.make_dataclass(spec[0], list(spec[1]))
        state['output'] = klass(**state['output'])
        self.__dict__.update(state)


@dc.dataclass
class RuntimeSpec:
    outdir: ty.Optional[str] = None
    container: ty.Optional[str] = 'shell'
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

