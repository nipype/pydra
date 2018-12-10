import dataclasses as dc
from hashlib import sha256
from pathlib import Path
import typing as ty

File = ty.NewType('File', Path)
Directory = ty.NewType('Directory', Path)


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


@dc.dataclass
class ShellSpec(BaseSpec):
    executable: ty.Union[str, ty.List[str]]


@dc.dataclass
class ShellOutSpec(BaseSpec):
    return_code: int
    stdout: ty.Union[File, str]
    stderr: ty.Union[File, str]


@dc.dataclass
class ContainerSpec(BaseSpec):
    image: ty.Union[File, str, None]
    bind_mounts: ty.Optional[ty.Tuple[Path,  # local path
                                      ty.Optional[Path],  # container path
                                      ty.Optional[str]]]  # mount mode
