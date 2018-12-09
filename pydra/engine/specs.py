import dataclasses as dc
from hashlib import sha256
from pathlib import Path
import typing as ty

File = ty.NewType('File', Path)
Directory = ty.NewType('Directory', Path)


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
class BaseSpec:
    @property
    def hash(self):
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

