import os
import typing as ty
import logging
from pathlib import Path
import re
import subprocess as sp

import attrs
from pydra.compose import shell
from pydra.environments import base
from pydra.utils.general import ensure_list


logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


@attrs.define
class Lmod(base.Environment):
    """Lmod environment."""

    modules: list[str] = attrs.field(converter=ensure_list)

    @modules.validator
    def _validate_modules(self, _, value: ty.Any) -> None:
        if not value:
            raise ValueError("At least one module must be specified")
        if not all(isinstance(v, str) for v in value):
            raise ValueError("All module names must be strings")

    def execute(self, job: "Job[shell.Task]") -> dict[str, int | str]:
        env_src = self.run_lmod_cmd("python", "load", *self.modules)
        env = {}
        for key, value in re.findall(
            r"""os\.environ\[['"](.*?)['"]\]\s*=\s*['"](.*?)['"]""", env_src
        ):
            env[key] = value
        cmd_args = job.task._command_args(values=job.inputs)
        values = base.execute(cmd_args, env=env)
        return_code, stdout, stderr = values
        if return_code:
            msg = f"Error running '{job.name}' job with {cmd_args}:"
            if stderr:
                msg += "\n\nstderr:\n" + stderr
            if stdout:
                msg += "\n\nstdout:\n" + stdout
            raise RuntimeError(msg)
        return {"return_code": return_code, "stdout": stdout, "stderr": stderr}

    @classmethod
    def modules_are_installed(cls) -> bool:
        return "MODULESHOME" in os.environ

    @classmethod
    def run_lmod_cmd(cls, *args: str) -> str:
        if not cls.modules_are_installed():
            raise RuntimeError(
                "Could not find Lmod installation, please ensure it is installed and MODULESHOME is set"
            )
        lmod_exec = Path(os.environ["MODULESHOME"]) / "libexec" / "lmod"

        try:
            output_bytes, error_bytes = sp.Popen(
                [str(lmod_exec)] + list(args),
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            ).communicate()
        except (sp.CalledProcessError, OSError) as e:
            raise RuntimeError(f"Error running 'lmod': {e}")

        output = output_bytes.decode("utf-8")
        error = error_bytes.decode("utf-8")

        if output == "_mlstatus = False\n":
            raise RuntimeError(f"Error running module cmd '{' '.join(args)}':\n{error}")

        return output


# Alias so it can be referred to as lmod.Environment
Environment = Lmod
