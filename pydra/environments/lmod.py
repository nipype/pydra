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

_ASSIGNMENT = re.compile(r"""os\.environ\[['"](.*?)['"]\]\s*=\s*['"](.*?)['"]""")
_REMOVAL = re.compile(
    r"""(?:del\s+os\.environ\[['"](.*?)['"]\]|os\.environ\.pop\(['"](.*?)['"])"""
)


@attrs.define
class Lmod(base.Environment):
    """Lmod environment."""

    modules: list[str] = attrs.field(converter=ensure_list)

    resolves: ty.ClassVar[bool] = True

    @modules.validator
    def _validate_modules(self, _, value: ty.Any) -> None:
        if not value:
            raise ValueError("At least one module must be specified")
        if not all(isinstance(v, str) for v in value):
            raise ValueError("All module names must be strings")

    def _ref_key(self) -> ty.Hashable:
        return ("lmod", tuple(self.modules))

    def _warm_handle(
        self,
        ref: base.ResolvedRef,
        roots: frozenset[Path],
        device: ty.Any,
        writable: frozenset[Path],
        scratch: Path | None,
    ) -> "WarmLmod":
        return WarmLmod(self, ref, frozenset(), device)

    def resolve(self) -> base.ResolvedRef:
        """Resolve the modules to their full names once (RFC 0004 section 5.3): in a child process,
        ``lmod python purge`` with the calling shell's modules in place, then ``lmod python load``
        of the requested names, reading ``LOADEDMODULES``. A loaded name is kept for a requested one
        when it equals it, its name part (before the last ``/``) equals it, or the requested name,
        holding a ``/``, is a prefix of it up to a version separator.

        The key carries the kept full names in the requested order, and the worker purges and loads
        them. Dependencies, the calling shell's modules, the purged ``MODULEPATH`` and the Lmod tree
        are recorded, not keyed. Without Lmod, or when a name does not load or map here, the names
        are keyed as written and unresolved.
        """
        written = " ".join(self.modules)
        unresolved = base.ResolvedRef(f"unresolved:lmod:{written}", written)
        if not self.modules_are_installed():
            return unresolved
        try:
            purged, loaded, files, _ = self._purge_and_load(self.modules)
        except RuntimeError as e:
            logger.warning("Lmod modules %s are not resolved: %s", written, e)
            return unresolved
        kept: list[str] = []
        mapping = {}
        for requested in self.modules:
            full = self._match(requested, loaded)
            if full is None:
                logger.warning("Lmod module %r is not among %s", requested, loaded)
                return unresolved
            mapping[requested] = full
            if full not in kept:
                kept.append(full)
        shell_modules = os.environ.get("LOADEDMODULES", "")
        recorded = {
            "modules": mapping,
            "dependencies": [m for m in loaded if m not in kept],
            "shell_modules": [m for m in shell_modules.split(":") if m],
            "modulepath": purged.get("MODULEPATH"),
            "tree": os.environ["MODULESHOME"],
            "files": dict(zip(loaded, files)) if len(files) == len(loaded) else files,
        }
        names = " ".join(kept)
        return base.ResolvedRef(f"lmod@1:{names}", names, recorded=recorded)

    @classmethod
    def _purge_and_load(
        cls, names: ty.Sequence[str]
    ) -> tuple[dict[str, str], list[str], list[str], dict[str, str]]:
        """Purge from this process's environment, then load ``names``: the purged
        environment, the loaded full names and their files, and the loaded environment
        """
        env = dict(os.environ)
        purged = cls._applied(cls.run_lmod_cmd("python", "purge", env=env), env)
        output = cls.run_lmod_cmd("python", "load", *names, env=purged)
        assigned = dict(_ASSIGNMENT.findall(output))
        loaded = [m for m in assigned.get("LOADEDMODULES", "").split(":") if m]
        files = [f for f in assigned.get("_LMFILES_", "").split(":") if f]
        return purged, loaded, files, cls._applied(output, purged)

    @staticmethod
    def _match(requested: str, loaded: list[str]) -> str | None:
        for full in loaded:
            if full == requested or full.rpartition("/")[0] == requested:
                return full
        if "/" in requested:
            for full in loaded:
                rest = full[len(requested) :]
                if full.startswith(requested) and rest[:1] in ".-_":
                    return full
        return None

    @staticmethod
    def _applied(output: str, env: dict[str, str]) -> dict[str, str]:
        """``env`` with the assignments and removals of an ``lmod python`` output"""
        env = dict(env)
        for a, b in _REMOVAL.findall(output):
            env.pop(a or b, None)
        env.update(_ASSIGNMENT.findall(output))
        return env

    def execute(self, job: "Job[shell.Task]") -> dict[str, int | str]:
        if job.environment is not self:
            raise base.ImageResolutionError(
                f"{self!r} was asked to run a job built with the environment "
                f"{job.environment!r}, whose modules are the ones carried with it"
            )
        # the full names the submitting process resolved and carried with the job,
        # loaded after a purge as they were resolved, so the dependencies match
        return self._run(job, self._loaded_environment(job.resolved))

    def _loaded_environment(self, resolved: base.ResolvedRef) -> dict[str, str]:
        """The environment after a purge and a load of the resolved full names"""
        names = resolved.run.split()
        _, loaded, _, env = self._purge_and_load(names)
        dependencies = [m for m in loaded if m not in names]
        recorded = resolved.recorded.get("dependencies")
        if recorded is not None and dependencies != recorded:
            logger.warning(
                "Lmod modules %s loaded the dependencies %s here, but %s where the job "
                "was resolved",
                names,
                dependencies,
                recorded,
            )
        return env

    @staticmethod
    def _run(job: "Job[shell.Task]", env: dict[str, str]) -> dict[str, int | str]:
        cmd_args = job.task._command_args(values=job.inputs)
        job.cache_dir.mkdir(parents=True, exist_ok=True)
        values = base.execute(cmd_args, env=env, cwd=job.cache_dir)
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
    def run_lmod_cmd(cls, *args: str, env: dict[str, str] | None = None) -> str:
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
                env=env,
            ).communicate()
        except (sp.CalledProcessError, OSError) as e:
            raise RuntimeError(f"Error running 'lmod': {e}")

        output = output_bytes.decode("utf-8")
        error = error_bytes.decode("utf-8")

        if output == "_mlstatus = False\n":
            raise RuntimeError(f"Error running module cmd '{' '.join(args)}':\n{error}")

        return output


class WarmLmod(base.WarmEnvironment):
    """The resolved full names purged and loaded once, the resulting environment
    mapping kept and each job run with it (RFC 0005 section 3.1). No process is kept;
    the saving is the resolution."""

    loaded: dict[str, str] | None = None

    def _start(self) -> None:
        self.loaded = self.environment._loaded_environment(self.ref)

    def _execute(self, job: "Job[shell.Task]") -> dict[str, int | str]:
        return Lmod._run(job, dict(self.loaded))

    def _stop(self) -> None:
        self.loaded = None


# Alias so it can be referred to as lmod.Environment
Environment = Lmod
