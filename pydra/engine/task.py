"""
Implement processing nodes.

.. admonition :: Notes:

    * Environment specs

        1. neurodocker json
        2. singularity file+hash
        3. docker hash
        4. conda env
        5. niceman config
        6. environment variables

    * Monitors/Audit

        1. internal monitor
        2. external monitor
        3. callbacks

    * Resuming

        1. internal tracking
        2. external tracking (DMTCP)

    * Provenance

        1. Local fragments
        2. Remote server

    * Isolation

        1. Working directory
        2. File (copy to local on write)
        3. read only file system

    * `Original implementation
      <https://colab.research.google.com/drive/1RRV1gHbGJs49qQB1q1d5tQEycVRtuhw6>`__

"""

from __future__ import annotations

import attrs
import json
from pathlib import Path
from fileformats.core import FileSet
from .core import Task
from pydra.utils.messenger import AuditFlag
from .specs import (
    PythonSpec,
    ShellSpec,
    attrs_fields,
)
from .helpers import (
    attrs_values,
    parse_copyfile,
)
from pydra.engine.helpers_file import is_local_file
from pydra.utils.typing import TypeParser
from .environments import Native


class PythonTask(Task):
    """Wrap a Python callable as a task element."""

    spec: PythonSpec

    def _run_task(self, environment=None):
        inputs = attrs_values(self.spec)
        del inputs["function"]
        self.output_ = None
        output = self.spec.function(**inputs)
        output_names = [f.name for f in attrs.fields(self.spec.Outputs)]
        if output is None:
            self.output_ = {nm: None for nm in output_names}
        elif len(output_names) == 1:
            # if only one element in the fields, everything should be returned together
            self.output_ = {output_names[0]: output}
        elif isinstance(output, tuple) and len(output_names) == len(output):
            self.output_ = dict(zip(output_names, output))
        elif isinstance(output, dict):
            self.output_ = {key: output.get(key, None) for key in output_names}
        else:
            raise RuntimeError(
                f"expected {len(self.output_spec.fields)} elements, "
                f"but {output} were returned"
            )


class ShellTask(Task):
    """Wrap a shell command as a task element."""

    spec: ShellSpec

    def __init__(
        self,
        spec: ShellSpec,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cont_dim=None,
        messenger_args=None,
        messengers=None,
        name=None,
        rerun=False,
        strip=False,
        environment=Native(),
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        cont_dim : :obj:`dict`, or `None`
            Container dimensions for input fields,
            if any of the container should be treated as a container
        messenger_args :
            TODO
        messengers :
            TODO
        name : :obj:`str`
            Name of this task.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO
        """
        self.return_code = None
        self.stdout = None
        self.stderr = None
        super().__init__(
            spec=spec,
            name=name,
            inputs=kwargs,
            cont_dim=cont_dim,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            rerun=rerun,
        )
        self.strip = strip
        self.environment = environment
        self.bindings = {}
        self.inputs_mod_root = {}

    def get_bindings(self, root: str | None = None) -> dict[str, tuple[str, str]]:
        """Return bindings necessary to run task in an alternative root.

        This is primarily intended for contexts when a task is going
        to be run in a container with mounted volumes.

        Arguments
        ---------
        root: str

        Returns
        -------
        bindings: dict
          Mapping from paths in the host environment to the target environment
        """

        if root is None:
            return {}
        else:
            self._prepare_bindings(root=root)
            return self.bindings

    def command_args(self, root: Path | None = None) -> list[str]:
        return self.spec._command_args(input_updates=self.inputs_mod_root, root=root)

    def _run_task(self, environment=None):
        if environment is None:
            environment = self.environment
        self.output_ = environment.execute(self)

    def _prepare_bindings(self, root: str):
        """Prepare input files to be passed to the task

        This updates the ``bindings`` attribute of the current task to make files available
        in an ``Environment``-defined ``root``.
        """
        for fld in attrs_fields(self.spec):
            if TypeParser.contains_type(FileSet, fld.type):
                fileset = getattr(self.spec, fld.name)
                copy = parse_copyfile(fld)[0] == FileSet.CopyMode.copy

                host_path, env_path = fileset.parent, Path(f"{root}{fileset.parent}")

                # Default to mounting paths as read-only, but respect existing modes
                old_mode = self.bindings.get(host_path, ("", "ro"))[1]
                self.bindings[host_path] = (env_path, "rw" if copy else old_mode)

                # Provide in-container paths without type-checking
                self.inputs_mod_root[fld.name] = tuple(
                    env_path / rel for rel in fileset.relative_fspaths
                )

    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.adjacent


class BoshTask(ShellTask):

    def _command_args_single(self, state_ind=None, index=None):
        """Get command line arguments for a single state"""
        input_filepath = self._bosh_invocation_file(state_ind=state_ind, index=index)
        cmd_list = (
            self.spec.executable
            + [str(self.bosh_file), input_filepath]
            + self.spec.args
            + self.bindings
        )
        return cmd_list

    def _bosh_invocation_file(self, state_ind=None, index=None):
        """creating bosh invocation file - json file with inputs values"""
        input_json = {}
        for f in attrs_fields(self.spec, exclude_names=("executable", "args")):
            if self.state and f"{self.name}.{f.name}" in state_ind:
                value = getattr(self.spec, f.name)[state_ind[f"{self.name}.{f.name}"]]
            else:
                value = getattr(self.spec, f.name)
            # adding to the json file if specified by the user
            if value is not attrs.NOTHING and value != "NOTHING":
                if is_local_file(f):
                    value = Path(value)
                    self.bindings.extend(["-v", f"{value.parent}:{value.parent}:ro"])
                    value = str(value)

                input_json[f.name] = value

        filename = self.cache_dir / f"{self.name}-{index}.json"
        with open(filename, "w") as jsonfile:
            json.dump(input_json, jsonfile)

        return str(filename)
