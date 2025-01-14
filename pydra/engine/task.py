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
import typing as ty
import json
from pathlib import Path
import inspect
from fileformats.core import FileSet
from .core import Task
from pydra.utils.messenger import AuditFlag
from .specs import (
    PythonDef,
    ShellDef,
    is_set,
    attrs_fields,
)
from .helpers import (
    attrs_values,
    list_fields,
)
from pydra.engine.helpers_file import is_local_file, template_update_single
from pydra.utils.typing import TypeParser
from .environments import Native

if ty.TYPE_CHECKING:
    from pydra.design import shell


class PythonTask(Task):
    """Wrap a Python callable as a task element."""

    definition: PythonDef

    def _run_task(self, environment=None):
        inputs = attrs_values(self.definition)
        del inputs["function"]
        self.output_ = None
        output = self.definition.function(**inputs)
        output_names = [f.name for f in attrs.fields(self.definition.Outputs)]
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

    definition: ShellDef

    def __init__(
        self,
        definition: ShellDef,
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
        output_spec : :obj:`pydra.engine.specs.BaseDef`
            Specification of inputs.
        strip : :obj:`bool`
            TODO
        """
        self.return_code = None
        self.stdout = None
        self.stderr = None
        super().__init__(
            definition=definition,
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
        return self.definition._command_args(
            input_updates=self.inputs_mod_root, root=root
        )

    def _run_task(self, environment=None):
        if environment is None:
            environment = self.environment
        self.output_ = environment.execute(self)

    def _prepare_bindings(self, root: str):
        """Prepare input files to be passed to the task

        This updates the ``bindings`` attribute of the current task to make files available
        in an ``Environment``-defined ``root``.
        """
        for fld in attrs_fields(self.definition):
            if TypeParser.contains_type(FileSet, fld.type):
                fileset = getattr(self.definition, fld.name)
                copy = fld.copy_mode == FileSet.CopyMode.copy

                host_path, env_path = fileset.parent, Path(f"{root}{fileset.parent}")

                # Default to mounting paths as read-only, but respect existing modes
                old_mode = self.bindings.get(host_path, ("", "ro"))[1]
                self.bindings[host_path] = (env_path, "rw" if copy else old_mode)

                # Provide in-container paths without type-checking
                self.inputs_mod_root[fld.name] = tuple(
                    env_path / rel for rel in fileset.relative_fspaths
                )

    def resolve_output_value(
        self,
        fld: "shell.out",
        stdout: str,
        stderr: str,
    ) -> ty.Any:
        """Collect output file if metadata specified."""
        from pydra.design import shell

        if not self.definition.Outputs._required_fields_satisfied(fld, self.definition):
            return None
        elif isinstance(fld, shell.outarg) and fld.path_template:
            return template_update_single(
                fld,
                definition=self.definition,
                output_dir=self.output_dir,
                spec_type="output",
            )
        elif fld.callable:
            callable_ = fld.callable
            if isinstance(fld.callable, staticmethod):
                # In case callable is defined as a static method,
                # retrieve the function wrapped in the descriptor.
                callable_ = fld.callable.__func__
            call_args = inspect.getfullargspec(callable_)
            call_args_val = {}
            for argnm in call_args.args:
                if argnm == "field":
                    call_args_val[argnm] = fld
                elif argnm == "output_dir":
                    call_args_val[argnm] = self.output_dir
                elif argnm == "inputs":
                    call_args_val[argnm] = self.inputs
                elif argnm == "stdout":
                    call_args_val[argnm] = stdout
                elif argnm == "stderr":
                    call_args_val[argnm] = stderr
                else:
                    try:
                        call_args_val[argnm] = self.inputs[argnm]
                    except KeyError as e:
                        e.add_note(
                            f"arguments of the callable function from {fld.name} "
                            f"has to be in inputs or be field or output_dir, "
                            f"but {argnm} is used"
                        )
                        raise
            return callable_(**call_args_val)
        else:
            raise Exception(
                f"Metadata for '{fld.name}', does not not contain any of the required fields "
                f'("callable", "output_file_template" or "value"): {fld}.'
            )

    def generated_output_names(self, stdout: str, stderr: str):
        """Returns a list of all outputs that will be generated by the task.
        Takes into account the task input and the requires list for the output fields.
        TODO: should be in all Output specs?
        """
        # checking the input (if all mandatory fields are provided, etc.)
        self.definition._check_rules()
        output_names = ["return_code", "stdout", "stderr"]
        for fld in list_fields(self):
            # assuming that field should have either default or metadata, but not both
            if is_set(fld.default):
                output_names.append(fld.name)
            elif is_set(self.resolve_output_value(fld, stdout, stderr)):
                output_names.append(fld.name)
        return output_names

    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.adjacent


class BoshTask(ShellTask):

    def _command_args_single(self, state_ind=None, index=None):
        """Get command line arguments for a single state"""
        input_filepath = self._bosh_invocation_file(state_ind=state_ind, index=index)
        cmd_list = (
            self.definition.executable
            + [str(self.bosh_file), input_filepath]
            + self.definition.args
            + self.bindings
        )
        return cmd_list

    def _bosh_invocation_file(self, state_ind=None, index=None):
        """creating bosh invocation file - json file with inputs values"""
        input_json = {}
        for f in attrs_fields(self.definition, exclude_names=("executable", "args")):
            if self.state and f"{self.name}.{f.name}" in state_ind:
                value = getattr(self.definition, f.name)[
                    state_ind[f"{self.name}.{f.name}"]
                ]
            else:
                value = getattr(self.definition, f.name)
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
