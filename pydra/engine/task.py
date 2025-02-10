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
from .core import Task
from pydra.utils.messenger import AuditFlag
from .specs import (
    PythonDef,
    ShellDef,
    attrs_fields,
)
from .helpers import (
    attrs_values,
)
from pydra.engine.helpers_file import is_local_file
from .environments import Native


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


class BoshTask(ShellDef):

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
