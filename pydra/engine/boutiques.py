import typing as ty
import json
import attr
import subprocess as sp
import os
from pathlib import Path

from ..utils.messenger import AuditFlag
from ..engine import ShellCommandTask
from ..engine.specs import (
    SpecInfo,
    ShellSpec,
    ShellOutSpec,
    File,
    Directory,
    attr_fields,
)
from .helpers import ensure_list, execute
from .helpers_file import template_update, is_local_file


class BoshTask(ShellCommandTask):
    """Shell Command Task based on the Boutiques descriptor"""

    def __init__(
        self,
        zenodo=None,
        bosh_file=None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec: ty.Optional[SpecInfo] = None,
        rerun=False,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        zenodo: :obj: str
            Zenodo ID
        bosh_file : : str
            json file with the boutiques descriptors
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
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
        if (bosh_file and zenodo) or not (bosh_file or zenodo):
            raise Exception("either bosh or zenodo has to be specified")
        elif zenodo:
            bosh_file = self._download_spec(zenodo)

        with bosh_file.open() as f:
            self.bosh_spec = json.load(f)

        if input_spec is None:
            input_spec = self._prepare_input_spec()
        self.input_spec = input_spec
        if output_spec is None:
            output_spec = self._prepare_output_spec()
        self.output_spec = output_spec

        super(BoshTask, self).__init__(
            name=name,
            input_spec=input_spec,
            output_spec=output_spec,
            executable=["bosh", "exec", "launch", str(bosh_file)],
            args=["-s"],
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            strip=strip,
            rerun=rerun,
            **kwargs,
        )
        self.strip = strip

    def _download_spec(self, zenodo):
        """ usind bosh pull to download the zenodo file"""
        spec_file = (
            Path(os.environ["HOME"])
            / ".cache/boutiques/production"
            / (zenodo.replace(".", "-") + ".json")
        )
        if not spec_file.exists():
            sp.run(["bosh", "pull", zenodo])
        return spec_file

    def _prepare_input_spec(self):
        """ creating input spec from the zenodo file"""
        binputs = self.bosh_spec["inputs"]
        fields = []
        for input in binputs:
            name = input["id"]
            if input["type"] == "File":
                tp = File
            elif input["type"] == "String":
                tp = str
            elif input["type"] == "Number":
                tp = float
            elif input["type"] == "Flag":
                tp = bool
            else:
                tp = None
            # adding list
            if tp and "list" in input and input["list"]:
                tp = ty.List[tp]

            mdata = {
                "help_string": input["description"],
                "mandatory": not input["optional"],
                "argstr": input.get("command-line-flag", None),
            }
            fields.append((name, tp, mdata))

        spec = SpecInfo(name="Inputs", fields=fields, bases=(ShellSpec,))
        return spec

    def _prepare_output_spec(self):
        """ creating output spec from the zenodo file"""
        boutputs = self.bosh_spec["output-files"]
        fields = []
        for output in boutputs:
            name = output["id"]
            mdata = {
                "help_string": output["description"],
                "mandatory": not output["optional"],
            }
            fields.append((name, File, mdata))

        spec = SpecInfo(name="Outputs", fields=fields, bases=(ShellOutSpec,))
        return spec

    def _command_args_single(self, state_ind, ind=None):
        """Get command line arguments for a single state"""
        input_filepath = self._input_file(state_ind=state_ind, ind=ind)
        cmd_list = self.inputs.executable + [input_filepath] + self.inputs.args
        return cmd_list

    def _input_file(self, state_ind, ind=None):
        input_json = {}
        for f in attr_fields(self.inputs):
            if f.name in ["executable", "args"]:
                continue
            if self.state and f"{self.name}.{f.name}" in state_ind:
                value = getattr(self.inputs, f.name)[state_ind[f"{self.name}.{f.name}"]]
            else:
                value = getattr(self.inputs, f.name)
            if is_local_file(f):
                value = str(value)
            # adding to the json file if specified by the user
            if value is not attr.NOTHING and value != "NOTHING":
                input_json[f.name] = value

        filename = self.cache_dir / f"{self.name}-{ind}.json"
        with open(filename, "w") as jsonfile:
            json.dump(input_json, jsonfile)

        return str(filename)

    def _run_task(self):
        self.output_ = None
        args = self.command_args
        if args:
            # removing empty strings
            args = [str(el) for el in args if el not in ["", " "]]
            keys = ["return_code", "stdout", "stderr"]
            values = execute(args, strip=self.strip)
            self.output_ = dict(zip(keys, values))
            if self.output_["return_code"]:
                if self.output_["stderr"]:
                    raise RuntimeError(self.output_["stderr"])
                else:
                    raise RuntimeError(self.output_["stdout"])
