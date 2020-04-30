import typing as ty
import json
import attr
from urllib.request import urlretrieve
from pathlib import Path
from functools import reduce

from ..utils.messenger import AuditFlag
from ..engine import ShellCommandTask
from ..engine.specs import SpecInfo, ShellSpec, ShellOutSpec, File, attr_fields
from .helpers_file import is_local_file


class BoshTask(ShellCommandTask):
    """Shell Command Task based on the Boutiques descriptor"""

    def __init__(
        self,
        zenodo_id=None,
        bosh_file=None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec_names: ty.Optional[ty.List] = None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec_names: ty.Optional[ty.List] = None,
        rerun=False,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        zenodo_id: :obj: str
            Zenodo ID
        bosh_file : : str
            json file with the boutiques descriptors
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec_names : :obj: list
            Input names for input_spec.
        messenger_args :
            TODO
        messengers :
            TODO
        name : :obj:`str`
            Name of this task.
        output_spec_names : :obj: list
            Output names for output_spec.
        strip : :obj:`bool`
            TODO

        """
        self.cache_dir = cache_dir
        if (bosh_file and zenodo_id) or not (bosh_file or zenodo_id):
            raise Exception("either bosh or zenodo_id has to be specified")
        elif zenodo_id:
            self.bosh_file = self._download_spec(zenodo_id)
        else:  # bosh_file
            self.bosh_file = bosh_file

        with self.bosh_file.open() as f:
            self.bosh_spec = json.load(f)

        self.input_spec = self._prepare_input_spec(names_subset=input_spec_names)
        self.output_spec = self._prepare_output_spec(names_subset=output_spec_names)
        self.bindings = ["-v", f"{self.bosh_file.parent}:{self.bosh_file.parent}:ro"]

        super(BoshTask, self).__init__(
            name=name,
            input_spec=self.input_spec,
            output_spec=self.output_spec,
            executable=["bosh", "exec", "launch"],
            args=["-s"],
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=self.cache_dir,
            strip=strip,
            rerun=rerun,
            **kwargs,
        )
        self.strip = strip

    def _download_spec(self, zenodo_id):
        """
        usind boutiques Searcher to find url of zenodo file for a specific id,
        and download the file to self.cache_dir
        """
        from boutiques.searcher import Searcher

        searcher = Searcher(zenodo_id, exact_match=True)
        hits = searcher.zenodo_search().json()["hits"]["hits"]
        if len(hits) == 0:
            raise Exception(f"can't find zenodo spec for {zenodo_id}")
        elif len(hits) > 1:
            raise Exception(f"too many hits for {zenodo_id}")
        else:
            zenodo_url = hits[0]["files"][0]["links"]["self"]
            zenodo_file = self.cache_dir / f"zenodo.{zenodo_id}.json"
            urlretrieve(zenodo_url, zenodo_file)
            return zenodo_file

    def _prepare_input_spec(self, names_subset=None):
        """ creating input spec from the zenodo file
            if name_subset provided, only names from the subset will be used in the spec
        """
        binputs = self.bosh_spec["inputs"]
        self._input_spec_keys = {}
        fields = []
        for input in binputs:
            name = input["id"]
            if names_subset is None:
                pass
            elif name not in names_subset:
                continue
            else:
                names_subset.remove(name)
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
                "help_string": input.get("description", None) or input["name"],
                "mandatory": not input["optional"],
                "argstr": input.get("command-line-flag", None),
            }
            fields.append((name, tp, mdata))
            self._input_spec_keys[input["value-key"]] = "{" + f"{name}" + "}"
        if names_subset:
            raise RuntimeError(f"{names_subset} are not in the zenodo input spec")
        spec = SpecInfo(name="Inputs", fields=fields, bases=(ShellSpec,))
        return spec

    def _prepare_output_spec(self, names_subset=None):
        """ creating output spec from the zenodo file
            if name_subset provided, only names from the subset will be used in the spec
        """
        boutputs = self.bosh_spec["output-files"]
        fields = []
        for output in boutputs:
            name = output["id"]
            if names_subset is None:
                pass
            elif name not in names_subset:
                continue
            else:
                names_subset.remove(name)
            path_template = reduce(
                lambda s, r: s.replace(*r),
                self._input_spec_keys.items(),
                output["path-template"],
            )
            mdata = {
                "help_string": output.get("description", None) or output["name"],
                "mandatory": not output["optional"],
                "output_file_template": path_template,
            }
            fields.append((name, attr.ib(type=File, metadata=mdata)))

        if names_subset:
            raise RuntimeError(f"{names_subset} are not in the zenodo output spec")
        spec = SpecInfo(name="Outputs", fields=fields, bases=(ShellOutSpec,))
        return spec

    def _command_args_single(self, state_ind, ind=None):
        """Get command line arguments for a single state"""
        input_filepath = self._bosh_invocation_file(state_ind=state_ind, ind=ind)
        cmd_list = (
            self.inputs.executable
            + [str(self.bosh_file), input_filepath]
            + self.inputs.args
            + self.bindings
        )
        return cmd_list

    def _bosh_invocation_file(self, state_ind, ind=None):
        """creating bosh invocation file - json file with inputs values"""
        input_json = {}
        for f in attr_fields(self.inputs):
            if f.name in ["executable", "args"]:
                continue
            if self.state and f"{self.name}.{f.name}" in state_ind:
                value = getattr(self.inputs, f.name)[state_ind[f"{self.name}.{f.name}"]]
            else:
                value = getattr(self.inputs, f.name)
            # adding to the json file if specified by the user
            if value is not attr.NOTHING and value != "NOTHING":
                if is_local_file(f):
                    value = Path(value)
                    self.bindings.extend(["-v", f"{value.parent}:{value.parent}:ro"])
                    value = str(value)

                input_json[f.name] = value

        filename = self.cache_dir / f"{self.name}-{ind}.json"
        with open(filename, "w") as jsonfile:
            json.dump(input_json, jsonfile)

        return str(filename)
