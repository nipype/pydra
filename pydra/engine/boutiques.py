import typing as ty
import json
import attr
from urllib.request import urlretrieve
from pathlib import Path
from functools import reduce
from copy import deepcopy

from ..utils.messenger import AuditFlag
from ..engine import ShellCommandTask
from ..engine.specs import SpecInfo, ShellSpec, ShellOutSpec, File, attr_fields
from .helpers_file import is_local_file, template_update_single
from .specs import attr_fields
from .helpers import make_klass


class BoshTask(ShellCommandTask):
    """Shell Command Task based on the Boutiques descriptor"""

    def __init__(
        self,
        zenodo_id=None,
        bosh_file=None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        bindings=[],
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
            prefix = "zenodo."
            if zenodo_id.startswith(prefix):
                zenodo_id = zenodo_id[len(prefix) :]
            self.bosh_file = self._download_spec(zenodo_id)
        else:  # bosh_file
            if isinstance(bosh_file, str):
                self.bosh_file = Path(bosh_file)
            elif isinstance(bosh_file, Path):
                self.bosh_file = bosh_file
            else:
                raise Exception(
                    "the given bosh_file is neither a string nor a path object"
                )

        with self.bosh_file.open() as f:
            self.bosh_spec = json.load(f)

        self._post_run_changes = {}
        self.input_spec = self._prepare_input_spec(names_subset=input_spec_names)
        self.output_spec = self._prepare_output_spec(names_subset=output_spec_names)
        self.bindings = ["-v", f"{self.bosh_file.parent}:{self.bosh_file.parent}:ro"]
        self.add_input_bindigs(bindings)

        super().__init__(
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
        # self.hooks.__setattr__("post_run_task",post_run_patches)

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
        """creating input spec from the zenodo file
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
                if input.get("integer", False):
                    tp = int
            elif input["type"] == "Flag":
                tp = bool
            else:
                tp = None
            ti = tp  # copy in case tp gets changed to a list
            # adding list
            if tp and "list" in input and input["list"]:
                tp = ty.List[tp]

            mdata = {
                "help_string": input.get("description", None) or input["name"],
                "mandatory": not input["optional"],
                "argstr": input.get("command-line-flag", None),
            }

            if "default-value" in input:
                default_val = input["default-value"]
                # Setting the type of the default value
                if ti is not File:
                    if isinstance(default_val, ty.List):
                        default_val = [ty.cast(ti, v) for v in default_val]
                    else:
                        default_val = ty.cast(tp, default_val)
                # Cannot have mandatory spec with a default value.
                mdata["mandatory"] = False
                fields.append((name, tp, default_val, mdata))
            else:
                fields.append((name, tp, mdata))

            if "list-separator" in input:
                mdata["sep"] = input["list-separator"]

            if "command-line-flag-separator" in input and tp != bool:
                # overwriting previously set value
                if mdata["argstr"] is not None:
                    mdata["argstr"] = (
                        mdata["argstr"]
                        + input["command-line-flag-separator"]
                        + "{name}"
                    )

            self._input_spec_keys[input["value-key"]] = "{" + f"{name}" + "}"
        if names_subset:
            raise RuntimeError(f"{names_subset} are not in the zenodo input spec")
        spec = SpecInfo(name="Inputs", fields=fields, bases=(ShellSpec,))
        return spec

    def _prepare_output_spec(self, names_subset=None):
        """creating output spec from the zenodo file
        if name_subset provided, only names from the subset will be used in the spec
        """

        boutputs = self.bosh_spec.get("output-files", None)
        if not boutputs:
            return SpecInfo(name="Outputs", fields=[], bases=(ShellOutSpec,))
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
            if "uses-absolute-path" in output:
                mdata["absolute_path"] = output["uses-absolute-path"]

            if "path-template-stripped-extensions" in output:
                exts = output["path-template-stripped-extensions"]
                self._post_run_changes[name] = {
                    "path-template-stripped-extensions": exts
                }

                def find_special_out(field, output_dir, inputs):
                    mdata["output_file_template"] = path_template
                    fields = [(name, attr.ib(type=File, metadata=mdata))]
                    spec = SpecInfo(
                        name="Outputs", fields=fields, bases=(ShellOutSpec,)
                    )
                    spec = make_klass(spec)
                    bspec = attr_fields(
                        spec, exclude_names=("return_code", "stdout", "stderr")
                    )
                    fld = bspec[0]
                    inputs = deepcopy(inputs)
                    dict_ = attr.asdict(inputs)
                    for k, v in dict_.items():
                        if isinstance(v, str):
                            for ext in exts:
                                v = v.replace(ext, "")
                            dict_[k] = v
                    inputs = attr.evolve(inputs, **dict_)
                    new_value = template_update_single(
                        fld, inputs, output_dir=output_dir, spec_type="output"
                    )
                    ret = Path(new_value)
                    if ret.exists():
                        return ret
                    else:
                        return attr.NOTHING

                mdata["callable"] = find_special_out
                del mdata["output_file_template"]  # = None

            fields.append((name, attr.ib(type=File, metadata=mdata)))
        if names_subset:
            raise RuntimeError(f"{names_subset} are not in the zenodo output spec")
        spec = SpecInfo(name="Outputs", fields=fields, bases=(ShellOutSpec,))
        return spec

    def _command_args_single(self, state_ind=None, index=None):
        """Get command line arguments for a single state"""
        input_filepath = self._bosh_invocation_file(state_ind=state_ind, index=index)
        cmd_list = (
            self.inputs.executable
            + [str(self.bosh_file), input_filepath]
            + self.inputs.args
            + self.bindings
        )
        return cmd_list

    def _bosh_invocation_file(self, state_ind=None, index=None):
        """creating bosh invocation file - json file with inputs values"""
        input_json = {}
        for f in attr_fields(self.inputs, exclude_names=("executable", "args")):
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

        filename = self.cache_dir / f"{self.name}-{index}.json"
        with open(filename, "w") as jsonfile:
            json.dump(input_json, jsonfile)

        return str(filename)

    def add_input_bindigs(self, binings):
        for binding in binings:
            if len(binding) == 3:
                lpath, cpath, mode = binding
            elif len(binding) == 2:
                lpath, cpath, mode = binding + ["rw"]
            else:
                raise Exception(
                    f"binding should have length 2, 3, or 4, it has {len(binding)}"
                )
            self.bindings.extend(["-v", f"{lpath}:{cpath}:{mode}"])
