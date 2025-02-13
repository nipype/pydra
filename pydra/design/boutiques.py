import typing as ty
import json
import tempfile
from urllib.request import urlretrieve
from pathlib import Path
from functools import reduce
from fileformats.generic import File
from pydra.engine.specs import ShellDef
from .base import make_task_def
from . import shell


class arg(shell.arg):
    """Class for input fields of Boutiques task definitions

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    default : Any, optional
        the default value for the field, by default it is NO_DEFAULT
    help: str
        A short description of the input field.
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    xor: list[str], optional
        Names of the inputs that are mutually exclusive with the field.
    copy_mode: File.CopyMode, optional
        The mode of copying the file, by default it is File.CopyMode.any
    copy_collation: File.CopyCollation, optional
        The collation of the file, by default it is File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition, optional
        The extension decomposition of the file, by default it is
        File.ExtensionDecomposition.single
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    """


class out(shell.out):
    """Class for output fields of Boutiques task definitions

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    default : Any, optional
        the default value for the field, by default it is NO_DEFAULT
    help: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    """


def define(
    zenodo_id=None,
    bosh_file=None,
    input_spec_names: list[str] | None = None,
    output_spec_names: list[str] | None = None,
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
    if (bosh_file and zenodo_id) or not (bosh_file or zenodo_id):
        raise Exception("either bosh or zenodo_id has to be specified")
    elif zenodo_id:
        bosh_file = _download_spec(zenodo_id)

    with bosh_file.open() as f:
        bosh_spec = json.load(f)

    inputs, input_keys = _prepare_input_spec(bosh_spec, names_subset=input_spec_names)
    outputs = _prepare_output_spec(
        bosh_spec, input_keys, names_subset=output_spec_names
    )
    return make_task_def(
        spec_type=ShellDef,
        out_type=out,
        arg_type=arg,
        inputs=inputs,
        outputs=outputs,
    )


def _download_spec(zenodo_id):
    """
    using boutiques Searcher to find url of zenodo file for a specific id,
    and download the file to self.cache_dir
    """
    from boutiques.searcher import Searcher

    tmp_dir = Path(tempfile.mkdtemp())

    searcher = Searcher(zenodo_id, exact_match=True)
    hits = searcher.zenodo_search().json()["hits"]["hits"]
    if len(hits) == 0:
        raise Exception(f"can't find zenodo definition for {zenodo_id}")
    elif len(hits) > 1:
        raise Exception(f"too many hits for {zenodo_id}")
    else:
        zenodo_url = hits[0]["files"][0]["links"]["self"]
        zenodo_file = tmp_dir / f"zenodo.{zenodo_id}.json"
        urlretrieve(zenodo_url, zenodo_file)
        return zenodo_file


def _prepare_input_spec(bosh_spec: dict[str, ty.Any], names_subset=None):
    """creating input definition from the zenodo file
    if name_subset provided, only names from the subset will be used in the definition
    """
    binputs = bosh_spec["inputs"]
    input_keys = {}
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

        fields.append(
            arg(
                name=name,
                type=tp,
                help=input.get("description", None) or input["name"],
                mandatory=not input["optional"],
                argstr=input.get("command-line-flag", None),
            )
        )
        input_keys[input["value-key"]] = "{" + f"{name}" + "}"
    if names_subset:
        raise RuntimeError(f"{names_subset} are not in the zenodo input definition")
    return fields, input_keys


def _prepare_output_spec(bosh_spec: dict[str, ty.Any], input_keys, names_subset=None):
    """creating output definition from the zenodo file
    if name_subset provided, only names from the subset will be used in the definition
    """
    boutputs = bosh_spec["output-files"]
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
            input_keys.items(),
            output["path-template"],
        )
        fields.append(
            out(
                name=name,
                type=File,
                help=output.get("description", None) or output["name"],
                mandatory=not output["optional"],
                output_file_template=path_template,
            )
        )

    if names_subset:
        raise RuntimeError(f"{names_subset} are not in the zenodo output definition")
    return fields
