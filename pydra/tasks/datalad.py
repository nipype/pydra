"""A :obj:`~nipype.interfaces.utility.base.IdentityInterface` with a grafted Datalad getter."""
import os
import logging
import typing as ty
from pathlib import Path
from ..engine.specs import (
    File,
    Directory,
    SpecInfo,
    BaseSpec,
)
from ..engine.core import TaskBase
from ..engine.helpers import output_from_inputfields
from ..utils.messenger import AuditFlag

logger = logging.getLogger("pydra.tasks.datalad")

input_fields = [
    (
        "in_file",
        str,
        {
            "help_string": "Path to the data to be downloaded through datalad",
            "mandatory": True,
        },
    ),
    (
        "dataset_path",
        Directory,
        {
            "help_string": "Path to the dataset that will be used to get data",
            "mandatory": True,
        },
    ),
    (
        "dataset_url",
        str,
        {
            "help_string": "URL to the dataset that will be used to get data",
        },
    ),
]


output_fields = [
    (
        "out_file",
        File,
        {
            "help_string": "file downloaded through datalad",
            "requires": ["in_file"],
            "output_file_template": "{in_file}",
        },
    )
]

# define a TaskBase calss
class DataladInterface(TaskBase):
    """A :obj:`~nipype.interfaces.utility.base.IdentityInterface` with a grafted Datalad getter."""

    def __init__(
        self,
        name: str,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        input_spec: ty.Optional[ty.Union[SpecInfo, BaseSpec]] = None,
        output_spec: ty.Optional[ty.Union[SpecInfo, BaseSpec]] = None,
        cont_dim=None,
        messenger_args=None,
        messengers=None,
        rerun=False,
        **kwargs,
    ):
        """Initialize a DataladInterface instance."""

        self.input_spec = input_spec or SpecInfo(
            name="Inputs", fields=input_fields, bases=(BaseSpec,)
        )
        self.output_spec = output_spec or SpecInfo(
            name="Output", fields=output_fields, bases=(BaseSpec,)
        )
        self.output_spec = output_from_inputfields(self.output_spec, self.input_spec)
        super().__init__(
            name=name,
            inputs=kwargs,
            audit_flags=audit_flags,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            cont_dim=cont_dim,
            messenger_args=messenger_args,
            messengers=messengers,
            rerun=rerun,
        )

    def _run_task(self):
        in_file = self.inputs.in_file
        dataset_path = self.inputs.dataset_path

        _dl_found = False
        try:
            import datalad.api as dl

            _dl_found = True
        except:
            raise ImportError("Datalad is not installed.")

        # checking if the dataset is already downloaded

        if not (Path(dataset_path) / ".datalad").exists():
            logger.info("Datalad interface without dataset path defined.")
            try:
                dataset_url = self.inputs.dataset_url
                os.makedirs(dataset_path, exist_ok=True)
                dl.install(source=dataset_url, path=dataset_path)
            except Exception as e:
                logger.error(e)
        else:
            ds = dl.Dataset(self.inputs.dataset_path)

        # getting the file
        ds.get(self.inputs.in_file)

        # checking if the file was downloaded
        if not Path(dataset_path, in_file).exists():
            raise FileNotFoundError(f"File {in_file} not found in {dataset_path}")

        _pth = Path(in_file)
        if not _pth.is_absolute():
            _pth = dataset_path / _pth

        _datalad_candidate = _pth.is_symlink() and not _pth.exists()
        if not _datalad_candidate:
            logger.warning("datalad was required but not found")

        if _datalad_candidate:
            try:
                result = dl.get(_pth, dataset=dataset_path)
            except Exception as exc:
                logger.warning(f"datalad get on {_pth} failed.")
                ## discussed with @djarecka, we keep it commented here for now
                ## do we still need it for pydra?
                # if (
                #     config.environment.exec_env == "docker"
                #     and ("This repository is not initialized for use by git-annex, "
                #          "but .git/annex/objects/ exists") in f"{exc}"
                # ):
                #     logger.warning(
                #         "Execution seems containerirzed with Docker, please make sure "
                #         "you are not running as root. To do so, please add the argument "
                #         "``-u $(id -u):$(id -g)`` to your command line."
                #     )
                # else:
                #     logger.warning(str(exc))
            else:
                if result[0]["status"] == "error":
                    logger.warning(f"datalad get failed: {result}")

        self.output_ = None
        output = os.path.abspath(
            os.path.join(self.inputs.dataset_path, self.inputs.in_file)
        )
        output_names = [el[0] for el in self.output_spec.fields]
        if output is None:
            self.output_ = {nm: None for nm in output_names}
        elif len(output_names) == 1:
            # if only one element in the fields, everything should be returned together
            self.output_ = {output_names[0]: output}
        elif isinstance(output, tuple) and len(output_names) == len(output):
            self.output_ = dict(zip(output_names, output))
        else:
            raise RuntimeError(
                f"expected {len(self.output_spec.fields)} elements, "
                f"but {output} were returned"
            )
        # outputs = self.output_spec().get()
        # outputs["out_file"] = os.path.abspath(os.path.join(self.inputs.dataset_path, self.inputs.in_file))

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(
            os.path.join(self.inputs.dataset_path, self.inputs.in_file)
        )
        return outputs
