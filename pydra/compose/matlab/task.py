import typing as ty
import attrs

# from pydra.utils.general import get_fields, asdict
from pydra.compose import base

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MatlabOutputs(base.Outputs):

    @classmethod
    def _from_job(cls, job: "Job[MatlabTask]") -> ty.Self:
        """Collect the outputs of a job from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        job : Job[Task]
            The job whose outputs are being collected.
        outputs_dict : dict[str, ty.Any]
            The outputs of the job, as a dictionary

        Returns
        -------
        outputs : Outputs
            The outputs of the job in dataclass
        """
        raise NotImplementedError


MatlabOutputsType = ty.TypeVar("MatlabOutputsType", bound=MatlabOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MatlabTask(base.Task[MatlabOutputsType]):

    _executor_name = "function"

    FUCNTION_HELP = (
        "a string containing the definition of the MATLAB function to run in the task"
    )

    def _run(self, job: "Job[MatlabTask]", rerun: bool = True) -> None:
        raise NotImplementedError


# Alias ShellTask to Task so we can refer to it by shell.Task
Task = MatlabTask
Outputs = MatlabOutputs
