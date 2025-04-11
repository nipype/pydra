from pydra.compose import python, workflow, shell
from fileformats.generic import File
from pydra.utils.general import task_class_as_dict, task_class_from_dict, task_fields
from pydra.utils.tests.utils import SpecificFuncTask, Concatenate


def test_python_task_class_as_dict():

    @python.define(outputs=["out_int"], xor=["b", "c"])
    def Add(a: int, b: int | None = None, c: int | None = None) -> int:
        """
        Parameters
        ----------
        a: int
            the first arg
        b : int, optional
            the optional second arg
        c : int, optional
            the optional third arg

        Returns
        -------
        out_int : int
            the sum of a and b
        """
        return a + (b if b is not None else c)

    dct = task_class_as_dict(Add)
    Reloaded = task_class_from_dict(dct)
    assert task_fields(Add) == task_fields(Reloaded)


def test_shell_task_class_as_dict():

    MyCmd = shell.define(
        "my-cmd <in_file> <out|out_file> --an-arg <an_arg:int=2> --a-flag<a_flag>"
    )

    dct = task_class_as_dict(MyCmd)
    Reloaded = task_class_from_dict(dct)
    assert task_fields(MyCmd) == task_fields(Reloaded)


def test_workflow_task_class_as_dict():

    @workflow.define
    def AWorkflow(in_file: File, a_param: int) -> tuple[File, File]:
        spec_func = workflow.add(SpecificFuncTask(in_file))
        concatenate = workflow.add(
            Concatenate(
                in_file1=in_file, in_file2=spec_func.out_file, duplicates=a_param
            )
        )
        return concatenate.out_file

    dct = task_class_as_dict(AWorkflow)
    Reloaded = task_class_from_dict(dct)
    assert task_fields(AWorkflow) == task_fields(Reloaded)
