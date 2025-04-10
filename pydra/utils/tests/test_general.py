from pydra.compose import python  # , workflow, shell
from pydra.utils.general import task_def_as_dict, task_def_from_dict, task_fields


def test_python_task_def_as_dict():

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

    dct = task_def_as_dict(Add)
    Reloaded = task_def_from_dict(dct)
    assert task_fields(Add) == task_fields(Reloaded)
