import attr
import typing as ty
import pydra
from pydra.engine.specs import BaseSpec, SpecInfo, MultiInputObj, MultiOutputObj
from pydra.engine.core import TaskBase
from pydra.engine.helpers import ensure_list

try:
    from typing import Literal
except ImportError:  # PY37
    from typing_extensions import Literal


@attr.s(kw_only=True)
class MergeInputSpec(BaseSpec):
    axis: Literal["vstack", "hstack"] = attr.ib(
        default="vstack",
        metadata={
            "help_string": "Direction in which to merge, hstack requires same number of elements in each input."
        },
    )
    no_flatten: bool = attr.ib(
        default=False,
        metadata={
            "help_string": "Append to outlist instead of extending in vstack mode."
        },
    )
    ravel_inputs: bool = attr.ib(
        default=False,
        metadata={"help_string": "Ravel inputs when no_flatten is False."},
    )


def _ravel(in_val):
    if not isinstance(in_val, list):
        return in_val
    flat_list = []
    for val in in_val:
        raveled_val = _ravel(val)
        if isinstance(raveled_val, list):
            flat_list.extend(raveled_val)
        else:
            flat_list.append(raveled_val)
    return flat_list


class Merge(TaskBase):
    """
    Task to merge inputs into a single list

    ``Merge(1)`` will merge a list of lists

    Examples
    --------
    >>> from pydra.tasks.core.sequences import Merge
    >>> mi = Merge(3, name="mi")
    >>> mi.inputs.in1 = 1
    >>> mi.inputs.in2 = [2, 5]
    >>> mi.inputs.in3 = 3
    >>> out = mi()
    >>> out.output.out
    [1, 2, 5, 3]

    >>> merge = Merge(1, name="merge")
    >>> merge.inputs.in1 = [1, [2, 5], 3]
    >>> out = merge()
    >>> out.output.out
    [1, [2, 5], 3]

    >>> merge = Merge(1, name="merge")
    >>> merge.inputs.in1 = [1, [2, 5], 3]
    >>> merge.inputs.ravel_inputs = True
    >>> out = merge()
    >>> out.output.out
    [1, 2, 5, 3]

    >>> merge = Merge(1, name="merge")
    >>> merge.inputs.in1 = [1, [2, 5], 3]
    >>> merge.inputs.no_flatten = True
    >>> out = merge()
    >>> out.output.out
    [[1, [2, 5], 3]]
    """

    _task_version = "1"
    output_spec = SpecInfo(name="Outputs", fields=[("out", ty.List)], bases=(BaseSpec,))

    def __init__(self, numinputs, *args, **kwargs):
        self._numinputs = max(numinputs, 0)
        self.input_spec = SpecInfo(
            name="Inputs",
            fields=[(f"in{i + 1}", ty.List) for i in range(self._numinputs)],
            bases=(MergeInputSpec,),
        )
        super().__init__(*args, **kwargs)

    def _run_task(self):
        self.output_ = {"out": []}
        if self._numinputs < 1:
            return

        values = [
            getattr(self.inputs, f"in{i + 1}")
            for i in range(self._numinputs)
            if getattr(self.inputs, f"in{i + 1}") is not attr.NOTHING
        ]

        if self.inputs.axis == "vstack":
            for value in values:
                if isinstance(value, list) and not self.inputs.no_flatten:
                    self.output_["out"].extend(
                        _ravel(value) if self.inputs.ravel_inputs else value
                    )
                else:
                    self.output_["out"].append(value)
        else:
            lists = [ensure_list(val) for val in values]
            self.output_["out"] = [
                [val[i] for val in lists] for i in range(len(lists[0]))
            ]


@pydra.mark.task
def Select(inlist: MultiInputObj, index: MultiInputObj) -> MultiOutputObj:
    """
    Task to select specific elements from a list

    Examples
    --------

    >>> from pydra.tasks.core.sequences import Select
    >>> sl = Select(name="sl")
    >>> sl.inputs.inlist = [1, 2, 3, 4, 5]
    >>> sl.inputs.index = [3]
    >>> out = sl()
    >>> out.output.out
    4

    >>> sl = Select(name="sl")
    >>> out = sl(inlist=[1, 2, 3, 4, 5], index=[3, 4])
    >>> out.output.out
    [4, 5]

    """
    return [inlist[i] for i in index]
