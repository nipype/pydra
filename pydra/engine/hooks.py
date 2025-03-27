import typing as ty
import attrs
from attrs.converters import default_if_none


def donothing(*args: ty.Any, **kwargs: ty.Any) -> None:
    return None


@attrs.define(kw_only=True)
class TaskHooks:
    """Callable job hooks."""

    pre_run_task: ty.Callable = attrs.field(
        default=donothing, converter=default_if_none(donothing)
    )
    post_run_task: ty.Callable = attrs.field(
        default=donothing, converter=default_if_none(donothing)
    )
    pre_run: ty.Callable = attrs.field(
        default=donothing, converter=default_if_none(donothing)
    )
    post_run: ty.Callable = attrs.field(
        default=donothing, converter=default_if_none(donothing)
    )

    def reset(self):
        for val in ["pre_run_task", "post_run_task", "pre_run", "post_run"]:
            setattr(self, val, donothing)
