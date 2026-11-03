import sys

import cloudpickle
import pytest
from pydra.compose import python, shell, workflow
from pydra.engine.hooks import TaskHooks
from pydra.utils.general import get_fields
from pydra.utils.resources import ComputeResources

SMALL = ComputeResources(cpus=1, mem="100M")
BIG = ComputeResources(cpus=8, mem="16G", time=600)

cloudpickle.register_pickle_by_value(sys.modules[__name__])


def add_one(x: int) -> int:
    return x + 1


def test_python_define_sets_resources():
    Task = python.define(add_one, resources=SMALL)
    assert Task._resources == SMALL
    assert Task(x=1)._resources == SMALL


def test_decorator_form_sets_resources():
    @python.define(resources=BIG)
    def Decorated(x: int) -> int:
        return x + 1

    assert Decorated(x=1)._resources == BIG


def test_shell_define_sets_resources():
    Echo = shell.define("echo <text:str>", resources=SMALL)
    assert Echo(text="hi")._resources == SMALL


def test_workflow_define_does_not_accept_resources():
    # a workflow's resources would never reach a scheduler: its nodes are submitted
    # individually, each with its own resources
    with pytest.raises(TypeError):
        workflow.define(outputs=["out"], resources=BIG)


def test_default_is_no_resources():
    assert python.define(add_one)(x=1)._resources is None


def test_resources_are_not_a_field_and_never_in_the_checksum():
    plain = python.define(add_one)(x=1)
    small = python.define(add_one, resources=SMALL)(x=1)
    big = python.define(add_one, resources=BIG)(x=1)
    assert "_resources" not in {f.name for f in get_fields(small)}
    assert plain._compute_hashes()[1].keys() == big._compute_hashes()[1].keys()
    assert plain._checksum == small._checksum == big._checksum


def _record_to(directory, name):
    """A hook that writes the job's resources to a file, so it is visible from the
    process the job runs in"""
    return TaskHooks(
        pre_run_task=lambda job, *a: (directory / name).write_text(repr(job.resources))
    )


def test_job_takes_the_definition_resources(worker, tmp_path):
    Task = python.define(add_one, resources=SMALL)
    hooks = _record_to(tmp_path, "seen")
    assert Task(x=1)(cache_root=tmp_path / "cache", worker=worker, hooks=hooks).out == 2
    assert (tmp_path / "seen").read_text() == repr(SMALL)


def test_node_resources_override_the_definition(worker, tmp_path):
    # each node feeds the next, so no node is a cache hit on another's result
    Inner = python.define(add_one, resources=SMALL)

    @workflow.define(outputs=["a", "b", "c"])
    def Wf(x: int):
        a = workflow.add(Inner(x=x), name="a", hooks=_record_to(tmp_path, "a"))
        b = workflow.add(
            Inner(x=a.out), name="b", hooks=_record_to(tmp_path, "b"), resources=BIG
        )
        c = workflow.add(
            python.define(add_one)(x=b.out),
            name="c",
            hooks=_record_to(tmp_path, "c"),
            resources=BIG,
        )
        return a.out, b.out, c.out

    outputs = Wf(x=1)(cache_root=tmp_path / "cache", worker=worker)
    assert (outputs.a, outputs.b, outputs.c) == (2, 3, 4)
    assert {n: (tmp_path / n).read_text() for n in "abc"} == {
        "a": repr(SMALL),
        "b": repr(BIG),
        "c": repr(BIG),
    }
