"""``class@1`` and ``type@1`` (RFC 0004 sections 4.2, 5.1; I4).

The checksum cases of RFC 0004 8.3 and 8.8 run a user package in a fresh process per
variant, so no memo of one edit is seen by the next. The type cases run in this process
and have no dependence on the Python minor, which the ``full-ci`` label checks on every
minor pydra supports.
"""

import copy
import dataclasses
import enum
import json
import re
import subprocess
import sys
import textwrap
import typing as ty
from pathlib import Path

import attrs
import cloudpickle as cp
import pytest
from fileformats.generic import File

from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.portable.identity import PydraIdentityV2
from pydra.engine.submitter import Submitter
from pydra.utils import hash as hash_module
from pydra.utils.digest import digest_form, register_codec
from pydra.utils.hash import class_digest, type_digest
from pydra.utils.identity import task_record

# ---------------------------------------------------------------------------
# a user package: one task per way the walk reaches code
# ---------------------------------------------------------------------------

PACKAGE = {
    "proj/__init__.py": "",
    "proj/bases.py": """
        class Base:
            def scale(self, x):
                return x * 2
    """,
    "proj/models.py": """
        from proj import helpers
        from proj.bases import Base


        class Proc:
            def __init__(self, k):
                self.k = k

            def __call__(self, x):
                return x * self.k


        class Helped:
            def __init__(self, k):
                self.k = k

            def __call__(self, x):
                return helpers.method_helper(x) + self.k


        class Derived(Base):
            def __init__(self, k):
                self.k = k


        class Unrelated:
            def run(self, x):
                return x - 1
    """,
    "proj/helpers.py": """
        import functools

        SCALE = 3


        def direct_helper(x):
            return x + 1


        def trans_b(x):
            return x + 2


        def trans_a(x):
            return trans_b(x)


        @functools.lru_cache(maxsize=None)
        def cached_helper(x):
            return x + 4


        def user_deco(func):
            def wrapper(*args):
                return func(*args)

            return wrapper


        @user_deco
        def decorated_helper(x):
            return x + 5


        double = lambda x: x * 2  # noqa: E731


        def make_adder(n):
            def adder(x):
                return x + n

            return adder


        adder = make_adder(3)


        def method_helper(x):
            return x + 6


        def dyn_helper(x):
            return x + 7
    """,
    "proj/dyn.py": """
        def run(x):
            return x + 8
    """,
    "proj/tasks.py": """
        import importlib

        from pydra.compose import python
        from proj import helpers
        from proj.helpers import (
            adder,
            cached_helper,
            decorated_helper,
            direct_helper,
            double,
            trans_a,
        )


        @python.define
        def Direct(x: int) -> int:
            return direct_helper(x)


        @python.define
        def Transitive(x: int) -> int:
            return trans_a(x)


        @python.define
        def Constant(x: int) -> int:
            return x * helpers.SCALE


        @python.define
        def Cached(x: int) -> int:
            return cached_helper(x)


        @python.define
        def Decorated(x: int) -> int:
            return decorated_helper(x)


        @python.define
        def Lambda(x: int) -> int:
            return double(x)


        @python.define
        def Factory(x: int) -> int:
            return adder(x)


        @python.define
        def BodyClass(x: int) -> int:
            class Local:
                def run(self, y):
                    return y + 9

            return Local().run(x)


        @python.define
        def ByGetattr(x: int) -> int:
            return getattr(helpers, "dyn_helper")(x)


        @python.define
        def ByImportlib(x: int) -> int:
            return importlib.import_module("proj.dyn").run(x)


        @python.define
        def OnInstance(p: object, x: int) -> int:
            return p(x)


        @python.define
        def OnHelped(p: object, x: int) -> int:
            return p(x)


        @python.define
        def OnDerived(p: object, x: int) -> int:
            return p.scale(x)


        @python.define
        def OnUnrelated(x: int) -> int:
            return x
    """,
}

SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from proj import tasks
from proj.models import Derived, Helped, Proc

checksums = {
    "Direct": tasks.Direct(x=1),
    "Transitive": tasks.Transitive(x=1),
    "Constant": tasks.Constant(x=1),
    "Cached": tasks.Cached(x=1),
    "Decorated": tasks.Decorated(x=1),
    "Lambda": tasks.Lambda(x=1),
    "Factory": tasks.Factory(x=1),
    "BodyClass": tasks.BodyClass(x=1),
    "ByGetattr": tasks.ByGetattr(x=1),
    "ByImportlib": tasks.ByImportlib(x=1),
    "OnInstance": tasks.OnInstance(p=Proc(2), x=1),
    "OnHelped": tasks.OnHelped(p=Helped(2), x=1),
    "OnDerived": tasks.OnDerived(p=Derived(2), x=1),
    "OnUnrelated": tasks.OnUnrelated(x=1),
}
print(json.dumps({k: v._checksum for k, v in checksums.items()}))
"""


def write_package(root: Path, edit: tuple[str, str, str] | None = None) -> Path:
    for name, source in PACKAGE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = textwrap.dedent(source).lstrip("\n")
        if edit is not None and edit[0] == name:
            assert edit[1] in text, edit
            text = text.replace(edit[1], edit[2])
        path.write_text(text)
    return root


def checksums(root: Path) -> dict[str, str]:
    """The checksums of the package's tasks, computed in a fresh process"""
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(root)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return checksums(write_package(tmp_path_factory.mktemp("base")))


# (task whose checksum must change, file, text to replace, replacement)
CHANGES = {
    "direct helper": ("Direct", "proj/helpers.py", "return x + 1", "return x + 10"),
    "transitive helper": (
        "Transitive",
        "proj/helpers.py",
        "return x + 2",
        "return x + 20",
    ),
    "module constant": ("Constant", "proj/helpers.py", "SCALE = 3", "SCALE = 30"),
    "lru_cache helper": ("Cached", "proj/helpers.py", "return x + 4", "return x + 40"),
    "decorated helper": (
        "Decorated",
        "proj/helpers.py",
        "return x + 5",
        "return x + 50",
    ),
    "decorator": (
        "Decorated",
        "proj/helpers.py",
        "return func(*args)",
        "return func(*args) + 0",
    ),
    "lambda": ("Lambda", "proj/helpers.py", "lambda x: x * 2", "lambda x: x * 3"),
    "factory closure": ("Factory", "proj/helpers.py", "return x + n", "return x - n"),
    "factory argument": (
        "Factory",
        "proj/helpers.py",
        "make_adder(3)",
        "make_adder(4)",
    ),
    "method of a class built in the body": (
        "BodyClass",
        "proj/tasks.py",
        "return y + 9",
        "return y + 90",
    ),
    # gap 1: the class of an input instance
    "method of an input instance": (
        "OnInstance",
        "proj/models.py",
        "return x * self.k",
        "return x + self.k",
    ),
    # I4's own cases: a base class method and a helper a method calls
    "method inherited from a user base class": (
        "OnDerived",
        "proj/bases.py",
        "return x * 2",
        "return x * 3",
    ),
    "helper called by a method": (
        "OnHelped",
        "proj/helpers.py",
        "return x + 6",
        "return x + 60",
    ),
}


@pytest.mark.parametrize("case", sorted(CHANGES))
def test_single_edit_changes_exactly_its_task(case, base, tmp_path):
    task, name, old, new = CHANGES[case]
    edited = checksums(write_package(tmp_path, (name, old, new)))
    assert edited[task] != base[task], case
    unchanged = {k for k in base if edited[k] == base[k]}
    assert task not in unchanged
    assert "OnUnrelated" in unchanged


@pytest.mark.parametrize(
    "case, name, old, new",
    [
        ("getattr", "proj/helpers.py", "return x + 7", "return x + 70"),
        ("importlib", "proj/dyn.py", "return x + 8", "return x + 80"),
    ],
)
def test_dynamic_dispatch_is_not_seen(case, name, old, new, base, tmp_path):
    """RFC 0004 4.2: the target of ``getattr`` or ``import_module`` is not reached; the
    override is to pass the callable as an input or to declare a ``version=``"""
    edited = checksums(write_package(tmp_path, (name, old, new)))
    assert edited["ByGetattr"] == base["ByGetattr"]
    assert edited["ByImportlib"] == base["ByImportlib"]


def test_call_edit_changes_checksum(base, tmp_path):
    """Editing ``Proc.__call__`` changes the checksum of a task taking a ``Proc``"""
    edited = checksums(
        write_package(
            tmp_path, ("proj/models.py", "return x * self.k", "return x + self.k")
        )
    )
    assert edited["OnInstance"] != base["OnInstance"]


def test_unrelated_class_edit_no_change(base, tmp_path):
    edited = checksums(
        write_package(tmp_path, ("proj/models.py", "return x - 1", "return x - 2"))
    )
    assert edited == base


def test_relocated_package_has_the_same_checksums(base, tmp_path):
    moved = tmp_path / "elsewhere" / "deeper"
    moved.mkdir(parents=True)
    write_package(moved)
    assert checksums(moved) == base


# ---------------------------------------------------------------------------
# class@1
# ---------------------------------------------------------------------------


class Plain:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x * self.k


def module_helper(x):
    return x + 1


class UsesHelper:
    def __call__(self, x):
        return module_helper(x)


class Color(enum.Enum):
    RED = 1
    GREEN = 2


def test_class_digest_form():
    digest = class_digest(Plain)
    assert re.fullmatch(r"class@1:[0-9a-f]{64}", digest)
    assert digest == class_digest(Plain)
    assert digest != class_digest(UsesHelper)


def test_class_digest_independent_of_code_hash_tier_default():
    assert class_digest(Plain) == class_digest(Plain, code_hash="user")


def test_instance_input_carries_class_code():
    """An instance input's digest includes its class's digest (gap 1)"""

    @python.define
    def Apply(p: object, x: int) -> int:
        return p(x)

    one = task_record(Apply(p=Plain(2), x=1))["inputs"]["p"]
    assert one == task_record(Apply(p=Plain(2), x=1))["inputs"]["p"]
    assert one != task_record(Apply(p=Plain(3), x=1))["inputs"]["p"]
    assert one != task_record(Apply(p=UsesHelper(), x=1))["inputs"]["p"]


# ---------------------------------------------------------------------------
# $class on codec values
# ---------------------------------------------------------------------------


@attrs.define
class Params:
    scale: int = 1


register_codec(
    Params,
    "test.identity_classes.Params",
    lambda p: {"scale": p.scale},
    lambda d: Params(**d),
)


def test_list_of_codec_class_carries_class():
    form = digest_form([Params(2)])
    assert form == [
        {
            "$codec": "test.identity_classes.Params",
            "value": {"scale": {"$int": "2"}},
            "$class": class_digest(Params),
        }
    ]


def test_codec_class_form_has_class_at_top_level():
    assert digest_form(Params(1))["$class"] == class_digest(Params)


def test_codec_of_library_type_has_no_class():
    import datetime

    assert "$class" not in json.dumps(digest_form(datetime.datetime(2020, 1, 1)))


# ---------------------------------------------------------------------------
# type@1
# ---------------------------------------------------------------------------


def td(tp):
    return type_digest(tp)


def test_type_digest_form():
    assert re.fullmatch(r"type@1:[0-9a-f]{64}", td(int))
    assert td(int) != td(str)
    assert td(list[int]) != td(list[str])
    assert td(list[int]) != td(tuple[int])


def test_union_order_free():
    assert td(int | str) == td(str | int)
    assert td(ty.Union[int, str]) == td(ty.Union[str, int])
    assert td(int | str) == td(ty.Union[int, str])
    assert td(int | str | None) == td(None | str | int)


def test_optional_spellings_agree():
    """``int | None`` and ``Optional[int]`` have one digest on every minor, though
    ``get_origin`` differs between 3.13 and 3.14"""
    assert td(int | None) == td(ty.Optional[int]) == td(ty.Union[int, None])
    assert td(int | None) != td(int)


def test_union_flattens_and_deduplicates():
    assert td(ty.Union[int, ty.Union[str, int]]) == td(int | str)
    assert td(ty.Union[int, int]) == td(ty.Union[int, int]) != td(str)
    assert td(list[int | str]) == td(list[str | int])


def test_literal_order_free():
    assert td(ty.Literal["a", 1]) == td(ty.Literal[1, "a"])
    assert td(ty.Literal["a", 1]) != td(ty.Literal["a", 2])
    assert td(ty.Literal[1]) != td(ty.Literal["1"])
    assert td(ty.Literal[True]) != td(ty.Literal[1])


def test_tuple_ellipsis():
    assert td(tuple[int, ...]) != td(tuple[int])
    assert td(tuple[int, ...]) != td(tuple[int, int])
    assert td(tuple[int, ...]) == td(ty.Tuple[int, ...])


def test_callable_list_arguments():
    assert td(ty.Callable[[int], str]) != td(ty.Callable[[str], str])
    assert td(ty.Callable[[int, str], str]) != td(ty.Callable[[int], str])
    assert td(ty.Callable[[int], str]) != td(ty.Callable[[], str])


def test_dict_argument_order_matters():
    assert td(dict[str, int]) != td(dict[int, str])


def test_annotated_carries_its_metadata():
    assert td(ty.Annotated[int, "a"]) != td(int)
    assert td(ty.Annotated[int, "a"]) != td(ty.Annotated[int, "b"])
    assert td(ty.Annotated[int, "a", "b"]) != td(ty.Annotated[int, "b", "a"])
    assert td(ty.Annotated[int, "a"]) == td(ty.Annotated[int, "a"])


def test_typevar_digest():
    T = ty.TypeVar("T")
    U = ty.TypeVar("U")
    B = ty.TypeVar("B", bound=int)
    assert td(T) == td(ty.TypeVar("T"))
    assert td(T) != td(U)
    assert td(T) != td(B)
    assert td(B) != td(ty.TypeVar("B", bound=str))
    assert td(ty.TypeVar("C", int, str)) != td(ty.TypeVar("C", int, bytes))


def test_stdlib_type_is_its_qualified_name():
    from pathlib import Path as P

    assert td(P) == td(P)
    assert td(P) != td(str)


def test_user_class_type_carries_class_code(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(write_package(tmp_path)))
    from proj.models import Derived, Proc

    assert td(Proc) != td(Derived)
    assert td(Proc) == type_digest(Proc)
    # the class's code is in it, not its __dict__ object identity
    assert td(Proc).startswith("type@1:")


def test_user_enum_digest_is_member_based_and_stable():
    assert td(Color) == td(Color)
    assert td(Color) != td(enum.Enum)


def test_outputs_of_a_task_is_type_at_1():
    @python.define
    def Add(a: int, b: int = 2) -> int:
        return a + b

    record = task_record(Add(a=1))
    assert record["outputs"] == type_digest(Add.Outputs)
    assert record["outputs"].startswith("type@1:")


def test_fields_class_digest_follows_field_types():
    @python.define
    def One(a: int) -> int:
        return a

    @python.define
    def Two(a: str) -> int:
        return 1

    assert td(One.Outputs) == td(One.Outputs)
    assert td(One) != td(Two)


# ---------------------------------------------------------------------------
# The enum golden: one literal, compared on every minor under ``full-ci``
# ---------------------------------------------------------------------------

GOLDEN_PACKAGE = {
    "goldpkg/__init__.py": "",
    "goldpkg/colors.py": """
        import enum


        class Shade(enum.Enum):
            LIGHT = 1
            DARK = 2

            def label(self):
                return self.name.lower()
    """,
}

GOLDEN_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from goldpkg.colors import Shade
from pydra.compose import python
from pydra.utils.hash import class_digest, type_digest


@python.define
def Pick(shade: Shade) -> int:
    return 1


record = Pick(shade=Shade.DARK)
from pydra.utils.identity import task_record

rec = task_record(record)
print(json.dumps({
    "class": class_digest(Shade),
    "type": type_digest(Shade),
    "input": rec["inputs"]["shade"],
}))
"""

ENUM_GOLDEN = {
    "class": "class@1:3c354204c51f4cef19126e2aa20b8814ed95c5cd68debab2fa33fbf02a133159",
    "type": "type@1:604799f13920222af5cb6fbb55ad31efb7708cf311ea8ef107709dcf71fd9fca",
    "input": "py@1:de3b874f1b6c31d77d80cd05058b163f",
}


# Computed under 3.11, 3.12, 3.13, 3.14 and 3.15 and equal on all of them
TYPE_GOLDEN = {
    "int": "type@1:201b7421badf40e8eb7da5965b86464c1ba804afbcbe82871a907abb86cf3cf7",
    "optional": "type@1:618098c29702ad1e1bad8ca0af10f27ac005496008e0b38863cf3322d0f2f0eb",
    "union": "type@1:fa4b21e9436afe67532a22907b087b4aa797c2a24125a65189c6eaf42bd60d1e",
    "literal": "type@1:6ea97794099343829e6009883cb7f93fe76d2eb655aaf07700fc4a160f58d4a1",
    "tuple": "type@1:85f82f0e6d745475409f9254eae52469dac0a16d4f2634b364dec66d664e0c14",
    "callable": "type@1:4276e45ee787cda0315bd3035354efc79c9837abd4698a593bb5cca533ffe6ff",
    "path": "type@1:cb095e6d923c102e7b09b757ee563dc8a46b9dcd4560e044a323edc7c5cf55eb",
    "dict": "type@1:c08369faf3402b5e78ad9a1b046fd82830467b64870cde40ecbb0fe07ab74a30",
    "annotated": "type@1:fc1a2abc46d4b9d5bbc596308bd8c11021dce8ca09082b08cc82ac5beb9f07b6",
}


def test_type_digest_goldens():
    from pathlib import Path

    assert {
        "int": td(int),
        "optional": td(int | None),
        "union": td(ty.Union[str, int, None]),
        "literal": td(ty.Literal["a", 1]),
        "tuple": td(tuple[int, ...]),
        "callable": td(ty.Callable[[int], str]),
        "path": td(Path),
        "dict": td(dict[str, list[int | None]]),
        "annotated": td(ty.Annotated[int, "a"]),
    } == TYPE_GOLDEN


def test_enum_member_golden(tmp_path):
    for name, source in GOLDEN_PACKAGE.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"))
    out = subprocess.run(
        [sys.executable, "-c", GOLDEN_SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(out.strip().splitlines()[-1]) == ENUM_GOLDEN


# ---------------------------------------------------------------------------
# #936: a worker without source uses the digests carried from the submitting process
# ---------------------------------------------------------------------------


@python.define
def Carried(a: int, b: int = 2) -> int:
    return a + b


def no_source(*args, **kwargs):
    raise OSError("could not get source code")


def test_worker_without_source_uses_the_carried_digests(monkeypatch):
    """A function pickled by value under cloudpickle 3.1.3 has ``co_filename``
    ``"<dynamic-code>"`` and no readable source. The worker takes the code digest and the
    ``Outputs`` digest the submitting process computed."""
    job = Job(name="t", task=Carried(a=1), submitter=Submitter())
    submitted = job.task._compute_hashes()[1]
    checksum, record, task_checksum = job.checksum, job.record, job.task._checksum
    clone = cp.loads(cp.dumps(job))

    for name in ("getsource", "getsourcefile", "findsource", "getsourcelines"):
        monkeypatch.setattr(hash_module.inspect, name, no_source)
    walked = []

    def walk(*args, **kwargs):
        walked.append(args)
        raise AssertionError("the worker walked the code")

    monkeypatch.setattr(hash_module, "_CodeWalk", walk)

    assert clone.checksum == checksum
    assert clone.record == record
    assert clone.task._checksum == task_checksum
    assert clone.task._compute_hashes()[1] == submitted
    assert clone.task._compute_hashes(exclude=clone.task._executor_name)[1] == {
        k: v for k, v in submitted.items() if k != clone.task._executor_name
    }
    assert clone.task._hash
    assert clone.task._hash_changes() == []
    assert not walked
    assert PydraIdentityV2().check_key(clone.checksum, PydraIdentityV2().key(clone))


def test_unpickled_task_checksum_is_the_submitters(monkeypatch):
    job = Job(name="t", task=Carried(a=1), submitter=Submitter())
    checksum = job.task._checksum
    job.checksum
    clone = cp.loads(cp.dumps(job))
    for name in ("getsource", "getsourcefile", "findsource", "getsourcelines"):
        monkeypatch.setattr(hash_module.inspect, name, no_source)
    assert clone.task._checksum == checksum


def test_only_an_unpickled_job_carries_digests():
    """The submitting process walks its code each time; the job a worker unpickles
    keeps the digests the submitting process computed"""
    job = Job(name="t", task=Carried(a=1), submitter=Submitter())
    job.checksum
    assert job.task._carried is None
    clone = cp.loads(cp.dumps(job))
    assert set(clone.task._carried) == set(job.task._hashes)
    assert {"a", "b", "Outputs", clone.task._executor_name} <= set(clone.task._carried)
    assert clone.task._carried["Outputs"][1] == type_digest(job.task.Outputs)


def test_run_through_a_worker(any_worker, tmp_path, monkeypatch):
    outputs = Carried(a=1)(cache_root=tmp_path, worker=any_worker)
    assert outputs.out == 3


# ---------------------------------------------------------------------------
# #936: inputs a worker without source cannot re-hash. Each task has a File field, so
# the worker re-checks its input hashes after the run.
# ---------------------------------------------------------------------------


def _helper(x):
    return x + 1


class Scaler:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return _helper(x) * self.k


def double(x: int) -> int:
    return x * 2


@python.define
def ApplyObj(f: File, p: object, x: int) -> int:
    return p(x) if isinstance(p, Scaler) else x


@python.define
def ApplyFn(f: File, p: ty.Callable, x: int) -> int:
    return p(x)


@python.define
def InnerAdd(a: int) -> int:
    return a + 1


@workflow.define
def InnerWf(a: int) -> int:
    return workflow.add(InnerAdd(a=a)).out


@pytest.fixture
def script_like():
    """Pickle this module by value, as a script's functions and classes are, so a worker
    imports nothing from it"""
    module = sys.modules[__name__]
    cp.register_pickle_by_value(module)
    yield
    cp.unregister_pickle_by_value(module)


def _file(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("x")
    return path


def _no_crash(root):
    assert list(Path(root).rglob("_error.pklz")) == []


def _wf_with(task_factory):
    @workflow.define
    def Wf(f: File, x: int) -> int:
        return workflow.add(task_factory(f)).out

    return Wf


INPUTS = {
    "user-class instance": lambda f: ApplyObj(f=f, p=Scaler(2), x=3),
    "function": lambda f: ApplyFn(f=f, p=double, x=3),
    "standalone task": lambda f: ApplyObj(f=f, p=InnerAdd(a=1), x=3),
    "workflow": lambda f: ApplyObj(f=f, p=InnerWf(a=1), x=3),
}


@pytest.mark.parametrize("case", sorted(INPUTS))
def test_standalone_job_with_file_and_unhashable_without_source(
    case, any_worker, tmp_path, script_like
):
    """A non-file input is never rehashed in a worker: a worker without source (cloudpickle
    3.1.3) neither fails nor writes a crash report"""
    outputs = INPUTS[case](_file(tmp_path))(
        cache_root=tmp_path / "cache", worker=any_worker
    )
    assert outputs.out in (3, 6, 8)
    _no_crash(tmp_path / "cache")


@pytest.mark.parametrize("case", sorted(INPUTS))
def test_workflow_job_with_file_and_unhashable_without_source(
    case, any_worker, tmp_path, script_like
):
    Wf = _wf_with(lambda f, c=INPUTS[case]: c(f))
    outputs = Wf(f=_file(tmp_path), x=1)(
        cache_root=tmp_path / "cache", worker=any_worker
    )
    assert outputs.out in (3, 6, 8)
    _no_crash(tmp_path / "cache")


@python.define
def Mutate(f: File) -> int:
    with open(f, "a") as handle:
        handle.write("x")
    return 1


def test_file_changed_by_the_job_is_still_detected(
    any_worker, tmp_path, script_like, caplog
):
    """A job that changes its File input fails with the change reported: the debug worker
    raises, and a pool worker raises after it committed the result, which the submitter
    logs"""
    raised = ""
    try:
        Mutate(f=_file(tmp_path))(cache_root=tmp_path / "cache", worker=any_worker)
    except RuntimeError as e:
        raised = str(e)
    assert "hashes have changed" in raised or "Task execution failed" in caplog.text


def test_file_changed_by_the_job_is_reported_by_the_unpickled_job(tmp_path):
    """What a worker does with the job it unpickles"""
    job = Job(
        name="t",
        task=Mutate(f=_file(tmp_path)),
        submitter=Submitter(cache_root=tmp_path / "cache"),
    )
    job.checksum
    clone = cp.loads(cp.dumps(job))
    with pytest.raises(RuntimeError, match="hashes have changed"):
        clone.run()


def test_outputs_digest_is_taken_from_the_carried_ones():
    job = Job(name="t", task=Carried(a=1), submitter=Submitter())
    job.checksum
    clone = cp.loads(cp.dumps(job))
    marker = "type@1:" + "0" * 64
    clone.task._carried["Outputs"] = (None, marker)
    assert clone.task._compute_hashes()[1]["Outputs"] == marker


def test_only_file_fields_are_rehashed_in_a_worker(tmp_path, monkeypatch):
    job = Job(
        name="t",
        task=ApplyObj(f=_file(tmp_path), p=Scaler(2), x=3),
        submitter=Submitter(),
    )
    job.checksum
    clone = cp.loads(cp.dumps(job))
    monkeypatch.setattr(
        hash_module,
        "_class_digest",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rehashed a class")),
    )
    assert clone.task._hash_changes() == []


# ---------------------------------------------------------------------------
# class@1 is stable under compiler-generated class members (3.14 annotate functions)
# ---------------------------------------------------------------------------

ANNOTATED_PACKAGE = {
    "annpkg/__init__.py": "",
    "annpkg/models.py": """
        import dataclasses


        @dataclasses.dataclass
        class Point:
            x: int = 0
            y: "list[int] | None" = None

            def norm(self):
                return abs(self.x)


        class Plain:
            scale: int = 2

            def go(self, a):
                return a * self.scale
    """,
}

ANNOTATED_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from annpkg.models import Plain, Point
from pydra.utils.hash import class_digest, type_digest
print(json.dumps({
    "point": class_digest(Point), "plain": class_digest(Plain),
    "point_type": type_digest(Point),
}))
"""


def _annotated(root, comment=False):
    for name, source in ANNOTATED_PACKAGE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = textwrap.dedent(source).lstrip("\n")
        if comment:
            text = text.replace("@dataclasses", "# a comment\n\n@dataclasses")
            text = text.replace("class Plain:", "# another\nclass Plain:")
        path.write_text(text)
    out = subprocess.run(
        [sys.executable, "-c", ANNOTATED_SCRIPT, str(root)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_annotated_class_digest_ignores_comments_and_line_numbers(tmp_path):
    assert _annotated(tmp_path / "a") == _annotated(tmp_path / "b", comment=True)


# equal on 3.11, 3.12, 3.13, 3.14 and 3.15
ANNOTATED_GOLDEN = {
    "point": "class@1:8ffc7c649507d0bd75fc401285c1fa754915991fc1377e0d3050e611b6adb2cb",
    "plain": "class@1:8b306e9d52580909f8a6dd20e280b201014a83bcb1ec521204eb675b23f9cacc",
    "point_type": "type@1:df5c8552ce10899ad12ceb2121c858a022cfa72b50c71a9d9939298cafcf6dfe",
}


def test_annotated_class_golden(tmp_path):
    assert _annotated(tmp_path) == ANNOTATED_GOLDEN


# ---------------------------------------------------------------------------
# type@1 of the field types of a fields class
# ---------------------------------------------------------------------------


def _outputs_digest(tp):
    @python.define(outputs={"out": python.out(type=tp)})
    def F(a: int):
        return a

    return type_digest(F.Outputs)


def test_pep585_field_types_are_digested_by_their_arguments():
    assert _outputs_digest(list[int]) != _outputs_digest(list[str])
    assert _outputs_digest(list[File]) != _outputs_digest(list[ty.Any])
    from fileformats.generic import Directory

    assert _outputs_digest(list[File]) != _outputs_digest(list[Directory])
    assert _outputs_digest(dict[str, int]) != _outputs_digest(dict[str, list[File]])
    assert _outputs_digest(ty.List[int]) == _outputs_digest(list[int])


def test_bare_typing_aliases_are_digested_by_name():
    assert _outputs_digest(ty.List) != _outputs_digest(ty.Dict)
    assert _outputs_digest(ty.Callable) != _outputs_digest(ty.Sequence)


def test_typevar_field_digest_has_name_bound_constraints_and_variance():
    T, U = ty.TypeVar("T"), ty.TypeVar("U")
    assert _outputs_digest(T) == _outputs_digest(ty.TypeVar("T"))
    assert _outputs_digest(T) != _outputs_digest(U)
    assert _outputs_digest(T) != _outputs_digest(ty.TypeVar("T", covariant=True))
    assert _outputs_digest(T) != _outputs_digest(ty.TypeVar("T", bound=int))


def test_forward_ref_is_its_string():
    assert td(ty.ForwardRef("Foo")) == td(ty.ForwardRef("Foo"))
    assert td(ty.ForwardRef("Foo")) != td(ty.ForwardRef("Bar"))
    assert _outputs_digest(ty.ForwardRef("Foo")) != _outputs_digest(
        ty.ForwardRef("Bar")
    )


def test_none_argument_is_nonetype():
    assert td(dict[str, None]) == td(ty.Dict[str, None]) == td(dict[str, type(None)])


def test_library_type_carries_its_owner():
    import numpy

    from pydra.utils.hash import Cache, bytes_repr_type

    cache = Cache()
    data = b"".join(bytes_repr_type(numpy.ndarray, cache))
    assert data.startswith(b"type:(lib:numpy==")
    assert data.endswith(b":numpy.ndarray)")
    assert "numpy" in cache.modules
    assert any(r.startswith("lib:numpy==") for r in cache.refs)
    assert b"lib:" not in b"".join(bytes_repr_type(Path, Cache()))


# equal on 3.11, 3.12, 3.13, 3.14 and 3.15
TYPE_GOLDEN_2 = {
    "list": "type@1:708f32d818ad14f99d19e487639b0ef2502b9b8f21a8efe8504434a91866ae5c",
    "tlist": "type@1:708f32d818ad14f99d19e487639b0ef2502b9b8f21a8efe8504434a91866ae5c",
    "paramspec": "type@1:0ba0266553649af4552feb205bfd1c49e0ac724289aa6efee3afe8b3a41fcaa2",
    "callable_p": "type@1:b216773194079f0c07bc19ceac6d99059b2d9b568a2628c622c12e0f3f1b2290",
    "typevar": "type@1:6dadccec566fc278b6009ef9d8b79f341724847b35e226cb98afe81cff4cf8f1",
    "forward": "type@1:825b1802a065b69799034e1cd5022b56da737c11276fe78ea482dbe94519842a",
    "dict_none": "type@1:baf77c7f0ebbc1de56d155ec80e8d9d4b8b1c1c1f1c44a052a619d1039949ea1",
}


def test_type_digest_goldens_2():
    P = ty.ParamSpec("P")
    T = ty.TypeVar("T", bound=int)
    assert {
        "list": _outputs_digest(list[int]),
        "tlist": _outputs_digest(ty.List[int]),
        "paramspec": td(P),
        "callable_p": td(ty.Callable[P, int]),
        "typevar": td(T),
        "forward": td(ty.ForwardRef("Foo")),
        "dict_none": td(dict[str, None]),
    } == TYPE_GOLDEN_2


# ---------------------------------------------------------------------------
# carried digests never outlive the worker run and never survive a change
# ---------------------------------------------------------------------------


@python.define
def AddXY(x: int, y: int) -> int:
    return x + y


def test_task_of_a_worker_result_is_not_carried(any_worker, tmp_path):
    with Submitter(cache_root=tmp_path, worker=any_worker) as sub:
        res = sub(AddXY(x=1, y=1))
    assert res.outputs.out == 2
    assert res.task._carried is None
    res.task.x = 40
    assert res.task(cache_root=tmp_path, worker="debug").out == 41
    other = copy.copy(res.task)
    other.y = 100
    assert other(cache_root=tmp_path, worker="debug").out == 140


def _carried_job():
    job = Job(name="t", task=AddXY(x=1, y=1), submitter=Submitter())
    job.checksum
    return job


@pytest.mark.parametrize(
    "clone",
    [lambda j: cp.loads(cp.dumps(j)), copy.deepcopy],
    ids=["unpickle", "deepcopy"],
)
def test_assignment_after_unpickling_is_seen(clone):
    # expected values first: unpickling a function by value can change how it is walked
    expected = AddXY(x=40, y=1)._checksum, AddXY(x=40, y=100)._checksum
    job = clone(_carried_job())
    assert job.task._carried
    job.task.x = 40
    assert job.task._checksum == expected[0]
    other = copy.copy(job.task)
    other.y = 100
    assert other._checksum == expected[1]
    assert job.task._checksum == expected[0]


# ---------------------------------------------------------------------------
# one predicate: what a worker rehashes
# ---------------------------------------------------------------------------


class External:
    """A value whose hash depends on state outside it"""

    def __init__(self, path):
        self.path = path

    def __bytes_repr__(self, cache):
        yield Path(self.path).read_bytes()


@python.define
def AnyBeside(f: File, v: ty.Any) -> int:
    return 1


@pytest.mark.parametrize("kind", ["bytes_repr", "file"])
def test_value_that_depends_on_external_state_is_rechecked(kind, tmp_path):
    inner = tmp_path / "inner.txt"
    inner.write_text("a")
    value = External(inner) if kind == "bytes_repr" else File(inner)
    job = Job(
        name="t", task=AnyBeside(f=_file(tmp_path), v=value), submitter=Submitter()
    )
    job.checksum
    clone = cp.loads(cp.dumps(job))
    assert clone.task._hash_changes() == []
    inner.write_text("abc")
    assert clone.task._hash_changes() == ["v"]


def test_plain_values_are_never_recomputed_in_a_worker(monkeypatch, tmp_path):
    job = Job(
        name="t", task=AnyBeside(f=_file(tmp_path), v=Scaler(2)), submitter=Submitter()
    )
    job.checksum
    clone = cp.loads(cp.dumps(job))
    monkeypatch.setattr(
        hash_module,
        "_class_digest",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rehashed a class")),
    )
    assert clone.task._hash_changes() == []


# ---------------------------------------------------------------------------
# no repr in a digest
# ---------------------------------------------------------------------------

SENTINEL_PACKAGE = {
    "sentpkg/__init__.py": "",
    "sentpkg/models.py": """
        import dataclasses
        import typing as ty


        @dataclasses.dataclass
        class Opts:
            a: int = 0
            _: dataclasses.KW_ONLY
            b: int = 1
            c: dataclasses.InitVar[int] = 2
            d: ty.ClassVar[int] = 3
            e: ty.Final[int] = 4
    """,
}

SENTINEL_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from sentpkg.models import Opts
from pydra.utils.hash import Cache, _CodeWalk, class_digest, type_digest
chunks = b"".join(_CodeWalk(Opts, Cache(), True))
print(json.dumps({"class": class_digest(Opts), "type": type_digest(Opts), "hex": b"0x" in chunks}))
"""


def _sentinels(root):
    for name, source in SENTINEL_PACKAGE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"))
    out = subprocess.run(
        [sys.executable, "-c", SENTINEL_SCRIPT, str(root)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_kw_only_dataclass_digest_is_stable_across_processes(tmp_path):
    one, two = _sentinels(tmp_path / "a"), _sentinels(tmp_path / "b")
    assert one == two
    assert one["hex"] is False


@pytest.mark.parametrize(
    "tp",
    [
        dataclasses.KW_ONLY,
        dataclasses.InitVar[int],
        ty.ClassVar[int],
        ty.Final[int],
        ty.ClassVar,
        ty.Final,
    ],
    ids=["KW_ONLY", "InitVar", "ClassVar[int]", "Final[int]", "ClassVar", "Final"],
)
def test_sentinel_types_have_no_address_in_their_bytes(tp):
    data = b"".join(hash_module.bytes_repr_type(tp, hash_module.Cache()))
    assert b"0x" not in data
    assert data == b"".join(hash_module.bytes_repr_type(tp, hash_module.Cache()))


def test_kw_only_and_initvar_are_distinct():
    assert td(dataclasses.InitVar[int]) != td(dataclasses.InitVar[str])
    assert td(dataclasses.KW_ONLY) != td(dataclasses.InitVar[int])


def test_value_without_a_stable_form_raises():
    with pytest.raises(TypeError, match="no stable"):
        td(object())


def test_class_annotation_without_a_stable_form_names_the_field():
    class Odd:
        x: object() = 1  # type: ignore[valid-type]

    with pytest.raises(TypeError, match="'x'"):
        class_digest(Odd)


# ---------------------------------------------------------------------------
# registrations: the serializers are bytes_repr_type itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp",
    [
        ty.List,
        ty.Callable,
        ty.ParamSpec("P"),
        ty.TypeVar("T"),
        list[int],
        ty.ForwardRef("X"),
    ],
    ids=["List", "Callable", "ParamSpec", "TypeVar", "list[int]", "ForwardRef"],
)
def test_type_forms_are_dispatched_to_bytes_repr_type(tp):
    assert hash_module.bytes_repr.dispatch(type(tp)) is hash_module.bytes_repr_type


# ---------------------------------------------------------------------------
# the hash cache memoises by id() and keeps what it memoises alive
# ---------------------------------------------------------------------------


def _versioned_task_classes(n):
    import types

    module = types.ModuleType("pydra_cache_alive_probe")
    sys.modules[module.__name__] = module
    exec(
        "\n".join(f"def f{i}(a: int) -> int:\n    return a + {i}\n" for i in range(n)),
        module.__dict__,
    )
    classes = []
    for i in range(n):
        function = getattr(module, f"f{i}")
        function.__module__ = module.__name__
        classes.append(python.define(function, version="1"))
    return classes


def _consume_list(x: list) -> int:
    return len(x)


def test_list_of_task_values_has_one_checksum():
    classes = _versioned_task_classes(40)
    consumer = python.define(_consume_list)
    checksums = {consumer(x=[cls(a=1) for cls in classes])._checksum for _ in range(20)}
    assert len(checksums) == 1


def test_one_cache_gives_every_versioned_task_class_its_digest():
    from pydra.utils.hash import Cache

    classes = _versioned_task_classes(40)
    cache = Cache()
    shared = [type_digest(cls, cache) for cls in classes]
    assert len(set(shared)) == 40
    assert shared == [type_digest(cls) for cls in classes]


# -- classes made by a factory -----------------------------------------------------------------------


def make(n):
    class Made:
        offset = n
        label = "made"

        def shift(self, x):
            return x + self.offset

    return Made


def test_factory_classes_differ_by_their_computed_attributes():
    assert class_digest(make(1)) != class_digest(make(2))
    assert class_digest(make(1)) == class_digest(make(1))


def make_unpacked(n):
    class Unpacked:  # a name unique in this file: before 3.13 a class is found by name
        lo, hi = n, n + 1
        [first, *rest] = [n, n, n]

    return Unpacked


def make_augmented(n):
    class Augmented:
        x = 0
        x += n

    return Augmented


@pytest.mark.parametrize("factory", [make_unpacked, make_augmented])
def test_factory_classes_differ_through_unpacking_and_augmented_assignment(factory):
    """Names bound by tuple, list and starred targets, and names an augmented assignment
    rebinds, are computed attributes too"""
    assert class_digest(factory(1)) != class_digest(factory(2))
    assert class_digest(factory(1)) == class_digest(factory(1))


def test_every_plain_attribute_value_is_hashed():
    from pydra.utils.hash import _attribute_values

    assert _attribute_values(make(1)) == [("label", "made"), ("offset", 1)]


def make_alike(value, first):
    if first:

        class Alike:
            offset = value

    else:

        class Alike:
            offset = -value

    return Alike


def test_definitions_computing_the_same_attributes_hash_them():
    """Two definitions of one qualified name computing the same attributes: the values
    are hashed whichever definition the class came from"""
    assert class_digest(make_alike(1, True)) != class_digest(make_alike(2, True))


def make_model_a(value):
    class Model:
        offset = value

    return Model


def make_model_b(value):
    class Model:
        offset = value + 0

    return Model


def test_same_named_classes_of_two_factories_are_told_apart():
    """Each factory's class is found by its qualified name, so its computed attributes are
    hashed where classes carry no ``__firstlineno__`` (before Python 3.13)"""
    assert class_digest(make_model_a(1)) != class_digest(make_model_a(2))
    assert class_digest(make_model_b(1)) != class_digest(make_model_b(2))


def test_classes_made_by_type_differ_by_their_values():
    assert class_digest(type("K", (), {"x": 1})) != class_digest(
        type("K", (), {"x": 2})
    )


def test_classes_made_by_attrs_make_class_differ_by_their_defaults():
    one = attrs.make_class("P", {"a": attrs.field(default=1)})
    two = attrs.make_class("P", {"a": attrs.field(default=2)})
    assert class_digest(one) != class_digest(two)


def test_classes_of_a_factory_compiled_from_a_string_differ_by_their_values():
    namespace = {}
    source = "def made(n):\n    class K:\n        x = n\n    return K\n"
    exec(compile(source, "<string>", "exec"), namespace)
    made = namespace["made"]
    assert class_digest(made(1)) != class_digest(made(2))


def make_literal(value, first):
    if first:

        class Literal:
            x = 1
            y = value

    else:

        class Literal:
            x = 2
            y = value

    return Literal


def make_extra(first):
    if first:

        class Extra:
            def f(self):
                return 1

    else:

        class Extra:
            z = 3

            def f(self):
                return 1

    return Extra


def test_same_named_definitions_differing_in_a_literal_differ():
    """On every Python minor, whether or not a class names its definition's line"""
    assert class_digest(make_literal(5, True)) != class_digest(make_literal(5, False))
    assert class_digest(make_extra(True)) != class_digest(make_extra(False))


def make_twice(value, first):
    if first:

        class Twice:
            offset = value

    else:

        class Twice:
            scale = value

    return Twice


def test_definitions_of_one_name_with_different_attributes_differ():
    assert class_digest(make_twice(1, True)) != class_digest(make_twice(1, False))
