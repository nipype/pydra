"""``class@1`` and ``type@1`` (RFC 0004 sections 4.2, 5.1; I4).

The checksum cases of RFC 0004 8.3 and 8.8 run a user package in a fresh process per
variant, so no memo of one edit is seen by the next. The type cases run in this process
and have no dependence on the Python minor, which the ``full-ci`` label checks on every
minor pydra supports.
"""

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

from pydra.compose import python
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
    "class": "class@1:43628a3420ea9ae68685a4e1a997048f933e54d7820e35d60d5a3f220afde774",
    "type": "type@1:d70f01555fee1f8b07c494f4fd94619431e06df545fa02ad2a68ad0320f1074e",
    "input": "py@1:fdd805a3bc26f7f52f4418c8ea39dc1e",
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
    ``"<dynamic-code>"`` and no readable source (pydra issue #936). The worker takes the
    code digest and the ``Outputs`` digest the submitting process computed."""
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
    assert set(clone.task._carried) == {"Outputs", clone.task._executor_name}
    assert clone.task._carried["Outputs"] == type_digest(job.task.Outputs)


def test_run_through_a_worker_without_source(any_worker, tmp_path, monkeypatch):
    outputs = Carried(a=1)(cache_root=tmp_path, worker=any_worker)
    assert outputs.out == 3


@python.define
def WithFile(f: File) -> int:
    return 1


def test_changed_outputs_digest_is_reported(tmp_path, monkeypatch):
    """A changed key that is not an input field (the ``Outputs`` digest) is reported as
    a change, not raised as an ``AttributeError`` (pydra issue #936)"""
    path = tmp_path / "x.txt"
    path.write_text("x")
    job = Job(name="t", task=WithFile(f=path), submitter=Submitter())
    job.checksum
    monkeypatch.setattr(type(job.task), "_hash_changes", lambda self: ["Outputs"])
    with pytest.raises(RuntimeError, match="Input field hashes have changed"):
        job._check_for_hash_changes()
