import typing as ty
from typing_extensions import dataclass_transform
import attrs
from fileformats.generic import File, Directory


OutSpec = ty.TypeVar("OutSpec")


class SpecBase(ty.Generic[OutSpec]):
    Out: ty.Type[OutSpec]


InSpec = ty.TypeVar("InSpec", bound=SpecBase)


@attrs.define
class LazyOut:
    name: str
    spec: type

    def __getattr___(self, field_name):
        try:
            field = self._fields[field_name]
        except KeyError as e:
            raise AttributeError(
                f"Lazy output interface of {self.name} task does not include "
                f"{field_name}"
            ) from e
        return LazyOutField[field.type](self.name, field_name, field.type)

    @property
    def _fields(self):
        return attrs.fields(self.spec)


@attrs.define
class LazyOutField:
    task_name: str
    field_name: str
    type: type


@attrs.define(auto_attribs=False)
class Task(ty.Generic[InSpec, OutSpec]):
    inputs: InSpec = attrs.field()
    name: str = attrs.field()

    @name.default
    def name_default(self):
        return type(self).__name__.lower()

    @property
    def lzout(self) -> OutSpec:
        return ty.cast(OutSpec, LazyOut(self.name, self.inputs.Out))


@attrs.define
class Workflow:
    name: str
    tasks: ty.List[Task] = attrs.field(factory=list)
    connections: ty.List[ty.Tuple[str, LazyOutField]] = attrs.field(factory=list)

    def add(
        self, spec: SpecBase[OutSpec], name: ty.Optional[str] = None
    ) -> Task[SpecBase[OutSpec], OutSpec]:
        task = Task[SpecBase[OutSpec], OutSpec](
            spec, **({"name": name} if name else {})
        )
        self.tasks.append(task)
        return task

    def set_output(self, connection):
        self.connections.append(connection)


def shell_arg(default=attrs.NOTHING, factory=None, argstr="", position=None, help=None):
    return attrs.field(
        default=default,
        factory=factory,
        metadata={"argstr": argstr, "position": position, "help_string": help},
    )


def shell_out(
    default=attrs.NOTHING,
    argstr="",
    position=None,
    help=None,
    file_template=None,
    callable=None,
):
    return attrs.field(
        default=default,
        metadata={
            "argstr": argstr,
            "position": position,
            "output_file_template": file_template,
            "help_string": help,
            "callable": callable,
        },
    )


@dataclass_transform(kw_only_default=True, field_specifiers=(shell_arg,))
def shell_task(klass):
    return attrs.define(kw_only=True, auto_attrib=False, slots=False)(klass)


@dataclass_transform(kw_only_default=True, field_specifiers=(shell_out,))
def shell_outputs(klass):
    return attrs.define(kw_only=True, auto_attrib=False, slots=False)(klass)


def func_arg(default=attrs.NOTHING, factory=None, help=None):
    return attrs.field(
        default=default,
        metadata={"factory": factory, "help_string": help},
    )


@dataclass_transform(kw_only_default=True, field_specifiers=(func_arg,))
def func_task(klass):
    return attrs.define(kw_only=True, auto_attrib=False, slots=False)(klass)


@shell_task
class MyShellSpec(SpecBase["MyShellSpec.Out"]):
    executable: str = "mycmd"

    in_file: File = shell_arg(argstr="", position=0)
    an_option: str = shell_arg(argstr="--opt", position=-1)

    @shell_outputs
    class Out:
        out_file: File = shell_out(file_template="")
        out_str: str = shell_out()


@func_task
class MyFuncSpec(SpecBase["MyFuncSpec.Out"]):
    @staticmethod
    def func(in_int: int, in_str: str) -> ty.Tuple[int, str]:
        return in_int, in_str

    in_int: int = func_arg(help="a dummy input int")
    in_str: str = func_arg(help="a dummy input int")

    @attrs.define
    class Out:
        out_int: int
        out_str: str


wf = Workflow(name="myworkflow")

mytask = wf.add(MyFuncSpec(in_int=1, in_str="hi"))

mytask2 = wf.add(
    MyFuncSpec(
        in_int=mytask.lzout.out_int,  # should be ok
        in_str=mytask.lzout.out_int,  # should show up as a mypy error
    ),
    name="mytask2",
)

wf.set_output(("out_str", mytask2.lzout.out_str))
