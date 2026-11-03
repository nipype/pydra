import shutil
import os
import pytest
import typing as ty
from click.testing import CliRunner, Result as CliResult

os.environ["NO_ET"] = "true"


def pytest_addoption(parser):
    parser.addoption("--with-dask", action="store_true", help="run all combinations")
    parser.addoption(
        "--only-worker",
        help="only run tests with provided worker",
    )


@pytest.fixture(autouse=True)
def isolated_user_cache(tmp_path_factory, monkeypatch):
    """Give each test its own user-level cache root (the default run cache and the
    persistent hash cache), so no test reads or writes another's results"""
    from pydra.utils.general import CACHE_ROOT_ENV_VAR
    from pydra.utils.hash import PersistentCache

    root = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setenv(CACHE_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(PersistentCache.LOCATION_ENV_VAR, raising=False)
    return root


def _is_loopback(host) -> bool:
    import ipaddress

    if host is None or host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture
def no_network(monkeypatch):
    """Make any connection or name lookup beyond the loopback interface fail in the
    test process; local sockets, including socketpair, keep working"""
    import socket

    def refuse(host):
        raise OSError(f"network access to {host} is disabled in this test")

    getaddrinfo = socket.getaddrinfo
    create_connection = socket.create_connection
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def local_getaddrinfo(host, *args, **kwargs):
        if not _is_loopback(host):
            refuse(host)
        return getaddrinfo(host, *args, **kwargs)

    def local_create_connection(address, *args, **kwargs):
        if not _is_loopback(address[0]):
            refuse(address[0])
        return create_connection(address, *args, **kwargs)

    def checked(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(
            address[0]
        ):
            refuse(address[0])

    def local_connect(sock, address):
        checked(sock, address)
        return connect(sock, address)

    def local_connect_ex(sock, address):
        checked(sock, address)
        return connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", local_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", local_create_connection)
    monkeypatch.setattr(socket.socket, "connect", local_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", local_connect_ex)


@pytest.fixture(scope="session", params=["debug", "cf"])
def worker(request):
    return request.param


def pytest_generate_tests(metafunc):
    if "any_worker" in metafunc.fixturenames:
        try:
            with_dask = metafunc.config.getoption("with_dask")
        except ValueError:
            with_dask = False
        try:
            only_worker = metafunc.config.getoption("only_worker")
        except ValueError:
            only_worker = None
        if only_worker is None:
            available_workers = ["debug", "cf"]
            if with_dask:
                available_workers.append("dask")
            if bool(shutil.which("sbatch")):
                available_workers.append("slurm")
        else:
            available_workers = [only_worker]
        # Set the available workers as a parameter to the
        # test function
        metafunc.parametrize("any_worker", available_workers)


def pytest_collection_modifyitems(items):
    """Tag the parametrisations that need a real worker, so the fast tier
    (``-m "not slow and not real_worker and not slurm"``) only runs the debug worker"""
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        for name in ("worker", "any_worker"):
            worker = callspec.params.get(name)
            if worker == "slurm":
                item.add_marker(pytest.mark.slurm)
            elif worker in ("cf", "dask"):
                item.add_marker(pytest.mark.real_worker)


@pytest.fixture
def cli_runner(catch_cli_exceptions: bool) -> ty.Callable[..., ty.Any]:
    def invoke(
        *args: ty.Any, catch_exceptions: bool = catch_cli_exceptions, **kwargs: ty.Any
    ) -> CliResult:
        runner = CliRunner()
        result = runner.invoke(*args, catch_exceptions=catch_exceptions, **kwargs)  # type: ignore[misc]
        return result

    return invoke


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call: pytest.CallInfo[ty.Any]) -> None:
        if call.excinfo is not None:
            raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
        raise excinfo.value

    CATCH_CLI_EXCEPTIONS = False
else:
    CATCH_CLI_EXCEPTIONS = True


@pytest.fixture
def catch_cli_exceptions() -> bool:
    return CATCH_CLI_EXCEPTIONS


# Example VSCode launch configuration for debugging unittests
# {
#     "name": "Test Config",
#     "type": "python",
#     "request": "launch",
#     "purpose": ["debug-test"],
#     "justMyCode": false,
#     "console": "internalConsole",
#     "env": {
#         "_PYTEST_RAISE": "1"
#     },
# }
