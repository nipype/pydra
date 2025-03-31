import typing as ty
import time
from pathlib import Path
from fileformats.generic import File
from pydra.compose import python
from pydra.engine.tests.utils import FunAddTwo, FunFile
from pydra.engine.submitter import Submitter
from pydra.engine.tests.utils import num_python_cache_roots


def test_task_state_cachelocations(worker, tmp_path):
    """
    Two identical tasks with a state and cache_root;
    the second task has readonly_caches and should not recompute the results
    """
    cache_root = tmp_path / "test_task_nostate"
    cache_root.mkdir()
    cache_root2 = tmp_path / "test_task_nostate2"
    cache_root2.mkdir()

    nn = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(worker=worker, cache_root=cache_root) as sub:
        sub(nn)

    nn2 = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results2.errors["error message"])

    # checking the results
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2.outputs.out[i] == res[1]

    # Would ideally check for all nodes of the workflows
    assert num_python_cache_roots(cache_root) == 2
    assert not num_python_cache_roots(cache_root2)


def test_task_state_cachelocations_forcererun(worker, tmp_path):
    """
    Two identical tasks with a state and cache_root;
    the second task has readonly_caches,
    but submitter is called with rerun=True, so should recompute
    """
    cache_root = tmp_path / "test_task_nostate"
    cache_root.mkdir()
    cache_root2 = tmp_path / "test_task_nostate2"
    cache_root2.mkdir()

    nn = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(worker=worker, cache_root=cache_root) as sub:
        sub(nn)

    nn2 = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root
    ) as sub:
        results2 = sub(nn2, rerun=True)

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2.outputs.out[i] == res[1]

    # both workflows should be run
    assert num_python_cache_roots(cache_root) == 2
    assert num_python_cache_roots(cache_root2) == 2


def test_task_state_cachelocations_updated(worker, tmp_path):
    """
    Two identical tasks with states and cache_root;
    the second task has readonly_caches in init,
     that is later overwritten in Submitter.__call__;
    the readonly_caches from call doesn't exist so the second task should run again
    """
    cache_root = tmp_path / "test_task_nostate"
    cache_root.mkdir()
    cache_root1 = tmp_path / "test_task_nostate1"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_task_nostate2"
    cache_root2.mkdir()

    nn = FunAddTwo().split("a", a=[3, 5])
    with Submitter(worker=worker, cache_root=cache_root) as sub:
        sub(nn)

    nn2 = FunAddTwo().split("a", a=[3, 5])
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results2.errors["error message"])

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2.outputs.out[i] == res[1]

    # both workflows should be run
    assert num_python_cache_roots(cache_root) == 2
    assert num_python_cache_roots(cache_root2) == 2


def test_task_files_cachelocations(worker, tmp_path):
    """
    Two identical tasks with provided cache_root that use file as an input;
    the second task has readonly_caches and should not recompute the results
    """
    cache_root = tmp_path / "test_task_nostate"
    cache_root.mkdir()
    cache_root2 = tmp_path / "test_task_nostate2"
    cache_root2.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    input1 = input_dir / "input1.txt"
    input1.write_text("test")
    input2 = input_dir / "input2.txt"
    input2.write_text("test")

    nn = FunFile(filename=input1)
    with Submitter(worker=worker, cache_root=cache_root) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    nn2 = FunFile(filename=input2)
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results2.outputs.out == "test"

    # checking if the second task didn't run the interface again
    assert results.output_dir == results2.output_dir


class OverriddenContentsFile(File):
    """A class for testing purposes, to that enables you to override the contents
    of the file to allow you to check whether the persistent cache is used."""

    def __init__(
        self,
        fspaths: ty.Iterator[Path],
        contents: ty.Optional[bytes] = None,
        metadata: ty.Dict[str, ty.Any] = None,
    ):
        super().__init__(fspaths, metadata=metadata)
        self._contents = contents

    def byte_chunks(self, **kwargs) -> ty.Generator[ty.Tuple[str, bytes], None, None]:
        if self._contents is not None:
            yield (str(self.fspath), iter([self._contents]))
        else:
            yield from super().byte_chunks(**kwargs)

    @property
    def raw_contents(self):
        if self._contents is not None:
            return self._contents
        return super().raw_contents


def test_task_files_persistentcache(tmp_path):
    """
    Two identical tasks with provided cache_root that use file as an input;
    the second task has readonly_caches and should not recompute the results
    """
    test_file_path = tmp_path / "test_file.txt"
    test_file_path.write_bytes(b"foo")
    cache_root = tmp_path / "cache-dir"
    cache_root.mkdir()
    test_file = OverriddenContentsFile(test_file_path)

    @python.define
    def read_contents(x: OverriddenContentsFile) -> bytes:
        return x.raw_contents

    assert read_contents(x=test_file)(cache_root=cache_root).out == b"foo"
    test_file._contents = b"bar"
    # should return result from the first run using the persistent cache
    assert read_contents(x=test_file)(cache_root=cache_root).out == b"foo"
    time.sleep(2)  # Windows has a 2-second resolution for mtime
    test_file_path.touch()  # update the mtime to invalidate the persistent cache value
    assert (
        read_contents(x=test_file)(cache_root=cache_root).out == b"bar"
    )  # returns the overridden value
