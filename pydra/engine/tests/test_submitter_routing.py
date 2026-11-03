"""Worker routing: the submitter's router picks a worker for each job at dispatch."""

import asyncio
import os
import threading
from types import SimpleNamespace
import pytest
from pydra.compose import python, workflow
from pydra.engine.submitter import Submitter
from pydra.utils.resources import ComputeResources
from pydra.utils.routing import default_router, tag_router
from pydra.workers import debug

GPU = ComputeResources(gpus=1, tags=("gpu",))


class RecordingDebug(debug.Worker):
    """A debug worker that remembers the names of the jobs it was given."""

    _plugin_name = "debug"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    def run(self, job, rerun=False):
        self.seen.append(job.name)
        return super().run(job, rerun=rerun)


class RecordingAsync(RecordingDebug):
    """The same, but an asynchronous worker."""

    async def run(self, job, rerun=False):
        self.seen.append(job.name)
        return job.run(rerun=rerun)


@python.define
def AddOne(x: int) -> int:
    return x + 1


@workflow.define(outputs=["out"])
def Pipeline(x: int):
    # each node feeds the next, so no node is a cache hit on another's result
    prep = workflow.add(AddOne(x=x), name="prep")
    train = workflow.add(AddOne(x=prep.out), name="train", resources=GPU)
    post = workflow.add(AddOne(x=train.out), name="post")
    return post.out


def by_gpus(job):
    return "gpu" if job.resources is not None and job.resources.gpus else None


def test_default_router_keeps_a_single_worker(tmp_path):
    only = RecordingDebug()
    with Submitter(worker=only, cache_root=tmp_path) as sub:
        assert sub.router is default_router
        assert sub.workers == {"default": only}
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert only.seen == ["main", "prep", "train", "post"]


def test_default_router_ignores_tags_when_other_workers_exist(tmp_path):
    default, other = RecordingDebug(), RecordingDebug()
    with Submitter(worker=default, workers={"gpu": other}, cache_root=tmp_path) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert default.seen == ["main", "prep", "train", "post"]
    assert other.seen == []


def test_route_gpu_tagged_jobs_to_their_own_worker(tmp_path):
    default, gpu = RecordingDebug(), RecordingDebug()
    with Submitter(
        worker=default,
        workers={"gpu": gpu},
        router=tag_router({"gpu": "gpu"}),
        cache_root=tmp_path,
    ) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert default.seen == ["main", "prep", "post"]
    assert gpu.seen == ["train"]


def test_route_by_resources(tmp_path):
    default, gpu = RecordingDebug(), RecordingDebug()
    with Submitter(
        worker=default, workers={"gpu": gpu}, router=by_gpus, cache_root=tmp_path
    ) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert default.seen == ["main", "prep", "post"]
    assert gpu.seen == ["train"]


def test_router_can_name_the_default_worker(tmp_path):
    default, gpu = RecordingDebug(), RecordingDebug()
    with Submitter(
        worker=default,
        workers={"gpu": gpu},
        router=lambda job: "gpu" if job.name == "post" else "default",
        cache_root=tmp_path,
    ) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert default.seen == ["main", "prep", "train"]
    assert gpu.seen == ["post"]


def test_resources_set_by_the_definition_route_the_main_job(tmp_path):
    default, gpu = RecordingDebug(), RecordingDebug()

    @python.define(resources=GPU)
    def OnGpu(x: int) -> int:
        return x + 1

    out = OnGpu(x=1)(
        cache_root=tmp_path,
        worker=default,
        workers={"gpu": gpu},
        router=tag_router({"gpu": "gpu"}),
    )
    assert out.out == 2
    assert default.seen == []
    assert gpu.seen == ["main"]


def test_mixed_run_with_an_asynchronous_default_worker(tmp_path):
    default, gpu = RecordingAsync(), RecordingDebug()
    with Submitter(
        worker=default,
        workers={"gpu": gpu},
        router=tag_router({"gpu": "gpu"}),
        cache_root=tmp_path,
    ) as sub:
        assert default.is_async and not gpu.is_async
        assert sub(Pipeline(x=1)).outputs.out == 4
    # the workflow itself is expanded by Worker.submit, not run()
    assert default.seen == ["prep", "post"]
    assert gpu.seen == ["train"]


def test_mixed_run_with_an_asynchronous_routed_worker(tmp_path):
    default, gpu = RecordingDebug(), RecordingAsync()
    with Submitter(
        worker=default,
        workers={"gpu": gpu},
        router=tag_router({"gpu": "gpu"}),
        cache_root=tmp_path,
    ) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert default.seen == ["main", "prep", "post"]
    assert gpu.seen == ["train"]


@python.define
def Pid(x: int) -> int:
    return os.getpid()


@workflow.define(outputs=["prep", "train", "post"])
def PidPipeline(x: int):
    prep = workflow.add(Pid(x=x), name="prep")
    train = workflow.add(Pid(x=prep.out), name="train", resources=GPU)
    post = workflow.add(Pid(x=train.out), name="post")
    return prep.out, train.out, post.out


def test_mixed_run_on_a_process_pool_and_in_process(tmp_path):
    with Submitter(
        worker="cf",
        workers={"local": "debug"},
        router=tag_router({"gpu": "local"}),
        cache_root=tmp_path,
    ) as sub:
        outputs = sub(PidPipeline(x=0), raise_errors=True).outputs
    # the tagged node ran in this process, the others in pool processes
    assert outputs.train == os.getpid()
    assert outputs.prep != os.getpid()
    assert outputs.post != os.getpid()


@workflow.define(outputs=["out"])
def Inner(x: int):
    a = workflow.add(AddOne(x=x), name="a")
    b = workflow.add(AddOne(x=a.out), name="b")
    return b.out


@workflow.define(outputs=["out"])
def Outer(x: int):
    prep = workflow.add(AddOne(x=x), name="prep")
    inner = workflow.add(Inner(x=prep.out), name="inner", resources=GPU)
    post = workflow.add(AddOne(x=inner.out), name="post")
    return post.out


def test_workflow_node_routed_to_a_sync_worker_keeps_async_children(tmp_path):
    # the router places the leaf jobs; a workflow node is expanded in its parent's loop
    with Submitter(
        worker="cf",
        workers={"local": "debug"},
        router=tag_router({"gpu": "local"}),
        cache_root=tmp_path,
    ) as sub:
        assert sub(Outer(x=1), raise_errors=True).outputs.out == 5


def test_workflow_node_routed_to_an_async_worker_in_a_sync_run(tmp_path):
    with Submitter(
        worker="debug",
        workers={"pool": "cf"},
        router=tag_router({"gpu": "pool"}),
        cache_root=tmp_path,
    ) as sub:
        assert sub(Outer(x=1)).outputs.out == 5


def test_unknown_worker_name_is_an_error(tmp_path):
    with Submitter(
        worker="debug",
        workers={"gpu": "debug"},
        router=lambda job: "tpu",
        cache_root=tmp_path,
    ) as sub:
        with pytest.raises(ValueError, match=r"'tpu'.*\['default', 'gpu'\]"):
            sub.worker_for(SimpleNamespace(name="j", resources=None, task=AddOne(x=1)))


def test_default_name_is_reserved(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        Submitter(workers={"default": "debug"}, cache_root=tmp_path)


def test_router_must_be_callable(tmp_path):
    with pytest.raises(TypeError, match="callable"):
        Submitter(router="gpu", cache_root=tmp_path)


def test_extra_workers_are_built_from_names_and_classes(tmp_path):
    with Submitter(
        worker="debug",
        workers={"a": "debug", "b": debug.Worker},
        cache_root=tmp_path,
    ) as sub:
        assert isinstance(sub.workers["a"], debug.Worker)
        assert isinstance(sub.workers["b"], debug.Worker)
        assert sub.workers["a"] is not sub.workers["b"]
        assert all(w.loop is sub.loop for w in sub.workers.values())


def test_tag_router_matches_the_first_known_tag():
    route = tag_router({"gpu": "g", "big": "b"})
    job = lambda *tags: SimpleNamespace(resources=ComputeResources(tags=tags))  # noqa: E731
    assert route(job("x", "big", "gpu")) == "b"
    assert route(job("gpu")) == "g"
    assert route(job("x")) is None
    assert route(job()) is None
    assert route(SimpleNamespace(resources=None)) is None


def test_default_router_names_the_default_worker():
    assert default_router(SimpleNamespace(resources=GPU)) == "default"


def test_a_workflow_job_is_never_routed(tmp_path):
    default, gpu = RecordingDebug(), RecordingDebug()

    @workflow.define(outputs=["out"], lazy=[])
    def Tagged(x: int):
        return workflow.add(AddOne(x=x), name="only").out

    job = SimpleNamespace(name="w", resources=GPU, task=Tagged(x=1))
    with Submitter(
        worker=default,
        workers={"gpu": gpu},
        router=tag_router({"gpu": "gpu"}),
        cache_root=tmp_path,
    ) as sub:
        assert sub.worker_for(job) is default
        leaf = SimpleNamespace(name="l", resources=GPU, task=AddOne(x=1))
        assert sub.worker_for(leaf) is gpu


def test_top_level_workflow_with_a_routed_tag_is_expanded_by_the_default_worker(
    tmp_path,
):
    default, gpu = RecordingDebug(), RecordingDebug()

    @workflow.define(outputs=["out"])
    def Tagged(x: int):
        prep = workflow.add(AddOne(x=x), name="prep")
        train = workflow.add(AddOne(x=prep.out), name="train", resources=GPU)
        post = workflow.add(AddOne(x=train.out), name="post")
        return post.out

    Tagged._resources = GPU  # as a task base class can carry
    tagged = Tagged(x=1)
    with Submitter(
        worker=default,
        workers={"gpu": gpu},
        router=tag_router({"gpu": "gpu"}),
        cache_root=tmp_path,
    ) as sub:
        assert sub(tagged).outputs.out == 4
    assert default.seen == ["main", "prep", "post"]
    assert gpu.seen == ["train"]


SYNC_STARTED = threading.Event()
ASYNC_STARTED = threading.Event()


class SyncSide(debug.Worker):
    """Finishes only once the asynchronous side has started."""

    _plugin_name = "debug"

    def run(self, job, rerun=False):
        SYNC_STARTED.set()
        # bounded: on failure the async side never started, because the loop is held
        assert ASYNC_STARTED.wait(timeout=20), "event loop was blocked"
        return super().run(job, rerun=rerun)


class AsyncSide(debug.Worker):
    """Finishes only once the synchronous side has started."""

    _plugin_name = "debug"

    async def run(self, job, rerun=False):
        ASYNC_STARTED.set()
        await asyncio.wait_for(self._sync_started(), timeout=20)
        return job.run(rerun=rerun)

    async def _sync_started(self):
        while not SYNC_STARTED.is_set():
            await asyncio.sleep(0)


@workflow.define(outputs=["a", "b"])
def Siblings(x: int, y: int):
    # different inputs, or the two nodes share a checksum and only one is dispatched
    a = workflow.add(AddOne(x=x), name="a", resources=GPU)
    b = workflow.add(AddOne(x=y), name="b")
    return a.out, b.out


def test_a_synchronous_worker_does_not_block_the_event_loop(tmp_path):
    SYNC_STARTED.clear()
    ASYNC_STARTED.clear()
    with Submitter(
        worker=AsyncSide(),
        workers={"local": SyncSide()},
        router=tag_router({"gpu": "local"}),
        cache_root=tmp_path,
    ) as sub:
        outputs = sub(Siblings(x=1, y=5), raise_errors=True).outputs
    assert (outputs.a, outputs.b) == (2, 6)


def test_close_closes_every_worker_once_and_reports_the_first_failure(tmp_path):
    closed = []

    class Closing(debug.Worker):
        _plugin_name = "debug"

        def __init__(self, name, fail=False):
            super().__init__()
            self.name, self.fail = name, fail

        def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(self.name)

    shared = Closing("shared", fail=True)
    sub = Submitter(worker=shared, workers={"b": Closing("b")}, cache_root=tmp_path)
    sub.workers["alias"] = shared
    with pytest.raises(RuntimeError, match="shared"):
        sub.close()
    assert sorted(closed) == ["b", "shared"]
