"""RFC 0005 W1: worker classes, placement and DeclaredFit."""

import subprocess
import sys
import pytest
from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.utils.resources import ComputeResources
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.placement import (
    ClassTried,
    DeclaredFit,
    Placement,
    Unplaceable,
    place,
)

G = 1024**3


@python.define
def AddOne(x: int) -> int:
    return x + 1


@pytest.fixture
def make_job(tmp_path):
    with Submitter(cache_root=tmp_path) as sub:

        def make(**resources):
            return Job(
                AddOne(x=1),
                submitter=sub,
                name="j",
                resources=ComputeResources(**resources),
            )

        yield make


def ok(cls):
    return True


A40 = WorkerClass(
    name="a100-40", device="cuda", gpus=1, gpu_type="a100", gpu_mem=40 * G
)
A80 = WorkerClass(
    name="a100-80", device="cuda", gpus=1, gpu_type="a100", gpu_mem=80 * G, mem=256 * G
)
H100 = WorkerClass(
    name="h100", device="cuda", gpus=1, gpu_type="h100", gpu_mem=80 * G, mem=512 * G
)
CPU = WorkerClass(name="cpu", device="cpu", cpus=16, mem=64 * G, time=3600)


def test_oversized_job_placed_on_larger_class(make_job):
    job = make_job(gpus=1, gpu_mem="60G")
    got = place(job, [A80, A40], DeclaredFit(), ok)
    assert isinstance(got, Placement)
    assert got.worker_class == "a100-80"
    assert got.queue == "a100-80"
    assert got.batch == 1
    assert [t.verdict for t in got.tried] == ["too small", "fits"]
    assert "a100-40" == got.tried[0].worker_class


def test_ladder_crosses_gpu_type_when_unnamed(make_job):
    got = place(make_job(gpus=1, gpu_mem="30G"), [H100, A80, A40], DeclaredFit(), ok)
    assert got.worker_class == "a100-40"
    # smallest device memory first, then host memory: the 80G A100 before the H100
    got = place(make_job(gpus=1, gpu_mem="70G"), [H100, A80, A40], DeclaredFit(), ok)
    assert got.worker_class == "a100-80"
    assert [t.worker_class for t in got.tried] == ["a100-40", "a100-80"]


def test_ladder_stays_on_named_type(make_job):
    job = make_job(gpus=1, gpu_type="h100", gpu_mem="30G")
    got = place(job, [A40, A80, H100], DeclaredFit(), ok)
    assert got.worker_class == "h100"
    assert all(t.worker_class != "a100-40" for t in got.tried if t.verdict)
    got = place(
        make_job(gpus=1, gpu_type="a100", gpu_mem="90G"),
        [A40, A80, H100],
        DeclaredFit(),
        ok,
    )
    assert isinstance(got, Unplaceable)


def test_time_exceeds_class_is_unfit(make_job):
    job = make_job(time=7200)
    got = place(job, [CPU], DeclaredFit(), ok)
    assert isinstance(got, Unplaceable)
    (t,) = got.tried
    assert t.verdict == "unfit"
    assert "7200" in t.reason and "3600" in t.reason
    longer = WorkerClass(name="long", device="cpu", time=86400)
    got = place(job, [CPU, longer], DeclaredFit(), ok)
    assert got.worker_class == "long"
    assert got.tried[0].verdict == "unfit"


def test_gpu_mem_none_places_as_unknown(make_job):
    unk = WorkerClass(name="unk", device="cuda", gpus=1)
    # a job that declares no gpu_mem
    got = place(make_job(gpus=1), [A40, unk], DeclaredFit(), ok)
    assert got.worker_class == "a100-40"
    assert got.tried[-1].verdict == "unknown"
    # a class with unknown device memory sorts last and places as unknown
    got = place(make_job(gpus=1, gpu_mem="100G"), [unk, A40, A80], DeclaredFit(), ok)
    assert got.worker_class == "unk"
    assert [t.worker_class for t in got.tried] == ["a100-40", "a100-80", "unk"]
    assert got.tried[-1].verdict == "unknown"


def test_fits_nothing_unplaceable_names_every_class(make_job):
    got = place(
        make_job(gpus=1, gpu_mem="200G"), [CPU, A40, A80, H100], DeclaredFit(), ok
    )
    assert isinstance(got, Unplaceable)
    assert {t.worker_class for t in got.tried} == {"cpu", "a100-40", "a100-80", "h100"}
    msg = got.message
    for name in ("cpu", "a100-40", "a100-80", "h100"):
        assert name in msg
    assert str(40 * G) in msg and str(80 * G) in msg and str(200 * G) in msg


def test_gpu_request_never_on_cpu(make_job):
    got = place(make_job(gpus=1), [CPU], DeclaredFit(), ok)
    assert isinstance(got, Unplaceable)
    assert place(make_job(cpus=2), [CPU], DeclaredFit(), ok).worker_class == "cpu"


def test_can_start_refusal_skipped_and_recorded(make_job):
    def can_start(cls):
        return "partition cap reached" if cls.name == "a100-40" else True

    got = place(make_job(gpus=1, gpu_mem="10G"), [A40, A80], DeclaredFit(), can_start)
    assert got.worker_class == "a100-80"
    skipped = got.tried[0]
    assert skipped == ClassTried("a100-40", None, "cannot start: partition cap reached")
    refuse_all = place(make_job(gpus=1), [A40], DeclaredFit(), lambda c: False)
    assert isinstance(refuse_all, Unplaceable)


def test_placement_leaves_checksum_unchanged(make_job):
    plain = make_job()
    placed = make_job(gpus=1, gpu_mem="10G", residency="ignore", time=60)
    before = (plain.checksum, placed.checksum)
    place(placed, [A40], DeclaredFit(), ok)
    assert (plain.checksum, placed.checksum) == before
    assert plain.checksum == placed.checksum


def test_compute_resources_new_fields():
    r = ComputeResources(gpus=1, gpu_mem=40960)
    assert r.gpu_mem == "40960M"
    assert ComputeResources().gpu_mem is None
    assert ComputeResources().residency == "route"
    with pytest.raises(ValueError):
        ComputeResources(residency="sometimes")
    assert "gpu_mem" not in " ".join(
        ComputeResources(gpus=1, gpu_mem="40G").slurm_args()
    )


def test_worker_class_load(tmp_path):
    data = {
        "gpu-h100": {"device": "cuda", "gpus": 1, "gpu_type": "h100", "gpu_mem": "80G"},
        "cpu-large": {"device": "cpu", "cpus": 32, "mounts": ["/data"]},
    }
    classes = WorkerClass.load(data)
    assert [c.name for c in classes] == ["gpu-h100", "cpu-large"]
    assert classes[0].gpu_mem == 80 * G
    assert classes[0].quantum_jobs == 64 and classes[0].quantum_s == 300.0
    f = tmp_path / "classes.yaml"
    f.write_text(
        "classes:\n  - {name: a, device: cpu}\n  - {name: b, device: cuda, gpus: 2, workers_per_device: 2}\n"
    )
    assert [c.name for c in WorkerClass.load(f)] == ["a", "b"]
    assert WorkerClass.load([{"name": "z", "device": "cpu"}])[0].name == "z"
    with pytest.raises(TypeError):
        WorkerClass.load({"x": {"device": "cpu", "bogus": 1}})
    with pytest.raises(ValueError):
        WorkerClass.load({"x": {"device": "cpu", "gpus": 1}})
    with pytest.raises(ValueError):
        WorkerClass.load(
            [{"name": "a", "device": "cpu"}, {"name": "a", "device": "cpu"}]
        )


def test_workers_per_device_divides_device_memory(make_job):
    shared = WorkerClass(
        name="s", device="cuda", gpus=1, gpu_mem=80 * G, workers_per_device=4
    )
    got = place(make_job(gpus=1, gpu_mem="30G"), [shared], DeclaredFit(), ok)
    assert isinstance(got, Unplaceable)
    assert (
        place(make_job(gpus=1, gpu_mem="20G"), [shared], DeclaredFit(), ok).worker_class
        == "s"
    )


def test_pool_not_imported_by_import_pydra():
    code = (
        "import sys, pydra, pydra.engine, pydra.workers.cf\n"
        "assert not [m for m in sys.modules if m.startswith('pydra.workers.pool')], "
        "[m for m in sys.modules if m.startswith('pydra.workers.pool')]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tags_filter_classes_by_overlap(make_job):
    inter = WorkerClass(name="inter", device="cpu", tags=["interactive"])
    batch = WorkerClass(name="batch", device="cpu", mem=64 * G)
    # a job with no tags goes to any class; the smaller-memory-known class sorts first
    assert place(make_job(), [inter, batch], DeclaredFit(), ok).worker_class == "batch"
    assert place(make_job(), [inter], DeclaredFit(), ok).worker_class == "inter"
    # a tagged job skips a tagged class it does not overlap, and takes an untagged class
    got = place(make_job(tags=("other",)), [inter, batch], DeclaredFit(), ok)
    assert got.worker_class == "batch"
    only = place(make_job(tags=("other",)), [inter], DeclaredFit(), ok)
    assert isinstance(only, Unplaceable)
    assert "interactive" in only.tried[0].reason and "other" in only.tried[0].reason
    assert (
        place(
            make_job(tags=("interactive", "x")), [inter], DeclaredFit(), ok
        ).worker_class
        == "inter"
    )
    assert WorkerClass(name="t", device="cpu", tags="a").tags == frozenset({"a"})


def test_gpu_mem_requires_gpus():
    with pytest.raises(ValueError, match="gpu_mem requires gpus"):
        ComputeResources(gpu_mem="40G")


def test_load_toml_and_json(tmp_path):
    t = tmp_path / "c.toml"
    t.write_text('[gpu]\ndevice = "cuda"\ngpus = 1\ngpu_mem = "40G"\n')
    assert WorkerClass.load(t)[0].gpu_mem == 40 * G
    j = tmp_path / "c.json"
    j.write_text('{"classes": [{"name": "a", "device": "cpu"}]}')
    assert WorkerClass.load(j)[0].name == "a"


def test_load_yaml_without_pyyaml_names_the_package(tmp_path, monkeypatch):
    f = tmp_path / "c.yaml"
    f.write_text("a: {device: cpu}\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="pyyaml"):
        WorkerClass.load(f)


def test_named_worker_does_not_import_pool(tmp_path):
    code = (
        "import sys, tempfile\n"
        "from pydra.compose import python\n"
        "from pydra.engine.submitter import Submitter\n"
        "@python.define\n"
        "def F(x: int) -> int:\n    return x + 1\n"
        "with Submitter(worker='cf', cache_root=tempfile.mkdtemp()) as s:\n"
        "    assert s(F(x=1)).outputs.out == 2\n"
        "assert not [m for m in sys.modules if m.startswith('pydra.workers.pool')]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
