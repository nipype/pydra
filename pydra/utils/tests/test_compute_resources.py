import pytest
from pydra.utils.resources import ComputeResources


def test_defaults_request_nothing():
    res = ComputeResources()
    assert res.slurm_args() == []
    assert res.sge_args() == []
    assert res.tags == ()


def test_mem_normalised():
    assert ComputeResources(mem=512).mem == "512M"
    assert ComputeResources(mem="4g").mem == "4G"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpus": 0},
        {"gpus": -1},
        {"time": 0},
        {"mem": 0},
        {"mem": "4"},
        {"mem": "lots"},
        {"gpu_type": "a100"},
    ],
)
def test_invalid_rejected(kwargs):
    with pytest.raises(ValueError):
        ComputeResources(**kwargs)


def test_tags_become_a_tuple():
    assert ComputeResources(tags=["gpu", "big"]).tags == ("gpu", "big")
    assert ComputeResources(tags="gpu").tags == ("gpu",)


def test_slurm_args():
    res = ComputeResources(cpus=4, mem="8G", gpus=2, gpu_type="a100", time=90)
    assert res.slurm_args() == [
        "--cpus-per-task=4",
        "--mem=8G",
        "--gres=gpu:a100:2",
        "--time=0-00:01:30",
    ]


def test_slurm_gpus_without_type_and_long_time():
    res = ComputeResources(gpus=1, time=2 * 86400 + 3600)
    assert res.slurm_args() == ["--gres=gpu:1", "--time=2-01:00:00"]


def test_sge_args():
    res = ComputeResources(cpus=4, mem="8G", gpus=2, gpu_type="a100", time=3725)
    assert res.sge_args() == [
        "-pe",
        "smp",
        "4",
        "-l",
        "mem_free=8G",
        "-l",
        "gpu=2",
        "-l",
        "gpu_type=a100",
        "-l",
        "h_rt=01:02:05",
    ]


def test_tags_are_not_scheduler_arguments():
    res = ComputeResources(tags=("gpu",))
    assert res.slurm_args() == []
    assert res.sge_args() == []
