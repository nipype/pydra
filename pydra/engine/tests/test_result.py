from pydra.engine.result import Result, Runtime


def test_runtime():
    runtime = Runtime()
    assert hasattr(runtime, "rss_peak_gb")
    assert hasattr(runtime, "vms_peak_gb")
    assert hasattr(runtime, "cpu_peak_percent")


def test_result(tmp_path):
    result = Result(output_dir=tmp_path)
    assert hasattr(result, "runtime")
    assert hasattr(result, "outputs")
    assert hasattr(result, "errored")
    assert getattr(result, "errored") is False
