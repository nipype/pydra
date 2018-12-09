from ..specs import BaseSpec, RuntimeSpec, Runtime, Result


def test_basespec():
    spec = BaseSpec()
    assert spec.hash == '01c12e5004b5e311e6f37bc758727644c08a719e46ab794eba312338e1d38ab0'


def test_runtime():
    runtime = Runtime()
    assert hasattr(runtime, 'rss_peak_gb')
    assert hasattr(runtime, 'vms_peak_gb')
    assert hasattr(runtime, 'cpu_peak_percent')


def test_result():
    result = Result()
    assert hasattr(result, 'runtime')
    assert hasattr(result, 'output')
