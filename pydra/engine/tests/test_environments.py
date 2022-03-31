from ..environments import *


def test_native_pwd(tmpdir):
    """simple command, no arguments"""
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd, cache_dir=tmpdir)
    assert shelly.cmdline == " ".join(cmd)

    env_res = Native().execute(shelly)
    shelly()
    assert env_res == shelly.output_
