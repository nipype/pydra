import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

SCENARIO = textwrap.dedent("""
    import json, subprocess, sys
    from pydra.conftest import kill_descendants
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
    starter = (
        "import subprocess, sys; "
        f"g = subprocess.Popen({sleeper!r}); "
        "print(g.pid, flush=True); "
        "import time; time.sleep(600)"
    )
    child = subprocess.Popen([sys.executable, "-c", starter], stdout=subprocess.PIPE, text=True)
    grandchild = int(child.stdout.readline())
    killed = kill_descendants()
    child.wait()
    print(json.dumps({"child": child.pid, "grandchild": grandchild, "killed": killed}))
""")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process tree")
def test_a_timed_out_test_takes_down_the_processes_it_started():
    """What the timeout guard runs before ``os._exit``, run in a process of its own: the
    child and the grandchild it started (a pool's forkserver and a worker) are killed"""
    out = subprocess.run(
        [sys.executable, "-c", SCENARIO], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    pids = json.loads(out.stdout.splitlines()[-1])
    assert {pids["child"], pids["grandchild"]} <= set(pids["killed"])
    deadline = time.monotonic() + 60  # a bound only: the grandchild is reaped by init
    while _alive(pids["grandchild"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pids["grandchild"])
