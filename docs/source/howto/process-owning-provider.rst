Writing a process-owning provider
=================================

A *provider* loads resident state that a warm worker keeps between jobs (see
:doc:`gpu-model-on-slurm`). Most providers return an object that lives in the worker's own
process. Some state cannot: a model that needs another Python, other library versions or its own
locked environment, or a server written in another language. A *process-owning provider* starts
that state as a child process and returns a client of it. This page shows how to write one.


The base class
--------------

Subclass :class:`~pydra.engine.procgroup.ProcessProvider`, set ``name``, and implement ``argv``;
override ``connect`` to build the client, and ``version`` to name the loader's code:

.. code-block:: python

    import os
    import socket
    import tempfile
    import time

    from pydra.engine.procgroup import ProcessProvider
    from pydra.workers.pool.faults import WorkerLost


    class Client:
        def __init__(self, path):
            self.path = path

        def __call__(self, text):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(self.path)
                    s.sendall(text.encode() + b"\n")
                    return s.makefile().readline().strip()
            except OSError as err:
                # the server is gone: a crash of the worker's state, not the job's fault
                raise WorkerLost(f"the model server stopped: {err}") from err


    class ServedModel(ProcessProvider):
        name = "served-model"

        def load(self, spec, device):
            self._socket = os.path.join(tempfile.mkdtemp(prefix="served-"), "s")
            return super().load(spec, device)

        def argv(self, spec, device):
            # the server runs in its own environment, with its own interpreter
            return [spec["python"], "-m", "my_server", "--model", spec["model"],
                    "--socket", self._socket]

        def connect(self, group, spec):
            while not os.path.exists(self._socket):
                if not group.leader_running():
                    raise RuntimeError("the model server exited while starting")
                time.sleep(0.1)
            return Client(self._socket)

        def version(self):
            return "served-model 1"

``load(spec, device)`` (inherited) starts ``argv`` with ``env(spec, device)`` as the leader of a
new process group, then calls ``connect(group, spec)``; a ``connect`` that raises stops the group.
It returns a :class:`~pydra.engine.procgroup.ProcessHandle` holding the group and the client. The
task receives that handle and calls the client through it:

.. code-block:: python

    from pydra.compose import python
    from pydra.engine.resident import Provided


    @python.define
    def Summarise(model: Provided[object, "served-model"], text: str) -> str:
        return model.client(text)

Pass the provider to the ``Submitter`` (``providers=[ServedModel()]``), or register it under the
``pydra.providers`` entry point group so that every worker finds it by name. Like any provider, it
must be importable where the workers run.


The device
----------

``device`` is where the worker was told to load: ``Device("cuda", index)`` on a GPU worker. The
default ``env`` passes the parent's environment with ``CUDA_VISIBLE_DEVICES`` narrowed to that one
device (the index is read relative to the worker's own ``CUDA_VISIBLE_DEVICES``), so the child
sees a single GPU as device 0. Override ``env`` to add variables, and return ``None`` to inherit the
environment unchanged.


What ``close`` guarantees
-------------------------

``close(handle)`` stops the child's whole group and every descendant, including one that left the
group with ``setsid``: SIGTERM, a wait of ``term_timeout`` seconds, then SIGKILL and a wait of
``kill_timeout``. It then waits up to ``gpu_settle_timeout`` seconds until ``nvidia-smi`` lists no
device memory for any of those processes, so the next load does not race the dying one for memory.
Errors are logged, not raised. The three timeouts are arguments of the constructor (defaults 10, 10
and 30 seconds).

On POSIX the group's leader is a small shim that starts the real command. It holds a pipe from the
worker: if the worker dies without closing (SIGKILL, an out-of-memory kill), the pipe reaches end of
file and the shim stops the whole tree. On Linux the shim is also a child subreaper, so a server
that daemonised itself is still stopped with the rest.

A worker closes its resident state when it evicts it to make room for another model, when it exits
(idle, drained after SIGTERM, or recycled), and after a crash of the state.

**Under Slurm, a cancel reaches the child too.** ``scancel``, a pool's stop and a job's time limit
send SIGTERM to every process of the worker's job at once, the provider's child included (and the
shim relays it to its tree). The worker's drain cannot keep the server alive for the call in
flight: that call fails, as a ``WorkerLost`` when the client raises it (a crash, charged to the job
within ``crashes_charged``), and the job runs again elsewhere. A server that must finish its
request on SIGTERM has to handle the signal itself.


Faults
------

The worker classifies what a call raises:

* An out-of-memory error (``MemoryError``, ``torch.OutOfMemoryError``,
  ``torch.cuda.OutOfMemoryError``, anywhere in the exception's chain) is a resource fault: the
  state is reloaded and a lone job that still runs out moves up its ladder of classes.
* :class:`~pydra.workers.pool.faults.WorkerLost` in the chain is a crash of the state: the
  residency is reloaded and the job retried, charged against ``FaultBudgets.crashes_charged``. Raise
  it (or raise from it) in the client when the child has ended or answered out of protocol.
* Anything else raised by the task is the job's own failure.

A failure inside ``load`` is never charged to a job: it counts as an infrastructure fault of the
residency, and after ``FaultBudgets.infrastructure_replacements`` such faults in one coordinator
epoch the residency is held on its queue and its jobs are reported not run.

``measure(handle)`` reports the device bytes held by the group's processes (by pid, through
``nvidia-smi``); the worker also records each load's growth in host and device memory in the
pool's calibration, attributed to the worker's own processes and its providers' groups.


Checking a provider
-------------------

Run it once outside a pool, then on a pool with one worker, and check what is left after the
worker exits:

.. code-block:: python

    from pydra.engine.resident import Device

    provider = ServedModel()
    handle = provider.load({"python": "/envs/served/bin/python", "model": "m"}, Device("cuda", 0))
    try:
        print(handle.client("hello"))
    finally:
        provider.close(handle)
    assert handle.group.members() == []

``pydra/workers/tests/test_pool_cluster.py::test_process_provider_warm_gpu`` does the same on a
cluster: one child start for many jobs, and no process and no device memory left once the worker
has closed it.
