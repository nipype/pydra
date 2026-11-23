GPU model on Slurm
==================

A task that uses a model pays for loading it. Under the plain ``slurm`` worker each job is its own
``sbatch`` and loads the model again. A *warm pool* starts a few long-running workers instead, one
Slurm job each, which load the model once and then run job after job. This page shows how to run a
workflow of such tasks on GPUs of a Slurm cluster.

The pool lives in ``pydra.workers.pool``; ``import pydra`` does not import it, and a workflow that
does not use it is unchanged.


Declare the model as resident state
-----------------------------------

A field annotated ``Provided[T, name]`` holds a *spec*: a small JSON-able value that names the
model (a path, an identifier and revision). The spec is an input like any other and is part of the
job's checksum. What the task body receives is the object the *provider* named ``name`` loaded from
that spec, which a worker keeps between jobs.

.. code-block:: python

    from pydra.compose import python, workflow
    from pydra.engine.resident import Provided
    from pydra.utils.resources import ComputeResources


    class Embedder:
        """Loads a model from a spec; one instance per process, passed to the Submitter."""

        name = "embedder"

        def load(self, spec, device):
            import torch

            model = torch.load(spec["path"], weights_only=False)
            return model.to(torch.device("cuda", device.index or 0)).eval()

        def close(self, model):
            import torch

            del model
            torch.cuda.empty_cache()

        def footprint(self, spec):
            return None  # declared bytes, when known

        def measure(self, model):
            return None  # bytes held now, when known

        def version(self):
            return "1"  # the loader's code: change it when loading changes


    @python.define
    def Embed(model: Provided[object, "embedder"], text: str) -> list:
        return model.encode(text)


    @workflow.define
    def EmbedAll(spec: dict, texts: list) -> list:
        need = ComputeResources(gpus=1, gpu_mem="20G")
        node = Embed(model=spec).split(text=texts)
        return workflow.add(node, name="embed", resources=need).out

The provider must be importable by the workers (define it in a module on their ``PYTHONPATH``,
not in a notebook cell), and ``version()`` must return the same string wherever it runs: placement
decisions recorded for a residency bind only the code that recorded them.


Describe the workers
--------------------

A :class:`~pydra.workers.pool.classes.WorkerClass` is a kind of worker the substrate can start.
For Slurm, a ``cuda`` class becomes ``sbatch --gres=gpu:<gpu_type>:<gpus>``, plus
``--cpus-per-task``, ``--mem`` and ``--time`` from its fields, plus its ``substrate_args`` (each
``key: value`` as ``--key=value``):

.. code-block:: python

    from pydra.workers.pool.classes import WorkerClass

    h100 = WorkerClass(
        name="h100",
        device="cuda",
        gpus=1,
        gpu_type="h100",      # always name the type: the grant is checked against it
        cpus=8,
        mem="64G",
        time=4 * 3600,        # seconds
        max_workers=4,
        substrate_args={"partition": "<gpu partition>", "account": "<account>"},
    )

``SlurmSubstrate`` passes a ``--qos`` only when a class's ``substrate_args`` set ``qos`` (a site
that requires one names it there). A class that sets an option the substrate sets itself (``gres``, ``cpus-per-task``, ``mem``, ``time``, ``job-name``,
``output`` and the others listed in its docstring) cannot start. Slurm gives each worker its GPUs
and sets ``CUDA_VISIBLE_DEVICES``; a worker that finds hardware other than its class names (an
``a100`` class granted an H100) refuses to start. A class is broken, and its jobs are reported not
run (held, never failed) with the latest start error, after ``FaultBudgets.start_failures`` failed
starts, or at a failed start once ``PoolSettings.start_deadline_failures`` starts have failed in a
row over ``PoolSettings.start_deadline_s`` with no worker of the class ready or starting, whichever
comes first. ``WarmPool.release_hold``, or a ``retry`` of a held job, gives the class a fresh
attempt.

Classes may also be read from a TOML, JSON or YAML file with
:meth:`~pydra.workers.pool.classes.WorkerClass.load`.


Run the workflow
----------------

The pool directory, the store and the module that defines the tasks and the provider must be on
storage that every node sees; ``SlurmSubstrate`` refuses a pool under ``/tmp``, ``/var/tmp``,
``/dev/shm`` or ``$TMPDIR``, or on a memory filesystem, when the pool is opened. Start the run from
a node that can call ``sbatch`` (a compute node of an allocation, not a login node):

.. code-block:: python

    from pydra.engine.lease import LeaseSettings
    from pydra.engine.submitter import Submitter
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.settings import PoolSettings
    from pydra.workers.pool.substrate import SlurmSubstrate

    store = Submitter(cache_root="/shared/user/cache").store
    settings = PoolSettings(
        lease=LeaseSettings(stale_after=300, attribute_delay=60),  # NFS: acdirmax 60 s
        poll_s=30,
    )
    pool = WarmPool("/shared/user/pool", [h100], SlurmSubstrate(), store=store,
                    settings=settings)
    try:
        with Submitter(worker=pool, providers=[Embedder()]) as sub:
            result = sub(EmbedAll(spec={"path": "/shared/models/m.pt"}, texts=texts))
    finally:
        pool.close()

Each job is placed on a class at submission, written into a queue of the pool as part of a ticket,
claimed by a worker and run there; its result is published to the store and read back by the
submitting process. ``pool.close()`` cancels the workers' Slurm jobs. A worker also exits by itself
after ``idle_polls`` polls with nothing to claim, and after ``orphan_after_s`` with no live
coordinator.


Who closes the pool
-------------------

Whoever makes the pool closes it:

* A pool you construct and pass as ``Submitter(worker=pool)`` is yours. Neither a run nor the
  ``Submitter``'s own close closes it, so its workers stay resident, with their models loaded, from
  one ``sub(task)`` to the next. Close it yourself (``pool.close()``, as above) when you are done.
* A pool the ``Submitter`` makes by name belongs to that ``Submitter`` and is closed when it
  closes (the end of the ``with`` block), never at the end of a single run:

  .. code-block:: python

      with Submitter(worker="pool", root="/shared/user/pool", classes=[h100],
                     substrate=SlurmSubstrate(), settings=settings,
                     providers=[Embedder()]) as sub:
          first = sub(EmbedAll(spec=spec, texts=batch_one))
          second = sub(EmbedAll(spec=spec, texts=batch_two))   # the same warm workers

  ``substrate`` defaults to ``LocalSubstrate()`` (workers on this host).

``drive`` and ``drive_inline``, for callers that drive a runner themselves, never close the
executors they are handed.


Several runs on one pool
------------------------

Runs may share one pool, one after another or at once: two ``Submitter`` objects on two threads,
or two ``drive`` calls on one event loop. Each job's outcome goes to the run that submitted it. A
job that another run already has open on the pool is not put again: it waits on the other run's
ticket and receives the same outcome. A run that is cancelled withdraws only its own jobs. All runs
share the pool's workers and the models they hold.

Only one coordinator may hold a pool directory at a time: a second ``WarmPool`` on the same
directory waits until the first is closed or its lease goes stale (``PoolSettings.coordinator_wait``
bounds the wait), then takes the next *epoch*. A coordinator that was killed is replaced this way,
from any host: the new one follows the jobs the old one had already put, reads the finished ones
from the store, and adopts its still-running Slurm workers.

A resumed run also waits out the *store's* leases that the killed process held: the lease of each
workflow job it was running is taken over only once it is ``stale_after`` old on the store's own
settings, not the pool's. With ``PortableStore`` (the default) that is the lease default, 300 s,
plus the attribute delay. ``PickleStore`` takes ``lease_settings``; give it the pool's to make both
waits the same:

.. code-block:: python

    from pydra.engine.store import PickleStore

    lease = LeaseSettings(stale_after=300, attribute_delay=60)
    store = PickleStore("/shared/user/cache", lease_settings=lease)
    pool = WarmPool("/shared/user/pool", [h100], SlurmSubstrate(), store=store,
                    settings=PoolSettings(lease=lease, poll_s=30))


Settings that matter on a cluster
---------------------------------

``LeaseSettings.attribute_delay``
    How late a client may see a directory change on the shared storage: 60 s (the NFS default
    ``acdirmax``) unless you know your mount. The pool probes the storage when it is created and
    refuses a root whose measured delay is larger. ``stale_after`` must exceed two renewals plus
    this delay; a worker that dies is taken over ``stale_after`` after its last heartbeat.
``PoolSettings.poll_s``
    How often the coordinator lists the queues and the Slurm queue (one ``squeue`` per poll for the
    whole process).
``WorkerClass.quantum_jobs`` and ``quantum_s``
    How long a worker serving several models stays on one before yielding to the queue whose work
    has waited longest.
``SlurmSubstrate(per_user_cpu_cap=...)``
    The most CPUs the pool's live workers may hold together.

The pool records how much memory each model's load and calls take, per GPU type, in
``<pool>/calibration``. Pass ``fit=MeasuredFit(DirectoryCalibration(<pool>/"calibration"), headroom=<fraction>)``
(the share of a device's memory the site lets a worker plan for) to
place jobs from those measurements: a load that ran out of memory on one class is recorded, its
jobs move to the next class of their ladder without being charged, and later placements skip the
class that was too small.


Inspecting a pool
-----------------

``pydracli pool status <root>`` summarises queues, workers and holds; ``pydracli pool release-hold``
lifts a hold placed after repeated infrastructure faults; ``pydracli pool collect`` removes resolved
tickets and old payloads. Worker logs are under ``<root>/private/logs/<token>.log``. The layout and
the protocol are described in :doc:`../reference/worker-protocol`.
