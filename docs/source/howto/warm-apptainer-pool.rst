Shell workflow on a warm Apptainer pool
=======================================

A shell task in a container pays for starting the container. Under the plain workers each job runs
``apptainer exec`` (or ``docker run``) of its image, which for a short command costs more than the
command. A *warm pool* (:doc:`gpu-model-on-slurm`) starts a few long-running workers instead; a
worker that runs shell jobs of an environment declared ``warm=True`` starts one Apptainer instance
(or Docker container) and runs job after job in it with ``apptainer exec instance://...``. An
``Lmod`` environment declared warm is resolved once per worker rather than once per command.

Warmth is execution detail. A job run in a warm instance has the checksum, the context and the
stored result of the same job run cold under ``worker="debug"``: a later run of either kind reuses
it.


Declare the environment warm
----------------------------

``warm=True`` is your statement that the tool keeps no state between invocations inside the
container (no licence daemon, no cache written into the image's own paths), so one started instance
may serve many jobs. It never changes a checksum.

.. code-block:: python

    from pydra.compose import shell, workflow
    from pydra.environments.singularity import Singularity

    tool = Singularity(image="docker://<registry>/<tool>", tag="<version>", warm=True)

    Convert = shell.define("convert-tool <in_file> <out|out_file>")


    @workflow.define
    def ConvertAll(in_files: list) -> list:
        node = Convert().split(in_file=in_files)
        return workflow.add(node, name="convert", environment=tool).out

The submitting process resolves the image once (a ``docker://`` reference to its digest, a ``.sif``
to the hash of the file) and every job carries the resolved reference; a worker runs exactly that
reference, from a local SIF, and never resolves or pulls again. The image must contain ``env``,
which sets each job's ``HOME`` and ``TMPDIR`` inside the instance.

``Docker(..., warm=True)`` and ``Lmod(modules=[...], warm=True)`` are declared the same way.


Describe the workers and run
----------------------------

Shell jobs need no GPU; a CPU class is enough. ``mounts`` names directories every warm instance
binds besides the ones the jobs need (a reference data tree, say):

.. code-block:: python

    from pydra.engine.submitter import Submitter
    from pydra.workers.pool.classes import WorkerClass
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import SlurmSubstrate

    cpu = WorkerClass(
        name="cpu",
        device="cpu",
        cpus=2,
        mem="8G",
        time=4 * 3600,
        max_workers=8,
        mounts=["/shared/reference"],
        substrate_args={"partition": "<cpu partition>"},
    )
    pool = WarmPool("/shared/user/pool", [cpu], SlurmSubstrate(),
                    store=Submitter(cache_root="/shared/user/cache").store)
    try:
        with Submitter(worker=pool) as sub:
            result = sub(ConvertAll(in_files=files))
    finally:
        pool.close()

Each worker's job script must find ``apptainer`` on its ``PATH`` (``module load apptainer``, for
instance, in the environment the substrate starts workers from). ``LocalSubstrate`` runs the same
workers on the current host.


What a worker does with a warm environment
------------------------------------------

* It starts the instance on the first ticket of shell jobs in that environment, bound to the union
  of the directories the ticket's jobs read and write, reduced to top-level roots, plus the class's
  ``mounts``. Every root is bound read-only under ``<root>-ro`` and the writable ones (the cache
  and work directories) also writable, as a cold run binds them, so a job that edits an input in
  place fails warm as it fails cold.
* A later ticket that needs a root the instance lacks restarts it once, bound to the union of the
  old roots and the new ones. A job that would need more than the environment's
  ``max_mount_roots`` roots (16 by default) runs cold, with ``apptainer exec`` of the image, in the
  same worker. A job keyed by another reference of the same environment also runs cold.
* Each job gets its own ``HOME``, removed when it ends, and the instance's ``/tmp`` is emptied after
  each job, so no job sees what another wrote there. ``max_jobs=<n>`` on the environment restarts
  the instance after ``n`` jobs, for a tool known to leak.
* The instance is stopped when the worker exits: idle, recycled, or on ``SIGTERM`` after its drain.
  A command still running when a claim is lost or the drain ends is stopped with its instance; the
  next job starts a fresh one. An instance a killed worker left behind on a host is removed by the
  next warm start there, never one of another user or another process that is still alive.
* A failed start does not fail any job: the environment's jobs run cold in that worker.

Each job's ``pool`` facts (``Result.runtime.pool``, and the execute context under a store that
records them) carry an ``environment`` entry: ``run`` (``warm`` or ``cold``), ``state`` (``cold``
for the first job of a start), ``since_load`` (jobs the instance ran before this one) and
``load_s`` (seconds the start took). These are recorded only; they never select a slot.


Checking it works
-----------------

Run a few jobs on the pool and look at the facts of their results, or at the worker's log under
``<root>/private/logs/<token>.log``: a warm worker logs one ``warm-start`` event per instance start,
not one per job. ``python benchmarks/bench_warm_environment.py --runtime apptainer`` measures, on
the host it runs on, ``apptainer exec`` of an image against ``exec`` in an instance.
