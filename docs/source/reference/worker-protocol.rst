Worker protocol
===============

A warm pool (``pydra.workers.pool``, RFC 0005) is a directory on shared storage. The submitting
process (the *coordinator*) and the workers never talk to each other directly: every exchange is a
file or a directory in the pool, and every exclusion is a
:class:`~pydra.engine.lease.DirectoryLease` judged on the storage's clock. This page is the
reference for that protocol: the layout, the records, and the rules each party follows. The
implementation is ``pydra.workers.pool.storage`` (the layout), ``pydra.workers.pool.pool`` (the
coordinator) and ``pydra.workers.pool.worker`` (the worker).


Layout
------

::

    <root>/                                   0700, owned by the pool's user
      pool.json                               "pool@1": pool id, the four LeaseSettings, codec,
                                              owner uid, segment size, probe results
      coordinator/<epoch>/                    one per coordinator, taken by mkdir:
                                              lease/, ended, faults.json
      classes/<name>.json                     a WorkerClass, as canonical JSON
      queues/<label>/                         label = sha256 of the queue name
        name                                  the queue name, written once
        tickets/<digest>.json                 a ticket; digest = sha256 of its canonical bytes
        segments/<seq>/slot-<n>               the index: a file holding a ticket digest
        segments/<seq>/jobs-<n>-<count>       the job count of slot n's ticket
        segments/<seq>/res-<n>                slot n's ticket is resolved
        segments/<seq>/retired                every slot of the segment is taken and resolved
        slotted/<digest>                      the ticket's first slot, "<seq>/<n>"
        claims/<digest>.lease/                the claim's DirectoryLease
        losses/<digest>/<n>                   a takeover of the ticket
        outcome/<digest>/<job>-<attempt>.json one per job and attempt; the first writer wins
        settled/<digest>-<count>              the ticket's outcome count reached <count>
        held-<epoch>-<label>                  a hold of a residency on this queue
        faults/<epoch>-<label>/<n>            an infrastructure fault of a residency in an epoch
      follow/<job>/<g>                        generation g of a job: the ticket that holds it
      follow/<job>/<g>.started                attempt <pool id>.<g> was started, and by whom
      workers/<token>/                        lease/, ready.json, samples.jsonl, calls/
      payloads/<sha256>                       a job's payload, read by every attempt of the job
      calibration/                            measurements (the default DirectoryCalibration)
      private/                                settings, Slurm logs; 0700

``<job>`` is ``<checksum>-<fingerprint>`` when that is a plain lower-case name of at most 200
characters, else ``h-<sha256>`` of the two. A queue is named ``<class>``, or
``<class>#<residency digest>`` when the jobs' resident state is routed (the default), so that the
jobs of one model share a queue.

**Every file is created once and never modified.** It is staged under a private name, flushed,
linked to its name and its directory flushed. A create that meets ``EEXIST`` reads the file back
and counts it as its own when the bytes, which carry a fresh uuid, are its own (a lost reply to a
retransmitted request). ``ESTALE`` on a read is read as absent. No symlink is made and no
host-bound lock (``flock``, a lock file) is taken, so a coordinator or a worker killed on one host
can be taken over from any other.


Records
-------

Ticket (``pool-ticket@1``)
    Up to ``quantum_jobs`` jobs of one node and one queue: for each job its ``checksum`` and
    ``fingerprint`` (its address in the store), the pool ``attempt`` that put it, the runner's
    attempt (for reporting), its payload (``{"codec", "path", "sha256"}``) and its units; the
    residency the jobs need; the batch size; the binding roots; and the ``ladder``, the classes the
    jobs may move up to.
Follow generation
    ``{"ticket", "queue", "attempt", "cause", "phase", "prior", "charged", "uuid"}``. ``cause`` is
    ``put``, ``replaced``, ``split``, ``takeover``, ``restart`` or ``ladder``. The newest generation
    names the ticket that holds the job and the attempt it runs under, ``<pool id>.<g>``. Moving a
    job writes generation ``g + 1``; nothing is removed while the pool lives (``collect`` keeps the
    highest).
Outcome (``pool-outcome@1``)
    One per job and attempt: ``done``, ``failed`` (with the typed fault), ``held``, ``withdrawn``
    (``cancel`` or ``refused``, the latter re-placed one class up the ladder) or ``expired`` (a call
    past its deadline).
Worker records
    ``ready.json`` (written once the worker has checked the pool's settings and its hardware: the
    class, the device memory read, the GPUs and CPUs it may use), ``samples.jsonl`` (one sample per
    job: wall time, peak host and device memory, whether the residency was cold), ``calls/<seq>.json``
    before each call and ``<seq>.done`` after it.
``faults.json``
    Written once by each coordinator into its epoch: the ``FaultBudgets``, the classifier as a
    ``module:qualname`` reference and the calibration directory. Every worker, however started,
    reads the newest epoch's.


Payloads and trust
------------------

The default codec, ``PrivatePickle``, cloudpickles each job once to ``payloads/<sha256>`` under the
pool's 0700 root. A worker reads a payload only after checking that ``payloads/`` belongs to its
own user and that no component of the path is writable by others, and that the digest matches the
ticket. A pool is single-user: ``pool.json`` records the owner's uid and a worker of another uid
refuses it. Another codec (a ``JobCodec``: ``encode(job)`` and ``decode(bytes, store)``) can carry
jobs as JSON in the ticket.


Leases and liveness
-------------------

Every exclusion is a named ``DirectoryLease`` (``DirectoryLease.at``): the coordinator's lease per
epoch, each worker's lease and each ticket's claim. A lease is taken by ``mkdir`` and kept alive by
a heartbeat that makes and removes an entry in the holder's token directory every
``renew_fraction * stale_after`` seconds. Another process takes it over once the newest mark in it
is older than ``stale_after`` on the storage clock; takeover renames the holder's token directory,
so of several takers exactly one succeeds, and a holder that was taken over learns it at its next
heartbeat or when it calls ``prove()``.

``pool.json`` records the four staleness fields of the pool's ``LeaseSettings`` (``stale_after``,
``renew_fraction``, ``attribute_delay``, ``timeout``); a process whose settings differ in any of
them refuses the pool. ``attribute_delay`` is how late a client may see a directory's mtime;
``pydracli pool probe <root>`` measures it and the pool refuses a root that is slower than it says.

A worker does not beat its leases from a thread of its own process, where a call holding the GIL
could starve the heartbeat. It starts a *beater* process (``LeaseBeater``) before importing any
provider, takes every lease with ``beat="external"`` and hands it to the beater, which beats any
number of token directories. Each call carries a deadline the beater enforces: past it, the beater
writes the call's ``expired`` outcomes and stops beating its claim. The worker pings its beater
while a call runs and proves its claims only while the beater answers; with a dead beater it writes
the expired outcomes itself and recycles.


The coordinator
---------------

1. **Epoch.** A new coordinator waits while another coordinator's lease is live
   (``PoolSettings.coordinator_wait`` bounds the wait), then takes the next epoch by ``mkdir`` and
   writes ``faults.json``. A coordinator that sees a newer epoch stops writing and reports its open
   jobs not run.
2. **Place.** Each job submitted is placed on a worker class and a queue (``place``, under the
   pool's ``Fit``); a job that fits no class is failed at once with its reasons.
3. **Follow or put.** A job already held by a live ticket (a predecessor's, on resume) is followed,
   not put again. Otherwise its payload is written, then its ticket, then its slot in the queue's
   index, at most ``max_queued_jobs`` unsettled jobs per queue; the rest wait in the process and
   are written as tickets settle.
4. **Scale.** Each poll the coordinator lists its substrate's workers and starts the ones each
   class's waiting jobs want, while the substrate allows. A worker that leaves without a ready mark
   is a failed start; the class backs off. It is broken, and its jobs are held with the latest
   start error, after ``start_failures`` failed starts in a row, or at a failed start once the
   streak has ``start_deadline_failures`` failures and ``start_deadline_s`` has passed since its
   first, with no worker of the class live. A streak ends when the class has no waiting jobs. A
   ticket write that fails puts its jobs back to wait for the next write, and a queue's writes
   failing by the same rule break its class too; a job its codec cannot encode is ``Failed`` on its
   own. ``WarmPool.release_hold``, or a ``retry`` of a held job, lifts the break with a fresh
   streak. Each failed start, listing, write or encoding, and each broken or released class, is a
   pool event: a ``pydra.pool`` log record whose ``event`` attribute names it.
5. **Settle.** Each poll it lists ``settled/`` of each queue and reads a ticket's outcomes only
   when its count moved, reporting each job's newest outcome: a ``done`` job's result is read back
   from the store.
6. **Close.** ``close`` writes the epoch's ``ended`` mark and, when the coordinator still holds the
   newest epoch, stops the workers its substrate lists.


The worker
----------

``pydracli pool worker --root <pool> --class <name>`` (or ``python -m pydra.workers.pool.worker``):

1. **Start.** Open the pool; refuse its settings if they differ; start the beater; take the worker
   lease; read the hardware and refuse a class whose ``gpu_type`` or ``gpus`` it does not match
   (exit 4); write ``ready.json``.
2. **Claim** tickets of the class's queues: the queue served last while its quantum
   (``quantum_jobs`` jobs or ``quantum_s`` seconds) lasts, then the queue whose oldest waiting
   ticket is oldest. A claim that takes a stale one over records the loss in ``losses/`` and places
   the open jobs again under fresh generations: a lone job under ``takeover``, two or more
   ``split``; a job whose attempt expired is looked up in the store first.
3. **Load** the ticket's residency through the process's resident set, on the granted device; a
   failure there is a fault of the residency, never of a job. A load out of memory moves the jobs
   up the ladder (``phase: "load"``, uncharged) and is recorded as an ``UnfitRecord``.
4. **Run** each job under the attempt of its newest generation, with the claim as its
   de-duplication lease, proved before the result is published to the store.
5. **Settle** each job as it finishes: its sample, its outcome, the ticket's settled mark.
6. **Exit** after ``idle_polls`` empty polls; on SIGTERM after settling the finished jobs and
   releasing the rest; when started by a substrate, after ``orphan_after_s`` with no live
   coordinator; and recycling (exit 5) after a call's deadline passed or the beater died. Resident
   state is closed on exit, process groups included.


Faults
------

The classifier (``pydra.workers.pool.faults.classify`` by default) types each failure as content
(the task's own error; retried up to ``content_attempts`` runs), resource (out of memory or a
timeout; the job moves up its ladder, at most ``resource_moves`` times), infrastructure (a crash of
the state, a failed load; after ``infrastructure_replacements`` in one epoch the residency is held
on its queue) or policy. A crash alone in its call is charged to the job, up to
``crashes_charged``. The budgets are the coordinator's ``FaultBudgets``, read by every worker from
``faults.json``.


Outcomes seen by the run
------------------------

Every job submitted to a pool is reported exactly once: as its ``Result``, as ``Failed`` (the
typed fault, or ``UnplaceableJob``), or as ``NotRun`` (``held=True`` for a broken class or a held
residency). Each call's facts (``pool-facts@1``: the class, the hardware, whether the residency was
cold, the calls since its load) reach the audit record and ``Result.runtime``; under a store whose
identity lists ``pool`` in ``recorded_only`` they also enter the execute context, where they are
recorded and never part of the job's identity, so a warm and a cold run share one slot in the
store. A pool refuses a store whose identity would key them.
