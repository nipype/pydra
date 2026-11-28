"""No new ``getattr``/``hasattr`` probes for a literal attribute in pydra/engine and pydra/workers.

A member that code needs from a pydra type is declared on that type (or its protocol) and read
directly. A probe stays only where the object is foreign or the attribute is optional by design,
listed in ``ALLOWED`` with its reason.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]  # pydra/
PACKAGES = ("engine", "workers")

# (path under pydra/, object expression, attribute): why a probe is right there
ALLOWED: dict[tuple[str, str, str], str] = {
    (
        "engine/audit.py",
        "job.task",
        "executable",
    ): "only shell tasks define an executable",
    ("engine/context.py", "module", "__file__"): "a module may have no file",
    (
        "engine/context.py",
        "sys.modules.get(module)",
        "__file__",
    ): "a module may have no file",
    ("engine/context.py", "task", "__checksum__"): "in the base branch, unchanged here",
    ("engine/graph.py", "nd", "_task"): "in the base branch, unchanged here",
    ("engine/graph.py", "self", "sorted_nodes"): "in the base branch, unchanged here",
    (
        "engine/job.py",
        "getattr(self.task, f.name)",
        "__bytes_repr__",
    ): "in the base branch, unchanged here",
    (
        "engine/job.py",
        "self.task",
        "__checksum__",
    ): "in the base branch, unchanged here",
    (
        "engine/lazy.py",
        "job.store",
        "readonly_caches",
    ): "in the base branch, unchanged here",
    ("engine/lease.py", "errno", "ESTALE"): "not defined on every platform",
    ("engine/lease.py", "os", "getuid"): "POSIX only",
    (
        "engine/portable/errors.py",
        "type",
        "__module__",
    ): "a type annotation may not have it",
    (
        "engine/portable/errors.py",
        "type",
        "__qualname__",
    ): "a type annotation may not have it",
    ("engine/portable/store.py", "errno", "ESTALE"): "not defined on every platform",
    (
        "engine/portable/store.py",
        "hook",
        "__qualname__",
    ): "a callable may have no qualified name",
    ("engine/portable/store.py", "os", "sync"): "POSIX only",
    (
        "engine/runner.py",
        "exc",
        "recorded_traceback",
    ): "set by result_error on an exception of any class",
    (
        "engine/state.py",
        "self",
        "_current_combiner_all",
    ): "in the base branch, unchanged here",
    (
        "engine/state.py",
        "self",
        "_current_splitter",
    ): "in the base branch, unchanged here",
    (
        "engine/state.py",
        "self",
        "_current_splitter_rpn",
    ): "in the base branch, unchanged here",
    (
        "engine/state.py",
        "self",
        "_prev_state_combiner",
    ): "in the base branch, unchanged here",
    (
        "engine/state.py",
        "self",
        "_prev_state_combiner_all",
    ): "in the base branch, unchanged here",
    (
        "engine/state.py",
        "self",
        "_prev_state_splitter",
    ): "in the base branch, unchanged here",
    (
        "engine/state.py",
        "st",
        "group_for_inputs_final",
    ): "in the base branch, unchanged here",
    ("engine/state.py", "st", "keys_final"): "in the base branch, unchanged here",
    ("engine/store.py", "error", "winerror"): "Windows only",
    ("engine/store.py", "lease", "lost"): "in the base branch, unchanged here",
    ("engine/store.py", "lease", "prove"): "in the base branch, unchanged here",
    ("engine/store.py", "raised", "winerror"): "Windows only",
    (
        "engine/submitter.py",
        "store",
        "cache_root",
    ): "in the base branch, unchanged here",
    (
        "engine/workflow.py",
        "node",
        "fut_combiner",
    ): "in the base branch, unchanged here",
    ("workers/cf.py", "os", "sched_getaffinity"): "Linux only",
    ("workers/slurm.py", "job", "state_index"): "in the base branch, unchanged here",
    ("workers/slurm.py", "os", "getuid"): "POSIX only",
}


def probes() -> dict[tuple[str, str, str], list[int]]:
    found: dict[tuple[str, str, str], list[int]] = {}
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "hasattr")
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    key = (
                        path.relative_to(ROOT).as_posix(),
                        ast.unparse(node.args[0]),
                        node.args[1].value,
                    )
                    found.setdefault(key, []).append(node.lineno)
    return found


def test_no_new_capability_probes():
    new = {k: v for k, v in probes().items() if k not in ALLOWED}
    assert (
        not new
    ), "declare the member and read it directly, or allow it with a reason: " + (
        "; ".join(
            f"{p}:{lines} {obj}.{attr}" for (p, obj, attr), lines in sorted(new.items())
        )
    )


def test_every_allowed_probe_exists():
    stale = sorted(set(ALLOWED) - set(probes()))
    assert not stale, f"remove from ALLOWED: {stale}"
