"""Fakes for warm environment tests (RFC 0005 W3, W5b), shared by the environment and the pool
worker tests. `FakeRuntime` stands for ``docker`` and ``apptainer``: it runs each command on the host,
mapping the container paths back, so that outputs, modes and isolation are those of real commands.
"""

import json
import os
import re
import shutil
import subprocess as sp
import threading
import time
import uuid
from pathlib import Path

from pydra.environments import base, images, singularity

D1 = "sha256:" + "1" * 64
IMAGE = f"org/tool@{D1}"


def _chmod_tree(path: Path, writable: bool) -> None:
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            f = Path(dirpath) / name
            if not f.is_symlink():
                f.chmod(0o644 if writable else 0o444)
        Path(dirpath).chmod(0o755 if writable else 0o555)


class FakeRuntime:
    """Stands for ``docker`` and ``apptainer``. A started container or instance is a
    list of binds ``(host, container path, mode)``; a command in it runs on the host
    with each container path mapped to its host path, or for a read-only bind to a
    read-only copy, and a path under ``/mnt/pydra`` outside every bind fails as a
    missing file. ``rm -f`` or ``instance stop`` during an exec kills it (137)."""

    def __init__(self, monkeypatch, tmp_path, fail_start=False, entrypoints=None):
        self.tmp = tmp_path
        self.calls: list[list[str]] = []
        self.started: dict[str, dict] = {}
        self.names: dict[str, str] = {}
        self.stopped: list[str] = []
        self.sifs: list[str] = []
        self.labels: dict[str, str] = {}  # container id -> owner label
        self.foreign: list[str] = []  # instances started by other processes
        self.fail_start = fail_start
        self.entrypoints = entrypoints or {}
        self.exec_delay = 0.0
        # per-exec delays, taken in order before ``exec_delay``, so a test changes one command's
        # run without racing the fake for a shared value
        self.exec_delays: list[float] = []
        self.running = threading.Event()  # set once a command has begun its delay
        self.lock = threading.Lock()
        self.cold_home = tmp_path / "image-home"
        self.cold_home.mkdir(exist_ok=True)
        monkeypatch.setattr(base, "execute", self.execute)
        monkeypatch.setattr(singularity, "runtime", lambda: "apptainer")
        monkeypatch.setattr(images, "local_sif", self.local_sif)

    def local_sif(self, uri, digest):
        self.sifs.append(uri)
        path = self.tmp / "images" / f"{digest.replace(':', '-')}.sif"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"SIF of {uri}")
        return path

    def of(self, *prefix):
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]

    @property
    def starts(self):
        return self.of("docker", "run", "-d") + self.of(
            "apptainer", "instance", "start"
        )

    @property
    def execs(self):
        return self.of("docker", "exec") + [
            c
            for c in self.of("apptainer", "exec")
            if any(a.startswith("instance://") for a in c)
        ]

    @property
    def cold(self):
        return [
            c
            for c in self.of("docker", "run") + self.of("singularity", "exec")
            if "-d" not in c
        ]

    def execute(self, cmd, strip=False, cwd=None, env=None, **kwargs):
        cmd = [str(c) for c in cmd]
        with self.lock:
            self.calls.append(cmd)
        tool, rest = cmd[0], cmd[1:]
        if tool == "docker" and rest[:2] == ["run", "-d"]:
            return self._start(cmd, "-v")
        if tool == "docker" and rest[0] == "run":
            return self._cold_docker(rest[1:])
        if tool == "docker" and rest[0] == "exec":
            return self._exec_docker(rest[1:])
        if tool == "docker" and rest[:2] == ["rm", "-f"]:
            self.stopped.append(self.names.get(rest[2], rest[2]))
            return 0, rest[2], ""
        if tool == "docker" and rest[:2] == ["image", "inspect"]:
            return 0, json.dumps(self.entrypoints.get(rest[-1])), ""
        if tool == "docker" and rest[:2] == ["ps", "-a"]:
            return 0, "\n".join(f"{c}\t{o}" for c, o in self.labels.items()), ""
        if tool == "apptainer" and rest[:2] == ["instance", "start"]:
            return self._start(cmd, "-B")
        if tool == "apptainer" and rest[:2] == ["instance", "stop"]:
            self.stopped.append(rest[2])
            return 0, "", ""
        if tool == "apptainer" and rest[:2] == ["instance", "list"]:
            names = list(self.started) + self.foreign
            listing = [{"instance": n} for n in names if n not in self.stopped]
            return 0, json.dumps({"instances": listing}), ""
        if tool == "apptainer" and rest[0] == "exec":
            return self._exec_apptainer(rest[1:])
        if tool == "singularity" and rest[0] == "exec":
            return self._cold_singularity(rest[1:])
        raise AssertionError(f"unexpected command {cmd}")

    @staticmethod
    def _bind(spec):
        host, ctr, mode = (spec.split(":") + ["rw"])[:3]
        return Path(host), ctr, mode

    def _start(self, cmd, bind_flag):
        valued = {
            bind_flag,
            "--name",
            "--user",
            "--gpus",
            "--tmpfs",
            "--entrypoint",
            "--label",
        }
        binds, args, given, i = [], [], None, 2
        while i < len(cmd):
            if cmd[i] in valued:
                if cmd[i] == bind_flag:
                    binds.append(self._bind(cmd[i + 1]))
                if cmd[i] == "--name":
                    given = cmd[i + 1]
                i += 2
            else:
                args.append(cmd[i])
                i += 1
        if cmd[0] == "docker":
            name = f"id{len(self.started)}"
            image = [a for a in args if not a.startswith("-")][0]
            self.names[given] = name
        else:
            name, image = args[-1], args[-2]
        self.started[name] = {"binds": binds, "image": image, "cmd": cmd}
        if self.fail_start:
            return 125, "", "cannot start"
        return 0, name, ""

    def _map(self, binds, value, shadows):
        for found in re.findall(r"(?<![\w./-])(/mnt/pydra[^\s\"':=]*)", value):
            if not any(found == c or found.startswith(c + "/") for _, c, _ in binds):
                raise FileNotFoundError(found)
        if not binds:
            return value
        ordered = sorted(binds, key=lambda b: len(b[1]), reverse=True)
        alternatives = "|".join(re.escape(c) for _, c, _ in ordered)
        pattern = rf"(?<![\w./-])({alternatives})(?=/|$|[\s\"'])"

        def repl(match):
            host, _, mode = next(b for b in ordered if b[1] == match.group(1))
            if mode == "rw":
                return str(host)
            if host not in shadows:
                shadow = self.tmp / "shadows" / uuid.uuid4().hex
                shutil.copytree(host, shadow, symlinks=True)
                _chmod_tree(shadow, writable=False)
                shadows[host] = shadow
            return str(shadows[host])

        return re.sub(pattern, repl, value)

    def _run(self, binds, workdir, env, args, alive=None):
        shadows: dict[Path, Path] = {}
        try:
            try:
                workdir = self._map(binds, workdir, shadows)
                env = {k: self._map(binds, v, shadows) for k, v in env.items()}
                args = [self._map(binds, a, shadows) for a in args]
            except FileNotFoundError as e:
                return 1, "", f"No such file or directory: {e}"
            # the command "runs" for exec_delay seconds; stopping its container ends it at once
            with self.lock:
                delay = self.exec_delays.pop(0) if self.exec_delays else self.exec_delay
            until = time.monotonic() + delay
            self.running.set()
            while time.monotonic() < until:
                if alive is not None and not alive():
                    return 137, "", "killed"
                time.sleep(min(0.02, max(until - time.monotonic(), 0)))
            if alive is not None and not alive():
                return 137, "", "killed"
            proc = sp.run(
                args,
                cwd=workdir,
                env={"PATH": os.environ["PATH"], **env},
                capture_output=True,
                text=True,
            )
            if alive is not None and not alive():
                return 137, "", "killed"
            return proc.returncode, proc.stdout, proc.stderr
        finally:
            for shadow in shadows.values():
                _chmod_tree(shadow, writable=True)
                shutil.rmtree(shadow, ignore_errors=True)

    def _exec_docker(self, rest):
        workdir, env, i = None, {}, 0
        while rest[i].startswith("-"):
            if rest[i] == "-w":
                workdir = rest[i + 1]
            elif rest[i] == "-e":
                k, _, v = rest[i + 1].partition("=")
                env[k] = v
            i += 2
        name, args = rest[i], rest[i + 1 :]
        if name in self.stopped or name not in self.started:
            return 1, "", f"No such container: {name}"
        return self._run(
            self.started[name]["binds"],
            workdir,
            env,
            args,
            alive=lambda: name not in self.stopped,
        )

    def _exec_apptainer(self, rest):
        workdir, i = None, 0
        while rest[i].startswith("-"):
            workdir = rest[i + 1] if rest[i] == "--pwd" else workdir
            i += 2
        name = rest[i].removeprefix("instance://")
        if name in self.stopped or name not in self.started:
            return 255, "", f"no instance found with name {name}"
        return self._run(
            self.started[name]["binds"],
            workdir,
            {},
            rest[i + 1 :],
            alive=lambda: name not in self.stopped,
        )

    def _parse_cold(self, rest, bind_flag, workdir_flag, bare):
        binds, workdir, entrypoint, i = [], None, None, 0
        while rest[i].startswith("-"):
            if rest[i] in bare:
                i += 1
                continue
            if rest[i] == bind_flag:
                binds.append(self._bind(rest[i + 1]))
            elif rest[i] == workdir_flag:
                workdir = rest[i + 1]
            elif rest[i] == "--entrypoint":
                entrypoint = [rest[i + 1]] if rest[i + 1] else []
            i += 2
        return binds, workdir, entrypoint, rest[i], rest[i + 1 :]

    def _cold_docker(self, rest):
        binds, workdir, entrypoint, image, args = self._parse_cold(
            rest, "-v", "-w", {"--rm"}
        )
        if entrypoint is None:
            entrypoint = self.entrypoints.get(image) or []
        env = {"HOME": str(self.cold_home)}
        return self._run(binds, workdir, env, entrypoint + args)

    def _cold_singularity(self, rest):
        binds, workdir, _, _, args = self._parse_cold(rest, "-B", "--pwd", set())
        return self._run(binds, workdir, {"HOME": str(self.cold_home)}, args)
