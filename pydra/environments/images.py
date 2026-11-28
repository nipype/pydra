"""Image reference helpers shared by the container environments (RFC 0004 section 5.3)."""

import hashlib
import logging
import os
import platform
from pathlib import Path


def split_reference(image: str) -> tuple[str, str | None]:
    """``(repository, digest)`` of ``image``, whose tag is dropped"""
    name, _, digest = image.partition("@")
    last = name.rsplit("/", 1)[-1]
    if ":" in last:
        name = name[: len(name) - len(last)] + last.split(":", 1)[0]
    return name, digest or None


def normal(repository: str) -> str:
    """``repository`` as Docker Hub names it, so ``library/busybox`` equals ``busybox``"""
    for prefix in ("docker.io/", "index.docker.io/", "registry-1.docker.io/"):
        if repository.startswith(prefix):
            repository = repository[len(prefix) :]
    return repository.removeprefix("library/")


_MANIFEST_TYPES = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


REGISTRY_TIMEOUT = 10
"""Seconds `registry_manifest` waits for each registry request: a manifest ``HEAD`` answers in
well under a second, and a registry slower than this leaves the image unresolved (reuse
then stays process-local) rather than stalling every job's keying. An unmeasured choice."""


_INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)


def registry_manifest(
    repository: str, tag: str, timeout: float = REGISTRY_TIMEOUT
) -> tuple[str, str | None] | None:
    """``(digest, platform digest)`` the registry reports for ``repository:tag``, from
    one ``HEAD`` of its manifest (after an anonymous token if asked for one). The
    platform digest is the digest itself when it names a single manifest, and ``None``
    when it names an index. ``None`` when the digest cannot be read. Nothing is pulled.
    """
    import re
    import json
    import http.client
    import urllib.error
    import urllib.parse
    import urllib.request

    first, _, rest = repository.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, name = first, rest
    else:
        host, name = "registry-1.docker.io", repository
        if "/" not in name:
            name = f"library/{name}"

    def head(token: str | None = None):
        headers = {"Accept": _MANIFEST_TYPES}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"https://{host}/v2/{name}/manifests/{tag}", method="HEAD", headers=headers
        )
        return urllib.request.urlopen(request, timeout=timeout)

    try:
        try:
            response = head()
        except urllib.error.HTTPError as e:
            if e.code != 401:
                return None
            challenge = e.headers.get("WWW-Authenticate", "")
            params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
            if "realm" not in params:
                return None
            query = {k: v for k, v in params.items() if k in ("service", "scope")}
            query.setdefault("scope", f"repository:{name}:pull")
            with urllib.request.urlopen(
                f"{params['realm']}?{urllib.parse.urlencode(query)}", timeout=timeout
            ) as auth:
                body = json.load(auth)
            response = head(body.get("token") or body.get("access_token"))
        with response:
            digest = response.headers.get("Docker-Content-Digest")
            media_type = (response.headers.get("Content-Type") or "").split(";")[0]
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
        return None
    if not (digest and digest.startswith("sha256:")):
        return None
    # an index's platform manifest would take a GET, which a registry such as Docker
    # Hub counts as a pull; a single manifest is its own platform manifest
    return digest, (digest if media_type and media_type not in _INDEX_TYPES else None)


_SIF_DIGESTS: dict[tuple, str] = {}
"""This process's ``.sif`` digests, by the file's stat"""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sif_digest(path: os.PathLike | str) -> str:
    """``sha256:<hex>`` of the content of the image file at ``path``: read in full the
    first time, then answered by its stat, in this process and through the
    ``PersistentCache`` in later ones."""
    from pydra.utils.hash import PersistentCache

    real = Path(path).resolve()
    st = real.stat()
    stat_key = ("pydra.sif", str(real), st.st_size, st.st_mtime_ns, st.st_ino)
    if stat_key not in _SIF_DIGESTS:
        hexdigest = PersistentCache().get_or_calculate_hash(
            stat_key, lambda: _sha256_file(real).encode()
        )
        _SIF_DIGESTS[stat_key] = "sha256:" + bytes(hexdigest).decode()
    return _SIF_DIGESTS[stat_key]


IMAGE_CACHE_ENV_VAR = "PYDRA_IMAGE_CACHE"

logger = logging.getLogger("pydra")
_LOCK_WARNED: list[bool] = []


def image_cache() -> Path:
    """Where pulled images are kept: ``PYDRA_IMAGE_CACHE`` if set and not empty, otherwise
    ``pydra-sif`` under ``APPTAINER_CACHEDIR`` (or ``SINGULARITY_CACHEDIR``) if set,
    otherwise ``images`` under pydra's user cache root, which is under ``$HOME``. On a
    cluster, set one of them to project or scratch storage: a SIF is the image's size.
    """
    from pydra.utils.general import user_cache_root

    if os.environ.get(IMAGE_CACHE_ENV_VAR):  # an empty value is unset
        return Path(os.environ[IMAGE_CACHE_ENV_VAR])
    for var in ("APPTAINER_CACHEDIR", "SINGULARITY_CACHEDIR"):
        if os.environ.get(var):
            return Path(os.environ[var]) / "pydra-sif"
    return user_cache_root() / "images"


def host_platform() -> tuple[str, str, str]:
    """``(os, architecture, variant)`` of this host, as image indexes name them"""
    machine = platform.machine().lower()
    arch, variant = {
        "x86_64": ("amd64", ""),
        "amd64": ("amd64", ""),
        "aarch64": ("arm64", ""),
        "arm64": ("arm64", ""),
        "armv7l": ("arm", "v7"),
        "armv6l": ("arm", "v6"),
    }.get(machine, (machine, ""))
    return platform.system().lower(), arch, variant


def local_sif_path(digest: str) -> Path:
    """Where `local_sif` keeps the image of ``digest`` for this host's platform: an
    index digest pulls the host's platform, so the platform is in the name"""
    plat = "-".join(p for p in host_platform() if p)
    return image_cache() / "sif" / f"{digest.replace(':', '-')}_{plat}.sif"


def _lock(path: Path) -> int | None:
    """An exclusive ``flock`` on ``path``, or ``None`` when the file system or the
    file's permissions do not allow one (the pull then runs unlocked: correctness
    rests on the rename, the lock only saves duplicate pulls)"""
    import fcntl

    fd = None
    try:
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        except PermissionError:
            fd = os.open(path, os.O_RDONLY)  # flock needs no write access
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except OSError as e:
        if fd is not None:
            os.close(fd)
        if not _LOCK_WARNED:
            _LOCK_WARNED.append(True)
            logger.info("Pulling images without a lock (%s: %s)", path, e)
        return None


def _cached(target: Path) -> bool:
    """Whether the SIF ``target`` is in the cache; one that cannot be read is refused."""
    if not target.is_file():
        return False
    if not os.access(target, os.R_OK):
        raise PermissionError(
            f"the cached image {target} cannot be read; fix its permissions or remove it"
        )
    return True


def local_sif(uri: str, digest: str) -> Path:
    """A local SIF of ``uri``, a ``docker://<repository>@<digest>`` reference, pulled
    the first time into `local_sif_path` and reused after that, so running it never
    reaches a registry again. A cached SIF is trusted as it is, because only an atomic
    rename of a complete pull creates it. Concurrent first uses pull once where the
    file system allows a lock, and otherwise each pulls and the last rename wins."""
    import tempfile
    from pydra.environments import base

    target = local_sif_path(digest)
    if _cached(target):
        return target
    folder = target.parent
    folder.mkdir(parents=True, exist_ok=True)
    lock = _lock(folder / f"{target.name}.lock")
    try:
        if _cached(target):  # another process pulled it while we waited
            return target
        fd, partial = tempfile.mkstemp(dir=folder, prefix=f".{target.stem}-")
        os.close(fd)
        os.unlink(partial)  # pull refuses to overwrite a file
        try:
            rc, out, err = base.execute(["singularity", "pull", partial, uri])
            if rc or not os.path.isfile(partial):
                raise RuntimeError(
                    f"singularity pull of {uri} exited with exit code {rc}:\n{err}{out}"
                )
            os.replace(partial, target)
        finally:
            if os.path.exists(partial):
                os.unlink(partial)
    finally:
        if lock is not None:
            os.close(lock)
    return target
