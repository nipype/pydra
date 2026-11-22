"""Image reference helpers shared by the container environments (RFC 0004 section 5.3)."""

import hashlib
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


_INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)


def host_platform() -> tuple[str, str]:
    """``(os, architecture)`` of this host as image indexes name them"""
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return "linux", arch


def registry_manifest(
    repository: str, tag: str, timeout: float = 10
) -> tuple[str, str | None] | None:
    """``(digest, platform digest)`` the registry reports for ``repository:tag``: the
    digest is from a ``HEAD`` of its manifest, after an anonymous token if asked for one;
    the platform digest is the manifest of this host's platform (`host_platform`), read
    from the index when the digest names one, or the digest itself when it names a
    single manifest, and ``None`` when it cannot be read. ``None`` when the digest
    cannot be read. Nothing is pulled."""
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

    def head(token: str | None = None, method: str = "HEAD", ref: str = tag):
        headers = {"Accept": _MANIFEST_TYPES}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"https://{host}/v2/{name}/manifests/{ref}", method=method, headers=headers
        )
        return urllib.request.urlopen(request, timeout=timeout)

    token = None
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
            token = body.get("token") or body.get("access_token")
            response = head(token)
        with response:
            digest = response.headers.get("Docker-Content-Digest")
            media_type = (response.headers.get("Content-Type") or "").split(";")[0]
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
        return None
    if not (digest and digest.startswith("sha256:")):
        return None
    if media_type not in _INDEX_TYPES:
        return digest, digest if media_type else None
    try:
        with head(token, "GET", digest) as response:
            index = json.load(response)
        wanted = host_platform()
        for entry in index.get("manifests", []):
            plat = entry.get("platform", {})
            if (plat.get("os"), plat.get("architecture")) == wanted:
                return digest, entry.get("digest")
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
        pass
    return digest, None


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
