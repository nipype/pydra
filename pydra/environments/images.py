"""Image reference helpers shared by the container environments (RFC 0004 section 5.3)."""


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
"""Seconds `registry_digest` waits for each registry request: a manifest ``HEAD`` answers in
well under a second, and a registry slower than this leaves the image unresolved (reuse
then stays process-local) rather than stalling every job's keying. An unmeasured choice."""


def registry_digest(
    repository: str, tag: str, timeout: float = REGISTRY_TIMEOUT
) -> str | None:
    """The digest the registry reports for ``repository:tag`` (a ``HEAD`` of its
    manifest, after an anonymous token if asked for one); ``None`` when it cannot be
    read. Nothing is pulled."""
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
    url = f"https://{host}/v2/{name}/manifests/{tag}"

    def head(token: str | None = None):
        headers = {"Accept": _MANIFEST_TYPES}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, method="HEAD", headers=headers)
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
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
        return None
    return digest if digest and digest.startswith("sha256:") else None
