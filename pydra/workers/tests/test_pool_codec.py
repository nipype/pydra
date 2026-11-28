"""Job payloads of a worker pool: `JobCodec`, `PrivatePickle` and the ownership walk (RFC 0005 5.2)."""

import base64
import hashlib
import json
import os
import sys

import attrs
import cloudpickle
import pytest
from pathlib import Path

from pydra.engine.portable.trust import EntryNotTrusted
from pydra.utils.digest import jcs
from pydra.workers.pool.codec import (
    PAYLOADS,
    JobCodec,
    PayloadMismatch,
    PrivatePickle,
    read_payload,
    write_payload,
)
from pydra.workers.pool.storage import Ticket, TicketJob

POSIX = sys.platform != "win32"


@attrs.define
class FakeJob:
    """Stands in for a `Job`: the codecs only need something they can encode and give a store."""

    name: str
    store: object = None


@attrs.frozen
class JsonCodec:
    """A codec whose jobs are JSON, carried in the ticket with no pickle."""

    name = "fake-json@1"

    def encode(self, job):
        return json.dumps({"name": job.name}).encode(), "application/json"

    def decode(self, data, store):
        return FakeJob(json.loads(data)["name"], store)


def _work_dir(tmp_path, name="w1"):
    path = tmp_path / "work" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_private_pickle_round_trip(tmp_path):
    """`PrivatePickle` writes ``payloads/<sha256>`` 0o600 under the pool root, in a 0o700
    directory no run uses; the payload names it and its sha256, and the read gives the job back with
    the reader's store."""
    codec = PrivatePickle()
    assert isinstance(codec, JobCodec)
    wd = _work_dir(tmp_path)
    payload = write_payload(codec, FakeJob("a"), wd)
    assert set(payload) == {"codec", "path", "sha256"}
    data = cloudpickle.dumps(FakeJob("a"))
    assert payload["codec"] == "private-pickle@1" and payload["path"] == str(
        wd / PAYLOADS / hashlib.sha256(data).hexdigest()
    )
    assert Path(payload["path"]).read_bytes() == data
    assert payload["sha256"] == hashlib.sha256(data).hexdigest()
    if POSIX:
        assert oct(os.stat(wd / PAYLOADS).st_mode & 0o777) == "0o700"
        assert oct(os.stat(payload["path"]).st_mode & 0o777) == "0o600"
    job = read_payload(codec, payload, store="the store")
    assert job == FakeJob("a", "the store")
    assert write_payload(codec, FakeJob("a"), wd) == payload  # the same job again


@pytest.mark.skipif(not POSIX, reason="owners and modes are POSIX")
def test_pickle_in_group_writable_dir_refused(tmp_path):
    """A pickle is read only after the ownership walk: a work directory or a payload file that group
    or others can write is refused, and so is a file whose digest is not the ticket's.
    """
    codec = PrivatePickle()
    wd = _work_dir(tmp_path)
    payload = write_payload(codec, FakeJob("a"), wd)
    os.chmod(wd / PAYLOADS, 0o770)
    with pytest.raises(EntryNotTrusted):
        read_payload(codec, payload, store=None)
    os.chmod(wd / PAYLOADS, 0o700)
    os.chmod(payload["path"], 0o620)
    with pytest.raises(EntryNotTrusted):
        read_payload(codec, payload, store=None)
    os.chmod(payload["path"], 0o600)
    assert read_payload(codec, payload, store=None) == FakeJob("a")
    with pytest.raises(PayloadMismatch):
        read_payload(codec, {**payload, "sha256": "0" * 64}, store=None)
    with pytest.raises(ValueError, match="codec"):
        read_payload(JsonCodec(), payload, store=None)


def test_ticket_never_contains_pickle_payload(tmp_path):
    """A `PrivatePickle` payload in a ticket is a path and a digest: the ticket's bytes hold neither
    the pickle nor its base64; a JSON codec's job travels in the ticket as JSON."""
    codec = PrivatePickle()
    wd = _work_dir(tmp_path)
    job = FakeJob("secret-" + "x" * 64)
    payload = write_payload(codec, job, wd)
    ticket = Ticket(
        queue="cpu",
        node="wf/n",
        jobs=(
            TicketJob(
                checksum="python-" + "a" * 64, fingerprint="b" * 64, payload=payload
            ),
        ),
    )
    data = jcs(ticket.to_json())
    pickled = Path(payload["path"]).read_bytes()
    assert pickled == cloudpickle.dumps(job)
    assert pickled not in data and base64.b64encode(pickled) not in data
    assert b"secret-" not in data

    inline = write_payload(JsonCodec(), FakeJob("j"), wd)
    assert set(inline) == {"codec", "sha256", "json"} and inline["json"] == {
        "name": "j"
    }
    assert not (wd / "_job.json").exists()
    assert read_payload(JsonCodec(), inline, store=1) == FakeJob("j", 1)
    with pytest.raises(PayloadMismatch):
        read_payload(JsonCodec(), {**inline, "json": {"name": "k"}}, store=1)
