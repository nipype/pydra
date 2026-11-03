from contextlib import redirect_stdout
import hashlib
import importlib.resources
import io
import json
import pytest
from pydra.utils.messenger import (
    PrintMessenger,
    FileMessenger,
    ProvenanceContextUnavailable,
    collect_messages,
    document_loader,
    make_message,
    pydra_context,
)

SCHEMA = importlib.resources.files("pydra") / "schema"


def test_print_messenger():
    f = io.StringIO()
    with redirect_stdout(f):
        msgr = PrintMessenger()
        msgr.send({"key": "value"})
    s = f.getvalue()
    assert s.startswith("id:")
    assert '"key": "value"' in s


def test_file_messenger(tmpdir):
    tmpdir.chdir()

    msgr = FileMessenger()
    msgr.send({"key": "value"})
    assert (tmpdir / "messages").exists()
    from glob import glob

    assert len(glob(str(tmpdir / "messages" / "*.jsonld"))) == 1

    newdir = tmpdir / "newdir"
    newdir.mkdir()

    msgr.send({"key": "value"}, message_dir=newdir)
    assert len(glob(str(newdir / "*.jsonld"))) == 1

    msgr.send({"key": "value"}, message_dir_badkwarg=newdir)
    assert len(glob(str(newdir / "*.jsonld"))) == 1
    assert len(glob(str(tmpdir / "messages" / "*.jsonld"))) == 2


def test_collect_messages(tmpdir, no_network):
    tmpdir.chdir()

    msgr = FileMessenger()
    msgr.send(make_message({"@context": {}, "key": "value"}))
    assert (tmpdir / "messages").exists()
    from glob import glob

    assert len(glob(str(tmpdir / "messages" / "*.jsonld"))) == 1
    collect_messages(tmpdir, tmpdir / "messages")
    assert (tmpdir / "messages.jsonld").exists()

    msgr.send(make_message({"@context": "http://example.org", "key": "value"}))
    with pytest.raises(ProvenanceContextUnavailable, match="http://example.org"):
        collect_messages(tmpdir, tmpdir / "messages")


def test_messages_carry_the_packaged_context():
    message = make_message({"key": "value"})
    assert message["@context"] == pydra_context()["@context"]
    assert message["key"] == "value"


def test_packaged_contexts_match_their_record(no_network):
    sources = json.loads((SCHEMA / "contexts.json").read_text())
    for iri, entry in sources.items():
        packaged = (SCHEMA / entry["file"]).read_bytes()
        assert document_loader(iri)["document"] == json.loads(packaged)
        if "sha256" in entry:
            assert hashlib.sha256(packaged).hexdigest() == entry["sha256"]
            assert entry["commit"] in iri


@pytest.mark.parametrize(
    "context",
    [
        None,
        "https://raw.githubusercontent.com/nipype/pydra/master/pydra/schema/context.jsonld",
    ],
)
def test_collect_messages_offline(tmp_path, no_network, context):
    messages = tmp_path / "messages"
    message = make_message({"@id": "uid:1", "@type": "activity"})
    if context is not None:
        message["@context"] = context
    FileMessenger().send(message, message_dir=messages)
    collect_messages(tmp_path, messages)
    collected = json.loads((tmp_path / "messages.jsonld").read_text())
    assert collected["@context"] == message["@context"]
