from contextlib import redirect_stdout
import io
import pytest
from ..messenger import PrintMessenger, FileMessenger, collect_messages, make_message


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


@pytest.mark.flaky(reruns=3)
def test_collect_messages(tmpdir):
    tmpdir.chdir()

    msgr = FileMessenger()
    msgr.send(make_message({"@context": {}, "key": "value"}))
    assert (tmpdir / "messages").exists()
    from glob import glob

    assert len(glob(str(tmpdir / "messages" / "*.jsonld"))) == 1
    collect_messages(tmpdir, tmpdir / "messages")
    assert (tmpdir / "messages.jsonld").exists()

    msgr.send(make_message({"@context": "http://example.org", "key": "value"}))
    import pyld

    with pytest.raises(pyld.jsonld.JsonLdError):
        collect_messages(tmpdir, tmpdir / "messages")
