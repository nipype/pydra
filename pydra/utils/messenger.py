"""Messaging of states."""

import abc
import datetime as dt
import enum
from pathlib import Path
import os


def gen_uuid():
    """Generate a unique identifier."""
    import uuid

    return uuid.uuid4().hex


def now():
    """Get a formatted timestamp."""
    return dt.datetime.utcnow().isoformat(timespec="microseconds")


class AuditFlag(enum.Flag):
    """Auditing flags."""

    NONE = 0
    """Do not track provenance or monitor resources."""
    PROV = enum.auto()  # 0x01
    """Track provenance only."""
    RESOURCE = enum.auto()  # 0x02
    """Monitor resource utilization only."""
    ALL = PROV | RESOURCE
    """Track provenance and resource utilization."""


class RuntimeHooks(enum.IntEnum):
    """Allowed points to hook into the process."""

    # TODO: Convert all messenger actions to hooks at the following stages
    task_run_entry = enum.auto()
    task_run_exit = enum.auto()
    resource_monitor_pre_start = enum.auto()
    resource_monitor_post_stop = enum.auto()
    task_execute_pre_entry = enum.auto()
    task_execute_post_exit = enum.auto()


class Messenger:
    """Base messenger class."""

    @abc.abstractmethod
    def send(self, message, **kwargs):
        """Send a message."""


class PrintMessenger(Messenger):
    """A messenger that redirects to standard output."""

    def send(self, message, **kwargs):
        """
        Send the message to standard output.

        Parameters
        ----------
        message : :obj:`dict`
            The message to be printed.

        """
        import json

        mid = gen_uuid()
        print(
            "id: {}\n{}".format(
                mid, json.dumps(message, ensure_ascii=False, indent=2, sort_keys=False)
            )
        )


class FileMessenger(Messenger):
    """A messenger that redirects to a file."""

    def send(self, message, append=True, **kwargs):
        """
        Append message to file.

        Parameters
        ----------
        message : :obj:`dict`
            The message to be printed.
        append : :obj:`bool`
            Do not truncate file when opening (i.e. append to it).

        Returns
        -------
        str
            Returns the unique identifier used in the file's name.
        """
        import json

        mid = gen_uuid()
        if kwargs.get("message_dir"):
            message_dir = Path(kwargs["message_dir"])
        else:
            message_dir = Path(os.getcwd()) / "messages"
        message_dir.mkdir(parents=True, exist_ok=True)
        with (message_dir / ("%s.jsonld" % mid)).open("wa"[append]) as fp:
            json.dump(message, fp, ensure_ascii=False, indent=2, sort_keys=False)
        return mid


class RemoteRESTMessenger(Messenger):
    """A messenger that redirects to remote REST endpoint."""

    def send(self, message, **kwargs):
        """
        Append message to file.

        Parameters
        ----------
        message : :obj:`dict`
            The message to be printed.

        Returns
        -------
        int
            The status code from the `request.post`

        """
        import requests

        r = requests.post(
            kwargs["post_url"],
            json=message,
            auth=(
                kwargs["auth"]()
                if getattr(kwargs["auth"], "__call__", None)
                else kwargs["auth"]
            ),
        )
        return r.status_code


def send_message(message, messengers=None, **kwargs):
    """Send NIDM messages for logging provenance and auditing."""
    for messenger in messengers:
        messenger.send(message, **kwargs)


def make_message(obj, context=None):
    """
    Build a message using the specific context

    Parameters
    ----------
    obj : :obj:`dict`
        A dictionary containing the non-context information of a message record.
    context : :obj:`dict`, optional
        Dictionary with the link to the context file or containing a JSON-LD context.

    Returns
    -------
    dict
        The message with the context.
    """
    if context is None:
        context = {
            "@context": "https://raw.githubusercontent.com/nipype/pydra/"
            "master/pydra/schema/context.jsonld"
        }
    message = context.copy()
    message.update(**obj)
    return message


def collect_messages(collected_path, message_path, ld_op="compact"):
    """
    Compile all messages into a single provenance graph.

    Parameters
    ----------
    collected_path : :obj:`os.pathlike`
        A place to write all of the collected messages. (?TODO)
    message_path : :obj:`os.pathlike`
        A path with the message file (?TODO)
    ld_op : :obj:`str`, optional
        Option used by pld.jsonld
    """

    import pyld as pld
    import json
    from glob import glob

    fl = glob(str(message_path / "*.jsonld"))
    data = []
    for f in fl:
        with open(f) as fp:
            data.append(json.load(fp))
    if data:
        records = getattr(pld.jsonld, ld_op)(
            pld.jsonld.from_rdf(pld.jsonld.to_rdf(data, {})), data[0]
        )
        records["@id"] = f"uid:{gen_uuid()}"
        with open(collected_path / "messages.jsonld", "w") as fp:
            json.dump(records, fp, ensure_ascii=False, indent=2, sort_keys=False)
