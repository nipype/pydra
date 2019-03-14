import abc
import datetime as dt
import enum
from pathlib import Path
import os


def gen_uuid():
    import uuid

    return uuid.uuid4().hex


def now():
    return dt.datetime.utcnow().isoformat(timespec="microseconds")


class AuditFlag(enum.Flag):
    NONE = 0
    PROV = enum.auto()  # 0x01
    RESOURCE = enum.auto()  # 0x02
    ALL = PROV | RESOURCE


class RuntimeHooks(enum.IntEnum):
    """Allowed points to hook into the process
    """

    """TODO: Convert all messenger actions to hooks at the following stages"""
    task_run_entry = enum.auto()
    task_run_exit = enum.auto()
    resource_monitor_pre_start = enum.auto()
    resource_monitor_post_stop = enum.auto()
    task_execute_pre_entry = enum.auto()
    task_execute_post_exit = enum.auto()


class Messenger:
    @abc.abstractmethod
    def send(self, message, **kwargs):
        pass


class PrintMessenger(Messenger):
    def send(self, message, **kwargs):
        import json

        mid = gen_uuid()
        print(
            "id: {0}\n{1}".format(
                mid, json.dumps(message, ensure_ascii=False, indent=2, sort_keys=False)
            )
        )


class FileMessenger(Messenger):
    def send(self, message, **kwargs):
        import json

        mid = gen_uuid()
        if kwargs.get("message_dir"):
            message_dir = Path(kwargs["message_dir"])
        else:
            message_dir = Path(os.getcwd()) / "messages"
        message_dir.mkdir(parents=True, exist_ok=True)
        with open(message_dir / (mid + ".jsonld"), "wt") as fp:
            json.dump(message, fp, ensure_ascii=False, indent=2, sort_keys=False)
        return mid


class RemoteRESTMessenger(Messenger):
    def send(self, message, **kwargs):
        import requests

        r = requests.post(
            kwargs["post_url"],
            json=message,
            auth=kwargs["auth"]()
            if getattr(kwargs["auth"], "__call__", None)
            else kwargs["auth"],
        )
        return r.status_code


def send_message(message, messengers=None, **kwargs):
    """Send nidm messages for logging provenance and auditing
    """
    for messenger in messengers:
        messenger.send(message, **kwargs)


def make_message(obj, context=None):
    if context is None:
        context = {
            "@context": "https://raw.githubusercontent.com/nipype/pydra/enh/task/pydra/schema/context.jsonld"
        }
    message = context.copy()
    message.update(**obj)
    return message


def collect_messages(collected_path, message_path, ld_op="compact"):
    import pyld as pld
    import json
    from glob import glob

    fl = glob(str(message_path / "*.jsonld"))
    data = []
    for f in fl:
        with open(f, "rt") as fp:
            data.append(json.load(fp))
    if data:
        records = getattr(pld.jsonld, ld_op)(
            pld.jsonld.from_rdf(pld.jsonld.to_rdf(data, {})), data[0]
        )
        records["@id"] = "uid:{}".format(gen_uuid())
        with open(collected_path / "messages.jsonld", "wt") as fp:
            json.dump(records, fp, ensure_ascii=False, indent=2, sort_keys=False)
