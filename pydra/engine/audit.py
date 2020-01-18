"""Module to keep track of provenance information."""
import os
from pathlib import Path
import json
import attr
from ..utils.messenger import send_message, make_message, gen_uuid, now, AuditFlag
from .helpers import ensure_list, gather_runtime_info


class Audit:
    """Handle provenance tracking and resource utilization."""

    def __init__(self, audit_flags, messengers, messenger_args, develop=None):
        """
        Initialize the auditing functionality.

        Parameters
        ----------
        audit_flags : :class:`AuditFlag`
            Base configuration of auditing.
        messengers :
            TODO
        messenger_args :
            TODO
        develop :
            TODO

        """
        self.audit_flags = audit_flags
        self.messengers = ensure_list(messengers)
        self.messenger_args = messenger_args
        self.develop = develop

    def start_audit(self, odir):
        """
        Start recording provenance.

        Monitored information is not sent until directory is created,
        in case message directory is inside task output directory.

        Parameters
        ----------
        odir : :obj:`os.pathlike`
            Message output directory.

        """
        self.odir = odir
        if self.audit_check(AuditFlag.PROV):
            self.aid = "uid:{}".format(gen_uuid())
            start_message = {"@id": self.aid, "@type": "task", "startedAtTime": now()}
        os.chdir(self.odir)
        if self.audit_check(AuditFlag.PROV):
            self.audit_message(start_message, AuditFlag.PROV)
        if self.audit_check(AuditFlag.RESOURCE):
            from ..utils.profiler import ResourceMonitor

            self.resource_monitor = ResourceMonitor(os.getpid(), logdir=self.odir)

    def monitor(self):
        """Start resource monitoring."""
        if self.audit_check(AuditFlag.RESOURCE):
            self.resource_monitor.start()
            if self.audit_check(AuditFlag.PROV):
                self.mid = "uid:{}".format(gen_uuid())
                self.audit_message(
                    {
                        "@id": self.mid,
                        "@type": "monitor",
                        "startedAtTime": now(),
                        "wasStartedBy": self.aid,
                    },
                    AuditFlag.PROV,
                )

    def finalize_audit(self, result):
        """End auditing."""
        if self.audit_check(AuditFlag.RESOURCE):
            self.resource_monitor.stop()
            result.runtime = gather_runtime_info(self.resource_monitor.fname)
            if self.audit_check(AuditFlag.PROV):
                self.audit_message(
                    {"@id": self.mid, "endedAtTime": now(), "wasEndedBy": self.aid},
                    AuditFlag.PROV,
                )
                # audit resources/runtime information
                self.eid = "uid:{}".format(gen_uuid())
                entity = attr.asdict(result.runtime)
                entity.update(
                    **{
                        "@id": self.eid,
                        "@type": "runtime",
                        "prov:wasGeneratedBy": self.aid,
                    }
                )
                self.audit_message(entity, AuditFlag.PROV)
                self.audit_message(
                    {
                        "@type": "prov:Generation",
                        "entity_generated": self.eid,
                        "hadActivity": self.mid,
                    },
                    AuditFlag.PROV,
                )
            self.resource_monitor = None
        if self.audit_check(AuditFlag.PROV):
            # audit outputs
            self.audit_message(
                {"@id": self.aid, "endedAtTime": now(), "errored": result.errored},
                AuditFlag.PROV,
            )

    def audit_message(self, message, flags=None):
        """
        Send auditing message.

        Parameters
        ----------
        message :
            TODO
        flags :
            TODO

        """
        if self.develop:
            with open(
                Path(os.path.dirname(__file__)) / ".." / "schema/context.jsonld", "rt"
            ) as fp:
                context = json.load(fp)
        else:
            context = {
                "@context": "https://raw.githubusercontent.com/nipype/pydra"
                "/master/pydra/schema/context.jsonld"
            }
        if self.audit_flags & flags:
            if self.messenger_args:
                send_message(
                    make_message(message, context=context),
                    messengers=self.messengers,
                    **self.messenger_args,
                )
            else:
                send_message(
                    make_message(message, context=context), messengers=self.messengers
                )

    def audit_check(self, flag):
        """
        Determine whether auditing is enabled for a particular flag.

        Parameters
        ----------
        flag :
            TODO

        """
        return self.audit_flags & flag
