from pathlib import Path
import pdb
import sys

import click
import cloudpickle as cp

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
ExistingFilePath = click.Path(exists=True, dir_okay=False, resolve_path=True)


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("crashfile", type=ExistingFilePath)
@click.option(
    "-r", "--rerun", is_flag=True, flag_value=True, help="Rerun crashed code."
)
@click.option(
    "-d",
    "--debugger",
    type=click.Choice([None, "ipython", "pdb"]),
    help="Debugger to use when rerunning",
)
def crash(crashfile, rerun, debugger=None):
    """Display a crash file and rerun if required."""
    if crashfile.endswith(("pkl", "pklz")):
        with open(crashfile, "rb") as f:
            crash_content = cp.load(f)
        print("".join(crash_content["error message"]))

        if rerun:
            jobfile = Path(crashfile).parent / "_job.pklz"
            if jobfile.exists():
                with open(jobfile, "rb") as f:
                    job_obj = cp.load(f)

                if debugger == "ipython":
                    try:
                        from IPython.core import ultratb

                        sys.excepthook = ultratb.FormattedTB(
                            mode="Verbose", theme_name="Linux", call_pdb=True
                        )
                    except ImportError:
                        raise ImportError(
                            "'Ipython' needs to be installed to use the 'ipython' debugger"
                        )

                try:
                    job_obj.run(rerun=True)
                except Exception:  # noqa: E722
                    if debugger == "pdb":
                        pdb.post_mortem()
                    elif debugger == "ipython":
                        raise
            else:
                raise FileNotFoundError(f"Job file {jobfile} not found")
    else:
        raise ValueError("Only pickled crashfiles are supported")
