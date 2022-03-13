import click
from glob import glob
import os
import shutil


@click.group()
def cli():
    pass


@cli.command()
@click.option("--cache_dir")
def clean(cache_dir):
    """cleaning lock directories from the cache dir"""
    paths_lock = glob(f"{cache_dir}/*.lock")
    for path in paths_lock:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


if __name__ == "__main__":
    cli()
