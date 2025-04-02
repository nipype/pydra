import cloudpickle as cp
import sys
from pathlib import Path
from pydra.engine.job import load_and_run

# To avoid issues when running pytest, where the namespace package "pydra" is dropped in
# the pickling process due to it being run from inside the source tree
sys.path.append(str(Path(__file__).parent.parent))


def run_pickled(*file_paths, rerun=False):
    loaded_objects = []

    for file_path in file_paths:
        with open(file_path, "rb") as file:
            loaded_objects.append(cp.load(file))

    if len(loaded_objects) == 1:
        result = loaded_objects[0](rerun=rerun)
    elif len(loaded_objects) == 2:
        result = load_and_run(loaded_objects[0], loaded_objects[1], rerun=rerun)
    else:
        raise ValueError("Unsupported number of loaded objects")

    print(f"Result: {result}")


if __name__ == "__main__":
    rerun = False  # Default value for rerun
    file_paths = sys.argv[1:]

    if "--rerun" in file_paths:
        rerun = True
        file_paths.remove("--rerun")

    run_pickled(*file_paths, rerun=rerun)
