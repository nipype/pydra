import pickle
import sys
from pydra.engine.helpers import load_and_run


def run_pickled(*file_paths, rerun=False):
    loaded_objects = []

    for file_path in file_paths:
        with open(file_path, "rb") as file:
            loaded_objects.append(pickle.load(file))

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
