import pickle
import pydra
import sys


def run_pickled(*file_paths):
    loaded_objects = []

    for file_path in file_paths:
        with open(file_path, "rb") as file:
            loaded_objects.append(pickle.load(file))

    if len(loaded_objects) == 1:
        result = loaded_objects[0](rerun=False)
    elif len(loaded_objects) == 3:
        result = loaded_objects[0](loaded_objects[1], loaded_objects[2], rerun=False)
    else:
        raise ValueError("Unsupported number of loaded objects")

    print(f"Result: {result}")


if __name__ == "__main__":
    run_pickled(*sys.argv[1:])
