from pathlib import Path
import json
from pydra.tasks.common import LoadJson


def test_load_json(tmp_path: Path):
    JSON_CONTENTS = {"a": True, "b": "two", "c": 3, "d": [7, 0.55, 6]}

    # Create a JSON file with some contents
    json_file = tmp_path / "test.json"
    with open(json_file, "w") as f:
        json.dump(JSON_CONTENTS, f)

    # Instantiate the task, providing the JSON file we want to load
    load_json = LoadJson(file=json_file)

    # Run the task to load the JSON file
    outputs = load_json()

    # Access the loaded JSON output contents and check they match original
    assert outputs.out == JSON_CONTENTS
