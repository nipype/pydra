import json
from fileformats.application import Json
from pydra.design import python


@python.define
def LoadJson(file: Json) -> dict | list:
    with open(file) as f:
        return json.load(f)
