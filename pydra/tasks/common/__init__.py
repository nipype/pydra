from fileformats.application import TextSerialization
from pydra.design import python


@python.define
def LoadJson(file: TextSerialization) -> dict | list:
    return file.load()
