def add_exc_note(e: Exception, note: str) -> Exception:
    """Adds a note to an exception in a Python <3.11 compatible way

    Parameters
    ----------
    e : Exception
        the exception to add the note to
    note : str
        the note to add

    Returns
    -------
    Exception
        returns the exception again
    """
    try:
        e.add_note(note)  # type: ignore
    except AttributeError:
        e.args = (e.args[0] + "\n" + note,)
    return e
