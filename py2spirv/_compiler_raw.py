import os

from ._module import SpirVModule


def bytes2spirv(bytes):
    """ Return a SpirVModule object, given the raw bytes of a SpirV module.
    """
    return SpirVModule(bytes, bytes, "raw")


def file2spirv(f):
    """ Loas a raw SpirV module from a filename or file object.
    Returns a SpirVModule object.
    """
    if isinstance(f, str):
        filename = f
        with open(filename, "rb") as f:
            bytes = f.read()
        return SpirVModule(bytes, bytes, "from " + os.path.basename(filename))
    else:
        bytes = f.read()
        return SpirVModule(bytes, bytes, "from file object")
