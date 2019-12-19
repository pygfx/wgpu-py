from .flags import *
from .enums import *
from .classes import *
from .utils import help
from . import classes


__version__ = "0.0.1"


def _register_backend(func):
    if not (callable(func) and func.__name__ == "requestAdapter"):
        raise RuntimeError(
            "WGPU backend must be registered as function called requestAdapter."
        )
    if globals()["requestAdapter"] is not classes.requestAdapter:
        raise RuntimeError("WGPU backend can only be set once.")
    globals()["requestAdapter"] = func
    # todo: auto-select upon using requestAdapter?


def requestAdapterSync(powerPreference: "enum PowerPreference"):
    """ A convenience function.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(requestAdapter(powerPreference))
