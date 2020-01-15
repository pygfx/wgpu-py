from .flags import *  # noqa: F403
from .enums import *  # noqa: F403
from .classes import *  # noqa: F403
from .utils import help  # noqa: F401
from . import classes


__version__ = "0.1.2"


def _register_backend(func):
    if not (callable(func) and func.__name__ == "requestAdapter"):
        raise RuntimeError(
            "WGPU backend must be registered as function called requestAdapter."
        )
    if globals()["requestAdapter"] is not classes.requestAdapter:
        raise RuntimeError("WGPU backend can only be set once.")
    globals()["requestAdapter"] = func
    # todo: auto-select upon using requestAdapter?


def requestAdapterSync(powerPreference: "enum PowerPreference"):  # noqa: F722
    """ A convenience function.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(requestAdapter(powerPreference))  # noqa: F405
