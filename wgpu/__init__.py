from ._constants import *
from ._api import *
from .utils import help
from . import _api


def _register_backend(func):
    if not (callable(func) and func.__name__ == "requestAdapter"):
        raise RuntimeError(
            "WGPU backend must be registered as function called requestAdapter."
        )
    if globals()["requestAdapter"] is not _api.requestAdapter:
        raise RuntimeError("WGPU backend can only be set once.")
    globals()["requestAdapter"] = func
    # todo: auto-select upon using requestAdapter?


def requestAdapterSync(options):
    """ A convenience function.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(requestAdapter(options))
