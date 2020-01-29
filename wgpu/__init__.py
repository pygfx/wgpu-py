from .flags import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .base import *  # noqa: F401,F403
from ._coreutils import help  # noqa: F401
from . import base


__version__ = "0.1.3"


def _register_backend(func, func_async):
    if not (callable(func) and func.__name__ == "requestAdapter"):
        raise RuntimeError(
            "WGPU backend must be registered with function called requestAdapterSync."
        )
    if not (callable(func_async) and func_async.__name__ == "requestAdapterAsync"):
        raise RuntimeError(
            "WGPU backend must be registered with function called requestAdapterAsync."
        )
    if globals()["requestAdapter"] is not base.requestAdapter:
        raise RuntimeError("WGPU backend can only be set once.")
    globals()["requestAdapter"] = func
    globals()["requestAdapterAsync"] = func_async

    # todo: auto-select upon using requestAdapter?
