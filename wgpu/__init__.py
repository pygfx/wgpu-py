"""
This a Python implementation of the next generation GPU API.
"""

from ._coreutils import help  # noqa: F401
from .flags import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .base import *  # noqa: F401,F403
from .gui import WgpuCanvasInterface  # noqa: F401,F403
from . import base


__version__ = "0.2.0"
version_info = tuple(map(int, __version__.split(".")))


def _register_backend(func, func_async):
    if not (callable(func) and func.__name__ == "request_adapter"):
        raise RuntimeError(
            "WGPU backend must be registered with function called request_adapter."
        )
    if not (callable(func_async) and func_async.__name__ == "request_adapter_async"):
        raise RuntimeError(
            "WGPU backend must be registered with function called request_adapter_async."
        )
    if globals()["request_adapter"] is not base.request_adapter:
        raise RuntimeError("WGPU backend can only be set once.")
    globals()["request_adapter"] = func
    globals()["request_adapter_async"] = func_async
