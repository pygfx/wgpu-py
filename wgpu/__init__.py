"""
This a Python implementation of the next generation GPU API.
"""

from ._coreutils import help  # noqa: F401
from .flags import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .base import *  # noqa: F401,F403
from .gui import WgpuCanvasInterface  # noqa: F401,F403


__version__ = "0.3.0"
version_info = tuple(map(int, __version__.split(".")))


_gpu_backend = None


def _register_backend(gpu_cls):
    global _gpu
    if not (
        hasattr(gpu_cls, "request_adapter")
        and callable(gpu_cls.request_adapter)
        and hasattr(gpu_cls, "request_adapter_async")
        and callable(gpu_cls.request_adapter_async)
    ):
        raise RuntimeError(
            "The registered WGPU backend object must have methods "
            + "'request_adapter' and 'request_adapter_async'"
        )

    # Set gpu object and reset request_adapter-functions
    if globals()["_gpu_backend"]:
        raise RuntimeError("WGPU backend can only be set once.")
    gpu = gpu_cls()
    globals()["_gpu_backend"] = gpu
    globals()["request_adapter"] = gpu.request_adapter
    globals()["request_adapter_async"] = gpu.request_adapter_async
