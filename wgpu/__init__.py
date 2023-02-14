"""
The wgpu library is a Python implementation of WebGPU.
"""

from ._coreutils import logger  # noqa: F401,F403
from .flags import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .base import *  # noqa: F401,F403
from .gui import WgpuCanvasInterface  # noqa: F401,F403

__version__ = "0.9.1"
version_info = tuple(map(int, __version__.split(".")))


def _register_backend(cls):
    """Backends call this to acticate themselves."""
    GPU = cls  # noqa: N806
    if not (
        hasattr(GPU, "request_adapter")
        and callable(GPU.request_adapter)
        and hasattr(GPU, "request_adapter_async")
        and callable(GPU.request_adapter_async)
    ):
        raise RuntimeError(
            "The registered WGPU backend object must have methods "
            + "'request_adapter' and 'request_adapter_async'"
        )

    # Set gpu object and reset request_adapter-functions
    if globals()["GPU"] is not _base_GPU:
        raise RuntimeError("WGPU backend can only be set once.")
    gpu = GPU()
    globals()["GPU"] = GPU
    globals()["request_adapter"] = gpu.request_adapter
    globals()["request_adapter_async"] = gpu.request_adapter_async
    if hasattr(gpu, "print_report"):
        globals()["print_report"] = gpu.print_report
    else:
        globals()["print_report"] = _base_GPU.print_report


_base_GPU = GPU  # noqa: F405, N816
_register_backend(_base_GPU)
