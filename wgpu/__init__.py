"""
The wgpu library is a Python implementation of WebGPU.
"""

from ._coreutils import logger  # noqa: F401,F403
from ._diagnostics import diagnostics  # noqa: F401,F403
from .flags import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .base import *  # noqa: F401,F403
from .gui import WgpuCanvasInterface  # noqa: F401,F403
from . import utils  # noqa: F401,F403
from . import backends  # noqa: F401,F403
from . import resources  # noqa: F401,F403


__version__ = "0.11.0"
version_info = tuple(map(int, __version__.split(".")))


# The API entrypoint from base.py - gets replaced when a backend loads.
gpu = GPU()  # noqa: F405


# Not sure yet whether we want below convenience functions or not.


def request_adapter(*args, **kwargs):
    """Convenience alias for ``gpu.request_adapter()``."""
    return gpu.request_adapter(*args, **kwargs)


def request_adapter_async(*args, **kwargs):
    """Convenience alias for ``gpu.request_adapter_async()``."""
    return gpu.request_adapter_async(*args, **kwargs)
