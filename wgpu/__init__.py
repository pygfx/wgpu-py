"""
WebGPU for Python.
"""

from ._coreutils import logger  # noqa: F401,F403
from ._diagnostics import diagnostics, DiagnosticsBase  # noqa: F401,F403
from .flags import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .classes import *  # noqa: F401,F403
from .gui import WgpuCanvasInterface  # noqa: F401,F403
from . import utils  # noqa: F401,F403
from . import backends  # noqa: F401,F403
from . import resources  # noqa: F401,F403


__version__ = "0.18.1"
version_info = tuple(map(int, __version__.split(".")))


# The API entrypoint, from wgpu.classes - gets replaced when a backend loads.
gpu = GPU()  # noqa: F405


# Temporary stub to help transitioning
def request_adapter(*args, **kwargs):
    """Deprecated!"""
    raise DeprecationWarning(
        "wgpu.request_adapter() is deprecated! Use wgpu.gpu.request_adapter() instead."
    )
