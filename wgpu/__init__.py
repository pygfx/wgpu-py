"""
WebGPU for Python.
"""

# ruff: noqa: F401, F403

from ._coreutils import logger
from ._version import __version__, version_info
from ._diagnostics import diagnostics, DiagnosticsBase
from .flags import *
from .enums import *
from .classes import *
from .gui import WgpuCanvasInterface
from . import utils
from . import backends
from . import resources

# The API entrypoint, from wgpu.classes - gets replaced when a backend loads.
gpu = GPU()  # noqa: F405


def rendercanvas_context_hook(canvas, present_methods):
    import sys

    backend_module = gpu.__module__
    if backend_module in ("", "wgpu._classes"):
        # Load backend now
        from .backends import auto

        backend_module = gpu.__module__

    return sys.modules[backend_module].GPUCanvasContext(canvas, present_methods)
