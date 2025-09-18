"""
WebGPU for Python.
"""

# ruff: noqa: F401, F403

from ._coreutils import logger
from ._version import __version__, version_info
from ._diagnostics import diagnostics, DiagnosticsBase
from .flags import *
from .enums import *
from .structs import *
from .classes import *
from . import utils
from . import backends
from . import resources

# The API entrypoint, from wgpu.classes - gets replaced when a backend loads.
gpu = GPU()  # noqa: F405


def rendercanvas_context_hook(canvas, present_methods):
    """Get a new GPUCanvasContext, given a canvas and present_methods dict.

    This is a hook for rendercanvas, so that it can support ``canvas.get_context("wgpu")``.

    See https://github.com/pygfx/wgpu-py/blob/main/wgpu/_canvas.py and https://rendercanvas.readthedocs.io/stable/contextapi.html.
    """

    import sys

    backend_module_name = gpu.__module__
    if backend_module_name in ("", "wgpu._classes"):
        # Load backend now
        from .backends import auto

        backend_module_name = gpu.__module__

    backend_module = sys.modules[backend_module_name]
    return backend_module.GPUCanvasContext(canvas, present_methods)
