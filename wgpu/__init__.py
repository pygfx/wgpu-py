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


def rendercanvas_context_hook(canvas, _):
    raise RuntimeError(
        "The rendercanvas_context_hook is deprecated. If you're using rendercanvas, please update to the latest version. Otherwise, use wgpu.gpu.get_canvas_context()"
    )
