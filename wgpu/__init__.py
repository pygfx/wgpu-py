"""
WebGPU for Python.
"""

# ruff: noqa: F401, F403

from ._coreutils import logger
from ._diagnostics import diagnostics, DiagnosticsBase
from .flags import *
from .enums import *
from .classes import *
from .gui import WgpuCanvasInterface
from . import utils
from . import backends
from . import resources


__version__ = "0.18.1"
version_info = tuple(map(int, __version__.split(".")))


# The API entrypoint, from wgpu.classes - gets replaced when a backend loads.
gpu = GPU()  # noqa: F405
