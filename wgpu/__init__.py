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
