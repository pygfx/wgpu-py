"""
Code to provide a canvas to render to.
"""

from .base import WgpuCanvasInterface, WgpuCanvasBase  # noqa: F401
from .events import EventTarget  # noqa: F401
from ._offscreen import WgpuOffscreenCanvas  # noqa: F401
