"""
Higher level utility functions. This module is not imported by default.
"""

# The purpose of wgpu-py is to provide a Pythonic wrapper around
# wgpu-native. In principal, a higher-level API is not within the scope
# of the project. However, by providing a few utility functions, other
# projects (like python-shader) can use wgpu (e.g. in their tests)
# without having to keep track of changes in wgpu itself.
#
# We should be conservative here: functionality added here should have an
# unopinionated API, providing tools that are still low-level (follow
# GPU/wgpu semantics), but without using low level details of the wgpu
# API itself.

from ._compute import compute_with_buffers  # noqa: F401
from ._device import create_device  # noqa: F401
