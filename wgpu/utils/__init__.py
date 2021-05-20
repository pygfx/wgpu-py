"""
Higher level utility functions. This module is not imported by default.
"""

# The purpose of wgpu-py is to provide a Pythonic wrapper around
# wgpu-native. In principal, a higher-level API is not within the scope
# of the project. However, by providing a few utility functions, other
# projects can use wgpu without having to keep track of changes in wgpu
# itself.
#
# We should be conservative here: functionality added here should have
# an unopinionated API, providing tools that are still low-level (follow
# GPU/wgpu semantics), but without using low level details of the wgpu
# API itself.

from ._device import get_default_device  # noqa: F401
from ._compute import compute_with_buffers  # noqa: F401
