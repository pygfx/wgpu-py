"""
Higher level utilities. Must be explicitly imported from ``wgpu.utils.xx``.
"""

# The purpose of wgpu-py is to provide a Pythonic wrapper around
# wgpu-native. In principle, a higher-level API is not within the scope
# of the project. However, by providing a few utility functions, other
# projects can use wgpu without having to keep track of changes in wgpu
# itself.
#
# We should be conservative here: functionality added here should have
# an unopinionated API, providing tools that are still low-level (follow
# GPU/wgpu semantics), but without using low level details of the wgpu
# API itself.

# ruff: noqa: F401

from .._coreutils import BaseEnum

# The get_default_device() is so small and generally convenient that we import it by default.
from .device import get_default_device
