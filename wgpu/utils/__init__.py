"""
Higher level utilities. Must be explicitly imported from ``wgpu.utils.xx``.
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

from .._coreutils import BaseEnum  # noqa: F401

# The get_default_device() is so small and generally convenient that we import it by default.
from .device import get_default_device  # noqa: F401


class _StubModule:
    def __init__(self, module):
        self._module = module
        self.must_be_explicitly_imported = True

    def __getattr__(self, *args, **kwargs):
        raise RuntimeError(f"wgpu.utils.{self._module} must be explicitly imported.")

    def __repr__(self):
        return f"<Stub for wgpu.utils.{self._module} - {self._module} must be explicitly imported>"


# Create stubs


def compute_with_buffers(*args, **kwargs):
    raise DeprecationWarning(
        "wgpu.utils.compute_with_buffers() must now be imported from wgpu.utils.compute"
    )


compute = _StubModule("compute")
