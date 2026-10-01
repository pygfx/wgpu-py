"""
The Dawn backend.

Uses the compiled ``wgpu_dawn`` bindings (see ``dawn_bindings/`` in the
repository), which target native Dawn on desktop, and Emdawnwebgpu (i.e. the
browser's WebGPU) in Pyodide.
"""

# ruff: noqa: F401, F403

from ._api import *
from ._api import process_events
from ._ffi import ffi, lib, lib_path, lib_version_info, IS_WEB
from .. import _register_backend


# The Dawn release that we target/expect
__version__ = ".".join(str(i) for i in lib_version_info)
version_info = lib_version_info

# Instantiate and register this backend
gpu = GPU()  # noqa: F405
_register_backend(gpu)
