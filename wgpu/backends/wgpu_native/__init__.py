"""
The wgpu-native backend.
"""

from ._api import *  # noqa: F401, F403
from ._ffi import ffi, lib, lib_path, lib_version_info  # noqa: F401
from ._ffi import _check_expected_version
from .. import _register_backend


# The wgpu-native version that we target/expect
__version__ = "22.1.0.5"
__commit_sha__ = "fad19f5990d8eb9a6e942eb957344957193fe66d"
version_info = tuple(map(int, __version__.split(".")))
_check_expected_version(version_info)  # produces a warning on mismatch

# Instantiate and register this backend
gpu = GPU()  # noqa: F405
_register_backend(gpu)  # noqa: F405

from .extras import enumerate_adapters, request_device_tracing  # noqa: F401, E402
