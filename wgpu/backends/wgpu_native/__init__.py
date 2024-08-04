"""
The wgpu-native backend.
"""

from ._api import *  # noqa: F401, F403
from ._ffi import ffi, lib, lib_path, lib_version_info  # noqa: F401
from ._ffi import _check_expected_version
from .. import _register_backend


# The wgpu-native version that we target/expect
__version__ = "22.1.0"
__commit_sha__ = "7304430548b11110b157bb29f620bd0b3e8e77f1"
version_info = tuple(map(int, __version__.split(".")))
_check_expected_version(version_info)  # produces a warning on mismatch

# Instantiate and register this backend
gpu = GPU()  # noqa: F405
_register_backend(gpu)  # noqa: F405

from .extras import enumerate_adapters, request_device_tracing  # noqa: F401, E402
