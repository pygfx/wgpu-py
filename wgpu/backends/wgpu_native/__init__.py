"""
The wgpu-native backend.
"""

# ruff: noqa: F401, E402, F403

from ._api import *
from ._ffi import ffi, lib, lib_path, lib_version_info
from ._ffi import _check_expected_version
from .. import _register_backend


# The wgpu-native version that we target/expect
__version__ = "25.0.2.1"
__commit_sha__ = "af9074edf144efe4f1432b2e42c477429c4964c1"
version_info = tuple(map(int, __version__.split(".")))  # noqa: RUF048
_check_expected_version(version_info)  # produces a warning on mismatch

# Instantiate and register this backend
gpu = GPU()  # noqa: F405
_register_backend(gpu)

from .extras import request_device_sync, request_device
from ._helpers import WgpuAwaitable
