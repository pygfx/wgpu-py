"""
The wgpu-native backend.
"""

# ruff: noqa: F401, E402, F403

from ._api import *
from ._ffi import ffi, lib, lib_path, lib_version_info
from ._ffi import _check_expected_version
from .. import _register_backend


# The wgpu-native version that we target/expect
__version__ = "27.0.4.0"
__commit_sha__ = "768f15f6ace8e4ec8e8720d5732b29e0b34250a8"
version_info = tuple(map(int, __version__.split(".")))  # noqa: RUF048
_check_expected_version(version_info)  # produces a warning on mismatch

# Instantiate and register this backend
gpu = GPU()  # noqa: F405
_register_backend(gpu)

from .extras import request_device_sync, request_device
