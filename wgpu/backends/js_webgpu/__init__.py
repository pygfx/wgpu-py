"""
WGPU backend implementation based on the JS WebGPU API.

Since the exposed Python API is the same as the JS API, except that
descriptors are arguments, this API can probably be fully automatically
generated.
"""

from .. import _register_backend
from ._api import * # includes gpu from _implementation?

gpu = GPU()
_register_backend(gpu)

print(help(gpu))
