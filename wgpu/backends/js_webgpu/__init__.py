"""
WGPU backend implementation based on the JS WebGPU API.

Since the exposed Python API is the same as the JS API, except that
descriptors are arguments, this API can probably be fully automatically
generated.
"""

from .. import _register_backend
from ._implementation import *



_register_backend(gpu)
