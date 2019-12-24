"""
Stub WGPU backend implementation based on JS WebGPU API.

Since the exposed Python API is the same as the JS API, except that
descriptors are arguments, this API can probably be fully automatically
generated.
"""

from .. import _register_backend

from pscript.stubs import window


async def requestAdapter(options):
    return await window.navigator.gpu.requestAdapter(options)


# Mark as the backend on import time
_register_backend(requestAdapter)
