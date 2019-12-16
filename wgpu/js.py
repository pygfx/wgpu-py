"""
Stub WGPU backend implementation based on JS WebGPU API.

If we manage to make our API (the method and attributes of the classes,
and enum and flag values) the same as WebGPU, then this should be all
we need to make vizualizations PScript-able.

If our API does deviate (for some reason), we need to write a thin
wrapper here. Also ok, but more work to write / maintain.
"""

from . import _register_backend


global window


async def requestAdapter(options):
    return await window.navigator.gpu.requestAdapter(options)


# Mark as the backend on import time
_register_backend(requestAdapter)
