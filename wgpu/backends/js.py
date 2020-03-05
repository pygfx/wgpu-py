"""
Stub WGPU backend implementation based on JS WebGPU API.

Since the exposed Python API is the same as the JS API, except that
descriptors are arguments, this API can probably be fully automatically
generated.
"""

from .. import _register_backend

from pscript.stubs import window


def request_adapter(options):
    raise NotImplementedError("Cannot use sync API functions in JS.")


async def request_adapter_async(options):
    return await window.navigator.gpu.request_adapter(options)


# Mark as the backend on import time
_register_backend(request_adapter, request_adapter_async)
