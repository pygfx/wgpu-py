"""
WGPU backend implementation based on the JS WebGPU API.

Since the exposed Python API is the same as the JS API, except that
descriptors are arguments, this API can probably be fully automatically
generated.
"""

# NOTE: this is just a stub for now!!

from .. import _register_backend


class GPU:
    def request_adapter(self, **parameters):
        raise NotImplementedError("Cannot use sync API functions in JS.")

    async def request_adapter_async(self, **parameters):
        gpu = window.navigator.gpu  # noqa
        return await gpu.request_adapter(**parameters)

    def get_preferred_canvas_format(self):
        raise NotImplementedError()

    @property
    def wgsl_language_features(self):
        return set()


gpu = GPU()
_register_backend(gpu)
