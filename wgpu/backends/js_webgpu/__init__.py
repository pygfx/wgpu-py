"""
WGPU backend implementation based on the JS WebGPU API.

Since the exposed Python API is the same as the JS API, except that
descriptors are arguments, this API can probably be fully automatically
generated.
"""

# NOTE: this is just a stub for now!!

from .. import _register_backend
from ... import classes, structs, enums, flags

from pyodide.ffi import run_sync, JsProxy, to_js

from js import window, Uint32Array, ArrayBuffer

def translate_python_methods(js_obj):
    # print("Translating methods for", js_obj)
    method_map = {}
    for attr in dir(js_obj):
        if "_" in attr:
            continue
        # print("attr", attr, type(attr))
        # print("attr value", getattr(js_obj, attr), type(getattr(js_obj, attr)))
        if isinstance(getattr(js_obj, attr), JsProxy):
            # print("maybe method", attr)
            #assume that is like a method?
            target = getattr(js_obj, attr)
            py_name = ''.join(['_' + c.lower() if c.isupper() else c for c in attr]).lstrip('_')
            method_map[py_name] = target
            py_name_async = py_name + '_async'
            method_map[py_name_async] = target
            py_name_sync = py_name + '_sync'
            method_map[py_name_sync] = lambda *args, target=target: run_sync(target(*args))

    for name, target in method_map.items():
        js_obj.name = target
        # setattr(js_obj, name, target)

    return js_obj

# for use in to_js() https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.ToJsConverter
def simple_js_accessor(value, convert, cache):
    if hasattr(value, "js"):
        value = value.js
        # print("converted to js", value)
    return convert(value)

# TODO: can we implement our own variant of JsProxy and PyProxy, to_js and to_py? to work with pyodide and not around it?
# https://pyodide.org/en/stable/usage/type-conversions.html#type-translations

class GPU(classes.GPU):
    def request_adapter_sync(self, **parameters):
        return run_sync(self.request_adapter_async(**parameters))
        # raise NotImplementedError("Cannot use sync API functions in JS.")

    async def request_adapter_async(self, **parameters):
        gpu = window.navigator.gpu  # noqa: F821
        self.js = gpu
        adapter = await gpu.requestAdapter(**parameters)
        # print(dir(adapter))
        # print(type(adapter.requestDevice))
        # adapter = translate_python_methods(adapter)
        # print(dir(adapter))
        py_adapter = GPUAdapter(adapter)
        # print(py_adapter, dir(py_adapter))
        return py_adapter

    @property
    def wgsl_language_features(self):
        return set()

class GPUAdapter(classes.GPUAdapter):
    def __init__(self, js_adapter):
        self.js = js_adapter

    def request_device_sync(self, **parameters):
        return run_sync(self.request_device_async(**parameters))
        # raise NotImplementedError("Cannot use sync API functions in JS.")

    async def request_device_async(self, **parameters):
        device = await self.js.requestDevice(**parameters)
        # device = translate_python_methods(device)
        return GPUDevice(device)

class GPUDevice(classes.GPUDevice):
    def __init__(self, js_device):
        self.js = js_device

    @property
    def queue(self):
        # TODO: maybe needs a class...
        return GPUQueue(self.js.queue, self)

    def create_shader_module(self, *args, **kwargs):
        # print("create_shader_module", args, kwargs)
        js_sm = self.js.createShaderModule(*args, **kwargs)
        return GPUShaderModule(js_sm)

    def create_buffer(self, *args, **kwargs):
        js_buf = self.js.createBuffer(*args, **kwargs)
        return GPUBuffer(js_buf)

    # TODO: apidiff
    def create_buffer_with_data(self, *args, **kwargs):
        kwargs["mappedAtCreation"] = True
        data = kwargs.get("data")
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
        kwargs["size"] = data_size
        print(data_size)
        kwargs["label"] = "input buffer"
        js_buf = self.js.createBuffer(*args, **kwargs)
        # TODO: dtype?
        Uint32Array.new(js_buf.getMappedRange()).set(kwargs["data"])
        js_buf.unmap()
        # print(dir(js_buf))
        return GPUBuffer(js_buf)

    def create_bind_group_layout(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_bgl = self.js.createBindGroupLayout(*args, **js_kwargs)
        return GPUBindGroupLayout(js_bgl)

    def create_compute_pipeline(self, *args, **kwargs):
        # TODO: can we automatically get the js object when it's called somehwere? maybe by implementing _to_js?
        # print("create_compute_pipeline", args, kwargs)
        # kwargs["compute"]["module"] = kwargs["compute"]["module"].to_js()
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        # print("create_compute_pipeline", args, js_kwargs)
        # print(dir(js_kwargs))
        js_cp = self.js.createComputePipeline(*args, **js_kwargs)
        return GPUComputePipeline(js_cp)

    def create_bind_group(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_bg = self.js.createBindGroup(*args, **js_kwargs)
        return GPUBindGroup(js_bg)

    def create_command_encoder(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_ce = self.js.createCommandEncoder(*args, **js_kwargs)
        return GPUCommandEncoder(js_ce)

    def create_pipeline_layout(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_pl = self.js.createPipelineLayout(*js_args, **js_kwargs)
        return GPUPipelineLayout(js_pl)

class GPUShaderModule(classes.GPUShaderModule):
    def __init__(self, js_sm):
        self.js = js_sm

class GPUBuffer(classes.GPUBuffer):
    def __init__(self, js_buf):
        self.js = js_buf

    # TODO apidiff
    def write_mapped(self, *args, **kwargs):
        raise NotImplementedError("write_mapped not implemented yet in JS backend")

    # TODO: idl attributes round trip -.-
    @property
    def _size(self):
        # print("getting size", dir(self.js), self.js.size)
        return self.js.size

class GPUBindGroupLayout(classes.GPUBindGroupLayout):
    def __init__(self, js_bgl):
        self.js = js_bgl

# TODO: mixin class
class GPUComputePipeline(classes.GPUComputePipeline):
    def __init__(self, js_cp):
        self.js = js_cp

    def get_bind_group_layout(self, *args, **kwargs):
        js_bgl = self.js.getBindGroupLayout(*args, **kwargs)
        return GPUBindGroupLayout(js_bgl)

class GPUBindGroup(classes.GPUBindGroup):
    def __init__(self, js_bg):
        self.js = js_bg

class GPUCommandEncoder(classes.GPUCommandEncoder):
    def __init__(self, js_ce):
        self.js = js_ce

    def begin_compute_pass(self, *args, **kwargs):
        # TODO: no args, should be empty maybe?
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cp = self.js.beginComputePass(*args, **js_kwargs)
        return GPUComputePassEncoder(js_cp)

    def finish(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cmd_buf = self.js.finish(*js_args, **js_kwargs)
        return GPUCommandBuffer(js_cmd_buf)

class GPUComputePassEncoder(classes.GPUComputePassEncoder):
    def __init__(self, js_cp):
        self.js = js_cp

    def set_pipeline(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self.js.setPipeline(*js_args, **js_kwargs)

    def set_bind_group(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self.js.setBindGroup(*js_args, **js_kwargs)

    def dispatch_workgroups(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self.js.dispatchWorkgroups(*js_args, **js_kwargs)

    def end(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self.js.end(*args, **js_kwargs)

class GPUCommandBuffer(classes.GPUCommandBuffer):
    def __init__(self, js_cb):
        self.js = js_cb

class GPUQueue(classes.GPUQueue):
    def __init__(self, js_queue, device: GPUDevice):
        self.js = js_queue
        self._device = device #needed for the read_buffer api diff I guess

    def submit(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self.js.submit(*js_args, **js_kwargs)

    # TODO: api diff
    def read_buffer(self, buffer: GPUBuffer, buffer_offset: int=0, size: int | None = None) -> memoryview:
        # largely copied from wgpu-native/_api.py
        # print(dir(self))
        device = self._device

        if not size:
            data_length = buffer.size - buffer_offset
        else:
            data_length = int(size)
        if not (0 <= buffer_offset < buffer.size):  # pragma: no cover
            raise ValueError("Invalid buffer_offset")
        if not (data_length <= buffer.size - buffer_offset):  # pragma: no cover
            raise ValueError("Invalid data_length")
        data_length = (data_length + 3) & ~3  # align to 4 bytes

        temp_buffer = device.js.createBuffer(
            size=data_length,
            usage=flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ,
            mappedAtCreation=True,
            label="output buffer temp"
        )
        res = temp_buffer.getMappedRange()
        res = res.slice(0)
        temp_buffer.unmap()
        return res.to_py() # should give a memoryview?

class GPUPipelineLayout(classes.GPUPipelineLayout):
    def __init__(self, js_pl):
        self.js = js_pl

gpu = GPU()
_register_backend(gpu)
