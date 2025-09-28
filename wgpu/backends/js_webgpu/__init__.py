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

from js import window, Uint32Array, ArrayBuffer, Float32Array

def to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


# for use in to_js() https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.ToJsConverter
# you have to do the recursion yourself...
def simple_js_accessor(value, convert, cache):
    if isinstance(value, classes.GPUObjectBase):
        return value._internal
        # print("converted to js", value)
    elif isinstance(value, structs.Struct):
        result = {}
        # cache(value, result)
        for k, v in value.__dict__.items():
            camel_key = to_camel_case(k)
            result[camel_key] = convert(v)
        return result
    elif isinstance(value, dict):
        result = {}
        # cache(value, result)
        for k, v in value.items():
            camel_key = to_camel_case(k) if isinstance(k, str) else k
            result[camel_key] = convert(v)
        return result
    elif isinstance(value, (tuple, list)):
        result = []
        # cache(value, result)
        for v in value:
            result.append(convert(v))
        return result

    return convert(value)

# TODO: can we implement our own variant of JsProxy and PyProxy, to_js and to_py? to work with pyodide and not around it?
# https://pyodide.org/en/stable/usage/type-conversions.html#type-translations

class GPU(classes.GPU):
    def __init__(self):
        self._internal = window.navigator.gpu  # noqa: F821

    def request_adapter_sync(self, **parameters):
        return run_sync(self.request_adapter_async(**parameters))
        # raise NotImplementedError("Cannot use sync API functions in JS.")

    async def request_adapter_async(self, **parameters):
        js_adapter = await self._internal.requestAdapter(**parameters)


        return GPUAdapter(
            js_adapter,
        )

    # api diff not really useful, but needed for compatibility I guess?
    def enumerate_adapters_sync(self):
        return run_sync(self.enumerate_adapters_async())

    async def enumerate_adapters_async(self):
        # bodge here: it blocks but we should await instead.
        return [self.request_adapter_sync()]

    @property
    def wgsl_language_features(self):
        return self._internal.wgslLanguageFeatures


class GPUAdapter(classes.GPUAdapter):
    def __init__(self, js_adapter):
        internal = js_adapter
        # manually turn these into useful python objects
        features = set(js_adapter.features)

        # TODO: _get_limits()?
        limits = js_adapter.limits
        py_limits = {}
        for limit in dir(limits):
            # we don't have the GPUSupportedLimits as a struct or list any where in the code right now, maybe we un skip it in the codegen?
            if isinstance(getattr(limits, limit), int) and "_" not in limit:
                py_limits[limit] = getattr(limits, limit)

        infos = ["vendor", "architecture", "device", "description", "subgroupMinSize", "subgroupMaxSize", "isFallbackAdapter"]
        adapter_info = js_adapter.info
        py_adapter_info = {}
        for info in infos:
            if hasattr(adapter_info, info):
                py_adapter_info[info] = getattr(adapter_info, info)

        #for compatibility, we fill the native-extra infos too:
        py_adapter_info["vendor_id"] = 0
        py_adapter_info["device_id"] = 0
        py_adapter_info["adapter_type"] = "browser"
        py_adapter_info["backend_type"] = "WebGPU"

        adapter_info = classes.GPUAdapterInfo(**py_adapter_info)

        super().__init__(internal=internal, features=features, limits=py_limits, adapter_info=adapter_info)



    def request_device_sync(self, **parameters):
        return run_sync(self.request_device_async(**parameters))
        # raise NotImplementedError("Cannot use sync API functions in JS.")

    async def request_device_async(self, **parameters):
        label = parameters.get("label", "")
        js_device = await self._internal.requestDevice(**parameters)
        default_queue = parameters.get("default_queue", {})
        return GPUDevice(label, js_device, adapter=self)


class GPUDevice(classes.GPUDevice):
    def __init__(self, label:str, js_device, adapter:GPUAdapter):
        features = set(js_device.features)

        js_limits = js_device.limits
        limits = {}
        for limit in dir(js_limits):
            if isinstance(getattr(js_limits, limit), int) and "_" not in limit:
                limits[limit] = getattr(js_limits, limit)

        queue = GPUQueue(label="default queue", internal=js_device.queue, device=self)
        super().__init__(label, internal=js_device, adapter=adapter, features=features, limits=limits, queue=queue)

    # API diff: useful to have?
    @property
    def adapter(self):
        return self._adapter

    def create_shader_module(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_sm = self._internal.createShaderModule(*js_args, js_kwargs)
        return GPUShaderModule(js_sm)

    def create_buffer(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_buf = self._internal.createBuffer(*js_args, js_kwargs)
        return GPUBuffer(js_buf)

    # TODO: apidiff
    def create_buffer_with_data(self, *args, **kwargs):
        kwargs["mappedAtCreation"] = True
        data = kwargs.get("data")
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
        kwargs["size"] = data_size
        # print(data_size)
        kwargs["label"] = "input buffer"
        js_buf = self._internal.createBuffer(*args, **kwargs)
        # TODO: dtype?
        Uint32Array.new(js_buf.getMappedRange()).set(kwargs["data"])
        js_buf.unmap()
        # print(dir(js_buf))
        return GPUBuffer(js_buf)

    # because there is no default and it has to be one of the binding group layouts this might need a custom check -.-
    def create_bind_group_layout(self, *args, **kwargs):
        # print("create_bind_group_layout", args, kwargs)
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        # print("JS create_bind_group_layout", js_args, js_kwargs, type(js_kwargs["entries"]))
        js_bgl = self._internal.createBindGroupLayout(*js_args, js_kwargs)
        return GPUBindGroupLayout(js_bgl)

    def create_compute_pipeline(self, *args, **kwargs):
        # TODO: can we automatically get the js object when it's called somehwere? maybe by implementing _to_js?
        # print("create_compute_pipeline", args, kwargs)
        # kwargs["compute"]["module"] = kwargs["compute"]["module"].to_js()
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        # print("create_compute_pipeline", args, js_kwargs)
        # print(dir(js_kwargs))
        js_cp = self._internal.createComputePipeline(*args, js_kwargs)
        return GPUComputePipeline(js_cp)

    def create_bind_group(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_bg = self._internal.createBindGroup(*js_args, js_kwargs)
        return GPUBindGroup(js_bg)

    def create_command_encoder(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_ce = self._internal.createCommandEncoder(*args, js_kwargs)
        return GPUCommandEncoder(js_ce)

    def create_pipeline_layout(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_pl = self._internal.createPipelineLayout(*js_args, js_kwargs)
        return GPUPipelineLayout(js_pl)

    def create_texture(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_tex = self._internal.createTexture(*args, js_kwargs)
        return GPUTexture(js_tex)

    def create_sampler(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_samp = self._internal.createSampler(*args, js_kwargs)
        return GPUSampler(js_samp)

    # breaks because we access the same module twice and might be losing it to GC or something -.-
    def create_render_pipeline(self, *args, **kwargs):
        # print("create_render_pipeline", args, kwargs)
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        # print("JS create_render_pipeline", js_args, js_kwargs)
        js_rp = self._internal.createRenderPipeline(*js_args, js_kwargs)
        return GPURenderPipeline(js_rp)

class GPUShaderModule(classes.GPUShaderModule):
    def __init__(self, js_sm):
        self._js = js_sm

    @property
    def js(self):
        return self._js  # hope we don't lose the reference after one access?

    # part of base object because we never call super().__init__() on ours
    @property
    def _label(self):
        return self._internal.label

class GPUBuffer(classes.GPUBuffer):
    def __init__(self, js_buf):
        self._internal = js_buf

    # TODO apidiff
    def write_mapped(self, data, buffer_offset: int | None = None):
        # TODO: get dtype
        js_buffer_array = Float32Array.new(data.nbytes)
        Float32Array.new(self._internal.getMappedRange(buffer_offset)).set(data)

    def map_sync(self, mode, offset=0, size=None):
        return run_sync(self.map_async(mode, offset, size))

    async def map_async(self, mode, offset: int = 0, size: int | None = None):
        js_mode = to_js(mode, eager_converter=simple_js_accessor)
        res = await self._internal.mapAsync(js_mode, offset, size)
        return res

    def unmap(self):
        self._internal.unmap()

    # TODO: idl attributes round trip -.-
    @property
    def _size(self):
        # print("getting size", dir(self._internal), self._internal.size)
        return self._internal.size

class GPUBindGroupLayout(classes.GPUBindGroupLayout):
    def __init__(self, js_bgl):
        self._internal = js_bgl

# TODO: mixin class
class GPUComputePipeline(classes.GPUComputePipeline):
    def __init__(self, js_cp):
        self._internal = js_cp

    def get_bind_group_layout(self, *args, **kwargs):
        js_bgl = self._internal.getBindGroupLayout(*args, **kwargs)
        return GPUBindGroupLayout(js_bgl)

class GPUBindGroup(classes.GPUBindGroup):
    def __init__(self, js_bg):
        self._internal = js_bg

class GPUCommandEncoder(classes.GPUCommandEncoder):
    def __init__(self, js_ce):
        self._internal = js_ce

    def begin_compute_pass(self, *args, **kwargs):
        # TODO: no args, should be empty maybe?
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cp = self._internal.beginComputePass(*args, js_kwargs)
        return GPUComputePassEncoder(js_cp)
    
    def begin_render_pass(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_rp = self._internal.beginRenderPass(*js_args, js_kwargs)
        return GPURenderPassEncoder(js_rp)

    def finish(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cmd_buf = self._internal.finish(*js_args, js_kwargs)
        return GPUCommandBuffer(js_cmd_buf)
    
    def copy_buffer_to_buffer(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.copyBufferToBuffer(*js_args, js_kwargs)

class GPUComputePassEncoder(classes.GPUComputePassEncoder):
    def __init__(self, js_cp):
        self._internal = js_cp

    def set_pipeline(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.setPipeline(*js_args, js_kwargs)

    def set_bind_group(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.setBindGroup(*js_args, js_kwargs)

    def dispatch_workgroups(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.dispatchWorkgroups(*js_args, js_kwargs)

    def end(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.end(*args, js_kwargs)

class GPUCommandBuffer(classes.GPUCommandBuffer):
    def __init__(self, js_cb):
        self._internal = js_cb

class GPUQueue(classes.GPUQueue):
    # should be part of the base class init actually...
    # def __init__(self, label:str, js_queue:JsProxy, device: GPUDevice):
    #     super().__init__(label, internal=js_queue, device=device)
    #     self._internal = js_queue
    #     self._device = device #needed for the read_buffer api diff I guess

    def submit(self, command_buffers: list[GPUCommandBuffer]):
        self._internal.submit([cb._internal for cb in command_buffers])

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

        temp_buffer = device._internal.createBuffer(
            size=data_length,
            usage=flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ,
            mappedAtCreation=True,
            label="output buffer temp"
        )
        res = temp_buffer.getMappedRange()
        res = res.slice(0)
        temp_buffer.unmap()
        return res.to_py() # should give a memoryview?

    # this one misbehaves with args or kwargs, like it seems the data gets unpacked?
    def write_texture(self, destination, data, data_layout, size):
        # print("GPUQueue.write_texture called with", destination, data, data_layout, size)
        js_destination = to_js(destination, eager_converter=simple_js_accessor)
        js_data = ArrayBuffer.new(data.nbytes) # does this actually hold any data?
        # js_data = Uint32Array.new(js_data).set(data) # maybe like this????
        # print("data js type", type(js_data))
        # print(js_data)
        js_data_layout = to_js(data_layout, eager_converter=simple_js_accessor)
        js_size = to_js(size, eager_converter=simple_js_accessor)
        self._internal.writeTexture(js_destination, js_data, js_data_layout, js_size)

    # def write_texture(self, *args, **kwargs):
    #     js_args = to_js(args, eager_converter=simple_js_accessor)
    #     js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
    #     self._internal.writeTexture(*js_args, js_kwargs)


class GPUPipelineLayout(classes.GPUPipelineLayout):
    def __init__(self, js_pl):
        self._internal = js_pl

class GPUTexture(classes.GPUTexture):
    def __init__(self, js_tex):
        self._internal = js_tex

    def create_view(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_view = self._internal.createView(*args, js_kwargs)
        return GPUTextureView(js_view)

class GPUTextureView(classes.GPUTextureView):
    def __init__(self, js_view):
        self._internal = js_view

class GPUSampler(classes.GPUSampler):
    def __init__(self, js_samp):
        self._internal = js_samp

class GPUCanvasContext(classes.GPUCanvasContext):
    # TODO update rendercanvas.html get_context to work here?
    def __init__(self, canvas, present_methods):
        # the super init also does this... maybe we can call it?
        super().__init__(canvas, present_methods)

    @property
    def _internal(self) -> JsProxy:
        return self.canvas.html_context

    # undo the api diff
    def get_preferred_format(self, adapter: GPUAdapter | None) -> enums.TextureFormat:
        return gpu._internal.getPreferredCanvasFormat()

    def configure(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.configure(*js_args, js_kwargs)

    def get_current_texture(self):
        js_texture = self._internal.getCurrentTexture()
        return GPUTexture(js_texture)

class GPURenderPipeline(classes.GPURenderPipeline):
    def __init__(self, js_rp):
        self._internal = js_rp

    def get_bind_group_layout(self, index: int) -> GPUBindGroupLayout:
        return GPUBindGroupLayout(self._internal.getBindGroupLayout(index))

class GPURenderPassEncoder(classes.GPURenderPassEncoder):
    def __init__(self, js_rp):
        self._internal = js_rp

    def set_pipeline(self, pipeline: GPURenderPipeline):
        self._internal.setPipeline(pipeline._internal)

    def set_bind_group(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.setBindGroup(*js_args, js_kwargs)

    # maybe it needs to be way simpler?
    def draw(self, *args, **kwargs):
        self._internal.draw(*args)

    def end(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.end(*args, js_kwargs)

# finally register the backend
gpu = GPU()
_register_backend(gpu)
