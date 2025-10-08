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
from js import window, Uint32Array, ArrayBuffer, Float32Array, Uint8Array, BigInt, Object, undefined


def to_camel_case(snake_str):
    components = snake_str.split('_')
    res = components[0] + ''.join(x.title() for x in components[1:])
    # maybe keywords are a problem? 
    # https://pyodide.org/en/stable/usage/faq.html#how-can-i-access-javascript-objects-attributes-in-python-if-their-names-are-python-keywords
    # if res in ["type", "format"]:
    #     res += "_"
    return res


# for use in to_js() https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.ToJsConverter
# you have to do the recursion yourself...
def simple_js_accessor(value, convert, cache):
    # print("simple_js_accessor", value, type(value), dir(value))
    if isinstance(value, classes.GPUObjectBase):
        # print("GPUObjectBase detected", value)
        return value._internal # type : JsProxy
    elif isinstance(value, structs.Struct):
        # print("struct detected", value, "\n")
        result = {}
        # cache(value, result)
        for k, v in value.items():
            # print("struct item", k, v, type(v))
            camel_key = to_camel_case(k)
            result[camel_key] = convert(v)
        return result
    # this might recursively call itself...
    # maybe use a map? or do a dict_converted?
    elif isinstance(value, dict):
        result = {}
        # cache(value, result)
        for k, v in value.items():
            camel_key = to_camel_case(k) if isinstance(k, str) else k
            result[camel_key] = convert(v)
        if len(result) == 0:
            return Object.new() # maybe this?
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
        label = kwargs.get("label", "")
        # we only need to implement the class if we implement the methods I guess?
        return classes.GPUShaderModule(label, js_sm, self)

    def create_buffer(self, *args, **kwargs):
        # js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_buf = self._internal.createBuffer(js_kwargs)

        label = kwargs.get("label", "")
        size = kwargs.get("size")
        usage = kwargs.get("usage")
        map_state = (
            enums.BufferMapState.mapped
            if kwargs.get("mapped_at_creation", False)
            else enums.BufferMapState.unmapped
        )
        return GPUBuffer(label, js_buf, self, size, usage, map_state)

    # TODO: apidiff rewritten so we avoid the buggy mess in map_write for a bit.
    def create_buffer_with_data(self, *, label="", data, usage: flags.BufferUsageFlags) -> classes.GPUBuffer:

        data = memoryview(data).cast("B") # unit8
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes

        # if it's a Descriptor you need the keywords
        js_buf = self._internal.createBuffer(label=label, size=data_size, usage=usage, mappedAtCreation=True)
        mapping_buffer = Uint8Array.new(js_buf.getMappedRange(0, data_size))
        mapping_buffer.assign(data) #.set only works with JS array I think...
        js_buf.unmap()
        return GPUBuffer(label, js_buf, self, data_size, usage, enums.BufferMapState.unmapped)

    # or here???
    def create_bind_group_layout(self, *, label: str = "", entries: list[structs.BindGroupLayoutEntryStruct]) -> classes.GPUBindGroupLayout:
        empty_value = undefined # figure out what pyodide is happy with
        js_entries = []
        for entry in entries:
            # we need exactly one of them needs to exist:
            # https://www.w3.org/TR/webgpu/#dictdef-gpubindgrouplayoutentry
            buffer = entry.get("buffer")
            sampler = entry.get("sampler")
            texture = entry.get("texture")
            storage_texture = entry.get("storage_texture")
            external_texture = entry.get("external_texture") # not sure if exists in wgpu-native, but let's have it anyway.

            if buffer is not None:
                sampler = texture = storage_texture = external_texture = empty_value
                # or struct.BufferBindingLayout?
                buffer = {
                    "type": buffer.get("type", enums.BufferBindingType.uniform),
                    "hasDynamicOffset": buffer.get("has_dynamic_offset", False),
                    "minBindingSize": buffer.get("min_binding_size", 0)
                }
                buffer = to_js(buffer, depth=1)
            elif sampler is not None:
                buffer = texture = storage_texture = external_texture = empty_value
                sampler = {
                    "type": sampler.get("type", enums.SamplerBindingType.filtering),
                }
                sampler = to_js(sampler, depth=1)
            elif texture is not None:
                buffer = sampler = storage_texture = external_texture = empty_value
                texture = {
                    "sampleType": texture.get("sample_type", enums.TextureSampleType.float),
                    "viewDimension": texture.get("view_dimension", enums.TextureViewDimension.d2),
                    "multisampled": texture.get("multisampled", False),
                }
                texture = to_js(texture, depth=1)
            elif storage_texture is not None:
                buffer = sampler = texture = external_texture = empty_value
                storage_texture = {
                    "access": storage_texture.get("access", enums.StorageTextureAccess.write_only),
                    "format": storage_texture.get("format"),
                    "viewDimension": storage_texture.get("view_dimension", enums.TextureViewDimension.d2),
                }
                storage_texture = to_js(storage_texture, depth=1)
            elif external_texture is not None:
                buffer = sampler = texture = storage_texture = empty_value
                external_texture = {
                    # https://www.w3.org/TR/webgpu/#dictdef-gpuexternaltexturebindinglayout
                    # there is nothing here... which makes this an empty dict/set?
                }
            else:
                raise ValueError(
                    "BindGroupLayoutEntry must have exactly one of buffer, sampler, texture, storage_texture, external_texture set. Got none."
                    )
            js_entry = {
                "binding": entry.get("binding"),
                "visibility": entry.get("visibility"),
                "buffer": buffer,
                "sampler": sampler,
                "texture": texture,
                "storage_texture": storage_texture,
                "external_texture": external_texture,
            }
            js_entries.append(js_entry)

        js_bgl = self._internal.createBindGroupLayout(label=label, entries=js_entries)
        return classes.GPUBindGroupLayout(label, js_bgl, self)

    def create_compute_pipeline(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cp = self._internal.createComputePipeline(*args, js_kwargs)

        label = kwargs.get("label", "")
        return GPUComputePipeline(label, js_cp, self)

    # I think the entries arg gers unpacked with a single dict inside, so trying to do the list around that manually
    def create_bind_group(self, **kwargs) -> classes.GPUBindGroup:
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_kwargs = to_js(js_kwargs) # to get the actual map? (should happen on the function call anyways...)
        js_bg = self._internal.createBindGroup(js_kwargs)

        label = kwargs.get("label", "")
        return classes.GPUBindGroup(label, js_bg, self)

    def create_command_encoder(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_ce = self._internal.createCommandEncoder(*args, js_kwargs)

        label = kwargs.get("label", "")
        return GPUCommandEncoder(label, js_ce, self)

    # or was it here?
    def create_pipeline_layout(self, *, label="", bind_group_layouts: list[classes.GPUBindGroupLayout]) -> classes.GPUPipelineLayout:
        js_bind_group_layouts = [to_js(bgl, eager_converter=simple_js_accessor) for bgl in bind_group_layouts]
        js_pl = self._internal.createPipelineLayout(label=label, bindGroupLayouts=js_bind_group_layouts)

        return classes.GPUPipelineLayout(label, js_pl, self)

    def create_texture(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_tex = self._internal.createTexture(*args, js_kwargs)

        label = kwargs.get("label", "")
        tex_info = {
            "size": kwargs.get("size"),
            "mip_level_count": kwargs.get("mip_level_count", 1),
            "sample_count": kwargs.get("sample_count", 1),
            "dimension": kwargs.get("dimension", "2d"),
            "format": kwargs.get("format"),
            "usage": kwargs.get("usage"),
        }

        return GPUTexture(label, js_tex, self, tex_info)

    def create_sampler(self, *args, **kwargs) -> classes.GPUSampler:
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_samp = self._internal.createSampler(*args, js_kwargs)

        label = kwargs.get("label", "")
        return classes.GPUSampler(label, js_samp, self)

    # breaks because we access the same module twice and might be losing it to GC or something -.-
    def create_render_pipeline(self, *args, **kwargs):
        # let's try to call to_js multiple times to maybe avoid caching,
        # vertex is a required argument, so can we just take it out like this?
        # maybe we can use a depth limit or something...
        kwargs["vertex"] = to_js(kwargs["vertex"], eager_converter=simple_js_accessor)
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)

        # js_kwargs = to_js(js_kwargs) # to get the actual map?
        js_rp = self._internal.createRenderPipeline(*js_args, js_kwargs)

        label = kwargs.get("label", "")
        return GPURenderPipeline(label, js_rp, self)

class GPUBuffer(classes.GPUBuffer):

    # since we actually got the real one:
    @property
    def map_state(self):
        return self._internal.mapState

    # TODO apidiff
    def write_mapped(self, data, buffer_offset: int | None = None):
        # TODO: get dtype
        if self.map_state != enums.BufferMapState.mapped:
            raise RuntimeError(f"Can only write to a buffer if its mapped: {self.map_state=}")

        # make sure it's in a known datatype???
        data = memoryview(data).cast("B")
        size = (data.nbytes + 3) & ~3

        # GPUSIze64 type works as pyton int, can't we just make 0 a default?
        if buffer_offset is None:
            buffer_offset = 0

        # these can't be passed as keyword arguments I guess...
        array_buf = self._internal.getMappedRange(buffer_offset, size)
        Uint8Array.new(array_buf).assign(data)

    def map_sync(self, mode=None, offset=0, size=None):
        return run_sync(self.map_async(mode, offset, size))

    async def map_async(self, mode: flags.MapModeFlags | None, offset: int = 0, size: int | None = None):
        res = await self._internal.mapAsync(mode, offset, size)
        return res

    def unmap(self):
        self._internal.unmap()

# TODO: mixin class
class GPUComputePipeline(classes.GPUComputePipeline):
    def get_bind_group_layout(self, *args, **kwargs):
        js_bgl = self._internal.getBindGroupLayout(*args, **kwargs)

        label = kwargs.get("label", "")
        return classes.GPUBindGroupLayout(label, js_bgl, self._device)

class GPUCommandEncoder(classes.GPUCommandEncoder):
    def begin_compute_pass(self, *args, **kwargs):
        # TODO: no args, should be empty maybe?
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cp = self._internal.beginComputePass(*args, js_kwargs)

        label = kwargs.get("label", "")
        return GPUComputePassEncoder(label, js_cp, self._device)

    def begin_render_pass(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_rp = self._internal.beginRenderPass(*js_args, js_kwargs)

        label = kwargs.get("label", "")
        return GPURenderPassEncoder(label, js_rp, self._device)

    def finish(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_cmd_buf = self._internal.finish(*js_args, js_kwargs)

        label = kwargs.get("label", "")
        return classes.GPUCommandBuffer(label, js_cmd_buf, self._device)

    def copy_buffer_to_buffer(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.copyBufferToBuffer(*js_args, js_kwargs)

class GPUComputePassEncoder(classes.GPUComputePassEncoder):
    def set_pipeline(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.setPipeline(*js_args, js_kwargs)

    # function has overloads!
    def set_bind_group(
            self,
            index:int,
            bind_group: classes.GPUBindGroup,
            dynamic_offsets_data: list[int] = (),
            dynamic_offsets_data_start = None,
            dynamic_offsets_data_length = None
            ) -> None:

        self._internal.setBindGroup(index, bind_group._internal, dynamic_offsets_data)

    def dispatch_workgroups(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.dispatchWorkgroups(*js_args, js_kwargs)

    def end(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.end(*args, js_kwargs)

class GPUQueue(classes.GPUQueue):
    def submit(self, command_buffers: list[classes.GPUCommandBuffer]):
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
    def write_texture(self,
        destination: structs.TexelCopyTextureInfoStruct | None = None,
        data: memoryview | None = None,
        data_layout: structs.TexelCopyBufferLayoutStruct | None = None,
        size: tuple[int, int, int] | structs.Extent3DStruct | None = None,
    ) -> None:
        js_destination = to_js(destination, eager_converter=simple_js_accessor)

        data = memoryview(data).cast("B")
        data_size = (data.nbytes + 3) & ~3  # align to
        js_data = Uint8Array.new(data_size)
        js_data.assign(data)


        js_data_layout = to_js(data_layout, eager_converter=simple_js_accessor)
        js_size = to_js(size, eager_converter=simple_js_accessor)

        self._internal.writeTexture(js_destination, js_data, js_data_layout, js_size)

    def write_buffer(
        self,
        buffer: GPUBuffer | None = None,
        buffer_offset: int | None = None,
        data: memoryview | None = None,
        data_offset: int = 0,
        size: int | None = None,
    ):
        data = memoryview(data).cast("B")
        if size is None:
            size = data.nbytes - data_offset
        size = (size + 3) & ~3  # align to 4 bytes

        if buffer_offset is None:
            buffer_offset = 0

        js_data = Uint8Array.new(size)
        js_data.assign(data[data_offset:data_offset+size])

        self._internal.writeBuffer(buffer._internal, buffer_offset, js_data, 0, size)



class GPUTexture(classes.GPUTexture):
    def create_view(self, *args, **kwargs):
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        js_view = self._internal.createView(*args, js_kwargs)

        label = kwargs.get("label", "")
        return classes.GPUTextureView(label, js_view, self._device, self, self.size)

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
        self._config = {
            "device": kwargs.get("device"),
            "format": kwargs.get("format"),
            "usage": kwargs.get("usage", 0x10),
            "view_formats": kwargs.get("view_formats", ()),
            "color_space": kwargs.get("color_space", "srgb"),
            "tone_mapping": kwargs.get("tone_mapping", None),
            "alpha_mode": kwargs.get("alpha_mode", "opaque"),
        }

    def get_current_texture(self):
        js_texture = self._internal.getCurrentTexture()

        label = "" # always empty?
        device = self._config["device"]
        text_info = {
            "size": (js_texture.width, js_texture.height, js_texture.depthOrArrayLayers),
            "mip_level_count": js_texture.mipLevelCount,
            "sample_count": js_texture.sampleCount,
            "dimension": js_texture.dimension,
            "format": js_texture.format,
            "usage": js_texture.usage,
        }
        return GPUTexture(label, js_texture, device, text_info)

class GPURenderPipeline(classes.GPURenderPipeline):
    def get_bind_group_layout(self, index: int | None = None) -> classes.GPUBindGroupLayout:
        return classes.GPUBindGroupLayout("", self._internal.getBindGroupLayout(index), self._device)

# TODO: abstract to mixin
class GPURenderPassEncoder(classes.GPURenderPassEncoder):
    def set_pipeline(self, pipeline: GPURenderPipeline):
        self._internal.setPipeline(pipeline._internal)

    def set_index_buffer(self, buffer: GPUBuffer, format: enums.IndexFormat, offset: int = 0, size: int | None= None):
        # for GPUSize64 you can't pass them as kwargs, as they get converted to something else...
        # they need to be position args and then it works.
        js_buffer = buffer._internal
        js_format = to_js(format)
        if size is None:
            size = buffer.size - offset

        self._internal.setIndexBuffer(js_buffer, js_format, offset, size)

    def set_vertex_buffer(self, slot, buffer: GPUBuffer, offset=0, size: int | None = None):
        # slot is a GPUsize32 so that works, but the others don't
        if size is None:
            size = buffer.size - offset

        self._internal.setVertexBuffer(slot, buffer._internal, offset, size)

    # function has overloads!
    def set_bind_group(
            self,
            index:int,
            bind_group: classes.GPUBindGroup,
            dynamic_offsets_data: list[int] = (),
            dynamic_offsets_data_start = None,
            dynamic_offsets_data_length = None
            ) -> None:

        self._internal.setBindGroup(index, bind_group._internal, dynamic_offsets_data)

    def set_viewport(self, *args):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        self._internal.setViewport(*js_args)

    def set_blend_constant(self, color = None):
        self._internal.setBlendConstant(color)

    def set_scissor_rect(self, *args):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        self._internal.setScissorRect(*js_args)

    def draw_indexed(self, *args, **kwargs):
        js_args = to_js(args, eager_converter=simple_js_accessor)
        js_kwargs = to_js(kwargs, eager_converter=simple_js_accessor)
        self._internal.drawIndexed(*js_args, js_kwargs)

    # maybe it needs to be way simpler?
    def draw(self, *args, **kwargs):
        self._internal.draw(*args)

    def end(self):
        self._internal.end()

# finally register the backend
gpu = GPU()
_register_backend(gpu)
