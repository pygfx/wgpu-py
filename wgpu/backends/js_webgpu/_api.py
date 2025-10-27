
# Auto-generated API for the JS WebGPU backend, based on the IDL and custom implementations.

from ... import classes, structs, enums, flags
from ...structs import ArrayLike, Sequence # for typing hints
from typing import Union

from pyodide.ffi import to_js, run_sync, JsProxy
from js import window, Uint8Array

from ._helpers import simple_js_accessor
from ._implementation import GPUPromise


class GPUCommandsMixin(classes.GPUCommandsMixin, ):

    pass

class GPUBindingCommandsMixin(classes.GPUBindingCommandsMixin, ):

    # Custom implementation for setBindGroup from _implementation.py:
    def set_bind_group(self, index: int, bind_group: classes.GPUBindGroup, dynamic_offsets_data: list[int] = (), dynamic_offsets_data_start=None, dynamic_offsets_data_length=None) -> None:
        self._internal.setBindGroup(index, bind_group._internal, dynamic_offsets_data)


class GPUDebugCommandsMixin(classes.GPUDebugCommandsMixin, ):

    def push_debug_group(self, group_label: Union[str, None] = None) -> None:
    
        self._internal.pushDebugGroup(group_label)

    def pop_debug_group(self) -> None:
        self._internal.popDebugGroup()

    def insert_debug_marker(self, marker_label: Union[str, None] = None) -> None:
    
        self._internal.insertDebugMarker(marker_label)


class GPURenderCommandsMixin(classes.GPURenderCommandsMixin, ):

    def set_pipeline(self, pipeline: Union["GPURenderPipeline", None] = None) -> None:
        js_pipeline = pipeline._internal
        self._internal.setPipeline(js_pipeline)

    def set_index_buffer(self, buffer: Union["GPUBuffer", None] = None, index_format: enums.IndexFormatEnum | None = None, offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
        self._internal.setIndexBuffer(js_buffer, index_format, offset, size)

    def set_vertex_buffer(self, slot: Union[int, None] = None, buffer: Union["GPUBuffer", None] = None, offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
        self._internal.setVertexBuffer(slot, js_buffer, offset, size)

    def draw(self, vertex_count: Union[int, None] = None, instance_count: int = 1, first_vertex: int = 0, first_instance: int = 0) -> None:
    
        self._internal.draw(vertex_count, instance_count, first_vertex, first_instance)

    def draw_indexed(self, index_count: Union[int, None] = None, instance_count: int = 1, first_index: int = 0, base_vertex: int = 0, first_instance: int = 0) -> None:
    
        self._internal.drawIndexed(index_count, instance_count, first_index, base_vertex, first_instance)

    def draw_indirect(self, indirect_buffer: Union["GPUBuffer", None] = None, indirect_offset: Union[int, None] = None) -> None:
        js_indirectBuffer = indirect_buffer._internal
        self._internal.drawIndirect(js_indirectBuffer, indirect_offset)

    def draw_indexed_indirect(self, indirect_buffer: Union["GPUBuffer", None] = None, indirect_offset: Union[int, None] = None) -> None:
        js_indirectBuffer = indirect_buffer._internal
        self._internal.drawIndexedIndirect(js_indirectBuffer, indirect_offset)


class GPUObjectBase(classes.GPUObjectBase, ):

    pass

class GPUAdapterInfo(classes.GPUAdapterInfo, ):

    pass

class GPU(classes.GPU, ):

    # TODO: requestAdapter sync variant likely taken from _classes.py directly!
    # TODO: implement codegen for getPreferredCanvasFormat with args [] or return type GPUTextureFormat
    # Additional custom methods from _implementation.py:
    def __init__(self):
        self._internal = window.navigator.gpu  # noqa: F821

    def request_adapter_async(self, loop=None, canvas=None, **options) -> GPUPromise["GPUAdapter"]:
        options = structs.RequestAdapterOptions(**options)
        js_options = to_js(options, eager_converter=simple_js_accessor)
        js_adapter_promise = self._internal.requestAdapter(js_options)

        if loop is None:
            # can we use this instead?
            webloop = js_adapter_promise.get_loop()
            loop = webloop

        def adapter_constructor(js_adapter):
            return GPUAdapter(js_adapter, loop=loop)

        promise = GPUPromise("request_adapter", adapter_constructor, loop=loop)

        js_adapter_promise.then(promise._set_input)  # we chain the js resolution to our promise
        return promise

    def enumerate_adapters_async(self, loop=None) -> GPUPromise[list["GPUAdapter"]]:
        adapter_hp = self.request_adapter_sync(power_preference="high-performance")
        adapter_lp = self.request_adapter_sync(power_preference="low-power")

        promise = GPUPromise("enumerate_adapters", None, loop=loop)
        promise._set_input([adapter_hp, adapter_lp])
        return promise

    @property
    def wgsl_language_features(self):
        return self._internal.wgslLanguageFeatures



class GPUAdapter(classes.GPUAdapter, ):

    # TODO: requestDevice sync variant likely taken from _classes.py directly!
    # Additional custom methods from _implementation.py:
    def __init__(self, js_adapter, loop):
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

        # for compatibility, we fill the native-extra infos too:
        py_adapter_info["vendor_id"] = 0
        py_adapter_info["device_id"] = 0
        py_adapter_info["adapter_type"] = "browser"
        py_adapter_info["backend_type"] = "WebGPU"

        adapter_info = classes.GPUAdapterInfo(**py_adapter_info)

        super().__init__(internal=internal, features=features, limits=py_limits, adapter_info=adapter_info, loop=loop)

    def request_device_async(self, **kwargs) -> GPUPromise["GPUDevice"]:
        descriptor = structs.DeviceDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_device_promise = self._internal.requestDevice(js_descriptor)

        label = kwargs.get("label", "")

        def device_constructor(js_device):
            # TODO: do we need to hand down a default_queue here?
            return GPUDevice(label, js_device, adapter=self)

        promise = GPUPromise("request_device", device_constructor, loop=self._loop)
        js_device_promise.then(promise._set_input)
        return promise



class GPUDevice(classes.GPUDevice, ):

    def destroy(self) -> None:
        self._internal.destroy()

    def create_buffer(self, **kwargs):
        descriptor = structs.BufferDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createBuffer(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUBuffer(label, js_obj, device=self)

    def create_texture(self, **kwargs):
        descriptor = structs.TextureDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createTexture(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUTexture(label, js_obj, device=self)

    def create_sampler(self, **kwargs):
        descriptor = structs.SamplerDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createSampler(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUSampler(label, js_obj, device=self)

    def import_external_texture(self, **kwargs):
        descriptor = structs.ExternalTextureDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.importExternalTexture(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUExternalTexture(label, js_obj, device=self)

    def create_bind_group_layout(self, **kwargs):
        descriptor = structs.BindGroupLayoutDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createBindGroupLayout(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUBindGroupLayout(label, js_obj, device=self)

    def create_pipeline_layout(self, **kwargs):
        descriptor = structs.PipelineLayoutDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createPipelineLayout(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUPipelineLayout(label, js_obj, device=self)

    def create_bind_group(self, **kwargs):
        descriptor = structs.BindGroupDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createBindGroup(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUBindGroup(label, js_obj, device=self)

    def create_shader_module(self, **kwargs):
        descriptor = structs.ShaderModuleDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createShaderModule(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUShaderModule(label, js_obj, device=self)

    def create_compute_pipeline(self, **kwargs):
        descriptor = structs.ComputePipelineDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createComputePipeline(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUComputePipeline(label, js_obj, device=self)

    def create_render_pipeline(self, **kwargs):
        descriptor = structs.RenderPipelineDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createRenderPipeline(js_descriptor)

        label = kwargs.pop("label", "")
        return GPURenderPipeline(label, js_obj, device=self)

    # TODO: was was there a redefinition for createComputePipelineAsync async variant?
    # TODO: was was there a redefinition for createRenderPipelineAsync async variant?
    def create_command_encoder(self, **kwargs):
        descriptor = structs.CommandEncoderDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createCommandEncoder(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUCommandEncoder(label, js_obj, device=self)

    def create_render_bundle_encoder(self, **kwargs):
        descriptor = structs.RenderBundleEncoderDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createRenderBundleEncoder(js_descriptor)

        label = kwargs.pop("label", "")
        return GPURenderBundleEncoder(label, js_obj, device=self)

    def create_query_set(self, **kwargs):
        descriptor = structs.QuerySetDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createQuerySet(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUQuerySet(label, js_obj, device=self)

    def push_error_scope(self, filter: enums.ErrorFilterEnum | None = None) -> None:
    
        self._internal.pushErrorScope(filter)

    # TODO: popErrorScope sync variant likely taken from _classes.py directly!
    # Additional custom methods from _implementation.py:
    def __init__(self, label: str, js_device, adapter: GPUAdapter):
        features = set(js_device.features)

        js_limits = js_device.limits
        limits = {}
        for limit in dir(js_limits):
            if isinstance(getattr(js_limits, limit), int) and "_" not in limit:
                limits[limit] = getattr(js_limits, limit)

        queue = GPUQueue(label="default queue", internal=js_device.queue, device=self)
        super().__init__(label, internal=js_device, adapter=adapter, features=features, limits=limits, queue=queue)

    def create_buffer_with_data_(self, *, label="", data, usage: flags.BufferUsageFlags) -> "GPUBuffer":
        data = memoryview(data).cast("B")  # unit8
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes

        # if it's a Descriptor you need the keywords
        # do we need to also need to modify the usages?
        js_buf = self._internal.createBuffer(label=label, size=data_size, usage=usage, mappedAtCreation=True)
        # print("created buffer", js_buf, dir(js_buf), js_buf.size)
        array_buf = js_buf.getMappedRange(0, data_size)
        Uint8Array.new(array_buf).assign(data)
        # print(array_buf.to_py().tolist())
        js_buf.unmap()
        # print("created buffer", js_buf, dir(js_buf), js_buf.size)
        return GPUBuffer(label, js_buf, self, data_size, usage, enums.BufferMapState.unmapped)

    def create_compute_pipeline_async(self, **kwargs):
        descriptor = structs.ComputePipelineDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_promise = self._internal.createComputePipelineAsync(js_descriptor)

        label = kwargs.get("label", "")

        def construct_compute_pipeline(js_cp):
            return classes.GPUComputePipeline(label, js_cp, self)

        promise = GPUPromise("create_compute_pipeline", construct_compute_pipeline, loop=self._loop)
        js_promise.then(promise._set_input)

        return promise

    def create_render_pipeline_async(self, **kwargs):
        descriptor = structs.RenderPipelineDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_promise = self._internal.createRenderPipelineAsync(js_descriptor)

        label = kwargs.get("label", "")

        def construct_render_pipeline(js_rp):
            return classes.GPURenderPipeline(label, js_rp, self)

        promise = GPUPromise("create_render_pipeline", construct_render_pipeline, loop=self._loop)
        js_promise.then(promise._set_input)

        return promise

    @property
    def adapter(self) -> GPUAdapter:
        return self._adapter



class GPUBuffer(classes.GPUBuffer, ):

    # TODO: mapAsync sync variant likely taken from _classes.py directly!
    def get_mapped_range(self, offset: int = 0, size: Union[int, None] = None) -> ArrayLike:
    
        self._internal.getMappedRange(offset, size)

    def unmap(self) -> None:
        self._internal.unmap()

    def destroy(self) -> None:
        self._internal.destroy()

    # Additional custom methods from _implementation.py:
    def __init__(self, label, internal, device):
        # can we just fill the _classes constructor with properties?
        super().__init__(internal.label, internal, device, internal.size, internal.usage, internal.mapState)

    def write_mapped(self, data, buffer_offset: int | None = None):
        if self.map_state != enums.BufferMapState.mapped:
            raise RuntimeError(f"Can only write to a buffer if its mapped: {self.map_state=}")

        # make sure it's in a known datatype???
        data = memoryview(data).cast("B")
        size = (data.nbytes + 3) & ~3

        # None default values become undefined in js, which should still work as the function can be overloaded.
        # TODO: try without this line
        if buffer_offset is None:
            buffer_offset = 0

        # these can't be passed as keyword arguments I guess...
        array_buf = self._internal.getMappedRange(buffer_offset, size)
        Uint8Array.new(array_buf).assign(data)

    def map_async(self, mode: flags.MapModeFlags | None, offset: int = 0, size: int | None = None) -> GPUPromise[None]:
        map_promise = self._internal.mapAsync(mode, offset, size)

        promise = GPUPromise("buffer.map_async", None, loop=self._device._loop)
        map_promise.then(promise._set_input)  # presumably this signals via a none callback to nothing?
        return promise

    @property
    def map_state(self) -> enums.BufferMapState:
        return self._internal.mapState

    @property
    def size(self) -> int:
        js_size = self._internal.size
        # print("GPUBuffer.size", js_size, type(js_size))
        return js_size

    @property
    def usage(self) -> flags.BufferUsageFlags:
        return self._internal.usage



class GPUTexture(classes.GPUTexture, ):

    # Custom implementation for createView from _implementation.py:
    def create_view(self, **kwargs):
        descriptor = structs.TextureViewDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createView(js_descriptor)

        label = kwargs.pop("label", "")
        return classes.GPUTextureView(label, js_obj, device=self._device, texture=self, size=self._tex_info["size"])

    def destroy(self) -> None:
        self._internal.destroy()

    # Additional custom methods from _implementation.py:
    def __init__(self, label: str, internal, device):
        # here we create the cached _tex_info dict

        tex_info = {
            "size": (internal.width, internal.height, internal.depthOrArrayLayers),
            "mip_level_count": internal.mipLevelCount,
            "sample_count": internal.sampleCount,
            "dimension": internal.dimension,
            "format": internal.format,
            "usage": internal.usage,
        }
        super().__init__(internal.label, internal, device, tex_info)



class GPUTextureView(classes.GPUTextureView, ):

    pass

class GPUSampler(classes.GPUSampler, ):

    pass

class GPUBindGroupLayout(classes.GPUBindGroupLayout, ):

    pass

class GPUBindGroup(classes.GPUBindGroup, ):

    pass

class GPUPipelineLayout(classes.GPUPipelineLayout, ):

    pass

class GPUShaderModule(classes.GPUShaderModule, ):

    # TODO: getCompilationInfo sync variant likely taken from _classes.py directly!
    pass

class GPUCompilationMessage(classes.GPUCompilationMessage, ):

    pass

class GPUCompilationInfo(classes.GPUCompilationInfo, ):

    pass

class GPUPipelineError(classes.GPUPipelineError, ):

    pass

class GPUPipelineBase(classes.GPUPipelineBase, ):

    # Custom implementation for getBindGroupLayout from _implementation.py:
    def get_bind_group_layout(self, **kwargs) -> classes.GPUBindGroupLayout:
        res = super().get_bind_group_layout(**kwargs)
        # returns the js object... so we call the constructor here manually - for now.
        label = res.label
        return classes.GPUBindGroupLayout(label, res, self._device)


class GPUComputePipeline(classes.GPUComputePipeline, ):

    pass

class GPURenderPipeline(classes.GPURenderPipeline, ):

    pass

class GPUCommandBuffer(classes.GPUCommandBuffer, ):

    pass

class GPUCommandEncoder(classes.GPUCommandEncoder, GPUCommandsMixin, GPUDebugCommandsMixin):

    def begin_render_pass(self, **kwargs):
        descriptor = structs.RenderPassDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.beginRenderPass(js_descriptor)

        label = kwargs.pop("label", "")
        return GPURenderPassEncoder(label, js_obj, device=self)

    def begin_compute_pass(self, **kwargs):
        descriptor = structs.ComputePassDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.beginComputePass(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUComputePassEncoder(label, js_obj, device=self)

    def copy_buffer_to_buffer(self, source: Union["GPUBuffer", None] = None, source_offset: Union[int, None] = None, destination: Union["GPUBuffer", None] = None, destination_offset: Union[int, None] = None, size: Union[int, None] = None) -> None:
        js_source = source._internal
        js_destination = destination._internal
        self._internal.copyBufferToBuffer(js_source, source_offset, js_destination, destination_offset, size)

    def copy_buffer_to_texture(self, source: structs.TexelCopyBufferInfoStruct | None = None, destination: structs.TexelCopyTextureInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.TexelCopyBufferInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.TexelCopyTextureInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        # TODO: argument copy_size of JS type GPUExtent3D, py type tuple[int, int, int] | structs.Extent3DStruct might need conversion
        self._internal.copyBufferToTexture(js_source, js_destination, copy_size)

    def copy_texture_to_buffer(self, source: structs.TexelCopyTextureInfoStruct | None = None, destination: structs.TexelCopyBufferInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.TexelCopyTextureInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.TexelCopyBufferInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        # TODO: argument copy_size of JS type GPUExtent3D, py type tuple[int, int, int] | structs.Extent3DStruct might need conversion
        self._internal.copyTextureToBuffer(js_source, js_destination, copy_size)

    def copy_texture_to_texture(self, source: structs.TexelCopyTextureInfoStruct | None = None, destination: structs.TexelCopyTextureInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.TexelCopyTextureInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.TexelCopyTextureInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        # TODO: argument copy_size of JS type GPUExtent3D, py type tuple[int, int, int] | structs.Extent3DStruct might need conversion
        self._internal.copyTextureToTexture(js_source, js_destination, copy_size)

    def clear_buffer(self, buffer: Union["GPUBuffer", None] = None, offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
        self._internal.clearBuffer(js_buffer, offset, size)

    def resolve_query_set(self, query_set: Union["GPUQuerySet", None] = None, first_query: Union[int, None] = None, query_count: Union[int, None] = None, destination: Union["GPUBuffer", None] = None, destination_offset: Union[int, None] = None) -> None:
        js_querySet = query_set._internal
        js_destination = destination._internal
        self._internal.resolveQuerySet(js_querySet, first_query, query_count, js_destination, destination_offset)

    def finish(self, **kwargs):
        descriptor = structs.CommandBufferDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.finish(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUCommandBuffer(label, js_obj, device=self)


class GPUComputePassEncoder(classes.GPUComputePassEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUBindingCommandsMixin):

    def set_pipeline(self, pipeline: Union["GPUComputePipeline", None] = None) -> None:
        js_pipeline = pipeline._internal
        self._internal.setPipeline(js_pipeline)

    def dispatch_workgroups(self, workgroup_count_x: Union[int, None] = None, workgroup_count_y: int = 1, workgroup_count_z: int = 1) -> None:
    
        self._internal.dispatchWorkgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z)

    def dispatch_workgroups_indirect(self, indirect_buffer: Union["GPUBuffer", None] = None, indirect_offset: Union[int, None] = None) -> None:
        js_indirectBuffer = indirect_buffer._internal
        self._internal.dispatchWorkgroupsIndirect(js_indirectBuffer, indirect_offset)

    def end(self) -> None:
        self._internal.end()


class GPURenderPassEncoder(classes.GPURenderPassEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUBindingCommandsMixin, GPURenderCommandsMixin):

    def set_viewport(self, x: Union[float, None] = None, y: Union[float, None] = None, width: Union[float, None] = None, height: Union[float, None] = None, min_depth: Union[float, None] = None, max_depth: Union[float, None] = None) -> None:
    
        self._internal.setViewport(x, y, width, height, min_depth, max_depth)

    def set_scissor_rect(self, x: Union[int, None] = None, y: Union[int, None] = None, width: Union[int, None] = None, height: Union[int, None] = None) -> None:
    
        self._internal.setScissorRect(x, y, width, height)

    def set_blend_constant(self, color: tuple[float, float, float, float] | structs.ColorStruct | None = None) -> None:
        color_desc = structs.Color(**color)
        js_color = to_js(color_desc, eager_converter=simple_js_accessor)
        self._internal.setBlendConstant(js_color)

    def set_stencil_reference(self, reference: Union[int, None] = None) -> None:
    
        self._internal.setStencilReference(reference)

    def begin_occlusion_query(self, query_index: Union[int, None] = None) -> None:
    
        self._internal.beginOcclusionQuery(query_index)

    def end_occlusion_query(self) -> None:
        self._internal.endOcclusionQuery()

    def execute_bundles(self, bundles: Sequence["GPURenderBundle"] | None = None) -> None:
        # TODO: argument bundles of JS type sequence<GPURenderBundle>, py type list[GPURenderBundle] might need conversion
        self._internal.executeBundles(bundles)

    def end(self) -> None:
        self._internal.end()


class GPURenderBundle(classes.GPURenderBundle, ):

    pass

class GPURenderBundleEncoder(classes.GPURenderBundleEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUBindingCommandsMixin, GPURenderCommandsMixin):

    def finish(self, **kwargs):
        descriptor = structs.RenderBundleDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.finish(js_descriptor)

        label = kwargs.pop("label", "")
        return GPURenderBundle(label, js_obj, device=self)


class GPUQueue(classes.GPUQueue, ):

    # Custom implementation for submit from _implementation.py:
    def submit(self, command_buffers: structs.Sequence["GPUCommandBuffer"]) -> None:
        js_command_buffers = [cb._internal for cb in command_buffers]
        self._internal.submit(js_command_buffers)

    # TODO: onSubmittedWorkDone sync variant likely taken from _classes.py directly!
    def write_buffer(self, buffer: Union["GPUBuffer", None] = None, buffer_offset: Union[int, None] = None, data: Union[ArrayLike, None] = None, data_offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
    
        if data is not None:
            data = memoryview(data).cast("B")
            data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
            js_data = Uint8Array.new(data_size)
            js_data.assign(data)
        else:
            js_data = None

        self._internal.writeBuffer(js_buffer, buffer_offset, js_data, data_offset, size)

    def write_texture(self, destination: structs.TexelCopyTextureInfoStruct | None = None, data: Union[ArrayLike, None] = None, data_layout: structs.TexelCopyBufferLayoutStruct | None = None, size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        destination_desc = structs.TexelCopyTextureInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
    
        if data is not None:
            data = memoryview(data).cast("B")
            data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
            js_data = Uint8Array.new(data_size)
            js_data.assign(data)
        else:
            js_data = None

        data_layout_desc = structs.TexelCopyBufferLayout(**data_layout)
        js_dataLayout = to_js(data_layout_desc, eager_converter=simple_js_accessor)
        # TODO: argument size of JS type GPUExtent3D, py type tuple[int, int, int] | structs.Extent3DStruct might need conversion
        self._internal.writeTexture(js_destination, js_data, js_dataLayout, size)

    def copy_external_image_to_texture(self, source: structs.CopyExternalImageSourceInfoStruct | None = None, destination: structs.CopyExternalImageDestInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.CopyExternalImageSourceInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.CopyExternalImageDestInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        # TODO: argument copy_size of JS type GPUExtent3D, py type tuple[int, int, int] | structs.Extent3DStruct might need conversion
        self._internal.copyExternalImageToTexture(js_source, js_destination, copy_size)

    # Additional custom methods from _implementation.py:
    def read_buffer(self, buffer: GPUBuffer, buffer_offset: int = 0, size: int | None = None) -> memoryview:
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

        js_temp_buffer = device._internal.createBuffer(size=data_length, usage=flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ, mappedAtCreation=False, label="output buffer temp")

        js_encoder = device._internal.createCommandEncoder()
        # TODO: somehow test if all the offset math is correct
        js_encoder.copyBufferToBuffer(buffer._internal, buffer_offset, js_temp_buffer, buffer_offset, data_length)
        self._internal.submit([js_encoder.finish()])

        # best way to await the promise directly?
        # TODO: can we do more steps async before waiting?
        run_sync(js_temp_buffer.mapAsync(flags.MapMode.READ, 0, data_length))
        array_buf = js_temp_buffer.getMappedRange()
        res = array_buf.slice(0)
        js_temp_buffer.unmap()
        return res.to_py()



class GPUQuerySet(classes.GPUQuerySet, ):

    def destroy(self) -> None:
        self._internal.destroy()


class GPUCanvasContext(classes.GPUCanvasContext, ):

    # Custom implementation for configure from _implementation.py:
    def configure(self, **kwargs):
        descriptor = structs.CanvasConfiguration(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)

        self._internal.configure(js_descriptor)
        self._config = {
            "device": kwargs.get("device"),
            "format": kwargs.get("format"),
            "usage": kwargs.get("usage", 0x10),
            "view_formats": kwargs.get("view_formats", ()),
            "color_space": kwargs.get("color_space", "srgb"),
            "tone_mapping": kwargs.get("tone_mapping", None),
            "alpha_mode": kwargs.get("alpha_mode", "opaque"),
        }

    def unconfigure(self) -> None:
        self._internal.unconfigure()

    # TODO: implement codegen for getConfiguration with args [] or return type GPUCanvasConfiguration?
    # Custom implementation for getCurrentTexture from _implementation.py:
    def get_current_texture(self) -> GPUTexture:
        js_texture = self._internal.getCurrentTexture()

        label = ""  # always empty?
        return GPUTexture(label, js_texture, self._config["device"])

    # Additional custom methods from _implementation.py:
    def get_preferred_format(self, adapter: GPUAdapter | None) -> enums.TextureFormat:
        return gpu._internal.getPreferredCanvasFormat()

    @property
    def _internal(self) -> JsProxy:
        return self.canvas.html_context



class GPUDeviceLostInfo(classes.GPUDeviceLostInfo, ):

    pass

class GPUError(classes.GPUError, ):

    pass

class GPUValidationError(classes.GPUValidationError, ):

    pass

class GPUOutOfMemoryError(classes.GPUOutOfMemoryError, ):

    pass

class GPUInternalError(classes.GPUInternalError, ):

    pass


gpu = GPU()
