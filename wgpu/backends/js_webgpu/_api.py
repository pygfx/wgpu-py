
# Auto-generated API for the JS WebGPU backend, based on the IDL.

from ... import classes, structs, enums, flags
from ...structs import ArrayLike, Sequence # for typing hints
from typing import Union

from pyodide.ffi import to_js
from js import Uint8Array

# TODO: move this to a new _helpers.py maybe?
from ._helpers import simple_js_accessor


class GPUCommandsMixin(classes.GPUCommandsMixin, ):
    pass

class GPUBindingCommandsMixin(classes.GPUBindingCommandsMixin, ):
    def set_bind_group(self, index: Union[int, None] = None, bind_group: Union["GPUBindGroup", None] = None, dynamic_offsets_data: Union[ArrayLike, None] = None, dynamic_offsets_data_start: Union[int, None] = None, dynamic_offsets_data_length: Union[int, None] = None) -> None:
        js_bindGroup = bind_group._internal
    
        data = memoryview(dynamic_offsets_data).cast("B")
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
        js_data = Uint8Array.new(data_size)
        js_data.assign(data)

        js_obj = self._internal.setBindGroup(index, js_bindGroup, js_data, dynamic_offsets_data_start, dynamic_offsets_data_length)
        return js_obj # TODO maybe instead? None


class GPUDebugCommandsMixin(classes.GPUDebugCommandsMixin, ):
    def push_debug_group(self, group_label: Union[str, None] = None) -> None:
    
        js_obj = self._internal.pushDebugGroup(group_label)
        return js_obj # TODO maybe instead? None

    def pop_debug_group(self) -> None:
        js_obj = self._internal.popDebugGroup()

    def insert_debug_marker(self, marker_label: Union[str, None] = None) -> None:
    
        js_obj = self._internal.insertDebugMarker(marker_label)
        return js_obj # TODO maybe instead? None


class GPURenderCommandsMixin(classes.GPURenderCommandsMixin, ):
    def set_pipeline(self, pipeline: Union["GPURenderPipeline", None] = None) -> None:
        js_pipeline = pipeline._internal
        js_obj = self._internal.setPipeline(js_pipeline)
        return js_obj # TODO maybe instead? None

    def set_index_buffer(self, buffer: Union["GPUBuffer", None] = None, index_format: enums.IndexFormatEnum | None = None, offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
        js_obj = self._internal.setIndexBuffer(js_buffer, index_format, offset, size)
        return js_obj # TODO maybe instead? None

    def set_vertex_buffer(self, slot: Union[int, None] = None, buffer: Union["GPUBuffer", None] = None, offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
        js_obj = self._internal.setVertexBuffer(slot, js_buffer, offset, size)
        return js_obj # TODO maybe instead? None

    def draw(self, vertex_count: Union[int, None] = None, instance_count: int = 1, first_vertex: int = 0, first_instance: int = 0) -> None:
    
        js_obj = self._internal.draw(vertex_count, instance_count, first_vertex, first_instance)
        return js_obj # TODO maybe instead? None

    def draw_indexed(self, index_count: Union[int, None] = None, instance_count: int = 1, first_index: int = 0, base_vertex: int = 0, first_instance: int = 0) -> None:
    
        js_obj = self._internal.drawIndexed(index_count, instance_count, first_index, base_vertex, first_instance)
        return js_obj # TODO maybe instead? None

    def draw_indirect(self, indirect_buffer: Union["GPUBuffer", None] = None, indirect_offset: Union[int, None] = None) -> None:
        js_indirectBuffer = indirect_buffer._internal
        js_obj = self._internal.drawIndirect(js_indirectBuffer, indirect_offset)
        return js_obj # TODO maybe instead? None

    def draw_indexed_indirect(self, indirect_buffer: Union["GPUBuffer", None] = None, indirect_offset: Union[int, None] = None) -> None:
        js_indirectBuffer = indirect_buffer._internal
        js_obj = self._internal.drawIndexedIndirect(js_indirectBuffer, indirect_offset)
        return js_obj # TODO maybe instead? None


class GPUObjectBase(classes.GPUObjectBase, ):
    pass

class GPUAdapterInfo(classes.GPUAdapterInfo, ):
    pass

class GPU(classes.GPU, ):
        # TODO: requestAdapter sync variant likely taken from _classes.py directly!
    # TODO: implement codegen for getPreferredCanvasFormat with args [] or return type GPUTextureFormat
    pass

class GPUAdapter(classes.GPUAdapter, ):
        # TODO: requestDevice sync variant likely taken from _classes.py directly!
    pass

class GPUDevice(classes.GPUDevice, ):
    def destroy(self) -> None:
        js_obj = self._internal.destroy()

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

    def create_compute_pipeline(self, **kwargs):
        descriptor = structs.ComputePipelineDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createComputePipelineAsync(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUComputePipeline(label, js_obj, device=self)

    def create_render_pipeline(self, **kwargs):
        descriptor = structs.RenderPipelineDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createRenderPipelineAsync(js_descriptor)

        label = kwargs.pop("label", "")
        return GPURenderPipeline(label, js_obj, device=self)

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
    
        js_obj = self._internal.pushErrorScope(filter)
        return js_obj # TODO maybe instead? None

        # TODO: popErrorScope sync variant likely taken from _classes.py directly!

class GPUBuffer(classes.GPUBuffer, ):
        # TODO: mapAsync sync variant likely taken from _classes.py directly!
    def get_mapped_range(self, offset: int = 0, size: Union[int, None] = None) -> ArrayLike:
    
        js_obj = self._internal.getMappedRange(offset, size)
        return js_obj # TODO maybe instead? ArrayBuffer

    def unmap(self) -> None:
        js_obj = self._internal.unmap()

    def destroy(self) -> None:
        js_obj = self._internal.destroy()


class GPUTexture(classes.GPUTexture, ):
    def create_view(self, **kwargs):
        descriptor = structs.TextureViewDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.createView(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUTextureView(label, js_obj, device=self)

    def destroy(self) -> None:
        js_obj = self._internal.destroy()


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
    def get_bind_group_layout(self, index: Union[int, None] = None) -> "GPUBindGroupLayout":
    
        js_obj = self._internal.getBindGroupLayout(index)
        return js_obj # TODO maybe instead? [NewObject]


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
        js_obj = self._internal.copyBufferToBuffer(js_source, source_offset, js_destination, destination_offset, size)
        return js_obj # TODO maybe instead? None

    def copy_buffer_to_texture(self, source: structs.TexelCopyBufferInfoStruct | None = None, destination: structs.TexelCopyTextureInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.TexelCopyBufferInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.TexelCopyTextureInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        copy_size_desc = structs.Extent3D(**copy_size)
        js_copySize = to_js(copy_size_desc, eager_converter=simple_js_accessor)
        js_obj = self._internal.copyBufferToTexture(js_source, js_destination, js_copySize)
        return js_obj # TODO maybe instead? None

    def copy_texture_to_buffer(self, source: structs.TexelCopyTextureInfoStruct | None = None, destination: structs.TexelCopyBufferInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.TexelCopyTextureInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.TexelCopyBufferInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        copy_size_desc = structs.Extent3D(**copy_size)
        js_copySize = to_js(copy_size_desc, eager_converter=simple_js_accessor)
        js_obj = self._internal.copyTextureToBuffer(js_source, js_destination, js_copySize)
        return js_obj # TODO maybe instead? None

    def copy_texture_to_texture(self, source: structs.TexelCopyTextureInfoStruct | None = None, destination: structs.TexelCopyTextureInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.TexelCopyTextureInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.TexelCopyTextureInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        copy_size_desc = structs.Extent3D(**copy_size)
        js_copySize = to_js(copy_size_desc, eager_converter=simple_js_accessor)
        js_obj = self._internal.copyTextureToTexture(js_source, js_destination, js_copySize)
        return js_obj # TODO maybe instead? None

    def clear_buffer(self, buffer: Union["GPUBuffer", None] = None, offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
        js_obj = self._internal.clearBuffer(js_buffer, offset, size)
        return js_obj # TODO maybe instead? None

    def resolve_query_set(self, query_set: Union["GPUQuerySet", None] = None, first_query: Union[int, None] = None, query_count: Union[int, None] = None, destination: Union["GPUBuffer", None] = None, destination_offset: Union[int, None] = None) -> None:
        js_querySet = query_set._internal
        js_destination = destination._internal
        js_obj = self._internal.resolveQuerySet(js_querySet, first_query, query_count, js_destination, destination_offset)
        return js_obj # TODO maybe instead? None

    def finish(self, **kwargs):
        descriptor = structs.CommandBufferDescriptor(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.finish(js_descriptor)

        label = kwargs.pop("label", "")
        return GPUCommandBuffer(label, js_obj, device=self)


class GPUComputePassEncoder(classes.GPUComputePassEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUBindingCommandsMixin):
    def set_pipeline(self, pipeline: Union["GPUComputePipeline", None] = None) -> None:
        js_pipeline = pipeline._internal
        js_obj = self._internal.setPipeline(js_pipeline)
        return js_obj # TODO maybe instead? None

    def dispatch_workgroups(self, workgroup_count_x: Union[int, None] = None, workgroup_count_y: int = 1, workgroup_count_z: int = 1) -> None:
    
        js_obj = self._internal.dispatchWorkgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z)
        return js_obj # TODO maybe instead? None

    def dispatch_workgroups_indirect(self, indirect_buffer: Union["GPUBuffer", None] = None, indirect_offset: Union[int, None] = None) -> None:
        js_indirectBuffer = indirect_buffer._internal
        js_obj = self._internal.dispatchWorkgroupsIndirect(js_indirectBuffer, indirect_offset)
        return js_obj # TODO maybe instead? None

    def end(self) -> None:
        js_obj = self._internal.end()


class GPURenderPassEncoder(classes.GPURenderPassEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUBindingCommandsMixin, GPURenderCommandsMixin):
    def set_viewport(self, x: Union[float, None] = None, y: Union[float, None] = None, width: Union[float, None] = None, height: Union[float, None] = None, min_depth: Union[float, None] = None, max_depth: Union[float, None] = None) -> None:
    
        js_obj = self._internal.setViewport(x, y, width, height, min_depth, max_depth)
        return js_obj # TODO maybe instead? None

    def set_scissor_rect(self, x: Union[int, None] = None, y: Union[int, None] = None, width: Union[int, None] = None, height: Union[int, None] = None) -> None:
    
        js_obj = self._internal.setScissorRect(x, y, width, height)
        return js_obj # TODO maybe instead? None

    def set_blend_constant(self, color: tuple[float, float, float, float] | structs.ColorStruct | None = None) -> None:
        color_desc = structs.Color(**color)
        js_color = to_js(color_desc, eager_converter=simple_js_accessor)
        js_obj = self._internal.setBlendConstant(js_color)
        return js_obj # TODO maybe instead? None

    def set_stencil_reference(self, reference: Union[int, None] = None) -> None:
    
        js_obj = self._internal.setStencilReference(reference)
        return js_obj # TODO maybe instead? None

    def begin_occlusion_query(self, query_index: Union[int, None] = None) -> None:
    
        js_obj = self._internal.beginOcclusionQuery(query_index)
        return js_obj # TODO maybe instead? None

    def end_occlusion_query(self) -> None:
        js_obj = self._internal.endOcclusionQuery()

    def execute_bundles(self, bundles: Sequence["GPURenderBundle"] | None = None) -> None:
        # TODO: argument bundles of JS type sequence<GPURenderBundle>, py type list[GPURenderBundle] might need conversion
        js_obj = self._internal.executeBundles(bundles)
        return js_obj # TODO maybe instead? None

    def end(self) -> None:
        js_obj = self._internal.end()


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
    def submit(self, command_buffers: Sequence[GPUCommandBuffer] | None = None) -> None:
        # TODO: argument command_buffers of JS type sequence<GPUCommandBuffer>, py type list[GPUCommandBuffer] might need conversion
        js_obj = self._internal.submit(command_buffers)
        return js_obj # TODO maybe instead? None

        # TODO: onSubmittedWorkDone sync variant likely taken from _classes.py directly!
    def write_buffer(self, buffer: Union["GPUBuffer", None] = None, buffer_offset: Union[int, None] = None, data: Union[ArrayLike, None] = None, data_offset: int = 0, size: Union[int, None] = None) -> None:
        js_buffer = buffer._internal
    
        data = memoryview(data).cast("B")
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
        js_data = Uint8Array.new(data_size)
        js_data.assign(data)

        js_obj = self._internal.writeBuffer(js_buffer, buffer_offset, js_data, data_offset, size)
        return js_obj # TODO maybe instead? None

    def write_texture(self, destination: structs.TexelCopyTextureInfoStruct | None = None, data: Union[ArrayLike, None] = None, data_layout: structs.TexelCopyBufferLayoutStruct | None = None, size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        destination_desc = structs.TexelCopyTextureInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
    
        data = memoryview(data).cast("B")
        data_size = (data.nbytes + 3) & ~3  # align to 4 bytes
        js_data = Uint8Array.new(data_size)
        js_data.assign(data)

        data_layout_desc = structs.TexelCopyBufferLayout(**data_layout)
        js_dataLayout = to_js(data_layout_desc, eager_converter=simple_js_accessor)
        size_desc = structs.Extent3D(**size)
        js_size = to_js(size_desc, eager_converter=simple_js_accessor)
        js_obj = self._internal.writeTexture(js_destination, js_data, js_dataLayout, js_size)
        return js_obj # TODO maybe instead? None

    def copy_external_image_to_texture(self, source: structs.CopyExternalImageSourceInfoStruct | None = None, destination: structs.CopyExternalImageDestInfoStruct | None = None, copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None) -> None:
        source_desc = structs.CopyExternalImageSourceInfo(**source)
        js_source = to_js(source_desc, eager_converter=simple_js_accessor)
        destination_desc = structs.CopyExternalImageDestInfo(**destination)
        js_destination = to_js(destination_desc, eager_converter=simple_js_accessor)
        copy_size_desc = structs.Extent3D(**copy_size)
        js_copySize = to_js(copy_size_desc, eager_converter=simple_js_accessor)
        js_obj = self._internal.copyExternalImageToTexture(js_source, js_destination, js_copySize)
        return js_obj # TODO maybe instead? None


class GPUQuerySet(classes.GPUQuerySet, ):
    def destroy(self) -> None:
        js_obj = self._internal.destroy()


class GPUCanvasContext(classes.GPUCanvasContext, ):
    def configure(self, **kwargs):
        descriptor = structs.CanvasConfiguration(**kwargs)
        js_descriptor = to_js(descriptor, eager_converter=simple_js_accessor)
        js_obj = self._internal.configure(js_descriptor)

        label = kwargs.pop("label", "")
        return undefined(label, js_obj, device=self)

    def unconfigure(self) -> None:
        js_obj = self._internal.unconfigure()

    # TODO: implement codegen for getConfiguration with args [] or return type GPUCanvasConfiguration?
    # TODO: implement codegen for getCurrentTexture with args [] or return type GPUTexture

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

