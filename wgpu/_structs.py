"""
All wgpu structs.
"""

# THIS CODE IS AUTOGENERATED - DO NOT EDIT


# There are 45 structs

RequestAdapterOptions = {"power_preference": "'enums.PowerPreference'"}

DeviceDescriptor = {
    "label": "str",
    "extensions": "'list(enums.ExtensionName)'",
    "limits": "'structs.Limits'",
}

Limits = {
    "max_bind_groups": "int",
    "max_dynamic_uniform_buffers_per_pipeline_layout": "int",
    "max_dynamic_storage_buffers_per_pipeline_layout": "int",
    "max_sampled_textures_per_shader_stage": "int",
    "max_samplers_per_shader_stage": "int",
    "max_storage_buffers_per_shader_stage": "int",
    "max_storage_textures_per_shader_stage": "int",
    "max_uniform_buffers_per_shader_stage": "int",
    "max_uniform_buffer_binding_size": "int",
}

BufferDescriptor = {
    "label": "str",
    "size": "int",
    "usage": "'flags.BufferUsage'",
    "mapped_at_creation": "bool",
}

TextureDescriptor = {
    "label": "str",
    "size": "'list(int) or structs.Extent3D'",
    "mip_level_count": "int",
    "sample_count": "int",
    "dimension": "'enums.TextureDimension'",
    "format": "'enums.TextureFormat'",
    "usage": "'flags.TextureUsage'",
}

TextureViewDescriptor = {
    "label": "str",
    "format": "'enums.TextureFormat'",
    "dimension": "'enums.TextureViewDimension'",
    "aspect": "'enums.TextureAspect'",
    "base_mip_level": "int",
    "mip_level_count": "int",
    "base_array_layer": "int",
    "array_layer_count": "int",
}

SamplerDescriptor = {
    "label": "str",
    "address_mode_u": "'enums.AddressMode'",
    "address_mode_v": "'enums.AddressMode'",
    "address_mode_w": "'enums.AddressMode'",
    "mag_filter": "'enums.FilterMode'",
    "min_filter": "'enums.FilterMode'",
    "mipmap_filter": "'enums.FilterMode'",
    "lod_min_clamp": "float",
    "lod_max_clamp": "float",
    "compare": "'enums.CompareFunction'",
}

BindGroupLayoutDescriptor = {
    "label": "str",
    "entries": "'list(structs.BindGroupLayoutEntry)'",
}

BindGroupLayoutEntry = {
    "binding": "int",
    "visibility": "'flags.ShaderStage'",
    "type": "'enums.BindingType'",
    "has_dynamic_offset": "bool",
    "min_buffer_binding_size": "int",
    "view_dimension": "'enums.TextureViewDimension'",
    "texture_component_type": "'enums.TextureComponentType'",
    "multisampled": "bool",
    "storage_texture_format": "'enums.TextureFormat'",
}

BindGroupDescriptor = {
    "label": "str",
    "layout": "'GPUBindGroupLayout'",
    "entries": "'list(structs.BindGroupEntry)'",
}

BindGroupEntry = {
    "binding": "int",
    "resource": "'GPUSampler or GPUTextureView or structs.BufferBinding'",
}

BufferBinding = {"buffer": "'GPUBuffer'", "offset": "int", "size": "int"}

PipelineLayoutDescriptor = {
    "label": "str",
    "bind_group_layouts": "'list(GPUBindGroupLayout)'",
}

ShaderModuleDescriptor = {"label": "str", "code": "str", "source_map": "dict"}

ProgrammableStageDescriptor = {"module": "'GPUShaderModule'", "entry_point": "str"}

ComputePipelineDescriptor = {
    "label": "str",
    "layout": "'GPUPipelineLayout'",
    "compute_stage": "'structs.ProgrammableStageDescriptor'",
}

RenderPipelineDescriptor = {
    "label": "str",
    "layout": "'GPUPipelineLayout'",
    "vertex_stage": "'structs.ProgrammableStageDescriptor'",
    "fragment_stage": "'structs.ProgrammableStageDescriptor'",
    "primitive_topology": "'enums.PrimitiveTopology'",
    "rasterization_state": "'structs.RasterizationStateDescriptor'",
    "color_states": "'list(structs.ColorStateDescriptor)'",
    "depth_stencil_state": "'structs.DepthStencilStateDescriptor'",
    "vertex_state": "'structs.VertexStateDescriptor'",
    "sample_count": "int",
    "sample_mask": "int",
    "alpha_to_coverage_enabled": "bool",
}

RasterizationStateDescriptor = {
    "front_face": "'enums.FrontFace'",
    "cull_mode": "'enums.CullMode'",
    "depth_bias": "int",
    "depth_bias_slope_scale": "float",
    "depth_bias_clamp": "float",
}

ColorStateDescriptor = {
    "format": "'enums.TextureFormat'",
    "alpha_blend": "'structs.BlendDescriptor'",
    "color_blend": "'structs.BlendDescriptor'",
    "write_mask": "'flags.ColorWrite'",
}

BlendDescriptor = {
    "src_factor": "'enums.BlendFactor'",
    "dst_factor": "'enums.BlendFactor'",
    "operation": "'enums.BlendOperation'",
}

DepthStencilStateDescriptor = {
    "format": "'enums.TextureFormat'",
    "depth_write_enabled": "bool",
    "depth_compare": "'enums.CompareFunction'",
    "stencil_front": "'structs.StencilStateFaceDescriptor'",
    "stencil_back": "'structs.StencilStateFaceDescriptor'",
    "stencil_read_mask": "int",
    "stencil_write_mask": "int",
}

StencilStateFaceDescriptor = {
    "compare": "'enums.CompareFunction'",
    "fail_op": "'enums.StencilOperation'",
    "depth_fail_op": "'enums.StencilOperation'",
    "pass_op": "'enums.StencilOperation'",
}

VertexStateDescriptor = {
    "index_format": "'enums.IndexFormat'",
    "vertex_buffers": "'list(structs.VertexBufferLayoutDescriptor)'",
}

VertexBufferLayoutDescriptor = {
    "array_stride": "int",
    "step_mode": "'enums.InputStepMode'",
    "attributes": "'list(structs.VertexAttributeDescriptor)'",
}

VertexAttributeDescriptor = {
    "format": "'enums.VertexFormat'",
    "offset": "int",
    "shader_location": "int",
}

CommandBufferDescriptor = {"label": "str"}

CommandEncoderDescriptor = {"label": "str"}

TextureDataLayout = {"offset": "int", "bytes_per_row": "int", "rows_per_image": "int"}

BufferCopyView = {
    "offset": "int",
    "bytes_per_row": "int",
    "rows_per_image": "int",
    "buffer": "'GPUBuffer'",
}

TextureCopyView = {
    "texture": "'GPUTexture'",
    "mip_level": "int",
    "origin": "'list(int) or structs.Origin3D'",
}

ImageBitmapCopyView = {
    "image_bitmap": "memoryview",
    "origin": "'list(int) or structs.Origin2D'",
}

ComputePassDescriptor = {"label": "str"}

RenderPassDescriptor = {
    "label": "str",
    "color_attachments": "'list(structs.RenderPassColorAttachmentDescriptor)'",
    "depth_stencil_attachment": "'structs.RenderPassDepthStencilAttachmentDescriptor'",
    "occlusion_query_set": "'GPUQuerySet'",
}

RenderPassColorAttachmentDescriptor = {
    "attachment": "'GPUTextureView'",
    "resolve_target": "'GPUTextureView'",
    "load_value": "'enums.LoadOp or list(float) or structs.Color'",
    "store_op": "'enums.StoreOp'",
}

RenderPassDepthStencilAttachmentDescriptor = {
    "attachment": "'GPUTextureView'",
    "depth_load_value": "'enums.LoadOp or float'",
    "depth_store_op": "'enums.StoreOp'",
    "depth_read_only": "bool",
    "stencil_load_value": "'enums.LoadOp or int'",
    "stencil_store_op": "'enums.StoreOp'",
    "stencil_read_only": "bool",
}

RenderBundleDescriptor = {"label": "str"}

RenderBundleEncoderDescriptor = {
    "label": "str",
    "color_formats": "'list(enums.TextureFormat)'",
    "depth_stencil_format": "'enums.TextureFormat'",
    "sample_count": "int",
}

FenceDescriptor = {"label": "str", "initial_value": "int"}

QuerySetDescriptor = {
    "label": "str",
    "type": "'enums.QueryType'",
    "count": "int",
    "pipeline_statistics": "'list(enums.PipelineStatisticName)'",
}

SwapChainDescriptor = {
    "label": "str",
    "device": "'GPUDevice'",
    "format": "'enums.TextureFormat'",
    "usage": "'flags.TextureUsage'",
}

UncapturedErrorEventInit = {"error": "'GPUOutOfMemoryError or GPUValidationError'"}

Color = {"r": "float", "g": "float", "b": "float", "a": "float"}

Origin2D = {"x": "int", "y": "int"}

Origin3D = {"x": "int", "y": "int", "z": "int"}

Extent3D = {"width": "int", "height": "int", "depth": "int"}
