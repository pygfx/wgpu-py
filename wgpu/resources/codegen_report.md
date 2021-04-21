# Code generatation report
## Preparing
* The webgpu.idl defines 33 classes with 81 functions
* The webgpu.idl defines 5 flags, 29 enums, 50 structs
* The wgpu.h defines 111 functions
* The wgpu.h defines 5 flags, 30 enums, 49 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 29 enums to enums.py
* Wrote 50 structs to structs.py
### Patching API for base.py
* Diffs for GPU: change request_adapter, change request_adapter_async
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add map_read, add map_write, add size, add usage, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTexture: add dimension, add format, add mip_level_count, add sample_count, add size, add usage
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUComputePassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPURenderPassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPUQueue: add read_buffer, add read_texture
* Validated 33 classes, 108 methods, 30 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 33 classes, 96 methods, 0 properties
## Validating rs.py
* Flag field BufferUsage.QUERY_RESOLVE missing in wgpu.h
* Flag MapMode missing in wgpu.h
* Flag ColorWrite missing in wgpu.h
* Enum FeatureName missing in wgpu.h
* Enum field TextureFormat.rgb9e5ufloat missing in wgpu.h
* Enum field TextureFormat.rgb10a2unorm missing in wgpu.h
* Enum field TextureFormat.rg11b10ufloat missing in wgpu.h
* Enum field TextureFormat.stencil8 missing in wgpu.h
* Enum field TextureFormat.depth16unorm missing in wgpu.h
* Enum field TextureFormat.bc6h-rgb-ufloat missing in wgpu.h
* Enum field TextureFormat.bc6h-rgb-float missing in wgpu.h
* Enum field TextureFormat.depth24unorm-stencil8 missing in wgpu.h
* Enum field TextureFormat.depth32float-stencil8 missing in wgpu.h
* Enum BufferBindingType missing in wgpu.h
* Enum SamplerBindingType missing in wgpu.h
* Enum TextureSampleType missing in wgpu.h
* Enum StorageTextureAccess missing in wgpu.h
* Enum CompilationMessageType missing in wgpu.h
* Enum field BlendFactor.src-component missing in wgpu.h
* Enum field BlendFactor.one-minus-src-component missing in wgpu.h
* Enum field BlendFactor.dst-component missing in wgpu.h
* Enum field BlendFactor.one-minus-dst-component missing in wgpu.h
* Enum field BlendFactor.blendcolor-component missing in wgpu.h
* Enum field BlendFactor.one-minus-blendcolor-component missing in wgpu.h
* Enum field VertexFormat.unorm8x2 missing in wgpu.h
* Enum field VertexFormat.unorm8x4 missing in wgpu.h
* Enum field VertexFormat.snorm8x2 missing in wgpu.h
* Enum field VertexFormat.snorm8x4 missing in wgpu.h
* Enum field VertexFormat.unorm16x2 missing in wgpu.h
* Enum field VertexFormat.unorm16x4 missing in wgpu.h
* Enum field VertexFormat.snorm16x2 missing in wgpu.h
* Enum field VertexFormat.snorm16x4 missing in wgpu.h
* Enum QueryType missing in wgpu.h
* Enum PipelineStatisticName missing in wgpu.h
* Enum DeviceLostReason missing in wgpu.h
* Enum ErrorFilter missing in wgpu.h
* Wrote 134 enum mappings and 38 struct-field mappings to rs_mappings.py
* Validated 81 C function calls
* Not using 26 C functions
* Validated 61 C structs
