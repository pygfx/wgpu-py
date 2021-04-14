# Code generatation report
## Preparing
* The webgpu.idl defines 33 classes with 81 functions
* The webgpu.idl defines 5 flags, 29 enums, 50 structs
* The wgpu.h defines 82 functions
* The wgpu.h defines 4 flags, 26 enums, 48 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 29 enums to enums.py
* Wrote 50 structs to structs.py
### Patching API for base.py
* Diffs for GPU: change request_adapter, change request_adapter_async
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add read_data, add read_data_async, add size, add usage, add write_data, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTexture: add dimension, add format, add mip_level_count, add sample_count, add size, add usage
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUComputePassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPURenderPassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Validated 33 classes, 107 methods, 30 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 33 classes, 90 methods, 0 properties
## Validating rs.py
* Flag field BufferUsage.QUERY_RESOLVE missing in wgpu.h
* Flag MapMode missing in wgpu.h
* Flag field TextureUsage.RENDER_ATTACHMENT missing in wgpu.h
* Flag ColorWrite missing in wgpu.h
* Enum FeatureName missing in wgpu.h
* Enum field TextureFormat.rgb9e5ufloat missing in wgpu.h
* Enum field TextureFormat.rgb10a2unorm missing in wgpu.h
* Enum field TextureFormat.rg11b10ufloat missing in wgpu.h
* Enum field TextureFormat.stencil8 missing in wgpu.h
* Enum field TextureFormat.depth16unorm missing in wgpu.h
* Enum field TextureFormat.bc1-rgba-unorm missing in wgpu.h
* Enum field TextureFormat.bc1-rgba-unorm-srgb missing in wgpu.h
* Enum field TextureFormat.bc2-rgba-unorm missing in wgpu.h
* Enum field TextureFormat.bc2-rgba-unorm-srgb missing in wgpu.h
* Enum field TextureFormat.bc3-rgba-unorm missing in wgpu.h
* Enum field TextureFormat.bc3-rgba-unorm-srgb missing in wgpu.h
* Enum field TextureFormat.bc4-r-unorm missing in wgpu.h
* Enum field TextureFormat.bc4-r-snorm missing in wgpu.h
* Enum field TextureFormat.bc5-rg-unorm missing in wgpu.h
* Enum field TextureFormat.bc5-rg-snorm missing in wgpu.h
* Enum field TextureFormat.bc6h-rgb-ufloat missing in wgpu.h
* Enum field TextureFormat.bc6h-rgb-float missing in wgpu.h
* Enum field TextureFormat.bc7-rgba-unorm missing in wgpu.h
* Enum field TextureFormat.bc7-rgba-unorm-srgb missing in wgpu.h
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
* Wrote 122 enum mappings and 44 struct-field mappings to rs_mappings.py
* Validated 71 C function calls
* Validated 62 C structs
