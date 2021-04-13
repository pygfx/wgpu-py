# Code generatation report
## Preparing
* The webgpu.idl defines 34 classes with 82 functions
* The webgpu.idl defines 5 flags, 26 enums, 45 structs
* The wgpu.h defines 82 functions
* The wgpu.h defines 4 flags, 26 enums, 48 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 26 enums to enums.py
* Wrote 45 structs to structs.py
### Patching API for base.py
* Diffs for GPU: change request_adapter, change request_adapter_async
* Diffs for GPUDevice: add create_buffer_with_data, hide create_buffer_mapped, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add read_data, add read_data_async, add size, add usage, add write_data, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTexture: add dimension, add format, add mip_level_count, add sample_count, add size, add usage
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUComputePassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPURenderPassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPUQueue: hide create_fence, hide signal
* Validated 34 classes, 108 methods, 27 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 34 classes, 88 methods, 0 properties
## Validating rs.py
* Flag field BufferUsage.QUERY_RESOLVE missing in wgpu.h
* Flag MapMode missing in wgpu.h
* Flag ColorWrite missing in wgpu.h
* Enum ExtensionName missing in wgpu.h
* Enum field TextureFormat.rgb10a2unorm missing in wgpu.h
* Enum field TextureFormat.rg11b10float missing in wgpu.h
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
* Enum field TextureFormat.bc6h-rgb-sfloat missing in wgpu.h
* Enum field TextureFormat.bc7-rgba-unorm missing in wgpu.h
* Enum field TextureFormat.bc7-rgba-unorm-srgb missing in wgpu.h
* Enum CompilationMessageType missing in wgpu.h
* Enum QueryType missing in wgpu.h
* Enum PipelineStatisticName missing in wgpu.h
* Enum ErrorFilter missing in wgpu.h
* Wrote 147 enum mappings and 46 struct-field mappings to rs_mappings.py
* Validated 71 C function calls
* Validated 62 C structs
