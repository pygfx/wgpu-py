# Code generatation report
## Preparing
* The webgpu.idl defines 33 classes with 83 functions
* The webgpu.idl defines 5 flags, 31 enums, 53 structs
* The wgpu.h defines 106 functions
* The wgpu.h defines 5 flags, 37 enums, 60 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 31 enums to enums.py
* Wrote 53 structs to structs.py
### Patching API for base.py
* Diffs for GPU: change request_adapter, change request_adapter_async
* Diffs for GPUPresentationContext: add present
* Diffs for GPUAdapter: add properties
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add map_read, add map_write, add size, add usage, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTexture: add dimension, add format, add mip_level_count, add sample_count, add size, add usage
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUComputePassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPURenderPassEncoder: hide begin_pipeline_statistics_query, hide end_pipeline_statistics_query
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 33 classes, 110 methods, 34 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 33 classes, 95 methods, 0 properties
## Validating rs.py
* Enum PredefinedColorSpace missing in wgpu.h
* Enum PowerPreference missing in wgpu.h
* Enum FeatureName missing in wgpu.h
* Enum field TextureFormat.depth16unorm missing in wgpu.h
* Enum field TextureFormat.depth24unorm-stencil8 missing in wgpu.h
* Enum field TextureFormat.depth32float-stencil8 missing in wgpu.h
* Enum CompilationMessageType missing in wgpu.h
* Enum CanvasCompositingAlphaMode missing in wgpu.h
* Enum DeviceLostReason missing in wgpu.h
* Wrote 175 enum mappings and 45 struct-field mappings to rs_mappings.py
* Validated 65 C function calls
* Not using 46 C functions
* Validated 66 C structs
