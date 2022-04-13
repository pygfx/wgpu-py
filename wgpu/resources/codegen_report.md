# Code generatation report
## Preparing
* The webgpu.idl defines 33 classes with 78 functions
* The webgpu.idl defines 5 flags, 32 enums, 57 structs
* The wgpu.h defines 131 functions
* The wgpu.h defines 5 flags, 48 enums, 74 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 32 enums to enums.py
* Wrote 57 structs to structs.py
### Patching API for base.py
* Diffs for GPU: change request_adapter, change request_adapter_async
* Diffs for GPUCanvasContext: add present
* Diffs for GPUAdapter: add properties
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add map_read, add map_write, add size, add usage, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTexture: add dimension, add format, add mip_level_count, add sample_count, add size, add usage
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 33 classes, 105 methods, 34 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 33 classes, 96 methods, 0 properties
## Validating rs.py
* Enum CanvasCompositingAlphaMode missing in wgpu.h
* Wrote 230 enum mappings and 49 struct-field mappings to rs_mappings.py
* Validated 80 C function calls
* Not using 56 C functions
* Validated 70 C structs
