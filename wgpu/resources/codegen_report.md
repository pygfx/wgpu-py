# Code generatation report
## Preparing
* The webgpu.idl defines 39 classes with 78 functions
* The webgpu.idl defines 5 flags, 35 enums, 59 structs
* The wgpu.h defines 131 functions
* The wgpu.h defines 5 flags, 48 enums, 74 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 35 enums to enums.py
* Wrote 59 structs to structs.py
### Patching API for base.py
* Diffs for GPU: change get_preferred_canvas_format, change request_adapter, change request_adapter_async
* Diffs for GPUCanvasContext: add get_preferred_format, add present
* Diffs for GPUAdapter: add properties
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add map_read, add map_write, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 39 classes, 109 methods, 44 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 39 classes, 96 methods, 0 properties
## Validating rs.py
* Enum field FeatureName.shader-f16 missing in wgpu.h
* Enum field FeatureName.rg11b10ufloat-renderable missing in wgpu.h
* Enum BufferMapState missing in wgpu.h
* Enum PipelineErrorReason missing in wgpu.h
* Enum AutoLayoutMode missing in wgpu.h
* Enum CanvasAlphaMode missing in wgpu.h
* Enum field ErrorFilter.internal missing in wgpu.h
* Wrote 228 enum mappings and 49 struct-field mappings to rs_mappings.py
* Validated 80 C function calls
* Not using 56 C functions
* Validated 70 C structs
