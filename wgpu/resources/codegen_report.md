# Code generatation report
## Preparing
* The webgpu.idl defines 39 classes with 78 functions
* The webgpu.idl defines 5 flags, 33 enums, 59 structs
* The wgpu.h defines 198 functions
* The wgpu.h defines 6 flags, 49 enums, 88 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 33 enums to enums.py
* Wrote 59 structs to structs.py
### Patching API for base.py
* Diffs for GPU: add print_report, change get_preferred_canvas_format, change request_adapter, change request_adapter_async
* Diffs for GPUCanvasContext: add get_preferred_format, add present
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add map_read, add map_write, hide get_mapped_range, hide map_async, hide unmap
* Diffs for GPUTexture: add size
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 39 classes, 111 methods, 44 properties
### Patching API for backends/rs.py
* Diffs for GPUAdapter: add request_device_tracing
* Validated 39 classes, 100 methods, 0 properties
## Validating rs.py
* Enum field TextureFormat.rgb10a2uint missing in wgpu.h
* Enum PipelineErrorReason missing in wgpu.h
* Enum AutoLayoutMode missing in wgpu.h
* Enum field VertexFormat.unorm10-10-10-2 missing in wgpu.h
* Enum CanvasAlphaMode missing in wgpu.h
* Enum field DeviceLostReason.unknown missing in wgpu.h
* Wrote 232 enum mappings and 47 struct-field mappings to rs_mappings.py
* Validated 91 C function calls
* Not using 115 C functions
* Validated 72 C structs
