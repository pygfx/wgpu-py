# Code generatation report
## Preparing
* The webgpu.idl defines 37 classes with 76 functions
* The webgpu.idl defines 5 flags, 33 enums, 59 structs
* The wgpu.h defines 198 functions
* The wgpu.h defines 7 flags, 50 enums, 92 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 33 enums to enums.py
* Wrote 59 structs to structs.py
### Patching API for _classes.py
* Diffs for GPU: change get_preferred_canvas_format, change request_adapter, change request_adapter_async
* Diffs for GPUCanvasContext: add get_preferred_format, add present
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide lost, hide onuncapturederror, hide pop_error_scope, hide push_error_scope
* Diffs for GPUBuffer: add map_read, add map_write, add read_mapped, add write_mapped, hide get_mapped_range
* Diffs for GPUTexture: add size
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 37 classes, 112 methods, 43 properties
### Patching API for backends/wgpu_native/_api.py
* Validated 37 classes, 107 methods, 0 properties
## Validating backends/wgpu_native/_api.py
* Enum field TextureFormat.rgb10a2uint missing in wgpu.h
* Enum field StorageTextureAccess.read-only missing in wgpu.h
* Enum field StorageTextureAccess.read-write missing in wgpu.h
* Enum PipelineErrorReason missing in wgpu.h
* Enum AutoLayoutMode missing in wgpu.h
* Enum field VertexFormat.unorm10-10-10-2 missing in wgpu.h
* Enum CanvasAlphaMode missing in wgpu.h
* Enum field DeviceLostReason.unknown missing in wgpu.h
* Wrote 232 enum mappings and 47 struct-field mappings to wgpu_native/_mappings.py
* Validated 105 C function calls
* Not using 97 C functions
* Validated 75 C structs
