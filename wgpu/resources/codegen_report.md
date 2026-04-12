# Code generation report
## Preparing
* The webgpu.idl defines 37 classes with 76 functions
* The webgpu.idl defines 5 flags, 34 enums, 60 structs
* webgpu.h/wgpu.h define 226 functions
* webgpu.h/wgpu.h define 7 flags, 68 enums, 114 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 34 enums to enums.py
* Wrote 60 structs to structs.py
### Patching API for _classes.py
* Diffs for GPU: add enumerate_adapters_async, add enumerate_adapters_sync, add get_canvas_context, change get_preferred_canvas_format, change request_adapter_async, change request_adapter_sync
* Diffs for GPUPromise: add GPUPromise
* Diffs for GPUCanvasContext: add get_preferred_format, add physical_size, add present, add set_physical_size, hide canvas
* Diffs for GPUAdapter: add summary
* Diffs for GPUObjectBase: add uid
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide lost_async, hide lost_sync, hide onuncapturederror, hide pop_error_scope_async, hide pop_error_scope_sync, hide push_error_scope
* Diffs for GPUBuffer: add read_mapped, add write_mapped, hide get_mapped_range
* Diffs for GPUTexture: add size
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUBindingCommandsMixin: change set_bind_group
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 38 classes, 120 methods, 51 properties
### Patching API for backends/wgpu_native/_api.py
* Validated 38 classes, 117 methods, 0 properties
## Validating backends/wgpu_native/_api.py
* Enum PipelineErrorReason missing in webgpu.h/wgpu.h
* Enum AutoLayoutMode missing in webgpu.h/wgpu.h
* Enum field VertexFormat.unorm10-10-10-2 missing in webgpu.h/wgpu.h
* Enum CanvasAlphaMode missing in webgpu.h/wgpu.h
* Enum CanvasToneMappingMode missing in webgpu.h/wgpu.h
* Wrote 266 enum mappings and 48 struct-field mappings to wgpu_native/_mappings.py
* Validated 154 C function calls
* Not using 83 C functions
* Validated 97 C structs
