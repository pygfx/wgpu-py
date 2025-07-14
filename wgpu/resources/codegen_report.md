# Code generation report
## Preparing
* The webgpu.idl defines 37 classes with 76 functions
* The webgpu.idl defines 5 flags, 34 enums, 60 structs
* webgpu.h/wgpu.h define 211 functions
* webgpu.h/wgpu.h define 7 flags, 60 enums, 101 structs
## Updating API
* Wrote 5 flags to flags.py
* Wrote 34 enums to enums.py
* Wrote 60 structs to structs.py
### Patching API for _classes.py
* Diffs for GPU: add enumerate_adapters_async, add enumerate_adapters_sync, change get_preferred_canvas_format, change request_adapter_async, change request_adapter_sync
* Diffs for GPUCanvasContext: add get_preferred_format, add present
* Diffs for GPUAdapter: add summary
* Diffs for GPUDevice: add adapter, add create_buffer_with_data, hide import_external_texture, hide lost_async, hide lost_sync, hide onuncapturederror, hide pop_error_scope_async, hide pop_error_scope_sync, hide push_error_scope
* Diffs for GPUBuffer: add read_mapped, add write_mapped, hide get_mapped_range
* Diffs for GPUTexture: add size
* Diffs for GPUTextureView: add size, add texture
* Diffs for GPUBindingCommandsMixin: change set_bind_group
* Diffs for GPUQueue: add read_buffer, add read_texture, hide copy_external_image_to_texture
* Validated 37 classes, 122 methods, 49 properties
### Patching API for backends/wgpu_native/_api.py
* Validated 37 classes, 124 methods, 0 properties
## Validating backends/wgpu_native/_api.py
* Enum field FeatureName.subgroups missing in webgpu.h/wgpu.h
* Enum PipelineErrorReason missing in webgpu.h/wgpu.h
* Enum AutoLayoutMode missing in webgpu.h/wgpu.h
* Enum field VertexFormat.unorm10-10-10-2 missing in webgpu.h/wgpu.h
* Enum CanvasAlphaMode missing in webgpu.h/wgpu.h
* Enum CanvasToneMappingMode missing in webgpu.h/wgpu.h
* Wrote 255 enum mappings and 47 struct-field mappings to wgpu_native/_mappings.py
* Validated 151 C function calls
* Not using 69 C functions
* Validated 95 C structs
