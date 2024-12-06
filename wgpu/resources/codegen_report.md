# Code generation report
## Preparing
* The webgpu.idl defines 37 classes with 75 functions
* The webgpu.idl defines 5 flags, 34 enums, 60 structs
* The wgpu.h defines 199 functions
* The wgpu.h defines 7 flags, 52 enums, 93 structs
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
* Validated 37 classes, 121 methods, 46 properties
### Patching API for backends/wgpu_native/_api.py
* Validated 37 classes, 121 methods, 0 properties
## Validating backends/wgpu_native/_api.py
* Enum field FeatureName.texture-compression-bc-sliced-3d missing in wgpu.h
* Enum field FeatureName.clip-distances missing in wgpu.h
* Enum field FeatureName.dual-source-blending missing in wgpu.h
* Enum PipelineErrorReason missing in wgpu.h
* Enum AutoLayoutMode missing in wgpu.h
* Enum field BlendFactor.src1 missing in wgpu.h
* Enum field BlendFactor.one-minus-src1 missing in wgpu.h
* Enum field BlendFactor.src1-alpha missing in wgpu.h
* Enum field BlendFactor.one-minus-src1-alpha missing in wgpu.h
* Enum field VertexFormat.unorm10-10-10-2 missing in wgpu.h
* Enum CanvasAlphaMode missing in wgpu.h
* Enum CanvasToneMappingMode missing in wgpu.h
* Wrote 236 enum mappings and 47 struct-field mappings to wgpu_native/_mappings.py
* Validated 142 C function calls
* Not using 64 C functions
* Validated 82 C structs
