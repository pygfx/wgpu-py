# Code generatation report
## Preparing
*  The webgpu.idl defines 34 classes with 82 functions
*  The webgpu.idl defines 5 flags, 26 enums, 45 structs
*  The wgpu.h defines 82 functions
*  The wgpu.h defines 4 flags, 26 enums, 48 structs
## Updating API
*  Wrote 5 flags to flags.py
*  Wrote 26 enums to enums.py
*  Wrote 45 structs to structs.py
### Patching API for base.py
*  Warning: unknown api: method GPUDevice.configure_swap_chain
*  Warning: unknown api: method GPUDevice.get_swap_chain_preferred_format
*  Warning: unknown api: prop GPUBuffer.size
*  Warning: unknown api: prop GPUBuffer.usage
*  Warning: unknown api: prop GPUTexture.texture_size
*  Warning: unknown api: prop GPUTexture.mip_level_count
*  Warning: unknown api: prop GPUTexture.sample_count
*  Warning: unknown api: prop GPUTexture.dimension
*  Warning: unknown api: prop GPUTexture.format
*  Warning: unknown api: prop GPUTexture.texture_usage
*  Warning: unknown api: prop GPUTextureView.size
*  Warning: unknown api: prop GPUTextureView.texture
*  Warning: unknown api: prop GPUPipelineBase.layout
*  Validated 34 classes, 109 methods, 28 properties
### Patching API for backends/rs.py
*  Validated 34 classes, 91 methods, 0 properties
## Validating rs.py
*  Flag field BufferUsage.QUERY_RESOLVE missing in wgpu.h
*  Flag MapMode missing in wgpu.h
*  Flag ColorWrite missing in wgpu.h
*  Enum ExtensionName missing in wgpu.h
*  Enum field TextureFormat.rgb10a2unorm missing in wgpu.h
*  Enum field TextureFormat.rg11b10float missing in wgpu.h
*  Enum field TextureFormat.bc1-rgba-unorm missing in wgpu.h
*  Enum field TextureFormat.bc1-rgba-unorm-srgb missing in wgpu.h
*  Enum field TextureFormat.bc2-rgba-unorm missing in wgpu.h
*  Enum field TextureFormat.bc2-rgba-unorm-srgb missing in wgpu.h
*  Enum field TextureFormat.bc3-rgba-unorm missing in wgpu.h
*  Enum field TextureFormat.bc3-rgba-unorm-srgb missing in wgpu.h
*  Enum field TextureFormat.bc4-r-unorm missing in wgpu.h
*  Enum field TextureFormat.bc4-r-snorm missing in wgpu.h
*  Enum field TextureFormat.bc5-rg-unorm missing in wgpu.h
*  Enum field TextureFormat.bc5-rg-snorm missing in wgpu.h
*  Enum field TextureFormat.bc6h-rgb-ufloat missing in wgpu.h
*  Enum field TextureFormat.bc6h-rgb-sfloat missing in wgpu.h
*  Enum field TextureFormat.bc7-rgba-unorm missing in wgpu.h
*  Enum field TextureFormat.bc7-rgba-unorm-srgb missing in wgpu.h
*  Enum CompilationMessageType missing in wgpu.h
*  Enum QueryType missing in wgpu.h
*  Enum PipelineStatisticName missing in wgpu.h
*  Enum ErrorFilter missing in wgpu.h
*  Wrote 147 enum mappings and 46 struct-field mappings to rs_mappings.py
*  Validated 71 C function calls
*  Validated 62 C structs
