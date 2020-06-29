# wgpu-py codegen report
*  Running codegen-script.py

## Comparing webgpu.idl with wgpu.h

### Comparing flags
*  MapMode flag missing in .h
*   ShaderStage
*  c: NONE:0, VERTEX:1, FRAGMENT:2, COMPUTE:4
*  i: VERTEX:1, FRAGMENT:2, COMPUTE:4
*   BufferUsage
*  c: MAP_READ:1, MAP_WRITE:2, COPY_SRC:4, COPY_DST:8, INDEX:16, VERTEX:32, UNIFORM:64, STORAGE:128, INDIRECT:256
*  i: MAP_READ:1, MAP_WRITE:2, COPY_SRC:4, COPY_DST:8, INDEX:16, VERTEX:32, UNIFORM:64, STORAGE:128, INDIRECT:256, QUERY_RESOLVE:512
*   ColorWrite
*  c: RED:1, GREEN:2, BLUE:4, ALPHA:8, COLOR:7, ALL:15
*  i: RED:1, GREEN:2, BLUE:4, ALPHA:8, ALL:15

### Comparing enums
*  BufferMapAsyncStatus enum missing in .idl
*  LogLevel enum missing in .idl
*  PresentMode enum missing in .idl
*  SwapChainStatus enum missing in .idl
*  BindingResource_Tag enum missing in .idl
*  ExtensionName enum missing in .h
*  CompilationMessageType enum missing in .h
*  QueryType enum missing in .h
*  PipelineStatisticName enum missing in .h
*  ErrorFilter enum missing in .h
*  TextureFormat.rgb10a2unorm missing in .h
*  TextureFormat.rg11b10float missing in .h
*  TextureFormat.bc1-rgba-unorm missing in .h
*  TextureFormat.bc1-rgba-unorm-srgb missing in .h
*  TextureFormat.bc2-rgba-unorm missing in .h
*  TextureFormat.bc2-rgba-unorm-srgb missing in .h
*  TextureFormat.bc3-rgba-unorm missing in .h
*  TextureFormat.bc3-rgba-unorm-srgb missing in .h
*  TextureFormat.bc4-r-unorm missing in .h
*  TextureFormat.bc4-r-snorm missing in .h
*  TextureFormat.bc5-rg-unorm missing in .h
*  TextureFormat.bc5-rg-snorm missing in .h
*  TextureFormat.bc6h-rgb-ufloat missing in .h
*  TextureFormat.bc6h-rgb-sfloat missing in .h
*  TextureFormat.bc7-rgba-unorm missing in .h
*  TextureFormat.bc7-rgba-unorm-srgb missing in .h

### Comparing structs
*  Extensions struct missing in .idl
*  RawPass struct missing in .idl
*  RenderPassColorAttachmentDescriptorBase_TextureViewId struct missing in .idl
*  RenderPassDepthStencilAttachmentDescriptorBase_TextureViewId struct missing in .idl
*  Origin3d struct missing in .idl
*  Extent3d struct missing in .idl
*  BindingResource_WGPUBuffer_Body struct missing in .idl
*  BindingResource_WGPUSampler_Body struct missing in .idl
*  BindingResource_WGPUTextureView_Body struct missing in .idl
*  BindingResource struct missing in .idl
*  U32Array struct missing in .idl
*  SwapChainOutput struct missing in .idl
*  ImageBitmapCopyView struct missing in .h
*  RenderPassColorAttachmentDescriptor struct missing in .h
*  RenderPassDepthStencilAttachmentDescriptor struct missing in .h
*  RenderBundleDescriptor struct missing in .h
*  RenderBundleEncoderDescriptor struct missing in .h
*  FenceDescriptor struct missing in .h
*  QuerySetDescriptor struct missing in .h
*  UncapturedErrorEventInit struct missing in .h
*  Origin2D struct missing in .h
*  Origin3D struct missing in .h
*  Extent3D struct missing in .h
*   Limits
*  c: ['max_bind_groups']
*  i: ['maxBindGroups', 'maxDynamicUniformBuffersPerPipelineLayout', 'maxDynamicStorageBuffersPerPipelineLayout', 'maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage', 'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage', 'maxUniformBufferBindingSize']
*   RenderPassDescriptor
*  c: ['color_attachments', 'color_attachments_length', 'depth_stencil_attachment']
*  i: ['label', 'colorAttachments', 'depthStencilAttachment', 'occlusionQuerySet']
*   BufferCopyView
*  c: ['buffer', 'layout']
*  i: ['offset', 'bytesPerRow', 'rowsPerImage', 'buffer']
*   BindGroupLayoutEntry
*  c: ['binding', 'visibility', 'ty', 'multisampled', 'has_dynamic_offset', 'view_dimension', 'texture_component_type', 'storage_texture_format']
*  i: ['binding', 'visibility', 'type', 'hasDynamicOffset', 'minBufferBindingSize', 'viewDimension', 'textureComponentType', 'multisampled', 'storageTextureFormat']
*   BufferDescriptor
*  c: ['label', 'size', 'usage']
*  i: ['label', 'size', 'usage', 'mappedAtCreation']
*   ShaderModuleDescriptor
*  c: ['code']
*  i: ['label', 'code', 'sourceMap']
*   SwapChainDescriptor
*  c: ['usage', 'format', 'width', 'height', 'present_mode']
*  i: ['label', 'device', 'format', 'usage']
*   RequestAdapterOptions
*  c: ['power_preference', 'compatible_surface']
*  i: ['powerPreference']
*   TextureViewDescriptor
*  c: ['label', 'format', 'dimension', 'aspect', 'base_mip_level', 'level_count', 'base_array_layer', 'array_layer_count']
*  i: ['label', 'format', 'dimension', 'aspect', 'baseMipLevel', 'mipLevelCount', 'baseArrayLayer', 'arrayLayerCount']

## Generate API code
*  Written to flags.py
*  Written to enums.py
*  Written to _mappings.py
*  Written to _structs.py

## Checking and patching hand-written API code

### Check functions in base.py
*  Found 60 functions already implemented
*  Not implemented: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor); (devicecreatebuffermapped)
*  Not implemented: GPUQuerySet createQuerySet(GPUQuerySetDescriptor descriptor); (devicecreatequeryset)
*  Not implemented: ArrayBuffer getMappedRange(optional GPUSize64 offset = 0, optional GPUSize64 size = 0); (buffergetmappedrange)
*  Not implemented: void writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex); (commandencoderwritetimestamp)
*  Not implemented: void resolveQuerySet( GPUQuerySet querySet, GPUSize32 firstQuery, GPUSize32 queryCount, GPUBuffer destination, GPUSize64 destinationOffset); (commandencoderresolvequeryset)
*  Not implemented: void beginPipelineStatisticsQuery(GPUQuerySet querySet, GPUSize32 queryIndex); (computepassencoderbeginpipelinestatisticsquery)
*  Not implemented: void endPipelineStatisticsQuery(); (computepassencoderendpipelinestatisticsquery)
*  Not implemented: void writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex); (computepassencoderwritetimestamp)
*  Not implemented: void beginOcclusionQuery(GPUSize32 queryIndex); (renderpassencoderbeginocclusionquery)
*  Not implemented: void endOcclusionQuery(); (renderpassencoderendocclusionquery)
*  Not implemented: void beginPipelineStatisticsQuery(GPUQuerySet querySet, GPUSize32 queryIndex); (renderpassencoderbeginpipelinestatisticsquery)
*  Not implemented: void endPipelineStatisticsQuery(); (renderpassencoderendpipelinestatisticsquery)
*  Not implemented: void writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex); (renderpassencoderwritetimestamp)
*  Not implemented: GPUFence createFence(optional GPUFenceDescriptor descriptor = {}); (queuecreatefence)
*  Not implemented: void signal(GPUFence fence, GPUFenceValue signalValue); (queuesignal)
*  Not implemented: void writeBuffer( GPUBuffer buffer, GPUSize64 bufferOffset, [AllowShared] ArrayBuffer data, optional GPUSize64 dataOffset = 0, optional GPUSize64 size); (queuewritebuffer)
*  Not implemented: void writeTexture( GPUTextureCopyView destination, [AllowShared] ArrayBuffer data, GPUTextureDataLayout dataLayout, GPUExtent3D size); (queuewritetexture)
*  Not implemented: GPUFenceValue getCompletedValue(); (fencegetcompletedvalue)
*  Not implemented: Promise<void> onCompletion(GPUFenceValue completionValue); (fenceoncompletion)
*  Not implemented: void destroy(); (querysetdestroy)
*  Found unknown function configure_swap_chain (deviceconfigureswapchain)
*  Found unknown function get_swap_chain_preferred_format (devicegetswapchainpreferredformat)
*  Injected IDL lines into base.py

### Check functions in backends/rs.py
*  Found 50 functions already implemented
*  Not implemented: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor); (devicecreatebuffermapped)
*  Not implemented: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor); (devicecreaterenderbundleencoder)
*  Not implemented: GPUQuerySet createQuerySet(GPUQuerySetDescriptor descriptor); (devicecreatequeryset)
*  Not implemented: ArrayBuffer getMappedRange(optional GPUSize64 offset = 0, optional GPUSize64 size = 0); (buffergetmappedrange)
*  Not implemented: GPUBindGroupLayout getBindGroupLayout(unsigned long index); (pipelinebasegetbindgrouplayout)
*  Not implemented: void pushDebugGroup(USVString groupLabel); (commandencoderpushdebuggroup)
*  Not implemented: void popDebugGroup(); (commandencoderpopdebuggroup)
*  Not implemented: void insertDebugMarker(USVString markerLabel); (commandencoderinsertdebugmarker)
*  Not implemented: void writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex); (commandencoderwritetimestamp)
*  Not implemented: void resolveQuerySet( GPUQuerySet querySet, GPUSize32 firstQuery, GPUSize32 queryCount, GPUBuffer destination, GPUSize64 destinationOffset); (commandencoderresolvequeryset)
*  Not implemented: void beginPipelineStatisticsQuery(GPUQuerySet querySet, GPUSize32 queryIndex); (computepassencoderbeginpipelinestatisticsquery)
*  Not implemented: void endPipelineStatisticsQuery(); (computepassencoderendpipelinestatisticsquery)
*  Not implemented: void writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex); (computepassencoderwritetimestamp)
*  Not implemented: void beginOcclusionQuery(GPUSize32 queryIndex); (renderpassencoderbeginocclusionquery)
*  Not implemented: void endOcclusionQuery(); (renderpassencoderendocclusionquery)
*  Not implemented: void beginPipelineStatisticsQuery(GPUQuerySet querySet, GPUSize32 queryIndex); (renderpassencoderbeginpipelinestatisticsquery)
*  Not implemented: void endPipelineStatisticsQuery(); (renderpassencoderendpipelinestatisticsquery)
*  Not implemented: void writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex); (renderpassencoderwritetimestamp)
*  Not implemented: void executeBundles(sequence<GPURenderBundle> bundles); (renderpassencoderexecutebundles)
*  Not implemented: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {}); (renderbundleencoderfinish)
*  Not implemented: GPUFence createFence(optional GPUFenceDescriptor descriptor = {}); (queuecreatefence)
*  Not implemented: void signal(GPUFence fence, GPUFenceValue signalValue); (queuesignal)
*  Not implemented: void writeBuffer( GPUBuffer buffer, GPUSize64 bufferOffset, [AllowShared] ArrayBuffer data, optional GPUSize64 dataOffset = 0, optional GPUSize64 size); (queuewritebuffer)
*  Not implemented: void writeTexture( GPUTextureCopyView destination, [AllowShared] ArrayBuffer data, GPUTextureDataLayout dataLayout, GPUExtent3D size); (queuewritetexture)
*  Not implemented: void copyImageBitmapToTexture( GPUImageBitmapCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize); (queuecopyimagebitmaptotexture)
*  Not implemented: GPUFenceValue getCompletedValue(); (fencegetcompletedvalue)
*  Not implemented: Promise<void> onCompletion(GPUFenceValue completionValue); (fenceoncompletion)
*  Not implemented: void destroy(); (querysetdestroy)
*  Not implemented: GPUTexture getCurrentTexture(); (swapchaingetcurrenttexture)
*  Found unknown function new_struct_p (newstructp)
*  Found unknown function configure_swap_chain (deviceconfigureswapchain)
*  Injected IDL lines into backends/rs.py
