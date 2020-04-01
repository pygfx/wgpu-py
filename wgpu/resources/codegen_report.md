# wgpu-py codegen report
*  Running codegen-script.py

## Comparing webgpu.idl with wgpu.h

### Comparing flags
*   DEFAULT flag missing in .idl
*   DESIRED flag missing in .idl
*   MAX flag missing in .idl
*   ShaderStage
*  c: NONE:0, VERTEX:1, FRAGMENT:2, COMPUTE:4
*  i: VERTEX:1, FRAGMENT:2, COMPUTE:4
*   BufferUsage
*  c: MAP_READ:1, MAP_WRITE:2, COPY_SRC:4, COPY_DST:8, INDEX:16, VERTEX:32, UNIFORM:64, STORAGE:128, STORAGE_READ:256, INDIRECT:512, NONE:0
*  i: MAP_READ:1, MAP_WRITE:2, COPY_SRC:4, COPY_DST:8, INDEX:16, VERTEX:32, UNIFORM:64, STORAGE:128, INDIRECT:256
*   ColorWrite
*  c: RED:1, GREEN:2, BLUE:4, ALPHA:8, COLOR:7, ALL:15
*  i: RED:1, GREEN:2, BLUE:4, ALPHA:8, ALL:15
*   TextureUsage
*  c: COPY_SRC:1, COPY_DST:2, SAMPLED:4, STORAGE:8, OUTPUT_ATTACHMENT:16, NONE:0, UNINITIALIZED:65535
*  i: COPY_SRC:1, COPY_DST:2, SAMPLED:4, STORAGE:8, OUTPUT_ATTACHMENT:16

### Comparing enums
*  BufferMapAsyncStatus enum missing in .idl
*  PresentMode enum missing in .idl
*  BindingResource_Tag enum missing in .idl
*  ExtensionNameenum missing in .h
*  TextureComponentTypeenum missing in .h
*  ErrorFilterenum missing in .h
*  TextureFormat.rgb10a2unorm is missing
*  TextureFormat.rg11b10float is missing

### Comparing structs
*  Extensions struct missing in .idl
*  RawPass struct missing in .idl
*  RenderPassColorAttachmentDescriptorBase_TextureViewId__TextureViewId struct missing in .idl
*  RenderPassDepthStencilAttachmentDescriptorBase_TextureViewId struct missing in .idl
*  RawRenderTargets struct missing in .idl
*  RawRenderPass struct missing in .idl
*  RenderPassColorAttachmentDescriptorBase_TextureViewId__OptionRef_TextureViewId struct missing in .idl
*  Origin3d struct missing in .idl
*  Extent3d struct missing in .idl
*  BindingResource_WGPUBuffer_Body struct missing in .idl
*  BindingResource_WGPUSampler_Body struct missing in .idl
*  BindingResource_WGPUTextureView_Body struct missing in .idl
*  BindingResource struct missing in .idl
*  VertexBufferDescriptor struct missing in .idl
*  VertexInputDescriptor struct missing in .idl
*  U32Array struct missing in .idl
*  SwapChainOutput struct missing in .idl
*  VertexStateDescriptor struct missing in .h
*  VertexBufferLayoutDescriptor struct missing in .h
*  ImageBitmapCopyView struct missing in .h
*  RenderPassColorAttachmentDescriptor struct missing in .h
*  RenderPassDepthStencilAttachmentDescriptor struct missing in .h
*  RenderBundleDescriptor struct missing in .h
*  RenderBundleEncoderDescriptor struct missing in .h
*  FenceDescriptor struct missing in .h
*  UncapturedErrorEventInit struct missing in .h
*  Origin2D struct missing in .h
*  Origin3D struct missing in .h
*  Extent3D struct missing in .h
*   Limits
*  c: ['max_bind_groups']
*  i: ['maxBindGroups', 'maxDynamicUniformBuffersPerPipelineLayout', 'maxDynamicStorageBuffersPerPipelineLayout', 'maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage', 'maxStorageTexturesPerShaderStage', 'maxUniformBuffersPerShaderStage']
*   BindGroupLayoutBinding
*  c: ['binding', 'visibility', 'ty', 'texture_dimension', 'multisampled', 'dynamic']
*  i: ['binding', 'visibility', 'type', 'textureDimension', 'textureComponentType', 'multisampled', 'hasDynamicOffset']
*   RenderPipelineDescriptor
*  c: ['layout', 'vertex_stage', 'fragment_stage', 'primitive_topology', 'rasterization_state', 'color_states', 'color_states_length', 'depth_stencil_state', 'vertex_input', 'sample_count', 'sample_mask', 'alpha_to_coverage_enabled']
*  i: ['label', 'layout', 'vertexStage', 'fragmentStage', 'primitiveTopology', 'rasterizationState', 'colorStates', 'depthStencilState', 'vertexState', 'sampleCount', 'sampleMask', 'alphaToCoverageEnabled']
*   SamplerDescriptor
*  c: ['address_mode_u', 'address_mode_v', 'address_mode_w', 'mag_filter', 'min_filter', 'mipmap_filter', 'lod_min_clamp', 'lod_max_clamp', 'compare_function']
*  i: ['label', 'addressModeU', 'addressModeV', 'addressModeW', 'magFilter', 'minFilter', 'mipmapFilter', 'lodMinClamp', 'lodMaxClamp', 'compare']
*   SwapChainDescriptor
*  c: ['usage', 'format', 'width', 'height', 'present_mode']
*  i: ['label', 'device', 'format', 'usage']
*   TextureViewDescriptor
*  c: ['format', 'dimension', 'aspect', 'base_mip_level', 'level_count', 'base_array_layer', 'array_layer_count']
*  i: ['label', 'format', 'dimension', 'aspect', 'baseMipLevel', 'mipLevelCount', 'baseArrayLayer', 'arrayLayerCount']

## Generate API code
*  Written to flags.py
*  Written to enums.py
*  Written to _mappings.py

## Checking and patching hand-written API code

### Check functions in base.py
*  Found 60 functions already implemented
*  Not implemented: GPUFence createFence(optional GPUFenceDescriptor descriptor = {});
*  Not implemented: void signal(GPUFence fence, GPUFenceValue signalValue);
*  Not implemented: GPUFenceValue getCompletedValue();
*  Not implemented: Promise<void> onCompletion(GPUFenceValue completionValue);
*  Found unknown function create_default_view (texturecreatedefaultview)
*  Found unknown function get_current_texture_view (swapchaingetcurrenttextureview)
*  Injected IDL lines into base.py

### Check functions in backends/rs.py
*  Found 48 functions already implemented
*  Not implemented: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor);
*  Not implemented: void pushDebugGroup(DOMString groupLabel);
*  Not implemented: void popDebugGroup();
*  Not implemented: void insertDebugMarker(DOMString markerLabel);
*  Not implemented: void pushDebugGroup(DOMString groupLabel);
*  Not implemented: void popDebugGroup();
*  Not implemented: void insertDebugMarker(DOMString markerLabel);
*  Not implemented: void executeBundles(sequence<GPURenderBundle> bundles);
*  Not implemented: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {});
*  Not implemented: GPUFence createFence(optional GPUFenceDescriptor descriptor = {});
*  Not implemented: void signal(GPUFence fence, GPUFenceValue signalValue);
*  Not implemented: void copyImageBitmapToTexture( GPUImageBitmapCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
*  Not implemented: GPUFenceValue getCompletedValue();
*  Not implemented: Promise<void> onCompletion(GPUFenceValue completionValue);
*  Not implemented: GPUTexture getCurrentTexture();
*  Found unknown function create_default_view (texturecreatedefaultview)
*  Found unknown function get_current_texture_view (swapchaingetcurrenttextureview)
*  Injected IDL lines into backends/rs.py
