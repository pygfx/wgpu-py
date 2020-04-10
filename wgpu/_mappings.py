"""
THIS CODE IS AUTOGENERATED - DO NOT EDIT

Mappings that help automate some things in the implementations.
"""
# flake8: noqa

enummap = {
    "AddressMode.clamp-to-edge": 0,
    "AddressMode.repeat": 1,
    "AddressMode.mirror-repeat": 2,
    "BindingType.uniform-buffer": 0,
    "BindingType.storage-buffer": 1,
    "BindingType.readonly-storage-buffer": 2,
    "BindingType.sampler": 3,
    "BindingType.comparison-sampler": 4,
    "BindingType.sampled-texture": 5,
    "BindingType.readonly-storage-texture": 6,
    "BindingType.writeonly-storage-texture": 7,
    "BlendFactor.zero": 0,
    "BlendFactor.one": 1,
    "BlendFactor.src-color": 2,
    "BlendFactor.one-minus-src-color": 3,
    "BlendFactor.src-alpha": 4,
    "BlendFactor.one-minus-src-alpha": 5,
    "BlendFactor.dst-color": 6,
    "BlendFactor.one-minus-dst-color": 7,
    "BlendFactor.dst-alpha": 8,
    "BlendFactor.one-minus-dst-alpha": 9,
    "BlendFactor.src-alpha-saturated": 10,
    "BlendFactor.blend-color": 11,
    "BlendFactor.one-minus-blend-color": 12,
    "BlendOperation.add": 0,
    "BlendOperation.subtract": 1,
    "BlendOperation.reverse-subtract": 2,
    "BlendOperation.min": 3,
    "BlendOperation.max": 4,
    "CompareFunction.never": 1,
    "CompareFunction.less": 2,
    "CompareFunction.equal": 3,
    "CompareFunction.less-equal": 4,
    "CompareFunction.greater": 5,
    "CompareFunction.not-equal": 6,
    "CompareFunction.greater-equal": 7,
    "CompareFunction.always": 8,
    "CullMode.none": 0,
    "CullMode.front": 1,
    "CullMode.back": 2,
    "FilterMode.nearest": 0,
    "FilterMode.linear": 1,
    "FrontFace.ccw": 0,
    "FrontFace.cw": 1,
    "IndexFormat.uint16": 0,
    "IndexFormat.uint32": 1,
    "InputStepMode.vertex": 0,
    "InputStepMode.instance": 1,
    "LoadOp.load": 1,
    "PowerPreference.low-power": 1,
    "PowerPreference.high-performance": 2,
    "PrimitiveTopology.point-list": 0,
    "PrimitiveTopology.line-list": 1,
    "PrimitiveTopology.line-strip": 2,
    "PrimitiveTopology.triangle-list": 3,
    "PrimitiveTopology.triangle-strip": 4,
    "StencilOperation.keep": 0,
    "StencilOperation.zero": 1,
    "StencilOperation.replace": 2,
    "StencilOperation.invert": 3,
    "StencilOperation.increment-clamp": 4,
    "StencilOperation.decrement-clamp": 5,
    "StencilOperation.increment-wrap": 6,
    "StencilOperation.decrement-wrap": 7,
    "StoreOp.store": 1,
    "StoreOp.clear": 0,
    "TextureAspect.all": 0,
    "TextureAspect.stencil-only": 1,
    "TextureAspect.depth-only": 2,
    "TextureComponentType.float": 0,
    "TextureComponentType.sint": 1,
    "TextureComponentType.uint": 2,
    "TextureDimension.1d": 0,
    "TextureDimension.2d": 1,
    "TextureDimension.3d": 2,
    "TextureFormat.r8unorm": 0,
    "TextureFormat.r8snorm": 1,
    "TextureFormat.r8uint": 2,
    "TextureFormat.r8sint": 3,
    "TextureFormat.r16uint": 4,
    "TextureFormat.r16sint": 5,
    "TextureFormat.r16float": 6,
    "TextureFormat.rg8unorm": 7,
    "TextureFormat.rg8snorm": 8,
    "TextureFormat.rg8uint": 9,
    "TextureFormat.rg8sint": 10,
    "TextureFormat.r32uint": 11,
    "TextureFormat.r32sint": 12,
    "TextureFormat.r32float": 13,
    "TextureFormat.rg16uint": 14,
    "TextureFormat.rg16sint": 15,
    "TextureFormat.rg16float": 16,
    "TextureFormat.rgba8unorm": 17,
    "TextureFormat.rgba8unorm-srgb": 18,
    "TextureFormat.rgba8snorm": 19,
    "TextureFormat.rgba8uint": 20,
    "TextureFormat.rgba8sint": 21,
    "TextureFormat.bgra8unorm": 22,
    "TextureFormat.bgra8unorm-srgb": 23,
    "TextureFormat.rg32uint": 26,
    "TextureFormat.rg32sint": 27,
    "TextureFormat.rg32float": 28,
    "TextureFormat.rgba16uint": 29,
    "TextureFormat.rgba16sint": 30,
    "TextureFormat.rgba16float": 31,
    "TextureFormat.rgba32uint": 32,
    "TextureFormat.rgba32sint": 33,
    "TextureFormat.rgba32float": 34,
    "TextureFormat.depth32float": 35,
    "TextureFormat.depth24plus": 36,
    "TextureFormat.depth24plus-stencil8": 37,
    "TextureViewDimension.1d": 0,
    "TextureViewDimension.2d": 1,
    "TextureViewDimension.2d-array": 2,
    "TextureViewDimension.cube": 3,
    "TextureViewDimension.cube-array": 4,
    "TextureViewDimension.3d": 5,
    "VertexFormat.uchar2": 1,
    "VertexFormat.uchar4": 3,
    "VertexFormat.char2": 5,
    "VertexFormat.char4": 7,
    "VertexFormat.uchar2norm": 9,
    "VertexFormat.uchar4norm": 11,
    "VertexFormat.char2norm": 14,
    "VertexFormat.char4norm": 16,
    "VertexFormat.ushort2": 18,
    "VertexFormat.ushort4": 20,
    "VertexFormat.short2": 22,
    "VertexFormat.short4": 24,
    "VertexFormat.ushort2norm": 26,
    "VertexFormat.ushort4norm": 28,
    "VertexFormat.short2norm": 30,
    "VertexFormat.short4norm": 32,
    "VertexFormat.half2": 34,
    "VertexFormat.half4": 36,
    "VertexFormat.float": 37,
    "VertexFormat.float2": 38,
    "VertexFormat.float3": 39,
    "VertexFormat.float4": 40,
    "VertexFormat.uint": 41,
    "VertexFormat.uint2": 42,
    "VertexFormat.uint3": 43,
    "VertexFormat.uint4": 44,
    "VertexFormat.int": 45,
    "VertexFormat.int2": 46,
    "VertexFormat.int3": 47,
    "VertexFormat.int4": 48,
}

cstructfield2enum = {
    "RenderPassColorAttachmentDescriptorBase_TextureViewId.load_op": "LoadOp",
    "RenderPassColorAttachmentDescriptorBase_TextureViewId.store_op": "StoreOp",
    "RenderPassDepthStencilAttachmentDescriptorBase_TextureViewId.depth_load_op": "LoadOp",
    "RenderPassDepthStencilAttachmentDescriptorBase_TextureViewId.depth_store_op": "StoreOp",
    "RenderPassDepthStencilAttachmentDescriptorBase_TextureViewId.stencil_load_op": "LoadOp",
    "RenderPassDepthStencilAttachmentDescriptorBase_TextureViewId.stencil_store_op": "StoreOp",
    "BindGroupLayoutEntry.ty": "BindingType",
    "BindGroupLayoutEntry.view_dimension": "TextureViewDimension",
    "BindGroupLayoutEntry.texture_component_type": "TextureComponentType",
    "BindGroupLayoutEntry.storage_texture_format": "TextureFormat",
    "RasterizationStateDescriptor.front_face": "FrontFace",
    "RasterizationStateDescriptor.cull_mode": "CullMode",
    "BlendDescriptor.src_factor": "BlendFactor",
    "BlendDescriptor.dst_factor": "BlendFactor",
    "BlendDescriptor.operation": "BlendOperation",
    "ColorStateDescriptor.format": "TextureFormat",
    "StencilStateFaceDescriptor.compare": "CompareFunction",
    "StencilStateFaceDescriptor.fail_op": "StencilOperation",
    "StencilStateFaceDescriptor.depth_fail_op": "StencilOperation",
    "StencilStateFaceDescriptor.pass_op": "StencilOperation",
    "DepthStencilStateDescriptor.format": "TextureFormat",
    "DepthStencilStateDescriptor.depth_compare": "CompareFunction",
    "VertexAttributeDescriptor.format": "VertexFormat",
    "VertexBufferLayoutDescriptor.step_mode": "InputStepMode",
    "VertexStateDescriptor.index_format": "IndexFormat",
    "RenderPipelineDescriptor.primitive_topology": "PrimitiveTopology",
    "SamplerDescriptor.address_mode_u": "AddressMode",
    "SamplerDescriptor.address_mode_v": "AddressMode",
    "SamplerDescriptor.address_mode_w": "AddressMode",
    "SamplerDescriptor.mag_filter": "FilterMode",
    "SamplerDescriptor.min_filter": "FilterMode",
    "SamplerDescriptor.mipmap_filter": "FilterMode",
    "SamplerDescriptor.compare": "CompareFunction",
    "SwapChainDescriptor.format": "TextureFormat",
    "TextureDescriptor.dimension": "TextureDimension",
    "TextureDescriptor.format": "TextureFormat",
    "RequestAdapterOptions.power_preference": "PowerPreference",
    "TextureViewDescriptor.format": "TextureFormat",
    "TextureViewDescriptor.dimension": "TextureViewDimension",
    "TextureViewDescriptor.aspect": "TextureAspect",
}
