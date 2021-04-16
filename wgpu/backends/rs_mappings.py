""" Mappings for the rs backend. """

# THIS CODE IS AUTOGENERATED - DO NOT EDIT

# flake8: noqa

# There are 134 enum mappings

enummap = {
    "AddressMode.clamp-to-edge": 0,
    "AddressMode.mirror-repeat": 2,
    "AddressMode.repeat": 1,
    "BlendFactor.dst-alpha": 8,
    "BlendFactor.one": 1,
    "BlendFactor.one-minus-dst-alpha": 9,
    "BlendFactor.one-minus-src-alpha": 5,
    "BlendFactor.src-alpha": 4,
    "BlendFactor.src-alpha-saturated": 10,
    "BlendFactor.zero": 0,
    "BlendOperation.add": 0,
    "BlendOperation.max": 4,
    "BlendOperation.min": 3,
    "BlendOperation.reverse-subtract": 2,
    "BlendOperation.subtract": 1,
    "CompareFunction.always": 8,
    "CompareFunction.equal": 6,
    "CompareFunction.greater": 4,
    "CompareFunction.greater-equal": 5,
    "CompareFunction.less": 2,
    "CompareFunction.less-equal": 3,
    "CompareFunction.never": 1,
    "CompareFunction.not-equal": 7,
    "CullMode.back": 2,
    "CullMode.front": 1,
    "CullMode.none": 0,
    "FilterMode.linear": 1,
    "FilterMode.nearest": 0,
    "FrontFace.ccw": 0,
    "FrontFace.cw": 1,
    "IndexFormat.uint16": 1,
    "IndexFormat.uint32": 2,
    "InputStepMode.instance": 1,
    "InputStepMode.vertex": 0,
    "LoadOp.load": 1,
    "PowerPreference.high-performance": 1,
    "PowerPreference.low-power": 0,
    "PrimitiveTopology.line-list": 1,
    "PrimitiveTopology.line-strip": 2,
    "PrimitiveTopology.point-list": 0,
    "PrimitiveTopology.triangle-list": 3,
    "PrimitiveTopology.triangle-strip": 4,
    "StencilOperation.decrement-clamp": 5,
    "StencilOperation.decrement-wrap": 7,
    "StencilOperation.increment-clamp": 4,
    "StencilOperation.increment-wrap": 6,
    "StencilOperation.invert": 3,
    "StencilOperation.keep": 0,
    "StencilOperation.replace": 2,
    "StencilOperation.zero": 1,
    "StoreOp.clear": 0,
    "StoreOp.store": 1,
    "TextureAspect.all": 0,
    "TextureAspect.depth-only": 2,
    "TextureAspect.stencil-only": 1,
    "TextureDimension.1d": 0,
    "TextureDimension.2d": 1,
    "TextureDimension.3d": 2,
    "TextureFormat.bc1-rgba-unorm": 38,
    "TextureFormat.bc1-rgba-unorm-srgb": 39,
    "TextureFormat.bc2-rgba-unorm": 40,
    "TextureFormat.bc2-rgba-unorm-srgb": 41,
    "TextureFormat.bc3-rgba-unorm": 42,
    "TextureFormat.bc3-rgba-unorm-srgb": 43,
    "TextureFormat.bc4-r-snorm": 45,
    "TextureFormat.bc4-r-unorm": 44,
    "TextureFormat.bc5-rg-snorm": 47,
    "TextureFormat.bc5-rg-unorm": 46,
    "TextureFormat.bc7-rgba-unorm": 50,
    "TextureFormat.bc7-rgba-unorm-srgb": 51,
    "TextureFormat.bgra8unorm": 22,
    "TextureFormat.bgra8unorm-srgb": 23,
    "TextureFormat.depth24plus": 36,
    "TextureFormat.depth24plus-stencil8": 37,
    "TextureFormat.depth32float": 35,
    "TextureFormat.r16float": 6,
    "TextureFormat.r16sint": 5,
    "TextureFormat.r16uint": 4,
    "TextureFormat.r32float": 13,
    "TextureFormat.r32sint": 12,
    "TextureFormat.r32uint": 11,
    "TextureFormat.r8sint": 3,
    "TextureFormat.r8snorm": 1,
    "TextureFormat.r8uint": 2,
    "TextureFormat.r8unorm": 0,
    "TextureFormat.rg16float": 16,
    "TextureFormat.rg16sint": 15,
    "TextureFormat.rg16uint": 14,
    "TextureFormat.rg32float": 28,
    "TextureFormat.rg32sint": 27,
    "TextureFormat.rg32uint": 26,
    "TextureFormat.rg8sint": 10,
    "TextureFormat.rg8snorm": 8,
    "TextureFormat.rg8uint": 9,
    "TextureFormat.rg8unorm": 7,
    "TextureFormat.rgba16float": 31,
    "TextureFormat.rgba16sint": 30,
    "TextureFormat.rgba16uint": 29,
    "TextureFormat.rgba32float": 34,
    "TextureFormat.rgba32sint": 33,
    "TextureFormat.rgba32uint": 32,
    "TextureFormat.rgba8sint": 21,
    "TextureFormat.rgba8snorm": 19,
    "TextureFormat.rgba8uint": 20,
    "TextureFormat.rgba8unorm": 17,
    "TextureFormat.rgba8unorm-srgb": 18,
    "TextureViewDimension.1d": 0,
    "TextureViewDimension.2d": 1,
    "TextureViewDimension.2d-array": 2,
    "TextureViewDimension.3d": 5,
    "TextureViewDimension.cube": 3,
    "TextureViewDimension.cube-array": 4,
    "VertexFormat.float16x2": 16,
    "VertexFormat.float16x4": 17,
    "VertexFormat.float32": 18,
    "VertexFormat.float32x2": 19,
    "VertexFormat.float32x3": 20,
    "VertexFormat.float32x4": 21,
    "VertexFormat.sint16x2": 10,
    "VertexFormat.sint16x4": 11,
    "VertexFormat.sint32": 26,
    "VertexFormat.sint32x2": 27,
    "VertexFormat.sint32x3": 28,
    "VertexFormat.sint32x4": 29,
    "VertexFormat.sint8x2": 2,
    "VertexFormat.sint8x4": 3,
    "VertexFormat.uint16x2": 8,
    "VertexFormat.uint16x4": 9,
    "VertexFormat.uint32": 22,
    "VertexFormat.uint32x2": 23,
    "VertexFormat.uint32x3": 24,
    "VertexFormat.uint32x4": 25,
    "VertexFormat.uint8x2": 0,
    "VertexFormat.uint8x4": 1,
}

# There are 38 struct-field enum mappings

cstructfield2enum = {
    "BindGroupLayoutEntry.storage_texture_format": "TextureFormat",
    "BindGroupLayoutEntry.view_dimension": "TextureViewDimension",
    "BlendDescriptor.dstFactor": "BlendFactor",
    "BlendDescriptor.operation": "BlendOperation",
    "BlendDescriptor.srcFactor": "BlendFactor",
    "ColorStateDescriptor.format": "TextureFormat",
    "DepthStencilStateDescriptor.depthCompare": "CompareFunction",
    "DepthStencilStateDescriptor.format": "TextureFormat",
    "PassChannel_Color.load_op": "LoadOp",
    "PassChannel_Color.store_op": "StoreOp",
    "PassChannel_f32.load_op": "LoadOp",
    "PassChannel_f32.store_op": "StoreOp",
    "PassChannel_u32.load_op": "LoadOp",
    "PassChannel_u32.store_op": "StoreOp",
    "RasterizationStateDescriptor.cullMode": "CullMode",
    "RasterizationStateDescriptor.frontFace": "FrontFace",
    "RenderPipelineDescriptor.primitiveTopology": "PrimitiveTopology",
    "RequestAdapterOptions.power_preference": "PowerPreference",
    "SamplerDescriptor.address_mode_u": "AddressMode",
    "SamplerDescriptor.address_mode_v": "AddressMode",
    "SamplerDescriptor.address_mode_w": "AddressMode",
    "SamplerDescriptor.compare": "CompareFunction",
    "SamplerDescriptor.mag_filter": "FilterMode",
    "SamplerDescriptor.min_filter": "FilterMode",
    "SamplerDescriptor.mipmap_filter": "FilterMode",
    "StencilStateFaceDescriptor.compare": "CompareFunction",
    "StencilStateFaceDescriptor.depthFailOp": "StencilOperation",
    "StencilStateFaceDescriptor.failOp": "StencilOperation",
    "StencilStateFaceDescriptor.passOp": "StencilOperation",
    "SwapChainDescriptor.format": "TextureFormat",
    "TextureDescriptor.dimension": "TextureDimension",
    "TextureDescriptor.format": "TextureFormat",
    "TextureViewDescriptor.aspect": "TextureAspect",
    "TextureViewDescriptor.dimension": "TextureViewDimension",
    "TextureViewDescriptor.format": "TextureFormat",
    "VertexAttributeDescriptor.format": "VertexFormat",
    "VertexBufferLayoutDescriptor.stepMode": "InputStepMode",
    "VertexStateDescriptor.indexFormat": "IndexFormat",
}
