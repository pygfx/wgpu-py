""" Mappings for the rs backend. """

# THIS CODE IS AUTOGENERATED - DO NOT EDIT

# flake8: noqa

# There are 188 enum mappings

enummap = {
    "AddressMode.clamp-to-edge": 2,
    "AddressMode.mirror-repeat": 1,
    "AddressMode.repeat": 0,
    "BlendFactor.constant": 11,
    "BlendFactor.dst": 6,
    "BlendFactor.dst-alpha": 8,
    "BlendFactor.one": 1,
    "BlendFactor.one-minus-constant": 12,
    "BlendFactor.one-minus-dst": 7,
    "BlendFactor.one-minus-dst-alpha": 9,
    "BlendFactor.one-minus-src": 3,
    "BlendFactor.one-minus-src-alpha": 5,
    "BlendFactor.src": 2,
    "BlendFactor.src-alpha": 4,
    "BlendFactor.src-alpha-saturated": 10,
    "BlendFactor.zero": 0,
    "BlendOperation.add": 0,
    "BlendOperation.max": 4,
    "BlendOperation.min": 3,
    "BlendOperation.reverse-subtract": 2,
    "BlendOperation.subtract": 1,
    "BufferBindingType.read-only-storage": 3,
    "BufferBindingType.storage": 2,
    "BufferBindingType.uniform": 1,
    "CompareFunction.always": 8,
    "CompareFunction.equal": 6,
    "CompareFunction.greater": 4,
    "CompareFunction.greater-equal": 5,
    "CompareFunction.less": 2,
    "CompareFunction.less-equal": 3,
    "CompareFunction.never": 1,
    "CompareFunction.not-equal": 7,
    "CompilationMessageType.error": 0,
    "CompilationMessageType.info": 2,
    "CompilationMessageType.warning": 1,
    "CullMode.back": 2,
    "CullMode.front": 1,
    "CullMode.none": 0,
    "DeviceLostReason.destroyed": 1,
    "ErrorFilter.out-of-memory": 2,
    "ErrorFilter.validation": 1,
    "FeatureName.depth24unorm-stencil8": 2,
    "FeatureName.depth32float-stencil8": 3,
    "FeatureName.pipeline-statistics-query": 5,
    "FeatureName.texture-compression-bc": 6,
    "FeatureName.timestamp-query": 4,
    "FilterMode.linear": 1,
    "FilterMode.nearest": 0,
    "FrontFace.ccw": 0,
    "FrontFace.cw": 1,
    "IndexFormat.uint16": 1,
    "IndexFormat.uint32": 2,
    "LoadOp.load": 1,
    "PipelineStatisticName.clipper-invocations": 1,
    "PipelineStatisticName.clipper-primitives-out": 2,
    "PipelineStatisticName.compute-shader-invocations": 4,
    "PipelineStatisticName.fragment-shader-invocations": 3,
    "PipelineStatisticName.vertex-shader-invocations": 0,
    "PowerPreference.high-performance": 2,
    "PowerPreference.low-power": 1,
    "PrimitiveTopology.line-list": 1,
    "PrimitiveTopology.line-strip": 2,
    "PrimitiveTopology.point-list": 0,
    "PrimitiveTopology.triangle-list": 3,
    "PrimitiveTopology.triangle-strip": 4,
    "QueryType.occlusion": 0,
    "QueryType.pipeline-statistics": 1,
    "QueryType.timestamp": 2,
    "SamplerBindingType.comparison": 3,
    "SamplerBindingType.filtering": 1,
    "SamplerBindingType.non-filtering": 2,
    "StencilOperation.decrement-clamp": 5,
    "StencilOperation.decrement-wrap": 7,
    "StencilOperation.increment-clamp": 4,
    "StencilOperation.increment-wrap": 6,
    "StencilOperation.invert": 3,
    "StencilOperation.keep": 0,
    "StencilOperation.replace": 2,
    "StencilOperation.zero": 1,
    "StorageTextureAccess.write-only": 1,
    "StoreOp.discard": 1,
    "StoreOp.store": 0,
    "TextureAspect.all": 0,
    "TextureAspect.depth-only": 2,
    "TextureAspect.stencil-only": 1,
    "TextureDimension.1d": 0,
    "TextureDimension.2d": 1,
    "TextureDimension.3d": 2,
    "TextureFormat.bc1-rgba-unorm": 44,
    "TextureFormat.bc1-rgba-unorm-srgb": 45,
    "TextureFormat.bc2-rgba-unorm": 46,
    "TextureFormat.bc2-rgba-unorm-srgb": 47,
    "TextureFormat.bc3-rgba-unorm": 48,
    "TextureFormat.bc3-rgba-unorm-srgb": 49,
    "TextureFormat.bc4-r-snorm": 51,
    "TextureFormat.bc4-r-unorm": 50,
    "TextureFormat.bc5-rg-snorm": 53,
    "TextureFormat.bc5-rg-unorm": 52,
    "TextureFormat.bc6h-rgb-float": 55,
    "TextureFormat.bc6h-rgb-ufloat": 54,
    "TextureFormat.bc7-rgba-unorm": 56,
    "TextureFormat.bc7-rgba-unorm-srgb": 57,
    "TextureFormat.bgra8unorm": 23,
    "TextureFormat.bgra8unorm-srgb": 24,
    "TextureFormat.depth16unorm": 38,
    "TextureFormat.depth24plus": 39,
    "TextureFormat.depth24plus-stencil8": 40,
    "TextureFormat.depth24unorm-stencil8": 41,
    "TextureFormat.depth32float": 42,
    "TextureFormat.depth32float-stencil8": 43,
    "TextureFormat.r16float": 7,
    "TextureFormat.r16sint": 6,
    "TextureFormat.r16uint": 5,
    "TextureFormat.r32float": 12,
    "TextureFormat.r32sint": 14,
    "TextureFormat.r32uint": 13,
    "TextureFormat.r8sint": 4,
    "TextureFormat.r8snorm": 2,
    "TextureFormat.r8uint": 3,
    "TextureFormat.r8unorm": 1,
    "TextureFormat.rg11b10ufloat": 26,
    "TextureFormat.rg16float": 17,
    "TextureFormat.rg16sint": 16,
    "TextureFormat.rg16uint": 15,
    "TextureFormat.rg32float": 28,
    "TextureFormat.rg32sint": 30,
    "TextureFormat.rg32uint": 29,
    "TextureFormat.rg8sint": 11,
    "TextureFormat.rg8snorm": 9,
    "TextureFormat.rg8uint": 10,
    "TextureFormat.rg8unorm": 8,
    "TextureFormat.rgb10a2unorm": 25,
    "TextureFormat.rgb9e5ufloat": 27,
    "TextureFormat.rgba16float": 33,
    "TextureFormat.rgba16sint": 32,
    "TextureFormat.rgba16uint": 31,
    "TextureFormat.rgba32float": 34,
    "TextureFormat.rgba32sint": 36,
    "TextureFormat.rgba32uint": 35,
    "TextureFormat.rgba8sint": 22,
    "TextureFormat.rgba8snorm": 20,
    "TextureFormat.rgba8uint": 21,
    "TextureFormat.rgba8unorm": 18,
    "TextureFormat.rgba8unorm-srgb": 19,
    "TextureFormat.stencil8": 37,
    "TextureSampleType.depth": 3,
    "TextureSampleType.float": 1,
    "TextureSampleType.sint": 4,
    "TextureSampleType.uint": 5,
    "TextureSampleType.unfilterable-float": 2,
    "TextureViewDimension.1d": 1,
    "TextureViewDimension.2d": 2,
    "TextureViewDimension.2d-array": 3,
    "TextureViewDimension.3d": 6,
    "TextureViewDimension.cube": 4,
    "TextureViewDimension.cube-array": 5,
    "VertexFormat.float16x2": 17,
    "VertexFormat.float16x4": 18,
    "VertexFormat.float32": 19,
    "VertexFormat.float32x2": 20,
    "VertexFormat.float32x3": 21,
    "VertexFormat.float32x4": 22,
    "VertexFormat.sint16x2": 11,
    "VertexFormat.sint16x4": 12,
    "VertexFormat.sint32": 27,
    "VertexFormat.sint32x2": 28,
    "VertexFormat.sint32x3": 29,
    "VertexFormat.sint32x4": 30,
    "VertexFormat.sint8x2": 3,
    "VertexFormat.sint8x4": 4,
    "VertexFormat.snorm16x2": 15,
    "VertexFormat.snorm16x4": 16,
    "VertexFormat.snorm8x2": 7,
    "VertexFormat.snorm8x4": 8,
    "VertexFormat.uint16x2": 9,
    "VertexFormat.uint16x4": 10,
    "VertexFormat.uint32": 23,
    "VertexFormat.uint32x2": 24,
    "VertexFormat.uint32x3": 25,
    "VertexFormat.uint32x4": 26,
    "VertexFormat.uint8x2": 1,
    "VertexFormat.uint8x4": 2,
    "VertexFormat.unorm16x2": 13,
    "VertexFormat.unorm16x4": 14,
    "VertexFormat.unorm8x2": 5,
    "VertexFormat.unorm8x4": 6,
    "VertexStepMode.instance": 1,
    "VertexStepMode.vertex": 0,
}

# There are 47 struct-field enum mappings

cstructfield2enum = {
    "BlendComponent.dstFactor": "BlendFactor",
    "BlendComponent.operation": "BlendOperation",
    "BlendComponent.srcFactor": "BlendFactor",
    "BufferBindingLayout.type": "BufferBindingType",
    "ColorTargetState.format": "TextureFormat",
    "CompilationMessage.type": "CompilationMessageType",
    "DepthStencilState.depthCompare": "CompareFunction",
    "DepthStencilState.format": "TextureFormat",
    "ImageCopyTexture.aspect": "TextureAspect",
    "PrimitiveState.cullMode": "CullMode",
    "PrimitiveState.frontFace": "FrontFace",
    "PrimitiveState.stripIndexFormat": "IndexFormat",
    "PrimitiveState.topology": "PrimitiveTopology",
    "QuerySetDescriptor.type": "QueryType",
    "RenderBundleEncoderDescriptor.depthStencilFormat": "TextureFormat",
    "RenderPassColorAttachment.loadOp": "LoadOp",
    "RenderPassColorAttachment.storeOp": "StoreOp",
    "RenderPassDepthStencilAttachment.depthLoadOp": "LoadOp",
    "RenderPassDepthStencilAttachment.depthStoreOp": "StoreOp",
    "RenderPassDepthStencilAttachment.stencilLoadOp": "LoadOp",
    "RenderPassDepthStencilAttachment.stencilStoreOp": "StoreOp",
    "RequestAdapterOptions.powerPreference": "PowerPreference",
    "SamplerBindingLayout.type": "SamplerBindingType",
    "SamplerDescriptor.addressModeU": "AddressMode",
    "SamplerDescriptor.addressModeV": "AddressMode",
    "SamplerDescriptor.addressModeW": "AddressMode",
    "SamplerDescriptor.compare": "CompareFunction",
    "SamplerDescriptor.magFilter": "FilterMode",
    "SamplerDescriptor.minFilter": "FilterMode",
    "SamplerDescriptor.mipmapFilter": "FilterMode",
    "StencilFaceState.compare": "CompareFunction",
    "StencilFaceState.depthFailOp": "StencilOperation",
    "StencilFaceState.failOp": "StencilOperation",
    "StencilFaceState.passOp": "StencilOperation",
    "StorageTextureBindingLayout.access": "StorageTextureAccess",
    "StorageTextureBindingLayout.format": "TextureFormat",
    "StorageTextureBindingLayout.viewDimension": "TextureViewDimension",
    "SwapChainDescriptor.format": "TextureFormat",
    "TextureBindingLayout.sampleType": "TextureSampleType",
    "TextureBindingLayout.viewDimension": "TextureViewDimension",
    "TextureDescriptor.dimension": "TextureDimension",
    "TextureDescriptor.format": "TextureFormat",
    "TextureViewDescriptor.aspect": "TextureAspect",
    "TextureViewDescriptor.dimension": "TextureViewDimension",
    "TextureViewDescriptor.format": "TextureFormat",
    "VertexAttribute.format": "VertexFormat",
    "VertexBufferLayout.stepMode": "VertexStepMode",
}

enum_str2int = {
    "BackendType": {
        "Null": 0,
        "WebGPU": 1,
        "D3D11": 2,
        "D3D12": 3,
        "Metal": 4,
        "Vulkan": 5,
        "OpenGL": 6,
        "OpenGLES": 7,
    }
}
enum_int2str = {
    "BackendType": {
        0: "Null",
        1: "WebGPU",
        2: "D3D11",
        3: "D3D12",
        4: "Metal",
        5: "Vulkan",
        6: "OpenGL",
        7: "OpenGLES",
    },
    "AdapterType": {
        0: "DiscreteGPU",
        1: "IntegratedGPU",
        2: "CPU",
        3: "Unknown",
    },
}
