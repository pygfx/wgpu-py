"""
All wgpu enums. Also available in the root wgpu namespace.
"""

# THIS CODE IS AUTOGENERATED - DO NOT EDIT

_use_sphinx_repr = False


class Enum:
    def __init__(self, name, **kwargs):
        self._name = name
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __iter__(self):
        return iter(
            [getattr(self, key) for key in dir(self) if not key.startswith("_")]
        )

    def __repr__(self):
        options = ", ".join(f"'{x}'" for x in self)
        if _use_sphinx_repr:
            return options
        return f"<{self.__class__.__name__} {self._name}: {options}>"


# %% Enums (26)

PowerPreference = Enum(
    "PowerPreference", low_power="low-power", high_performance="high-performance",
)  #:

ExtensionName = Enum(
    "ExtensionName",
    texture_compression_bc="texture-compression-bc",
    pipeline_statistics_query="pipeline-statistics-query",
    timestamp_query="timestamp-query",
)  #:

TextureDimension = Enum("TextureDimension", d1="1d", d2="2d", d3="3d",)  #:

TextureViewDimension = Enum(
    "TextureViewDimension",
    d1="1d",
    d2="2d",
    d2_array="2d-array",
    cube="cube",
    cube_array="cube-array",
    d3="3d",
)  #:

TextureAspect = Enum(
    "TextureAspect", all="all", stencil_only="stencil-only", depth_only="depth-only",
)  #:

TextureFormat = Enum(
    "TextureFormat",
    r8unorm="r8unorm",
    r8snorm="r8snorm",
    r8uint="r8uint",
    r8sint="r8sint",
    r16uint="r16uint",
    r16sint="r16sint",
    r16float="r16float",
    rg8unorm="rg8unorm",
    rg8snorm="rg8snorm",
    rg8uint="rg8uint",
    rg8sint="rg8sint",
    r32uint="r32uint",
    r32sint="r32sint",
    r32float="r32float",
    rg16uint="rg16uint",
    rg16sint="rg16sint",
    rg16float="rg16float",
    rgba8unorm="rgba8unorm",
    rgba8unorm_srgb="rgba8unorm-srgb",
    rgba8snorm="rgba8snorm",
    rgba8uint="rgba8uint",
    rgba8sint="rgba8sint",
    bgra8unorm="bgra8unorm",
    bgra8unorm_srgb="bgra8unorm-srgb",
    rgb10a2unorm="rgb10a2unorm",
    rg11b10float="rg11b10float",
    rg32uint="rg32uint",
    rg32sint="rg32sint",
    rg32float="rg32float",
    rgba16uint="rgba16uint",
    rgba16sint="rgba16sint",
    rgba16float="rgba16float",
    rgba32uint="rgba32uint",
    rgba32sint="rgba32sint",
    rgba32float="rgba32float",
    depth32float="depth32float",
    depth24plus="depth24plus",
    depth24plus_stencil8="depth24plus-stencil8",
    bc1_rgba_unorm="bc1-rgba-unorm",
    bc1_rgba_unorm_srgb="bc1-rgba-unorm-srgb",
    bc2_rgba_unorm="bc2-rgba-unorm",
    bc2_rgba_unorm_srgb="bc2-rgba-unorm-srgb",
    bc3_rgba_unorm="bc3-rgba-unorm",
    bc3_rgba_unorm_srgb="bc3-rgba-unorm-srgb",
    bc4_r_unorm="bc4-r-unorm",
    bc4_r_snorm="bc4-r-snorm",
    bc5_rg_unorm="bc5-rg-unorm",
    bc5_rg_snorm="bc5-rg-snorm",
    bc6h_rgb_ufloat="bc6h-rgb-ufloat",
    bc6h_rgb_sfloat="bc6h-rgb-sfloat",
    bc7_rgba_unorm="bc7-rgba-unorm",
    bc7_rgba_unorm_srgb="bc7-rgba-unorm-srgb",
)  #:

TextureComponentType = Enum(
    "TextureComponentType", float="float", sint="sint", uint="uint",
)  #:

AddressMode = Enum(
    "AddressMode",
    clamp_to_edge="clamp-to-edge",
    repeat="repeat",
    mirror_repeat="mirror-repeat",
)  #:

FilterMode = Enum("FilterMode", nearest="nearest", linear="linear",)  #:

CompareFunction = Enum(
    "CompareFunction",
    never="never",
    less="less",
    equal="equal",
    less_equal="less-equal",
    greater="greater",
    not_equal="not-equal",
    greater_equal="greater-equal",
    always="always",
)  #:

BindingType = Enum(
    "BindingType",
    uniform_buffer="uniform-buffer",
    storage_buffer="storage-buffer",
    readonly_storage_buffer="readonly-storage-buffer",
    sampler="sampler",
    comparison_sampler="comparison-sampler",
    sampled_texture="sampled-texture",
    readonly_storage_texture="readonly-storage-texture",
    writeonly_storage_texture="writeonly-storage-texture",
)  #:

CompilationMessageType = Enum(
    "CompilationMessageType", error="error", warning="warning", info="info",
)  #:

PrimitiveTopology = Enum(
    "PrimitiveTopology",
    point_list="point-list",
    line_list="line-list",
    line_strip="line-strip",
    triangle_list="triangle-list",
    triangle_strip="triangle-strip",
)  #:

FrontFace = Enum("FrontFace", ccw="ccw", cw="cw",)  #:

CullMode = Enum("CullMode", none="none", front="front", back="back",)  #:

BlendFactor = Enum(
    "BlendFactor",
    zero="zero",
    one="one",
    src_color="src-color",
    one_minus_src_color="one-minus-src-color",
    src_alpha="src-alpha",
    one_minus_src_alpha="one-minus-src-alpha",
    dst_color="dst-color",
    one_minus_dst_color="one-minus-dst-color",
    dst_alpha="dst-alpha",
    one_minus_dst_alpha="one-minus-dst-alpha",
    src_alpha_saturated="src-alpha-saturated",
    blend_color="blend-color",
    one_minus_blend_color="one-minus-blend-color",
)  #:

BlendOperation = Enum(
    "BlendOperation",
    add="add",
    subtract="subtract",
    reverse_subtract="reverse-subtract",
    min="min",
    max="max",
)  #:

StencilOperation = Enum(
    "StencilOperation",
    keep="keep",
    zero="zero",
    replace="replace",
    invert="invert",
    increment_clamp="increment-clamp",
    decrement_clamp="decrement-clamp",
    increment_wrap="increment-wrap",
    decrement_wrap="decrement-wrap",
)  #:

IndexFormat = Enum("IndexFormat", uint16="uint16", uint32="uint32",)  #:

VertexFormat = Enum(
    "VertexFormat",
    uchar2="uchar2",
    uchar4="uchar4",
    char2="char2",
    char4="char4",
    uchar2norm="uchar2norm",
    uchar4norm="uchar4norm",
    char2norm="char2norm",
    char4norm="char4norm",
    ushort2="ushort2",
    ushort4="ushort4",
    short2="short2",
    short4="short4",
    ushort2norm="ushort2norm",
    ushort4norm="ushort4norm",
    short2norm="short2norm",
    short4norm="short4norm",
    half2="half2",
    half4="half4",
    float="float",
    float2="float2",
    float3="float3",
    float4="float4",
    uint="uint",
    uint2="uint2",
    uint3="uint3",
    uint4="uint4",
    int="int",
    int2="int2",
    int3="int3",
    int4="int4",
)  #:

InputStepMode = Enum("InputStepMode", vertex="vertex", instance="instance",)  #:

LoadOp = Enum("LoadOp", load="load",)  #:

StoreOp = Enum("StoreOp", store="store", clear="clear",)  #:

QueryType = Enum(
    "QueryType",
    occlusion="occlusion",
    pipeline_statistics="pipeline-statistics",
    timestamp="timestamp",
)  #:

PipelineStatisticName = Enum(
    "PipelineStatisticName",
    vertex_shader_invocations="vertex-shader-invocations",
    clipper_invocations="clipper-invocations",
    clipper_primitives_out="clipper-primitives-out",
    fragment_shader_invocations="fragment-shader-invocations",
    compute_shader_invocations="compute-shader-invocations",
)  #:

ErrorFilter = Enum(
    "ErrorFilter", none="none", out_of_memory="out-of-memory", validation="validation",
)  #:
