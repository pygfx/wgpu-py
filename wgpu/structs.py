"""
The sructs in wgpu-py are represented as Python dictionaries.
Fields that have default values (as indicated below) may be omitted.
"""

# THIS CODE IS AUTOGENERATED - DO NOT EDIT

_use_sphinx_repr = False


class Struct:
    def __init__(self, name, **kwargs):
        self._name = name
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __iter__(self):
        return iter([key for key in dir(self) if not key.startswith("_")])

    def __repr__(self):
        if _use_sphinx_repr:  # no-cover
            return ""
        options = ", ".join(f"'{x}'" for x in self)
        return f"<{self.__class__.__name__} {self._name}: {options}>"


# There are 59 structs

#: * powerPreference :: :obj:`enums.PowerPreference <wgpu.enums.PowerPreference>` = None
#: * forceFallbackAdapter :: bool = false
RequestAdapterOptions = Struct(
    "RequestAdapterOptions",
    power_preference="enums.PowerPreference",
    force_fallback_adapter="bool",
)

#: * label :: str = None
#: * requiredFeatures :: List[:obj:`enums.FeatureName <wgpu.enums.FeatureName>`] = []
#: * requiredLimits :: Dict[str, int] = {}
#: * defaultQueue :: :obj:`structs.QueueDescriptor <QueueDescriptor>` = {}
DeviceDescriptor = Struct(
    "DeviceDescriptor",
    label="str",
    required_features="List[enums.FeatureName]",
    required_limits="Dict[str, int]",
    default_queue="structs.QueueDescriptor",
)

#: * label :: str = None
#: * size :: int
#: * usage :: :obj:`flags.BufferUsage <wgpu.flags.BufferUsage>`
#: * mappedAtCreation :: bool = false
BufferDescriptor = Struct(
    "BufferDescriptor",
    label="str",
    size="int",
    usage="flags.BufferUsage",
    mapped_at_creation="bool",
)

#: * label :: str = None
#: * size :: Union[List[int], :obj:`structs.Extent3D <Extent3D>`]
#: * mipLevelCount :: int = 1
#: * sampleCount :: int = 1
#: * dimension :: :obj:`enums.TextureDimension <wgpu.enums.TextureDimension>` = "2d"
#: * format :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`
#: * usage :: :obj:`flags.TextureUsage <wgpu.flags.TextureUsage>`
#: * viewFormats :: List[:obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`] = []
TextureDescriptor = Struct(
    "TextureDescriptor",
    label="str",
    size="Union[List[int], structs.Extent3D]",
    mip_level_count="int",
    sample_count="int",
    dimension="enums.TextureDimension",
    format="enums.TextureFormat",
    usage="flags.TextureUsage",
    view_formats="List[enums.TextureFormat]",
)

#: * label :: str = None
#: * format :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>` = None
#: * dimension :: :obj:`enums.TextureViewDimension <wgpu.enums.TextureViewDimension>` = None
#: * aspect :: :obj:`enums.TextureAspect <wgpu.enums.TextureAspect>` = "all"
#: * baseMipLevel :: int = 0
#: * mipLevelCount :: int = None
#: * baseArrayLayer :: int = 0
#: * arrayLayerCount :: int = None
TextureViewDescriptor = Struct(
    "TextureViewDescriptor",
    label="str",
    format="enums.TextureFormat",
    dimension="enums.TextureViewDimension",
    aspect="enums.TextureAspect",
    base_mip_level="int",
    mip_level_count="int",
    base_array_layer="int",
    array_layer_count="int",
)

#: * label :: str = None
#: * source :: object
#: * colorSpace :: str = "srgb"
ExternalTextureDescriptor = Struct(
    "ExternalTextureDescriptor",
    label="str",
    source="object",
    color_space="str",
)

#: * label :: str = None
#: * addressModeU :: :obj:`enums.AddressMode <wgpu.enums.AddressMode>` = "clamp-to-edge"
#: * addressModeV :: :obj:`enums.AddressMode <wgpu.enums.AddressMode>` = "clamp-to-edge"
#: * addressModeW :: :obj:`enums.AddressMode <wgpu.enums.AddressMode>` = "clamp-to-edge"
#: * magFilter :: :obj:`enums.FilterMode <wgpu.enums.FilterMode>` = "nearest"
#: * minFilter :: :obj:`enums.FilterMode <wgpu.enums.FilterMode>` = "nearest"
#: * mipmapFilter :: :obj:`enums.MipmapFilterMode <wgpu.enums.MipmapFilterMode>` = "nearest"
#: * lodMinClamp :: float = 0
#: * lodMaxClamp :: float = 32
#: * compare :: :obj:`enums.CompareFunction <wgpu.enums.CompareFunction>` = None
#: * maxAnisotropy :: int = 1
SamplerDescriptor = Struct(
    "SamplerDescriptor",
    label="str",
    address_mode_u="enums.AddressMode",
    address_mode_v="enums.AddressMode",
    address_mode_w="enums.AddressMode",
    mag_filter="enums.FilterMode",
    min_filter="enums.FilterMode",
    mipmap_filter="enums.MipmapFilterMode",
    lod_min_clamp="float",
    lod_max_clamp="float",
    compare="enums.CompareFunction",
    max_anisotropy="int",
)

#: * label :: str = None
#: * entries :: List[:obj:`structs.BindGroupLayoutEntry <BindGroupLayoutEntry>`]
BindGroupLayoutDescriptor = Struct(
    "BindGroupLayoutDescriptor",
    label="str",
    entries="List[structs.BindGroupLayoutEntry]",
)

#: * binding :: int
#: * visibility :: :obj:`flags.ShaderStage <wgpu.flags.ShaderStage>`
#: * buffer :: :obj:`structs.BufferBindingLayout <BufferBindingLayout>` = None
#: * sampler :: :obj:`structs.SamplerBindingLayout <SamplerBindingLayout>` = None
#: * texture :: :obj:`structs.TextureBindingLayout <TextureBindingLayout>` = None
#: * storageTexture :: :obj:`structs.StorageTextureBindingLayout <StorageTextureBindingLayout>` = None
#: * externalTexture :: :obj:`structs.ExternalTextureBindingLayout <ExternalTextureBindingLayout>` = None
BindGroupLayoutEntry = Struct(
    "BindGroupLayoutEntry",
    binding="int",
    visibility="flags.ShaderStage",
    buffer="structs.BufferBindingLayout",
    sampler="structs.SamplerBindingLayout",
    texture="structs.TextureBindingLayout",
    storage_texture="structs.StorageTextureBindingLayout",
    external_texture="structs.ExternalTextureBindingLayout",
)

#: * type :: :obj:`enums.BufferBindingType <wgpu.enums.BufferBindingType>` = "uniform"
#: * hasDynamicOffset :: bool = false
#: * minBindingSize :: int = 0
BufferBindingLayout = Struct(
    "BufferBindingLayout",
    type="enums.BufferBindingType",
    has_dynamic_offset="bool",
    min_binding_size="int",
)

#: * type :: :obj:`enums.SamplerBindingType <wgpu.enums.SamplerBindingType>` = "filtering"
SamplerBindingLayout = Struct(
    "SamplerBindingLayout",
    type="enums.SamplerBindingType",
)

#: * sampleType :: :obj:`enums.TextureSampleType <wgpu.enums.TextureSampleType>` = "float"
#: * viewDimension :: :obj:`enums.TextureViewDimension <wgpu.enums.TextureViewDimension>` = "2d"
#: * multisampled :: bool = false
TextureBindingLayout = Struct(
    "TextureBindingLayout",
    sample_type="enums.TextureSampleType",
    view_dimension="enums.TextureViewDimension",
    multisampled="bool",
)

#: * access :: :obj:`enums.StorageTextureAccess <wgpu.enums.StorageTextureAccess>` = "write-only"
#: * format :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`
#: * viewDimension :: :obj:`enums.TextureViewDimension <wgpu.enums.TextureViewDimension>` = "2d"
StorageTextureBindingLayout = Struct(
    "StorageTextureBindingLayout",
    access="enums.StorageTextureAccess",
    format="enums.TextureFormat",
    view_dimension="enums.TextureViewDimension",
)

ExternalTextureBindingLayout = Struct(
    "ExternalTextureBindingLayout",
)

#: * label :: str = None
#: * layout :: :class:`GPUBindGroupLayout <wgpu.GPUBindGroupLayout>`
#: * entries :: List[:obj:`structs.BindGroupEntry <BindGroupEntry>`]
BindGroupDescriptor = Struct(
    "BindGroupDescriptor",
    label="str",
    layout="GPUBindGroupLayout",
    entries="List[structs.BindGroupEntry]",
)

#: * binding :: int
#: * resource :: Union[:class:`GPUExternalTexture <wgpu.GPUExternalTexture>`, :class:`GPUSampler <wgpu.GPUSampler>`, :class:`GPUTextureView <wgpu.GPUTextureView>`, :obj:`structs.BufferBinding <BufferBinding>`]
BindGroupEntry = Struct(
    "BindGroupEntry",
    binding="int",
    resource="Union[GPUExternalTexture, GPUSampler, GPUTextureView, structs.BufferBinding]",
)

#: * buffer :: :class:`GPUBuffer <wgpu.GPUBuffer>`
#: * offset :: int = 0
#: * size :: int = None
BufferBinding = Struct(
    "BufferBinding",
    buffer="GPUBuffer",
    offset="int",
    size="int",
)

#: * label :: str = None
#: * bindGroupLayouts :: List[:class:`GPUBindGroupLayout <wgpu.GPUBindGroupLayout>`]
PipelineLayoutDescriptor = Struct(
    "PipelineLayoutDescriptor",
    label="str",
    bind_group_layouts="List[GPUBindGroupLayout]",
)

#: * label :: str = None
#: * code :: str
#: * sourceMap :: dict = None
#: * hints :: Dict[str, :obj:`structs.ShaderModuleCompilationHint <ShaderModuleCompilationHint>`] = None
ShaderModuleDescriptor = Struct(
    "ShaderModuleDescriptor",
    label="str",
    code="str",
    source_map="dict",
    hints="Dict[str, structs.ShaderModuleCompilationHint]",
)

#: * layout :: Union[:class:`GPUPipelineLayout <wgpu.GPUPipelineLayout>`, :obj:`enums.AutoLayoutMode <wgpu.enums.AutoLayoutMode>`] = None
ShaderModuleCompilationHint = Struct(
    "ShaderModuleCompilationHint",
    layout="Union[GPUPipelineLayout, enums.AutoLayoutMode]",
)

#: * reason :: :obj:`enums.PipelineErrorReason <wgpu.enums.PipelineErrorReason>`
PipelineErrorInit = Struct(
    "PipelineErrorInit",
    reason="enums.PipelineErrorReason",
)

#: * module :: :class:`GPUShaderModule <wgpu.GPUShaderModule>`
#: * entryPoint :: str
#: * constants :: Dict[str, float] = None
ProgrammableStage = Struct(
    "ProgrammableStage",
    module="GPUShaderModule",
    entry_point="str",
    constants="Dict[str, float]",
)

#: * label :: str = None
#: * layout :: Union[:class:`GPUPipelineLayout <wgpu.GPUPipelineLayout>`, :obj:`enums.AutoLayoutMode <wgpu.enums.AutoLayoutMode>`]
#: * compute :: :obj:`structs.ProgrammableStage <ProgrammableStage>`
ComputePipelineDescriptor = Struct(
    "ComputePipelineDescriptor",
    label="str",
    layout="Union[GPUPipelineLayout, enums.AutoLayoutMode]",
    compute="structs.ProgrammableStage",
)

#: * label :: str = None
#: * layout :: Union[:class:`GPUPipelineLayout <wgpu.GPUPipelineLayout>`, :obj:`enums.AutoLayoutMode <wgpu.enums.AutoLayoutMode>`]
#: * vertex :: :obj:`structs.VertexState <VertexState>`
#: * primitive :: :obj:`structs.PrimitiveState <PrimitiveState>` = {}
#: * depthStencil :: :obj:`structs.DepthStencilState <DepthStencilState>` = None
#: * multisample :: :obj:`structs.MultisampleState <MultisampleState>` = {}
#: * fragment :: :obj:`structs.FragmentState <FragmentState>` = None
RenderPipelineDescriptor = Struct(
    "RenderPipelineDescriptor",
    label="str",
    layout="Union[GPUPipelineLayout, enums.AutoLayoutMode]",
    vertex="structs.VertexState",
    primitive="structs.PrimitiveState",
    depth_stencil="structs.DepthStencilState",
    multisample="structs.MultisampleState",
    fragment="structs.FragmentState",
)

#: * topology :: :obj:`enums.PrimitiveTopology <wgpu.enums.PrimitiveTopology>` = "triangle-list"
#: * stripIndexFormat :: :obj:`enums.IndexFormat <wgpu.enums.IndexFormat>` = None
#: * frontFace :: :obj:`enums.FrontFace <wgpu.enums.FrontFace>` = "ccw"
#: * cullMode :: :obj:`enums.CullMode <wgpu.enums.CullMode>` = "none"
#: * unclippedDepth :: bool = false
PrimitiveState = Struct(
    "PrimitiveState",
    topology="enums.PrimitiveTopology",
    strip_index_format="enums.IndexFormat",
    front_face="enums.FrontFace",
    cull_mode="enums.CullMode",
    unclipped_depth="bool",
)

#: * count :: int = 1
#: * mask :: int = 0xFFFFFFFF
#: * alphaToCoverageEnabled :: bool = false
MultisampleState = Struct(
    "MultisampleState",
    count="int",
    mask="int",
    alpha_to_coverage_enabled="bool",
)

#: * module :: :class:`GPUShaderModule <wgpu.GPUShaderModule>`
#: * entryPoint :: str
#: * constants :: Dict[str, float] = None
#: * targets :: List[:obj:`structs.ColorTargetState <ColorTargetState>`]
FragmentState = Struct(
    "FragmentState",
    module="GPUShaderModule",
    entry_point="str",
    constants="Dict[str, float]",
    targets="List[structs.ColorTargetState]",
)

#: * format :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`
#: * blend :: :obj:`structs.BlendState <BlendState>` = None
#: * writeMask :: :obj:`flags.ColorWrite <wgpu.flags.ColorWrite>` = 0xF
ColorTargetState = Struct(
    "ColorTargetState",
    format="enums.TextureFormat",
    blend="structs.BlendState",
    write_mask="flags.ColorWrite",
)

#: * color :: :obj:`structs.BlendComponent <BlendComponent>`
#: * alpha :: :obj:`structs.BlendComponent <BlendComponent>`
BlendState = Struct(
    "BlendState",
    color="structs.BlendComponent",
    alpha="structs.BlendComponent",
)

#: * operation :: :obj:`enums.BlendOperation <wgpu.enums.BlendOperation>` = "add"
#: * srcFactor :: :obj:`enums.BlendFactor <wgpu.enums.BlendFactor>` = "one"
#: * dstFactor :: :obj:`enums.BlendFactor <wgpu.enums.BlendFactor>` = "zero"
BlendComponent = Struct(
    "BlendComponent",
    operation="enums.BlendOperation",
    src_factor="enums.BlendFactor",
    dst_factor="enums.BlendFactor",
)

#: * format :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`
#: * depthWriteEnabled :: bool = false
#: * depthCompare :: :obj:`enums.CompareFunction <wgpu.enums.CompareFunction>` = "always"
#: * stencilFront :: :obj:`structs.StencilFaceState <StencilFaceState>` = {}
#: * stencilBack :: :obj:`structs.StencilFaceState <StencilFaceState>` = {}
#: * stencilReadMask :: int = 0xFFFFFFFF
#: * stencilWriteMask :: int = 0xFFFFFFFF
#: * depthBias :: int = 0
#: * depthBiasSlopeScale :: float = 0
#: * depthBiasClamp :: float = 0
DepthStencilState = Struct(
    "DepthStencilState",
    format="enums.TextureFormat",
    depth_write_enabled="bool",
    depth_compare="enums.CompareFunction",
    stencil_front="structs.StencilFaceState",
    stencil_back="structs.StencilFaceState",
    stencil_read_mask="int",
    stencil_write_mask="int",
    depth_bias="int",
    depth_bias_slope_scale="float",
    depth_bias_clamp="float",
)

#: * compare :: :obj:`enums.CompareFunction <wgpu.enums.CompareFunction>` = "always"
#: * failOp :: :obj:`enums.StencilOperation <wgpu.enums.StencilOperation>` = "keep"
#: * depthFailOp :: :obj:`enums.StencilOperation <wgpu.enums.StencilOperation>` = "keep"
#: * passOp :: :obj:`enums.StencilOperation <wgpu.enums.StencilOperation>` = "keep"
StencilFaceState = Struct(
    "StencilFaceState",
    compare="enums.CompareFunction",
    fail_op="enums.StencilOperation",
    depth_fail_op="enums.StencilOperation",
    pass_op="enums.StencilOperation",
)

#: * module :: :class:`GPUShaderModule <wgpu.GPUShaderModule>`
#: * entryPoint :: str
#: * constants :: Dict[str, float] = None
#: * buffers :: List[:obj:`structs.VertexBufferLayout <VertexBufferLayout>`] = []
VertexState = Struct(
    "VertexState",
    module="GPUShaderModule",
    entry_point="str",
    constants="Dict[str, float]",
    buffers="List[structs.VertexBufferLayout]",
)

#: * arrayStride :: int
#: * stepMode :: :obj:`enums.VertexStepMode <wgpu.enums.VertexStepMode>` = "vertex"
#: * attributes :: List[:obj:`structs.VertexAttribute <VertexAttribute>`]
VertexBufferLayout = Struct(
    "VertexBufferLayout",
    array_stride="int",
    step_mode="enums.VertexStepMode",
    attributes="List[structs.VertexAttribute]",
)

#: * format :: :obj:`enums.VertexFormat <wgpu.enums.VertexFormat>`
#: * offset :: int
#: * shaderLocation :: int
VertexAttribute = Struct(
    "VertexAttribute",
    format="enums.VertexFormat",
    offset="int",
    shader_location="int",
)

#: * offset :: int = 0
#: * bytesPerRow :: int = None
#: * rowsPerImage :: int = None
ImageDataLayout = Struct(
    "ImageDataLayout",
    offset="int",
    bytes_per_row="int",
    rows_per_image="int",
)

#: * offset :: int = 0
#: * bytesPerRow :: int = None
#: * rowsPerImage :: int = None
#: * buffer :: :class:`GPUBuffer <wgpu.GPUBuffer>`
ImageCopyBuffer = Struct(
    "ImageCopyBuffer",
    offset="int",
    bytes_per_row="int",
    rows_per_image="int",
    buffer="GPUBuffer",
)

#: * texture :: :class:`GPUTexture <wgpu.GPUTexture>`
#: * mipLevel :: int = 0
#: * origin :: Union[List[int], :obj:`structs.Origin3D <Origin3D>`] = {}
#: * aspect :: :obj:`enums.TextureAspect <wgpu.enums.TextureAspect>` = "all"
ImageCopyTexture = Struct(
    "ImageCopyTexture",
    texture="GPUTexture",
    mip_level="int",
    origin="Union[List[int], structs.Origin3D]",
    aspect="enums.TextureAspect",
)

#: * source :: Union[memoryview, object]
#: * origin :: Union[List[int], :obj:`structs.Origin2D <Origin2D>`] = {}
#: * flipY :: bool = false
ImageCopyExternalImage = Struct(
    "ImageCopyExternalImage",
    source="Union[memoryview, object]",
    origin="Union[List[int], structs.Origin2D]",
    flip_y="bool",
)

#: * label :: str = None
CommandBufferDescriptor = Struct(
    "CommandBufferDescriptor",
    label="str",
)

#: * label :: str = None
CommandEncoderDescriptor = Struct(
    "CommandEncoderDescriptor",
    label="str",
)

#: * querySet :: :class:`GPUQuerySet <wgpu.GPUQuerySet>`
#: * queryIndex :: int
#: * location :: :obj:`enums.ComputePassTimestampLocation <wgpu.enums.ComputePassTimestampLocation>`
ComputePassTimestampWrite = Struct(
    "ComputePassTimestampWrite",
    query_set="GPUQuerySet",
    query_index="int",
    location="enums.ComputePassTimestampLocation",
)

#: * label :: str = None
#: * timestampWrites :: List[:obj:`structs.ComputePassTimestampWrite <ComputePassTimestampWrite>`] = []
ComputePassDescriptor = Struct(
    "ComputePassDescriptor",
    label="str",
    timestamp_writes="List[structs.ComputePassTimestampWrite]",
)

#: * querySet :: :class:`GPUQuerySet <wgpu.GPUQuerySet>`
#: * queryIndex :: int
#: * location :: :obj:`enums.RenderPassTimestampLocation <wgpu.enums.RenderPassTimestampLocation>`
RenderPassTimestampWrite = Struct(
    "RenderPassTimestampWrite",
    query_set="GPUQuerySet",
    query_index="int",
    location="enums.RenderPassTimestampLocation",
)

#: * label :: str = None
#: * colorAttachments :: List[:obj:`structs.RenderPassColorAttachment <RenderPassColorAttachment>`]
#: * depthStencilAttachment :: :obj:`structs.RenderPassDepthStencilAttachment <RenderPassDepthStencilAttachment>` = None
#: * occlusionQuerySet :: :class:`GPUQuerySet <wgpu.GPUQuerySet>` = None
#: * timestampWrites :: List[:obj:`structs.RenderPassTimestampWrite <RenderPassTimestampWrite>`] = []
#: * maxDrawCount :: int = 50000000
RenderPassDescriptor = Struct(
    "RenderPassDescriptor",
    label="str",
    color_attachments="List[structs.RenderPassColorAttachment]",
    depth_stencil_attachment="structs.RenderPassDepthStencilAttachment",
    occlusion_query_set="GPUQuerySet",
    timestamp_writes="List[structs.RenderPassTimestampWrite]",
    max_draw_count="int",
)

#: * view :: :class:`GPUTextureView <wgpu.GPUTextureView>`
#: * resolveTarget :: :class:`GPUTextureView <wgpu.GPUTextureView>` = None
#: * clearValue :: Union[List[float], :obj:`structs.Color <Color>`] = None
#: * loadOp :: :obj:`enums.LoadOp <wgpu.enums.LoadOp>`
#: * storeOp :: :obj:`enums.StoreOp <wgpu.enums.StoreOp>`
RenderPassColorAttachment = Struct(
    "RenderPassColorAttachment",
    view="GPUTextureView",
    resolve_target="GPUTextureView",
    clear_value="Union[List[float], structs.Color]",
    load_op="enums.LoadOp",
    store_op="enums.StoreOp",
)

#: * view :: :class:`GPUTextureView <wgpu.GPUTextureView>`
#: * depthClearValue :: float = 0
#: * depthLoadOp :: :obj:`enums.LoadOp <wgpu.enums.LoadOp>` = None
#: * depthStoreOp :: :obj:`enums.StoreOp <wgpu.enums.StoreOp>` = None
#: * depthReadOnly :: bool = false
#: * stencilClearValue :: int = 0
#: * stencilLoadOp :: :obj:`enums.LoadOp <wgpu.enums.LoadOp>` = None
#: * stencilStoreOp :: :obj:`enums.StoreOp <wgpu.enums.StoreOp>` = None
#: * stencilReadOnly :: bool = false
RenderPassDepthStencilAttachment = Struct(
    "RenderPassDepthStencilAttachment",
    view="GPUTextureView",
    depth_clear_value="float",
    depth_load_op="enums.LoadOp",
    depth_store_op="enums.StoreOp",
    depth_read_only="bool",
    stencil_clear_value="int",
    stencil_load_op="enums.LoadOp",
    stencil_store_op="enums.StoreOp",
    stencil_read_only="bool",
)

#: * label :: str = None
#: * colorFormats :: List[:obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`]
#: * depthStencilFormat :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>` = None
#: * sampleCount :: int = 1
RenderPassLayout = Struct(
    "RenderPassLayout",
    label="str",
    color_formats="List[enums.TextureFormat]",
    depth_stencil_format="enums.TextureFormat",
    sample_count="int",
)

#: * label :: str = None
RenderBundleDescriptor = Struct(
    "RenderBundleDescriptor",
    label="str",
)

#: * label :: str = None
#: * colorFormats :: List[:obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`]
#: * depthStencilFormat :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>` = None
#: * sampleCount :: int = 1
#: * depthReadOnly :: bool = false
#: * stencilReadOnly :: bool = false
RenderBundleEncoderDescriptor = Struct(
    "RenderBundleEncoderDescriptor",
    label="str",
    color_formats="List[enums.TextureFormat]",
    depth_stencil_format="enums.TextureFormat",
    sample_count="int",
    depth_read_only="bool",
    stencil_read_only="bool",
)

#: * label :: str = None
QueueDescriptor = Struct(
    "QueueDescriptor",
    label="str",
)

#: * label :: str = None
#: * type :: :obj:`enums.QueryType <wgpu.enums.QueryType>`
#: * count :: int
QuerySetDescriptor = Struct(
    "QuerySetDescriptor",
    label="str",
    type="enums.QueryType",
    count="int",
)

#: * device :: :class:`GPUDevice <wgpu.GPUDevice>`
#: * format :: :obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`
#: * usage :: :obj:`flags.TextureUsage <wgpu.flags.TextureUsage>` = 0x10
#: * viewFormats :: List[:obj:`enums.TextureFormat <wgpu.enums.TextureFormat>`] = []
#: * colorSpace :: str = "srgb"
#: * alphaMode :: :obj:`enums.CanvasAlphaMode <wgpu.enums.CanvasAlphaMode>` = "opaque"
CanvasConfiguration = Struct(
    "CanvasConfiguration",
    device="GPUDevice",
    format="enums.TextureFormat",
    usage="flags.TextureUsage",
    view_formats="List[enums.TextureFormat]",
    color_space="str",
    alpha_mode="enums.CanvasAlphaMode",
)

#: * error :: :class:`GPUError <wgpu.GPUError>`
UncapturedErrorEventInit = Struct(
    "UncapturedErrorEventInit",
    error="GPUError",
)

#: * r :: float
#: * g :: float
#: * b :: float
#: * a :: float
Color = Struct(
    "Color",
    r="float",
    g="float",
    b="float",
    a="float",
)

#: * x :: int = 0
#: * y :: int = 0
Origin2D = Struct(
    "Origin2D",
    x="int",
    y="int",
)

#: * x :: int = 0
#: * y :: int = 0
#: * z :: int = 0
Origin3D = Struct(
    "Origin3D",
    x="int",
    y="int",
    z="int",
)

#: * width :: int
#: * height :: int = 1
#: * depthOrArrayLayers :: int = 1
Extent3D = Struct(
    "Extent3D",
    width="int",
    height="int",
    depth_or_array_layers="int",
)
