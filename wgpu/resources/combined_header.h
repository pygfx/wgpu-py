// Cleaned version of webgpu.h ------------------------------------------------
#define WGPU_ARRAY_LAYER_COUNT_UNDEFINED 0xffffffff
#define WGPU_COPY_STRIDE_UNDEFINED 0xffffffff
#define WGPU_DEPTH_SLICE_UNDEFINED 0xffffffff
#define WGPU_LIMIT_U32_UNDEFINED 0xffffffff
#define WGPU_LIMIT_U64_UNDEFINED 0xffffffffffffffff
#define WGPU_MIP_LEVEL_COUNT_UNDEFINED 0xffffffff
#define WGPU_QUERY_SET_INDEX_UNDEFINED 0xffffffff
#define WGPU_STRLEN 0xffffffffffffffff
#define WGPU_WHOLE_MAP_SIZE 0xffffffffffffffff
#define WGPU_WHOLE_SIZE 0xffffffffffffffff
typedef struct WGPUStringView {
char const * data;
size_t length;
} WGPUStringView;
typedef uint64_t WGPUFlags;
typedef uint32_t WGPUBool;
typedef struct WGPUAdapterImpl* WGPUAdapter;
typedef struct WGPUBindGroupImpl* WGPUBindGroup;
typedef struct WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;
typedef struct WGPUBufferImpl* WGPUBuffer;
typedef struct WGPUCommandBufferImpl* WGPUCommandBuffer;
typedef struct WGPUCommandEncoderImpl* WGPUCommandEncoder;
typedef struct WGPUComputePassEncoderImpl* WGPUComputePassEncoder;
typedef struct WGPUComputePipelineImpl* WGPUComputePipeline;
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUExternalTextureImpl* WGPUExternalTexture;
typedef struct WGPUInstanceImpl* WGPUInstance;
typedef struct WGPUPipelineLayoutImpl* WGPUPipelineLayout;
typedef struct WGPUQuerySetImpl* WGPUQuerySet;
typedef struct WGPUQueueImpl* WGPUQueue;
typedef struct WGPURenderBundleImpl* WGPURenderBundle;
typedef struct WGPURenderBundleEncoderImpl* WGPURenderBundleEncoder;
typedef struct WGPURenderPassEncoderImpl* WGPURenderPassEncoder;
typedef struct WGPURenderPipelineImpl* WGPURenderPipeline;
typedef struct WGPUSamplerImpl* WGPUSampler;
typedef struct WGPUShaderModuleImpl* WGPUShaderModule;
typedef struct WGPUSurfaceImpl* WGPUSurface;
typedef struct WGPUTextureImpl* WGPUTexture;
typedef struct WGPUTextureViewImpl* WGPUTextureView;
struct WGPUAdapterInfo;
struct WGPUBlendComponent;
struct WGPUBufferBindingLayout;
struct WGPUBufferDescriptor;
struct WGPUColor;
struct WGPUCommandBufferDescriptor;
struct WGPUCommandEncoderDescriptor;
struct WGPUCompatibilityModeLimits;
struct WGPUCompilationMessage;
struct WGPUConstantEntry;
struct WGPUExtent3D;
struct WGPUExternalTextureBindingEntry;
struct WGPUExternalTextureBindingLayout;
struct WGPUFuture;
struct WGPUInstanceLimits;
struct WGPUMultisampleState;
struct WGPUOrigin3D;
struct WGPUPassTimestampWrites;
struct WGPUPipelineLayoutDescriptor;
struct WGPUPrimitiveState;
struct WGPUQuerySetDescriptor;
struct WGPUQueueDescriptor;
struct WGPURenderBundleDescriptor;
struct WGPURenderBundleEncoderDescriptor;
struct WGPURenderPassDepthStencilAttachment;
struct WGPURenderPassMaxDrawCount;
struct WGPURequestAdapterWebXROptions;
struct WGPUSamplerBindingLayout;
struct WGPUSamplerDescriptor;
struct WGPUShaderSourceSPIRV;
struct WGPUShaderSourceWGSL;
struct WGPUStencilFaceState;
struct WGPUStorageTextureBindingLayout;
struct WGPUSupportedFeatures;
struct WGPUSupportedInstanceFeatures;
struct WGPUSupportedWGSLLanguageFeatures;
struct WGPUSurfaceCapabilities;
struct WGPUSurfaceColorManagement;
struct WGPUSurfaceConfiguration;
struct WGPUSurfaceSourceAndroidNativeWindow;
struct WGPUSurfaceSourceMetalLayer;
struct WGPUSurfaceSourceWaylandSurface;
struct WGPUSurfaceSourceWindowsHWND;
struct WGPUSurfaceSourceXCBWindow;
struct WGPUSurfaceSourceXlibWindow;
struct WGPUSurfaceTexture;
struct WGPUTexelCopyBufferLayout;
struct WGPUTextureBindingLayout;
struct WGPUTextureBindingViewDimension;
struct WGPUTextureComponentSwizzle;
struct WGPUVertexAttribute;
struct WGPUBindGroupEntry;
struct WGPUBindGroupLayoutEntry;
struct WGPUBlendState;
struct WGPUCompilationInfo;
struct WGPUComputePassDescriptor;
struct WGPUComputeState;
struct WGPUDepthStencilState;
struct WGPUFutureWaitInfo;
struct WGPUInstanceDescriptor;
struct WGPULimits;
struct WGPURenderPassColorAttachment;
struct WGPURequestAdapterOptions;
struct WGPUShaderModuleDescriptor;
struct WGPUSurfaceDescriptor;
struct WGPUTexelCopyBufferInfo;
struct WGPUTexelCopyTextureInfo;
struct WGPUTextureComponentSwizzleDescriptor;
struct WGPUTextureDescriptor;
struct WGPUVertexBufferLayout;
struct WGPUBindGroupDescriptor;
struct WGPUBindGroupLayoutDescriptor;
struct WGPUColorTargetState;
struct WGPUComputePipelineDescriptor;
struct WGPUDeviceDescriptor;
struct WGPURenderPassDescriptor;
struct WGPUTextureViewDescriptor;
struct WGPUVertexState;
struct WGPUFragmentState;
struct WGPURenderPipelineDescriptor;
struct WGPUBufferMapCallbackInfo;
struct WGPUCompilationInfoCallbackInfo;
struct WGPUCreateComputePipelineAsyncCallbackInfo;
struct WGPUCreateRenderPipelineAsyncCallbackInfo;
struct WGPUDeviceLostCallbackInfo;
struct WGPUPopErrorScopeCallbackInfo;
struct WGPUQueueWorkDoneCallbackInfo;
struct WGPURequestAdapterCallbackInfo;
struct WGPURequestDeviceCallbackInfo;
struct WGPUUncapturedErrorCallbackInfo;
typedef enum WGPUAdapterType {
WGPUAdapterType_DiscreteGPU = 0x00000001,
WGPUAdapterType_IntegratedGPU = 0x00000002,
WGPUAdapterType_CPU = 0x00000003,
WGPUAdapterType_Unknown = 0x00000004,
WGPUAdapterType_Force32 = 0x7FFFFFFF
} WGPUAdapterType;
typedef enum WGPUAddressMode {
WGPUAddressMode_Undefined = 0x00000000,
WGPUAddressMode_ClampToEdge = 0x00000001,
WGPUAddressMode_Repeat = 0x00000002,
WGPUAddressMode_MirrorRepeat = 0x00000003,
WGPUAddressMode_Force32 = 0x7FFFFFFF
} WGPUAddressMode;
typedef enum WGPUBackendType {
WGPUBackendType_Undefined = 0x00000000,
WGPUBackendType_Null = 0x00000001,
WGPUBackendType_WebGPU = 0x00000002,
WGPUBackendType_D3D11 = 0x00000003,
WGPUBackendType_D3D12 = 0x00000004,
WGPUBackendType_Metal = 0x00000005,
WGPUBackendType_Vulkan = 0x00000006,
WGPUBackendType_OpenGL = 0x00000007,
WGPUBackendType_OpenGLES = 0x00000008,
WGPUBackendType_Force32 = 0x7FFFFFFF
} WGPUBackendType;
typedef enum WGPUBlendFactor {
WGPUBlendFactor_Undefined = 0x00000000,
WGPUBlendFactor_Zero = 0x00000001,
WGPUBlendFactor_One = 0x00000002,
WGPUBlendFactor_Src = 0x00000003,
WGPUBlendFactor_OneMinusSrc = 0x00000004,
WGPUBlendFactor_SrcAlpha = 0x00000005,
WGPUBlendFactor_OneMinusSrcAlpha = 0x00000006,
WGPUBlendFactor_Dst = 0x00000007,
WGPUBlendFactor_OneMinusDst = 0x00000008,
WGPUBlendFactor_DstAlpha = 0x00000009,
WGPUBlendFactor_OneMinusDstAlpha = 0x0000000A,
WGPUBlendFactor_SrcAlphaSaturated = 0x0000000B,
WGPUBlendFactor_Constant = 0x0000000C,
WGPUBlendFactor_OneMinusConstant = 0x0000000D,
WGPUBlendFactor_Src1 = 0x0000000E,
WGPUBlendFactor_OneMinusSrc1 = 0x0000000F,
WGPUBlendFactor_Src1Alpha = 0x00000010,
WGPUBlendFactor_OneMinusSrc1Alpha = 0x00000011,
WGPUBlendFactor_Force32 = 0x7FFFFFFF
} WGPUBlendFactor;
typedef enum WGPUBlendOperation {
WGPUBlendOperation_Undefined = 0x00000000,
WGPUBlendOperation_Add = 0x00000001,
WGPUBlendOperation_Subtract = 0x00000002,
WGPUBlendOperation_ReverseSubtract = 0x00000003,
WGPUBlendOperation_Min = 0x00000004,
WGPUBlendOperation_Max = 0x00000005,
WGPUBlendOperation_Force32 = 0x7FFFFFFF
} WGPUBlendOperation;
typedef enum WGPUBufferBindingType {
WGPUBufferBindingType_BindingNotUsed = 0x00000000,
WGPUBufferBindingType_Undefined = 0x00000001,
WGPUBufferBindingType_Uniform = 0x00000002,
WGPUBufferBindingType_Storage = 0x00000003,
WGPUBufferBindingType_ReadOnlyStorage = 0x00000004,
WGPUBufferBindingType_Force32 = 0x7FFFFFFF
} WGPUBufferBindingType;
typedef enum WGPUBufferMapState {
WGPUBufferMapState_Unmapped = 0x00000001,
WGPUBufferMapState_Pending = 0x00000002,
WGPUBufferMapState_Mapped = 0x00000003,
WGPUBufferMapState_Force32 = 0x7FFFFFFF
} WGPUBufferMapState;
typedef enum WGPUCallbackMode {
WGPUCallbackMode_WaitAnyOnly = 0x00000001,
WGPUCallbackMode_AllowProcessEvents = 0x00000002,
WGPUCallbackMode_AllowSpontaneous = 0x00000003,
WGPUCallbackMode_Force32 = 0x7FFFFFFF
} WGPUCallbackMode;
typedef enum WGPUCompareFunction {
WGPUCompareFunction_Undefined = 0x00000000,
WGPUCompareFunction_Never = 0x00000001,
WGPUCompareFunction_Less = 0x00000002,
WGPUCompareFunction_Equal = 0x00000003,
WGPUCompareFunction_LessEqual = 0x00000004,
WGPUCompareFunction_Greater = 0x00000005,
WGPUCompareFunction_NotEqual = 0x00000006,
WGPUCompareFunction_GreaterEqual = 0x00000007,
WGPUCompareFunction_Always = 0x00000008,
WGPUCompareFunction_Force32 = 0x7FFFFFFF
} WGPUCompareFunction;
typedef enum WGPUCompilationInfoRequestStatus {
WGPUCompilationInfoRequestStatus_Success = 0x00000001,
WGPUCompilationInfoRequestStatus_CallbackCancelled = 0x00000002,
WGPUCompilationInfoRequestStatus_Force32 = 0x7FFFFFFF
} WGPUCompilationInfoRequestStatus;
typedef enum WGPUCompilationMessageType {
WGPUCompilationMessageType_Error = 0x00000001,
WGPUCompilationMessageType_Warning = 0x00000002,
WGPUCompilationMessageType_Info = 0x00000003,
WGPUCompilationMessageType_Force32 = 0x7FFFFFFF
} WGPUCompilationMessageType;
typedef enum WGPUComponentSwizzle {
WGPUComponentSwizzle_Undefined = 0x00000000,
WGPUComponentSwizzle_Zero = 0x00000001,
WGPUComponentSwizzle_One = 0x00000002,
WGPUComponentSwizzle_R = 0x00000003,
WGPUComponentSwizzle_G = 0x00000004,
WGPUComponentSwizzle_B = 0x00000005,
WGPUComponentSwizzle_A = 0x00000006,
WGPUComponentSwizzle_Force32 = 0x7FFFFFFF
} WGPUComponentSwizzle;
typedef enum WGPUCompositeAlphaMode {
WGPUCompositeAlphaMode_Auto = 0x00000000,
WGPUCompositeAlphaMode_Opaque = 0x00000001,
WGPUCompositeAlphaMode_Premultiplied = 0x00000002,
WGPUCompositeAlphaMode_Unpremultiplied = 0x00000003,
WGPUCompositeAlphaMode_Inherit = 0x00000004,
WGPUCompositeAlphaMode_Force32 = 0x7FFFFFFF
} WGPUCompositeAlphaMode;
typedef enum WGPUCreatePipelineAsyncStatus {
WGPUCreatePipelineAsyncStatus_Success = 0x00000001,
WGPUCreatePipelineAsyncStatus_CallbackCancelled = 0x00000002,
WGPUCreatePipelineAsyncStatus_ValidationError = 0x00000003,
WGPUCreatePipelineAsyncStatus_InternalError = 0x00000004,
WGPUCreatePipelineAsyncStatus_Force32 = 0x7FFFFFFF
} WGPUCreatePipelineAsyncStatus;
typedef enum WGPUCullMode {
WGPUCullMode_Undefined = 0x00000000,
WGPUCullMode_None = 0x00000001,
WGPUCullMode_Front = 0x00000002,
WGPUCullMode_Back = 0x00000003,
WGPUCullMode_Force32 = 0x7FFFFFFF
} WGPUCullMode;
typedef enum WGPUDeviceLostReason {
WGPUDeviceLostReason_Unknown = 0x00000001,
WGPUDeviceLostReason_Destroyed = 0x00000002,
WGPUDeviceLostReason_CallbackCancelled = 0x00000003,
WGPUDeviceLostReason_FailedCreation = 0x00000004,
WGPUDeviceLostReason_Force32 = 0x7FFFFFFF
} WGPUDeviceLostReason;
typedef enum WGPUErrorFilter {
WGPUErrorFilter_Validation = 0x00000001,
WGPUErrorFilter_OutOfMemory = 0x00000002,
WGPUErrorFilter_Internal = 0x00000003,
WGPUErrorFilter_Force32 = 0x7FFFFFFF
} WGPUErrorFilter;
typedef enum WGPUErrorType {
WGPUErrorType_NoError = 0x00000001,
WGPUErrorType_Validation = 0x00000002,
WGPUErrorType_OutOfMemory = 0x00000003,
WGPUErrorType_Internal = 0x00000004,
WGPUErrorType_Unknown = 0x00000005,
WGPUErrorType_Force32 = 0x7FFFFFFF
} WGPUErrorType;
typedef enum WGPUFeatureLevel {
WGPUFeatureLevel_Undefined = 0x00000000,
WGPUFeatureLevel_Compatibility = 0x00000001,
WGPUFeatureLevel_Core = 0x00000002,
WGPUFeatureLevel_Force32 = 0x7FFFFFFF
} WGPUFeatureLevel;
typedef enum WGPUFeatureName {
WGPUFeatureName_CoreFeaturesAndLimits = 0x00000001,
WGPUFeatureName_DepthClipControl = 0x00000002,
WGPUFeatureName_Depth32FloatStencil8 = 0x00000003,
WGPUFeatureName_TextureCompressionBC = 0x00000004,
WGPUFeatureName_TextureCompressionBCSliced3D = 0x00000005,
WGPUFeatureName_TextureCompressionETC2 = 0x00000006,
WGPUFeatureName_TextureCompressionASTC = 0x00000007,
WGPUFeatureName_TextureCompressionASTCSliced3D = 0x00000008,
WGPUFeatureName_TimestampQuery = 0x00000009,
WGPUFeatureName_IndirectFirstInstance = 0x0000000A,
WGPUFeatureName_ShaderF16 = 0x0000000B,
WGPUFeatureName_RG11B10UfloatRenderable = 0x0000000C,
WGPUFeatureName_BGRA8UnormStorage = 0x0000000D,
WGPUFeatureName_Float32Filterable = 0x0000000E,
WGPUFeatureName_Float32Blendable = 0x0000000F,
WGPUFeatureName_ClipDistances = 0x00000010,
WGPUFeatureName_DualSourceBlending = 0x00000011,
WGPUFeatureName_Subgroups = 0x00000012,
WGPUFeatureName_TextureFormatsTier1 = 0x00000013,
WGPUFeatureName_TextureFormatsTier2 = 0x00000014,
WGPUFeatureName_PrimitiveIndex = 0x00000015,
WGPUFeatureName_TextureComponentSwizzle = 0x00000016,
WGPUFeatureName_Force32 = 0x7FFFFFFF
} WGPUFeatureName;
typedef enum WGPUFilterMode {
WGPUFilterMode_Undefined = 0x00000000,
WGPUFilterMode_Nearest = 0x00000001,
WGPUFilterMode_Linear = 0x00000002,
WGPUFilterMode_Force32 = 0x7FFFFFFF
} WGPUFilterMode;
typedef enum WGPUFrontFace {
WGPUFrontFace_Undefined = 0x00000000,
WGPUFrontFace_CCW = 0x00000001,
WGPUFrontFace_CW = 0x00000002,
WGPUFrontFace_Force32 = 0x7FFFFFFF
} WGPUFrontFace;
typedef enum WGPUIndexFormat {
WGPUIndexFormat_Undefined = 0x00000000,
WGPUIndexFormat_Uint16 = 0x00000001,
WGPUIndexFormat_Uint32 = 0x00000002,
WGPUIndexFormat_Force32 = 0x7FFFFFFF
} WGPUIndexFormat;
typedef enum WGPUInstanceFeatureName {
WGPUInstanceFeatureName_TimedWaitAny = 0x00000001,
WGPUInstanceFeatureName_ShaderSourceSPIRV = 0x00000002,
WGPUInstanceFeatureName_MultipleDevicesPerAdapter = 0x00000003,
WGPUInstanceFeatureName_Force32 = 0x7FFFFFFF
} WGPUInstanceFeatureName;
typedef enum WGPULoadOp {
WGPULoadOp_Undefined = 0x00000000,
WGPULoadOp_Load = 0x00000001,
WGPULoadOp_Clear = 0x00000002,
WGPULoadOp_Force32 = 0x7FFFFFFF
} WGPULoadOp;
typedef enum WGPUMapAsyncStatus {
WGPUMapAsyncStatus_Success = 0x00000001,
WGPUMapAsyncStatus_CallbackCancelled = 0x00000002,
WGPUMapAsyncStatus_Error = 0x00000003,
WGPUMapAsyncStatus_Aborted = 0x00000004,
WGPUMapAsyncStatus_Force32 = 0x7FFFFFFF
} WGPUMapAsyncStatus;
typedef enum WGPUMipmapFilterMode {
WGPUMipmapFilterMode_Undefined = 0x00000000,
WGPUMipmapFilterMode_Nearest = 0x00000001,
WGPUMipmapFilterMode_Linear = 0x00000002,
WGPUMipmapFilterMode_Force32 = 0x7FFFFFFF
} WGPUMipmapFilterMode;
typedef enum WGPUOptionalBool {
WGPUOptionalBool_False = 0x00000000,
WGPUOptionalBool_True = 0x00000001,
WGPUOptionalBool_Undefined = 0x00000002,
WGPUOptionalBool_Force32 = 0x7FFFFFFF
} WGPUOptionalBool;
typedef enum WGPUPopErrorScopeStatus {
WGPUPopErrorScopeStatus_Success = 0x00000001,
WGPUPopErrorScopeStatus_CallbackCancelled = 0x00000002,
WGPUPopErrorScopeStatus_Error = 0x00000003,
WGPUPopErrorScopeStatus_Force32 = 0x7FFFFFFF
} WGPUPopErrorScopeStatus;
typedef enum WGPUPowerPreference {
WGPUPowerPreference_Undefined = 0x00000000,
WGPUPowerPreference_LowPower = 0x00000001,
WGPUPowerPreference_HighPerformance = 0x00000002,
WGPUPowerPreference_Force32 = 0x7FFFFFFF
} WGPUPowerPreference;
typedef enum WGPUPredefinedColorSpace {
WGPUPredefinedColorSpace_SRGB = 0x00000001,
WGPUPredefinedColorSpace_DisplayP3 = 0x00000002,
WGPUPredefinedColorSpace_Force32 = 0x7FFFFFFF
} WGPUPredefinedColorSpace;
typedef enum WGPUPresentMode {
WGPUPresentMode_Undefined = 0x00000000,
WGPUPresentMode_Fifo = 0x00000001,
WGPUPresentMode_FifoRelaxed = 0x00000002,
WGPUPresentMode_Immediate = 0x00000003,
WGPUPresentMode_Mailbox = 0x00000004,
WGPUPresentMode_Force32 = 0x7FFFFFFF
} WGPUPresentMode;
typedef enum WGPUPrimitiveTopology {
WGPUPrimitiveTopology_Undefined = 0x00000000,
WGPUPrimitiveTopology_PointList = 0x00000001,
WGPUPrimitiveTopology_LineList = 0x00000002,
WGPUPrimitiveTopology_LineStrip = 0x00000003,
WGPUPrimitiveTopology_TriangleList = 0x00000004,
WGPUPrimitiveTopology_TriangleStrip = 0x00000005,
WGPUPrimitiveTopology_Force32 = 0x7FFFFFFF
} WGPUPrimitiveTopology;
typedef enum WGPUQueryType {
WGPUQueryType_Occlusion = 0x00000001,
WGPUQueryType_Timestamp = 0x00000002,
WGPUQueryType_Force32 = 0x7FFFFFFF
} WGPUQueryType;
typedef enum WGPUQueueWorkDoneStatus {
WGPUQueueWorkDoneStatus_Success = 0x00000001,
WGPUQueueWorkDoneStatus_CallbackCancelled = 0x00000002,
WGPUQueueWorkDoneStatus_Error = 0x00000003,
WGPUQueueWorkDoneStatus_Force32 = 0x7FFFFFFF
} WGPUQueueWorkDoneStatus;
typedef enum WGPURequestAdapterStatus {
WGPURequestAdapterStatus_Success = 0x00000001,
WGPURequestAdapterStatus_CallbackCancelled = 0x00000002,
WGPURequestAdapterStatus_Unavailable = 0x00000003,
WGPURequestAdapterStatus_Error = 0x00000004,
WGPURequestAdapterStatus_Force32 = 0x7FFFFFFF
} WGPURequestAdapterStatus;
typedef enum WGPURequestDeviceStatus {
WGPURequestDeviceStatus_Success = 0x00000001,
WGPURequestDeviceStatus_CallbackCancelled = 0x00000002,
WGPURequestDeviceStatus_Error = 0x00000003,
WGPURequestDeviceStatus_Force32 = 0x7FFFFFFF
} WGPURequestDeviceStatus;
typedef enum WGPUSamplerBindingType {
WGPUSamplerBindingType_BindingNotUsed = 0x00000000,
WGPUSamplerBindingType_Undefined = 0x00000001,
WGPUSamplerBindingType_Filtering = 0x00000002,
WGPUSamplerBindingType_NonFiltering = 0x00000003,
WGPUSamplerBindingType_Comparison = 0x00000004,
WGPUSamplerBindingType_Force32 = 0x7FFFFFFF
} WGPUSamplerBindingType;
typedef enum WGPUStatus {
WGPUStatus_Success = 0x00000001,
WGPUStatus_Error = 0x00000002,
WGPUStatus_Force32 = 0x7FFFFFFF
} WGPUStatus;
typedef enum WGPUStencilOperation {
WGPUStencilOperation_Undefined = 0x00000000,
WGPUStencilOperation_Keep = 0x00000001,
WGPUStencilOperation_Zero = 0x00000002,
WGPUStencilOperation_Replace = 0x00000003,
WGPUStencilOperation_Invert = 0x00000004,
WGPUStencilOperation_IncrementClamp = 0x00000005,
WGPUStencilOperation_DecrementClamp = 0x00000006,
WGPUStencilOperation_IncrementWrap = 0x00000007,
WGPUStencilOperation_DecrementWrap = 0x00000008,
WGPUStencilOperation_Force32 = 0x7FFFFFFF
} WGPUStencilOperation;
typedef enum WGPUStorageTextureAccess {
WGPUStorageTextureAccess_BindingNotUsed = 0x00000000,
WGPUStorageTextureAccess_Undefined = 0x00000001,
WGPUStorageTextureAccess_WriteOnly = 0x00000002,
WGPUStorageTextureAccess_ReadOnly = 0x00000003,
WGPUStorageTextureAccess_ReadWrite = 0x00000004,
WGPUStorageTextureAccess_Force32 = 0x7FFFFFFF
} WGPUStorageTextureAccess;
typedef enum WGPUStoreOp {
WGPUStoreOp_Undefined = 0x00000000,
WGPUStoreOp_Store = 0x00000001,
WGPUStoreOp_Discard = 0x00000002,
WGPUStoreOp_Force32 = 0x7FFFFFFF
} WGPUStoreOp;
typedef enum WGPUSType {
WGPUSType_ShaderSourceSPIRV = 0x00000001,
WGPUSType_ShaderSourceWGSL = 0x00000002,
WGPUSType_RenderPassMaxDrawCount = 0x00000003,
WGPUSType_SurfaceSourceMetalLayer = 0x00000004,
WGPUSType_SurfaceSourceWindowsHWND = 0x00000005,
WGPUSType_SurfaceSourceXlibWindow = 0x00000006,
WGPUSType_SurfaceSourceWaylandSurface = 0x00000007,
WGPUSType_SurfaceSourceAndroidNativeWindow = 0x00000008,
WGPUSType_SurfaceSourceXCBWindow = 0x00000009,
WGPUSType_SurfaceColorManagement = 0x0000000A,
WGPUSType_RequestAdapterWebXROptions = 0x0000000B,
WGPUSType_TextureComponentSwizzleDescriptor = 0x0000000C,
WGPUSType_ExternalTextureBindingLayout = 0x0000000D,
WGPUSType_ExternalTextureBindingEntry = 0x0000000E,
WGPUSType_CompatibilityModeLimits = 0x0000000F,
WGPUSType_TextureBindingViewDimension = 0x00000010,
WGPUSType_Force32 = 0x7FFFFFFF
} WGPUSType;
typedef enum WGPUSurfaceGetCurrentTextureStatus {
WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal = 0x00000001,
WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal = 0x00000002,
WGPUSurfaceGetCurrentTextureStatus_Timeout = 0x00000003,
WGPUSurfaceGetCurrentTextureStatus_Outdated = 0x00000004,
WGPUSurfaceGetCurrentTextureStatus_Lost = 0x00000005,
WGPUSurfaceGetCurrentTextureStatus_Error = 0x00000006,
WGPUSurfaceGetCurrentTextureStatus_Force32 = 0x7FFFFFFF
} WGPUSurfaceGetCurrentTextureStatus;
typedef enum WGPUTextureAspect {
WGPUTextureAspect_Undefined = 0x00000000,
WGPUTextureAspect_All = 0x00000001,
WGPUTextureAspect_StencilOnly = 0x00000002,
WGPUTextureAspect_DepthOnly = 0x00000003,
WGPUTextureAspect_Force32 = 0x7FFFFFFF
} WGPUTextureAspect;
typedef enum WGPUTextureDimension {
WGPUTextureDimension_Undefined = 0x00000000,
WGPUTextureDimension_1D = 0x00000001,
WGPUTextureDimension_2D = 0x00000002,
WGPUTextureDimension_3D = 0x00000003,
WGPUTextureDimension_Force32 = 0x7FFFFFFF
} WGPUTextureDimension;
typedef enum WGPUTextureFormat {
WGPUTextureFormat_Undefined = 0x00000000,
WGPUTextureFormat_R8Unorm = 0x00000001,
WGPUTextureFormat_R8Snorm = 0x00000002,
WGPUTextureFormat_R8Uint = 0x00000003,
WGPUTextureFormat_R8Sint = 0x00000004,
WGPUTextureFormat_R16Unorm = 0x00000005,
WGPUTextureFormat_R16Snorm = 0x00000006,
WGPUTextureFormat_R16Uint = 0x00000007,
WGPUTextureFormat_R16Sint = 0x00000008,
WGPUTextureFormat_R16Float = 0x00000009,
WGPUTextureFormat_RG8Unorm = 0x0000000A,
WGPUTextureFormat_RG8Snorm = 0x0000000B,
WGPUTextureFormat_RG8Uint = 0x0000000C,
WGPUTextureFormat_RG8Sint = 0x0000000D,
WGPUTextureFormat_R32Float = 0x0000000E,
WGPUTextureFormat_R32Uint = 0x0000000F,
WGPUTextureFormat_R32Sint = 0x00000010,
WGPUTextureFormat_RG16Unorm = 0x00000011,
WGPUTextureFormat_RG16Snorm = 0x00000012,
WGPUTextureFormat_RG16Uint = 0x00000013,
WGPUTextureFormat_RG16Sint = 0x00000014,
WGPUTextureFormat_RG16Float = 0x00000015,
WGPUTextureFormat_RGBA8Unorm = 0x00000016,
WGPUTextureFormat_RGBA8UnormSrgb = 0x00000017,
WGPUTextureFormat_RGBA8Snorm = 0x00000018,
WGPUTextureFormat_RGBA8Uint = 0x00000019,
WGPUTextureFormat_RGBA8Sint = 0x0000001A,
WGPUTextureFormat_BGRA8Unorm = 0x0000001B,
WGPUTextureFormat_BGRA8UnormSrgb = 0x0000001C,
WGPUTextureFormat_RGB10A2Uint = 0x0000001D,
WGPUTextureFormat_RGB10A2Unorm = 0x0000001E,
WGPUTextureFormat_RG11B10Ufloat = 0x0000001F,
WGPUTextureFormat_RGB9E5Ufloat = 0x00000020,
WGPUTextureFormat_RG32Float = 0x00000021,
WGPUTextureFormat_RG32Uint = 0x00000022,
WGPUTextureFormat_RG32Sint = 0x00000023,
WGPUTextureFormat_RGBA16Unorm = 0x00000024,
WGPUTextureFormat_RGBA16Snorm = 0x00000025,
WGPUTextureFormat_RGBA16Uint = 0x00000026,
WGPUTextureFormat_RGBA16Sint = 0x00000027,
WGPUTextureFormat_RGBA16Float = 0x00000028,
WGPUTextureFormat_RGBA32Float = 0x00000029,
WGPUTextureFormat_RGBA32Uint = 0x0000002A,
WGPUTextureFormat_RGBA32Sint = 0x0000002B,
WGPUTextureFormat_Stencil8 = 0x0000002C,
WGPUTextureFormat_Depth16Unorm = 0x0000002D,
WGPUTextureFormat_Depth24Plus = 0x0000002E,
WGPUTextureFormat_Depth24PlusStencil8 = 0x0000002F,
WGPUTextureFormat_Depth32Float = 0x00000030,
WGPUTextureFormat_Depth32FloatStencil8 = 0x00000031,
WGPUTextureFormat_BC1RGBAUnorm = 0x00000032,
WGPUTextureFormat_BC1RGBAUnormSrgb = 0x00000033,
WGPUTextureFormat_BC2RGBAUnorm = 0x00000034,
WGPUTextureFormat_BC2RGBAUnormSrgb = 0x00000035,
WGPUTextureFormat_BC3RGBAUnorm = 0x00000036,
WGPUTextureFormat_BC3RGBAUnormSrgb = 0x00000037,
WGPUTextureFormat_BC4RUnorm = 0x00000038,
WGPUTextureFormat_BC4RSnorm = 0x00000039,
WGPUTextureFormat_BC5RGUnorm = 0x0000003A,
WGPUTextureFormat_BC5RGSnorm = 0x0000003B,
WGPUTextureFormat_BC6HRGBUfloat = 0x0000003C,
WGPUTextureFormat_BC6HRGBFloat = 0x0000003D,
WGPUTextureFormat_BC7RGBAUnorm = 0x0000003E,
WGPUTextureFormat_BC7RGBAUnormSrgb = 0x0000003F,
WGPUTextureFormat_ETC2RGB8Unorm = 0x00000040,
WGPUTextureFormat_ETC2RGB8UnormSrgb = 0x00000041,
WGPUTextureFormat_ETC2RGB8A1Unorm = 0x00000042,
WGPUTextureFormat_ETC2RGB8A1UnormSrgb = 0x00000043,
WGPUTextureFormat_ETC2RGBA8Unorm = 0x00000044,
WGPUTextureFormat_ETC2RGBA8UnormSrgb = 0x00000045,
WGPUTextureFormat_EACR11Unorm = 0x00000046,
WGPUTextureFormat_EACR11Snorm = 0x00000047,
WGPUTextureFormat_EACRG11Unorm = 0x00000048,
WGPUTextureFormat_EACRG11Snorm = 0x00000049,
WGPUTextureFormat_ASTC4x4Unorm = 0x0000004A,
WGPUTextureFormat_ASTC4x4UnormSrgb = 0x0000004B,
WGPUTextureFormat_ASTC5x4Unorm = 0x0000004C,
WGPUTextureFormat_ASTC5x4UnormSrgb = 0x0000004D,
WGPUTextureFormat_ASTC5x5Unorm = 0x0000004E,
WGPUTextureFormat_ASTC5x5UnormSrgb = 0x0000004F,
WGPUTextureFormat_ASTC6x5Unorm = 0x00000050,
WGPUTextureFormat_ASTC6x5UnormSrgb = 0x00000051,
WGPUTextureFormat_ASTC6x6Unorm = 0x00000052,
WGPUTextureFormat_ASTC6x6UnormSrgb = 0x00000053,
WGPUTextureFormat_ASTC8x5Unorm = 0x00000054,
WGPUTextureFormat_ASTC8x5UnormSrgb = 0x00000055,
WGPUTextureFormat_ASTC8x6Unorm = 0x00000056,
WGPUTextureFormat_ASTC8x6UnormSrgb = 0x00000057,
WGPUTextureFormat_ASTC8x8Unorm = 0x00000058,
WGPUTextureFormat_ASTC8x8UnormSrgb = 0x00000059,
WGPUTextureFormat_ASTC10x5Unorm = 0x0000005A,
WGPUTextureFormat_ASTC10x5UnormSrgb = 0x0000005B,
WGPUTextureFormat_ASTC10x6Unorm = 0x0000005C,
WGPUTextureFormat_ASTC10x6UnormSrgb = 0x0000005D,
WGPUTextureFormat_ASTC10x8Unorm = 0x0000005E,
WGPUTextureFormat_ASTC10x8UnormSrgb = 0x0000005F,
WGPUTextureFormat_ASTC10x10Unorm = 0x00000060,
WGPUTextureFormat_ASTC10x10UnormSrgb = 0x00000061,
WGPUTextureFormat_ASTC12x10Unorm = 0x00000062,
WGPUTextureFormat_ASTC12x10UnormSrgb = 0x00000063,
WGPUTextureFormat_ASTC12x12Unorm = 0x00000064,
WGPUTextureFormat_ASTC12x12UnormSrgb = 0x00000065,
WGPUTextureFormat_Force32 = 0x7FFFFFFF
} WGPUTextureFormat;
typedef enum WGPUTextureSampleType {
WGPUTextureSampleType_BindingNotUsed = 0x00000000,
WGPUTextureSampleType_Undefined = 0x00000001,
WGPUTextureSampleType_Float = 0x00000002,
WGPUTextureSampleType_UnfilterableFloat = 0x00000003,
WGPUTextureSampleType_Depth = 0x00000004,
WGPUTextureSampleType_Sint = 0x00000005,
WGPUTextureSampleType_Uint = 0x00000006,
WGPUTextureSampleType_Force32 = 0x7FFFFFFF
} WGPUTextureSampleType;
typedef enum WGPUTextureViewDimension {
WGPUTextureViewDimension_Undefined = 0x00000000,
WGPUTextureViewDimension_1D = 0x00000001,
WGPUTextureViewDimension_2D = 0x00000002,
WGPUTextureViewDimension_2DArray = 0x00000003,
WGPUTextureViewDimension_Cube = 0x00000004,
WGPUTextureViewDimension_CubeArray = 0x00000005,
WGPUTextureViewDimension_3D = 0x00000006,
WGPUTextureViewDimension_Force32 = 0x7FFFFFFF
} WGPUTextureViewDimension;
typedef enum WGPUToneMappingMode {
WGPUToneMappingMode_Standard = 0x00000001,
WGPUToneMappingMode_Extended = 0x00000002,
WGPUToneMappingMode_Force32 = 0x7FFFFFFF
} WGPUToneMappingMode;
typedef enum WGPUVertexFormat {
WGPUVertexFormat_Uint8 = 0x00000001,
WGPUVertexFormat_Uint8x2 = 0x00000002,
WGPUVertexFormat_Uint8x4 = 0x00000003,
WGPUVertexFormat_Sint8 = 0x00000004,
WGPUVertexFormat_Sint8x2 = 0x00000005,
WGPUVertexFormat_Sint8x4 = 0x00000006,
WGPUVertexFormat_Unorm8 = 0x00000007,
WGPUVertexFormat_Unorm8x2 = 0x00000008,
WGPUVertexFormat_Unorm8x4 = 0x00000009,
WGPUVertexFormat_Snorm8 = 0x0000000A,
WGPUVertexFormat_Snorm8x2 = 0x0000000B,
WGPUVertexFormat_Snorm8x4 = 0x0000000C,
WGPUVertexFormat_Uint16 = 0x0000000D,
WGPUVertexFormat_Uint16x2 = 0x0000000E,
WGPUVertexFormat_Uint16x4 = 0x0000000F,
WGPUVertexFormat_Sint16 = 0x00000010,
WGPUVertexFormat_Sint16x2 = 0x00000011,
WGPUVertexFormat_Sint16x4 = 0x00000012,
WGPUVertexFormat_Unorm16 = 0x00000013,
WGPUVertexFormat_Unorm16x2 = 0x00000014,
WGPUVertexFormat_Unorm16x4 = 0x00000015,
WGPUVertexFormat_Snorm16 = 0x00000016,
WGPUVertexFormat_Snorm16x2 = 0x00000017,
WGPUVertexFormat_Snorm16x4 = 0x00000018,
WGPUVertexFormat_Float16 = 0x00000019,
WGPUVertexFormat_Float16x2 = 0x0000001A,
WGPUVertexFormat_Float16x4 = 0x0000001B,
WGPUVertexFormat_Float32 = 0x0000001C,
WGPUVertexFormat_Float32x2 = 0x0000001D,
WGPUVertexFormat_Float32x3 = 0x0000001E,
WGPUVertexFormat_Float32x4 = 0x0000001F,
WGPUVertexFormat_Uint32 = 0x00000020,
WGPUVertexFormat_Uint32x2 = 0x00000021,
WGPUVertexFormat_Uint32x3 = 0x00000022,
WGPUVertexFormat_Uint32x4 = 0x00000023,
WGPUVertexFormat_Sint32 = 0x00000024,
WGPUVertexFormat_Sint32x2 = 0x00000025,
WGPUVertexFormat_Sint32x3 = 0x00000026,
WGPUVertexFormat_Sint32x4 = 0x00000027,
WGPUVertexFormat_Unorm10_10_10_2 = 0x00000028,
WGPUVertexFormat_Unorm8x4BGRA = 0x00000029,
WGPUVertexFormat_Force32 = 0x7FFFFFFF
} WGPUVertexFormat;
typedef enum WGPUVertexStepMode {
WGPUVertexStepMode_Undefined = 0x00000000,
WGPUVertexStepMode_Vertex = 0x00000001,
WGPUVertexStepMode_Instance = 0x00000002,
WGPUVertexStepMode_Force32 = 0x7FFFFFFF
} WGPUVertexStepMode;
typedef enum WGPUWaitStatus {
WGPUWaitStatus_Success = 0x00000001,
WGPUWaitStatus_TimedOut = 0x00000002,
WGPUWaitStatus_Error = 0x00000003,
WGPUWaitStatus_Force32 = 0x7FFFFFFF
} WGPUWaitStatus;
typedef enum WGPUWGSLLanguageFeatureName {
WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures = 0x00000001,
WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct = 0x00000002,
WGPUWGSLLanguageFeatureName_UnrestrictedPointerParameters = 0x00000003,
WGPUWGSLLanguageFeatureName_PointerCompositeAccess = 0x00000004,
WGPUWGSLLanguageFeatureName_UniformBufferStandardLayout = 0x00000005,
WGPUWGSLLanguageFeatureName_SubgroupId = 0x00000006,
WGPUWGSLLanguageFeatureName_TextureAndSamplerLet = 0x00000007,
WGPUWGSLLanguageFeatureName_SubgroupUniformity = 0x00000008,
WGPUWGSLLanguageFeatureName_TextureFormatsTier1 = 0x00000009,
WGPUWGSLLanguageFeatureName_Force32 = 0x7FFFFFFF
} WGPUWGSLLanguageFeatureName;
typedef WGPUFlags WGPUBufferUsage;
static const WGPUBufferUsage WGPUBufferUsage_None = 0x0000000000000000;
static const WGPUBufferUsage WGPUBufferUsage_MapRead = 0x0000000000000001;
static const WGPUBufferUsage WGPUBufferUsage_MapWrite = 0x0000000000000002;
static const WGPUBufferUsage WGPUBufferUsage_CopySrc = 0x0000000000000004;
static const WGPUBufferUsage WGPUBufferUsage_CopyDst = 0x0000000000000008;
static const WGPUBufferUsage WGPUBufferUsage_Index = 0x0000000000000010;
static const WGPUBufferUsage WGPUBufferUsage_Vertex = 0x0000000000000020;
static const WGPUBufferUsage WGPUBufferUsage_Uniform = 0x0000000000000040;
static const WGPUBufferUsage WGPUBufferUsage_Storage = 0x0000000000000080;
static const WGPUBufferUsage WGPUBufferUsage_Indirect = 0x0000000000000100;
static const WGPUBufferUsage WGPUBufferUsage_QueryResolve = 0x0000000000000200;
typedef WGPUFlags WGPUColorWriteMask;
static const WGPUColorWriteMask WGPUColorWriteMask_None = 0x0000000000000000;
static const WGPUColorWriteMask WGPUColorWriteMask_Red = 0x0000000000000001;
static const WGPUColorWriteMask WGPUColorWriteMask_Green = 0x0000000000000002;
static const WGPUColorWriteMask WGPUColorWriteMask_Blue = 0x0000000000000004;
static const WGPUColorWriteMask WGPUColorWriteMask_Alpha = 0x0000000000000008;
static const WGPUColorWriteMask WGPUColorWriteMask_All = 0x000000000000000F;
typedef WGPUFlags WGPUMapMode;
static const WGPUMapMode WGPUMapMode_None = 0x0000000000000000;
static const WGPUMapMode WGPUMapMode_Read = 0x0000000000000001;
static const WGPUMapMode WGPUMapMode_Write = 0x0000000000000002;
typedef WGPUFlags WGPUShaderStage;
static const WGPUShaderStage WGPUShaderStage_None = 0x0000000000000000;
static const WGPUShaderStage WGPUShaderStage_Vertex = 0x0000000000000001;
static const WGPUShaderStage WGPUShaderStage_Fragment = 0x0000000000000002;
static const WGPUShaderStage WGPUShaderStage_Compute = 0x0000000000000004;
typedef WGPUFlags WGPUTextureUsage;
static const WGPUTextureUsage WGPUTextureUsage_None = 0x0000000000000000;
static const WGPUTextureUsage WGPUTextureUsage_CopySrc = 0x0000000000000001;
static const WGPUTextureUsage WGPUTextureUsage_CopyDst = 0x0000000000000002;
static const WGPUTextureUsage WGPUTextureUsage_TextureBinding = 0x0000000000000004;
static const WGPUTextureUsage WGPUTextureUsage_StorageBinding = 0x0000000000000008;
static const WGPUTextureUsage WGPUTextureUsage_RenderAttachment = 0x0000000000000010;
static const WGPUTextureUsage WGPUTextureUsage_TransientAttachment = 0x0000000000000020;
typedef void (*WGPUProc)(void);
typedef void (*WGPUBufferMapCallback)(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPUCompilationInfoCallback)(WGPUCompilationInfoRequestStatus status, struct WGPUCompilationInfo const * compilationInfo, void* userdata1, void* userdata2);
typedef void (*WGPUCreateComputePipelineAsyncCallback)(WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPUCreateRenderPipelineAsyncCallback)(WGPUCreatePipelineAsyncStatus status, WGPURenderPipeline pipeline, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPUDeviceLostCallback)(WGPUDevice const * device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPUPopErrorScopeCallback)(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPUQueueWorkDoneCallback)(WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPURequestAdapterCallback)(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPURequestDeviceCallback)(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2);
typedef void (*WGPUUncapturedErrorCallback)(WGPUDevice const * device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2);
typedef struct WGPUChainedStruct {
struct WGPUChainedStruct * next;
WGPUSType sType;
} WGPUChainedStruct;
typedef struct WGPUBufferMapCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUBufferMapCallback callback;
void* userdata1;
void* userdata2;
} WGPUBufferMapCallbackInfo;
typedef struct WGPUCompilationInfoCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUCompilationInfoCallback callback;
void* userdata1;
void* userdata2;
} WGPUCompilationInfoCallbackInfo;
typedef struct WGPUCreateComputePipelineAsyncCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUCreateComputePipelineAsyncCallback callback;
void* userdata1;
void* userdata2;
} WGPUCreateComputePipelineAsyncCallbackInfo;
typedef struct WGPUCreateRenderPipelineAsyncCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUCreateRenderPipelineAsyncCallback callback;
void* userdata1;
void* userdata2;
} WGPUCreateRenderPipelineAsyncCallbackInfo;
typedef struct WGPUDeviceLostCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUDeviceLostCallback callback;
void* userdata1;
void* userdata2;
} WGPUDeviceLostCallbackInfo;
typedef struct WGPUPopErrorScopeCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUPopErrorScopeCallback callback;
void* userdata1;
void* userdata2;
} WGPUPopErrorScopeCallbackInfo;
typedef struct WGPUQueueWorkDoneCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPUQueueWorkDoneCallback callback;
void* userdata1;
void* userdata2;
} WGPUQueueWorkDoneCallbackInfo;
typedef struct WGPURequestAdapterCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPURequestAdapterCallback callback;
void* userdata1;
void* userdata2;
} WGPURequestAdapterCallbackInfo;
typedef struct WGPURequestDeviceCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUCallbackMode mode;
WGPURequestDeviceCallback callback;
void* userdata1;
void* userdata2;
} WGPURequestDeviceCallbackInfo;
typedef struct WGPUUncapturedErrorCallbackInfo {
WGPUChainedStruct * nextInChain;
WGPUUncapturedErrorCallback callback;
void* userdata1;
void* userdata2;
} WGPUUncapturedErrorCallbackInfo;
typedef struct WGPUAdapterInfo {
WGPUChainedStruct * nextInChain;
WGPUStringView vendor;
WGPUStringView architecture;
WGPUStringView device;
WGPUStringView description;
WGPUBackendType backendType;
WGPUAdapterType adapterType;
uint32_t vendorID;
uint32_t deviceID;
uint32_t subgroupMinSize;
uint32_t subgroupMaxSize;
} WGPUAdapterInfo;
typedef struct WGPUBlendComponent {
WGPUBlendOperation operation;
WGPUBlendFactor srcFactor;
WGPUBlendFactor dstFactor;
} WGPUBlendComponent;
typedef struct WGPUBufferBindingLayout {
WGPUChainedStruct * nextInChain;
WGPUBufferBindingType type;
WGPUBool hasDynamicOffset;
uint64_t minBindingSize;
} WGPUBufferBindingLayout;
typedef struct WGPUBufferDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUBufferUsage usage;
uint64_t size;
WGPUBool mappedAtCreation;
} WGPUBufferDescriptor;
typedef struct WGPUColor {
double r;
double g;
double b;
double a;
} WGPUColor;
typedef struct WGPUCommandBufferDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
} WGPUCommandBufferDescriptor;
typedef struct WGPUCommandEncoderDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
} WGPUCommandEncoderDescriptor;
typedef struct WGPUCompatibilityModeLimits {
WGPUChainedStruct chain;
uint32_t maxStorageBuffersInVertexStage;
uint32_t maxStorageTexturesInVertexStage;
uint32_t maxStorageBuffersInFragmentStage;
uint32_t maxStorageTexturesInFragmentStage;
} WGPUCompatibilityModeLimits;
typedef struct WGPUCompilationMessage {
WGPUChainedStruct * nextInChain;
WGPUStringView message;
WGPUCompilationMessageType type;
uint64_t lineNum;
uint64_t linePos;
uint64_t offset;
uint64_t length;
} WGPUCompilationMessage;
typedef struct WGPUConstantEntry {
WGPUChainedStruct * nextInChain;
WGPUStringView key;
double value;
} WGPUConstantEntry;
typedef struct WGPUExtent3D {
uint32_t width;
uint32_t height;
uint32_t depthOrArrayLayers;
} WGPUExtent3D;
typedef struct WGPUExternalTextureBindingEntry {
WGPUChainedStruct chain;
WGPUExternalTexture externalTexture;
} WGPUExternalTextureBindingEntry;
typedef struct WGPUExternalTextureBindingLayout {
WGPUChainedStruct chain;
} WGPUExternalTextureBindingLayout;
typedef struct WGPUFuture {
uint64_t id;
} WGPUFuture;
typedef struct WGPUInstanceLimits {
WGPUChainedStruct * nextInChain;
size_t timedWaitAnyMaxCount;
} WGPUInstanceLimits;
typedef struct WGPUMultisampleState {
WGPUChainedStruct * nextInChain;
uint32_t count;
uint32_t mask;
WGPUBool alphaToCoverageEnabled;
} WGPUMultisampleState;
typedef struct WGPUOrigin3D {
uint32_t x;
uint32_t y;
uint32_t z;
} WGPUOrigin3D;
typedef struct WGPUPassTimestampWrites {
WGPUChainedStruct * nextInChain;
WGPUQuerySet querySet;
uint32_t beginningOfPassWriteIndex;
uint32_t endOfPassWriteIndex;
} WGPUPassTimestampWrites;
typedef struct WGPUPipelineLayoutDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
size_t bindGroupLayoutCount;
WGPUBindGroupLayout const * bindGroupLayouts;
uint32_t immediateSize;
} WGPUPipelineLayoutDescriptor;
typedef struct WGPUPrimitiveState {
WGPUChainedStruct * nextInChain;
WGPUPrimitiveTopology topology;
WGPUIndexFormat stripIndexFormat;
WGPUFrontFace frontFace;
WGPUCullMode cullMode;
WGPUBool unclippedDepth;
} WGPUPrimitiveState;
typedef struct WGPUQuerySetDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUQueryType type;
uint32_t count;
} WGPUQuerySetDescriptor;
typedef struct WGPUQueueDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
} WGPUQueueDescriptor;
typedef struct WGPURenderBundleDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
} WGPURenderBundleDescriptor;
typedef struct WGPURenderBundleEncoderDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
size_t colorFormatCount;
WGPUTextureFormat const * colorFormats;
WGPUTextureFormat depthStencilFormat;
uint32_t sampleCount;
WGPUBool depthReadOnly;
WGPUBool stencilReadOnly;
} WGPURenderBundleEncoderDescriptor;
typedef struct WGPURenderPassDepthStencilAttachment {
WGPUChainedStruct * nextInChain;
WGPUTextureView view;
WGPULoadOp depthLoadOp;
WGPUStoreOp depthStoreOp;
float depthClearValue;
WGPUBool depthReadOnly;
WGPULoadOp stencilLoadOp;
WGPUStoreOp stencilStoreOp;
uint32_t stencilClearValue;
WGPUBool stencilReadOnly;
} WGPURenderPassDepthStencilAttachment;
typedef struct WGPURenderPassMaxDrawCount {
WGPUChainedStruct chain;
uint64_t maxDrawCount;
} WGPURenderPassMaxDrawCount;
typedef struct WGPURequestAdapterWebXROptions {
WGPUChainedStruct chain;
WGPUBool xrCompatible;
} WGPURequestAdapterWebXROptions;
typedef struct WGPUSamplerBindingLayout {
WGPUChainedStruct * nextInChain;
WGPUSamplerBindingType type;
} WGPUSamplerBindingLayout;
typedef struct WGPUSamplerDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUAddressMode addressModeU;
WGPUAddressMode addressModeV;
WGPUAddressMode addressModeW;
WGPUFilterMode magFilter;
WGPUFilterMode minFilter;
WGPUMipmapFilterMode mipmapFilter;
float lodMinClamp;
float lodMaxClamp;
WGPUCompareFunction compare;
uint16_t maxAnisotropy;
} WGPUSamplerDescriptor;
typedef struct WGPUShaderSourceSPIRV {
WGPUChainedStruct chain;
uint32_t codeSize;
uint32_t const * code;
} WGPUShaderSourceSPIRV;
typedef struct WGPUShaderSourceWGSL {
WGPUChainedStruct chain;
WGPUStringView code;
} WGPUShaderSourceWGSL;
typedef struct WGPUStencilFaceState {
WGPUCompareFunction compare;
WGPUStencilOperation failOp;
WGPUStencilOperation depthFailOp;
WGPUStencilOperation passOp;
} WGPUStencilFaceState;
typedef struct WGPUStorageTextureBindingLayout {
WGPUChainedStruct * nextInChain;
WGPUStorageTextureAccess access;
WGPUTextureFormat format;
WGPUTextureViewDimension viewDimension;
} WGPUStorageTextureBindingLayout;
typedef struct WGPUSupportedFeatures {
size_t featureCount;
WGPUFeatureName const * features;
} WGPUSupportedFeatures;
typedef struct WGPUSupportedInstanceFeatures {
size_t featureCount;
WGPUInstanceFeatureName const * features;
} WGPUSupportedInstanceFeatures;
typedef struct WGPUSupportedWGSLLanguageFeatures {
size_t featureCount;
WGPUWGSLLanguageFeatureName const * features;
} WGPUSupportedWGSLLanguageFeatures;
typedef struct WGPUSurfaceCapabilities {
WGPUChainedStruct * nextInChain;
WGPUTextureUsage usages;
size_t formatCount;
WGPUTextureFormat const * formats;
size_t presentModeCount;
WGPUPresentMode const * presentModes;
size_t alphaModeCount;
WGPUCompositeAlphaMode const * alphaModes;
} WGPUSurfaceCapabilities;
typedef struct WGPUSurfaceColorManagement {
WGPUChainedStruct chain;
WGPUPredefinedColorSpace colorSpace;
WGPUToneMappingMode toneMappingMode;
} WGPUSurfaceColorManagement;
typedef struct WGPUSurfaceConfiguration {
WGPUChainedStruct * nextInChain;
WGPUDevice device;
WGPUTextureFormat format;
WGPUTextureUsage usage;
uint32_t width;
uint32_t height;
size_t viewFormatCount;
WGPUTextureFormat const * viewFormats;
WGPUCompositeAlphaMode alphaMode;
WGPUPresentMode presentMode;
} WGPUSurfaceConfiguration;
typedef struct WGPUSurfaceSourceAndroidNativeWindow {
WGPUChainedStruct chain;
void * window;
} WGPUSurfaceSourceAndroidNativeWindow;
typedef struct WGPUSurfaceSourceMetalLayer {
WGPUChainedStruct chain;
void * layer;
} WGPUSurfaceSourceMetalLayer;
typedef struct WGPUSurfaceSourceWaylandSurface {
WGPUChainedStruct chain;
void * display;
void * surface;
} WGPUSurfaceSourceWaylandSurface;
typedef struct WGPUSurfaceSourceWindowsHWND {
WGPUChainedStruct chain;
void * hinstance;
void * hwnd;
} WGPUSurfaceSourceWindowsHWND;
typedef struct WGPUSurfaceSourceXCBWindow {
WGPUChainedStruct chain;
void * connection;
uint32_t window;
} WGPUSurfaceSourceXCBWindow;
typedef struct WGPUSurfaceSourceXlibWindow {
WGPUChainedStruct chain;
void * display;
uint64_t window;
} WGPUSurfaceSourceXlibWindow;
typedef struct WGPUSurfaceTexture {
WGPUChainedStruct * nextInChain;
WGPUTexture texture;
WGPUSurfaceGetCurrentTextureStatus status;
} WGPUSurfaceTexture;
typedef struct WGPUTexelCopyBufferLayout {
uint64_t offset;
uint32_t bytesPerRow;
uint32_t rowsPerImage;
} WGPUTexelCopyBufferLayout;
typedef struct WGPUTextureBindingLayout {
WGPUChainedStruct * nextInChain;
WGPUTextureSampleType sampleType;
WGPUTextureViewDimension viewDimension;
WGPUBool multisampled;
} WGPUTextureBindingLayout;
typedef struct WGPUTextureBindingViewDimension {
WGPUChainedStruct chain;
WGPUTextureViewDimension textureBindingViewDimension;
} WGPUTextureBindingViewDimension;
typedef struct WGPUTextureComponentSwizzle {
WGPUComponentSwizzle r;
WGPUComponentSwizzle g;
WGPUComponentSwizzle b;
WGPUComponentSwizzle a;
} WGPUTextureComponentSwizzle;
typedef struct WGPUVertexAttribute {
WGPUChainedStruct * nextInChain;
WGPUVertexFormat format;
uint64_t offset;
uint32_t shaderLocation;
} WGPUVertexAttribute;
typedef struct WGPUBindGroupEntry {
WGPUChainedStruct * nextInChain;
uint32_t binding;
WGPUBuffer buffer;
uint64_t offset;
uint64_t size;
WGPUSampler sampler;
WGPUTextureView textureView;
} WGPUBindGroupEntry;
typedef struct WGPUBindGroupLayoutEntry {
WGPUChainedStruct * nextInChain;
uint32_t binding;
WGPUShaderStage visibility;
uint32_t bindingArraySize;
WGPUBufferBindingLayout buffer;
WGPUSamplerBindingLayout sampler;
WGPUTextureBindingLayout texture;
WGPUStorageTextureBindingLayout storageTexture;
} WGPUBindGroupLayoutEntry;
typedef struct WGPUBlendState {
WGPUBlendComponent color;
WGPUBlendComponent alpha;
} WGPUBlendState;
typedef struct WGPUCompilationInfo {
WGPUChainedStruct * nextInChain;
size_t messageCount;
WGPUCompilationMessage const * messages;
} WGPUCompilationInfo;
typedef struct WGPUComputePassDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUPassTimestampWrites const * timestampWrites;
} WGPUComputePassDescriptor;
typedef struct WGPUComputeState {
WGPUChainedStruct * nextInChain;
WGPUShaderModule module;
WGPUStringView entryPoint;
size_t constantCount;
WGPUConstantEntry const * constants;
} WGPUComputeState;
typedef struct WGPUDepthStencilState {
WGPUChainedStruct * nextInChain;
WGPUTextureFormat format;
WGPUOptionalBool depthWriteEnabled;
WGPUCompareFunction depthCompare;
WGPUStencilFaceState stencilFront;
WGPUStencilFaceState stencilBack;
uint32_t stencilReadMask;
uint32_t stencilWriteMask;
int32_t depthBias;
float depthBiasSlopeScale;
float depthBiasClamp;
} WGPUDepthStencilState;
typedef struct WGPUFutureWaitInfo {
WGPUFuture future;
WGPUBool completed;
} WGPUFutureWaitInfo;
typedef struct WGPUInstanceDescriptor {
WGPUChainedStruct * nextInChain;
size_t requiredFeatureCount;
WGPUInstanceFeatureName const * requiredFeatures;
WGPUInstanceLimits const * requiredLimits;
} WGPUInstanceDescriptor;
typedef struct WGPULimits {
WGPUChainedStruct * nextInChain;
uint32_t maxTextureDimension1D;
uint32_t maxTextureDimension2D;
uint32_t maxTextureDimension3D;
uint32_t maxTextureArrayLayers;
uint32_t maxBindGroups;
uint32_t maxBindGroupsPlusVertexBuffers;
uint32_t maxBindingsPerBindGroup;
uint32_t maxDynamicUniformBuffersPerPipelineLayout;
uint32_t maxDynamicStorageBuffersPerPipelineLayout;
uint32_t maxSampledTexturesPerShaderStage;
uint32_t maxSamplersPerShaderStage;
uint32_t maxStorageBuffersPerShaderStage;
uint32_t maxStorageTexturesPerShaderStage;
uint32_t maxUniformBuffersPerShaderStage;
uint64_t maxUniformBufferBindingSize;
uint64_t maxStorageBufferBindingSize;
uint32_t minUniformBufferOffsetAlignment;
uint32_t minStorageBufferOffsetAlignment;
uint32_t maxVertexBuffers;
uint64_t maxBufferSize;
uint32_t maxVertexAttributes;
uint32_t maxVertexBufferArrayStride;
uint32_t maxInterStageShaderVariables;
uint32_t maxColorAttachments;
uint32_t maxColorAttachmentBytesPerSample;
uint32_t maxComputeWorkgroupStorageSize;
uint32_t maxComputeInvocationsPerWorkgroup;
uint32_t maxComputeWorkgroupSizeX;
uint32_t maxComputeWorkgroupSizeY;
uint32_t maxComputeWorkgroupSizeZ;
uint32_t maxComputeWorkgroupsPerDimension;
uint32_t maxImmediateSize;
} WGPULimits;
typedef struct WGPURenderPassColorAttachment {
WGPUChainedStruct * nextInChain;
WGPUTextureView view;
uint32_t depthSlice;
WGPUTextureView resolveTarget;
WGPULoadOp loadOp;
WGPUStoreOp storeOp;
WGPUColor clearValue;
} WGPURenderPassColorAttachment;
typedef struct WGPURequestAdapterOptions {
WGPUChainedStruct * nextInChain;
WGPUFeatureLevel featureLevel;
WGPUPowerPreference powerPreference;
WGPUBool forceFallbackAdapter;
WGPUBackendType backendType;
WGPUSurface compatibleSurface;
} WGPURequestAdapterOptions;
typedef struct WGPUShaderModuleDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
} WGPUShaderModuleDescriptor;
typedef struct WGPUSurfaceDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
} WGPUSurfaceDescriptor;
typedef struct WGPUTexelCopyBufferInfo {
WGPUTexelCopyBufferLayout layout;
WGPUBuffer buffer;
} WGPUTexelCopyBufferInfo;
typedef struct WGPUTexelCopyTextureInfo {
WGPUTexture texture;
uint32_t mipLevel;
WGPUOrigin3D origin;
WGPUTextureAspect aspect;
} WGPUTexelCopyTextureInfo;
typedef struct WGPUTextureComponentSwizzleDescriptor {
WGPUChainedStruct chain;
WGPUTextureComponentSwizzle swizzle;
} WGPUTextureComponentSwizzleDescriptor;
typedef struct WGPUTextureDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUTextureUsage usage;
WGPUTextureDimension dimension;
WGPUExtent3D size;
WGPUTextureFormat format;
uint32_t mipLevelCount;
uint32_t sampleCount;
size_t viewFormatCount;
WGPUTextureFormat const * viewFormats;
} WGPUTextureDescriptor;
typedef struct WGPUVertexBufferLayout {
WGPUChainedStruct * nextInChain;
WGPUVertexStepMode stepMode;
uint64_t arrayStride;
size_t attributeCount;
WGPUVertexAttribute const * attributes;
} WGPUVertexBufferLayout;
typedef struct WGPUBindGroupDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUBindGroupLayout layout;
size_t entryCount;
WGPUBindGroupEntry const * entries;
} WGPUBindGroupDescriptor;
typedef struct WGPUBindGroupLayoutDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
size_t entryCount;
WGPUBindGroupLayoutEntry const * entries;
} WGPUBindGroupLayoutDescriptor;
typedef struct WGPUColorTargetState {
WGPUChainedStruct * nextInChain;
WGPUTextureFormat format;
WGPUBlendState const * blend;
WGPUColorWriteMask writeMask;
} WGPUColorTargetState;
typedef struct WGPUComputePipelineDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUPipelineLayout layout;
WGPUComputeState compute;
} WGPUComputePipelineDescriptor;
typedef struct WGPUDeviceDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
size_t requiredFeatureCount;
WGPUFeatureName const * requiredFeatures;
WGPULimits const * requiredLimits;
WGPUQueueDescriptor defaultQueue;
WGPUDeviceLostCallbackInfo deviceLostCallbackInfo;
WGPUUncapturedErrorCallbackInfo uncapturedErrorCallbackInfo;
} WGPUDeviceDescriptor;
typedef struct WGPURenderPassDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
size_t colorAttachmentCount;
WGPURenderPassColorAttachment const * colorAttachments;
WGPURenderPassDepthStencilAttachment const * depthStencilAttachment;
WGPUQuerySet occlusionQuerySet;
WGPUPassTimestampWrites const * timestampWrites;
} WGPURenderPassDescriptor;
typedef struct WGPUTextureViewDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUTextureFormat format;
WGPUTextureViewDimension dimension;
uint32_t baseMipLevel;
uint32_t mipLevelCount;
uint32_t baseArrayLayer;
uint32_t arrayLayerCount;
WGPUTextureAspect aspect;
WGPUTextureUsage usage;
} WGPUTextureViewDescriptor;
typedef struct WGPUVertexState {
WGPUChainedStruct * nextInChain;
WGPUShaderModule module;
WGPUStringView entryPoint;
size_t constantCount;
WGPUConstantEntry const * constants;
size_t bufferCount;
WGPUVertexBufferLayout const * buffers;
} WGPUVertexState;
typedef struct WGPUFragmentState {
WGPUChainedStruct * nextInChain;
WGPUShaderModule module;
WGPUStringView entryPoint;
size_t constantCount;
WGPUConstantEntry const * constants;
size_t targetCount;
WGPUColorTargetState const * targets;
} WGPUFragmentState;
typedef struct WGPURenderPipelineDescriptor {
WGPUChainedStruct * nextInChain;
WGPUStringView label;
WGPUPipelineLayout layout;
WGPUVertexState vertex;
WGPUPrimitiveState primitive;
WGPUDepthStencilState const * depthStencil;
WGPUMultisampleState multisample;
WGPUFragmentState const * fragment;
} WGPURenderPipelineDescriptor;
typedef WGPUInstance (*WGPUProcCreateInstance)(WGPUInstanceDescriptor const * descriptor);
typedef void (*WGPUProcGetInstanceFeatures)(WGPUSupportedInstanceFeatures * features);
typedef WGPUStatus (*WGPUProcGetInstanceLimits)(WGPUInstanceLimits * limits);
typedef WGPUBool (*WGPUProcHasInstanceFeature)(WGPUInstanceFeatureName feature);
typedef WGPUProc (*WGPUProcGetProcAddress)(WGPUStringView procName);
typedef void (*WGPUProcAdapterGetFeatures)(WGPUAdapter adapter, WGPUSupportedFeatures * features);
typedef WGPUStatus (*WGPUProcAdapterGetInfo)(WGPUAdapter adapter, WGPUAdapterInfo * info);
typedef WGPUStatus (*WGPUProcAdapterGetLimits)(WGPUAdapter adapter, WGPULimits * limits);
typedef WGPUBool (*WGPUProcAdapterHasFeature)(WGPUAdapter adapter, WGPUFeatureName feature);
typedef WGPUFuture (*WGPUProcAdapterRequestDevice)(WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor, WGPURequestDeviceCallbackInfo callbackInfo);
typedef void (*WGPUProcAdapterAddRef)(WGPUAdapter adapter);
typedef void (*WGPUProcAdapterRelease)(WGPUAdapter adapter);
typedef void (*WGPUProcAdapterInfoFreeMembers)(WGPUAdapterInfo adapterInfo);
typedef void (*WGPUProcBindGroupSetLabel)(WGPUBindGroup bindGroup, WGPUStringView label);
typedef void (*WGPUProcBindGroupAddRef)(WGPUBindGroup bindGroup);
typedef void (*WGPUProcBindGroupRelease)(WGPUBindGroup bindGroup);
typedef void (*WGPUProcBindGroupLayoutSetLabel)(WGPUBindGroupLayout bindGroupLayout, WGPUStringView label);
typedef void (*WGPUProcBindGroupLayoutAddRef)(WGPUBindGroupLayout bindGroupLayout);
typedef void (*WGPUProcBindGroupLayoutRelease)(WGPUBindGroupLayout bindGroupLayout);
typedef void (*WGPUProcBufferDestroy)(WGPUBuffer buffer);
typedef void const * (*WGPUProcBufferGetConstMappedRange)(WGPUBuffer buffer, size_t offset, size_t size);
typedef void * (*WGPUProcBufferGetMappedRange)(WGPUBuffer buffer, size_t offset, size_t size);
typedef WGPUBufferMapState (*WGPUProcBufferGetMapState)(WGPUBuffer buffer);
typedef uint64_t (*WGPUProcBufferGetSize)(WGPUBuffer buffer);
typedef WGPUBufferUsage (*WGPUProcBufferGetUsage)(WGPUBuffer buffer);
typedef WGPUFuture (*WGPUProcBufferMapAsync)(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo);
typedef WGPUStatus (*WGPUProcBufferReadMappedRange)(WGPUBuffer buffer, size_t offset, void * data, size_t size);
typedef void (*WGPUProcBufferSetLabel)(WGPUBuffer buffer, WGPUStringView label);
typedef void (*WGPUProcBufferUnmap)(WGPUBuffer buffer);
typedef WGPUStatus (*WGPUProcBufferWriteMappedRange)(WGPUBuffer buffer, size_t offset, void const * data, size_t size);
typedef void (*WGPUProcBufferAddRef)(WGPUBuffer buffer);
typedef void (*WGPUProcBufferRelease)(WGPUBuffer buffer);
typedef void (*WGPUProcCommandBufferSetLabel)(WGPUCommandBuffer commandBuffer, WGPUStringView label);
typedef void (*WGPUProcCommandBufferAddRef)(WGPUCommandBuffer commandBuffer);
typedef void (*WGPUProcCommandBufferRelease)(WGPUCommandBuffer commandBuffer);
typedef WGPUComputePassEncoder (*WGPUProcCommandEncoderBeginComputePass)(WGPUCommandEncoder commandEncoder, WGPUComputePassDescriptor const * descriptor);
typedef WGPURenderPassEncoder (*WGPUProcCommandEncoderBeginRenderPass)(WGPUCommandEncoder commandEncoder, WGPURenderPassDescriptor const * descriptor);
typedef void (*WGPUProcCommandEncoderClearBuffer)(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t offset, uint64_t size);
typedef void (*WGPUProcCommandEncoderCopyBufferToBuffer)(WGPUCommandEncoder commandEncoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size);
typedef void (*WGPUProcCommandEncoderCopyBufferToTexture)(WGPUCommandEncoder commandEncoder, WGPUTexelCopyBufferInfo const * source, WGPUTexelCopyTextureInfo const * destination, WGPUExtent3D const * copySize);
typedef void (*WGPUProcCommandEncoderCopyTextureToBuffer)(WGPUCommandEncoder commandEncoder, WGPUTexelCopyTextureInfo const * source, WGPUTexelCopyBufferInfo const * destination, WGPUExtent3D const * copySize);
typedef void (*WGPUProcCommandEncoderCopyTextureToTexture)(WGPUCommandEncoder commandEncoder, WGPUTexelCopyTextureInfo const * source, WGPUTexelCopyTextureInfo const * destination, WGPUExtent3D const * copySize);
typedef WGPUCommandBuffer (*WGPUProcCommandEncoderFinish)(WGPUCommandEncoder commandEncoder, WGPUCommandBufferDescriptor const * descriptor);
typedef void (*WGPUProcCommandEncoderInsertDebugMarker)(WGPUCommandEncoder commandEncoder, WGPUStringView markerLabel);
typedef void (*WGPUProcCommandEncoderPopDebugGroup)(WGPUCommandEncoder commandEncoder);
typedef void (*WGPUProcCommandEncoderPushDebugGroup)(WGPUCommandEncoder commandEncoder, WGPUStringView groupLabel);
typedef void (*WGPUProcCommandEncoderResolveQuerySet)(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer destination, uint64_t destinationOffset);
typedef void (*WGPUProcCommandEncoderSetLabel)(WGPUCommandEncoder commandEncoder, WGPUStringView label);
typedef void (*WGPUProcCommandEncoderWriteTimestamp)(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t queryIndex);
typedef void (*WGPUProcCommandEncoderAddRef)(WGPUCommandEncoder commandEncoder);
typedef void (*WGPUProcCommandEncoderRelease)(WGPUCommandEncoder commandEncoder);
typedef void (*WGPUProcComputePassEncoderDispatchWorkgroups)(WGPUComputePassEncoder computePassEncoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ);
typedef void (*WGPUProcComputePassEncoderDispatchWorkgroupsIndirect)(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
typedef void (*WGPUProcComputePassEncoderEnd)(WGPUComputePassEncoder computePassEncoder);
typedef void (*WGPUProcComputePassEncoderInsertDebugMarker)(WGPUComputePassEncoder computePassEncoder, WGPUStringView markerLabel);
typedef void (*WGPUProcComputePassEncoderPopDebugGroup)(WGPUComputePassEncoder computePassEncoder);
typedef void (*WGPUProcComputePassEncoderPushDebugGroup)(WGPUComputePassEncoder computePassEncoder, WGPUStringView groupLabel);
typedef void (*WGPUProcComputePassEncoderSetBindGroup)(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets);
typedef void (*WGPUProcComputePassEncoderSetLabel)(WGPUComputePassEncoder computePassEncoder, WGPUStringView label);
typedef void (*WGPUProcComputePassEncoderSetPipeline)(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline);
typedef void (*WGPUProcComputePassEncoderAddRef)(WGPUComputePassEncoder computePassEncoder);
typedef void (*WGPUProcComputePassEncoderRelease)(WGPUComputePassEncoder computePassEncoder);
typedef WGPUBindGroupLayout (*WGPUProcComputePipelineGetBindGroupLayout)(WGPUComputePipeline computePipeline, uint32_t groupIndex);
typedef void (*WGPUProcComputePipelineSetLabel)(WGPUComputePipeline computePipeline, WGPUStringView label);
typedef void (*WGPUProcComputePipelineAddRef)(WGPUComputePipeline computePipeline);
typedef void (*WGPUProcComputePipelineRelease)(WGPUComputePipeline computePipeline);
typedef WGPUBindGroup (*WGPUProcDeviceCreateBindGroup)(WGPUDevice device, WGPUBindGroupDescriptor const * descriptor);
typedef WGPUBindGroupLayout (*WGPUProcDeviceCreateBindGroupLayout)(WGPUDevice device, WGPUBindGroupLayoutDescriptor const * descriptor);
typedef WGPUBuffer (*WGPUProcDeviceCreateBuffer)(WGPUDevice device, WGPUBufferDescriptor const * descriptor);
typedef WGPUCommandEncoder (*WGPUProcDeviceCreateCommandEncoder)(WGPUDevice device, WGPUCommandEncoderDescriptor const * descriptor);
typedef WGPUComputePipeline (*WGPUProcDeviceCreateComputePipeline)(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor);
typedef WGPUFuture (*WGPUProcDeviceCreateComputePipelineAsync)(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor, WGPUCreateComputePipelineAsyncCallbackInfo callbackInfo);
typedef WGPUPipelineLayout (*WGPUProcDeviceCreatePipelineLayout)(WGPUDevice device, WGPUPipelineLayoutDescriptor const * descriptor);
typedef WGPUQuerySet (*WGPUProcDeviceCreateQuerySet)(WGPUDevice device, WGPUQuerySetDescriptor const * descriptor);
typedef WGPURenderBundleEncoder (*WGPUProcDeviceCreateRenderBundleEncoder)(WGPUDevice device, WGPURenderBundleEncoderDescriptor const * descriptor);
typedef WGPURenderPipeline (*WGPUProcDeviceCreateRenderPipeline)(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor);
typedef WGPUFuture (*WGPUProcDeviceCreateRenderPipelineAsync)(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor, WGPUCreateRenderPipelineAsyncCallbackInfo callbackInfo);
typedef WGPUSampler (*WGPUProcDeviceCreateSampler)(WGPUDevice device, WGPUSamplerDescriptor const * descriptor);
typedef WGPUShaderModule (*WGPUProcDeviceCreateShaderModule)(WGPUDevice device, WGPUShaderModuleDescriptor const * descriptor);
typedef WGPUTexture (*WGPUProcDeviceCreateTexture)(WGPUDevice device, WGPUTextureDescriptor const * descriptor);
typedef void (*WGPUProcDeviceDestroy)(WGPUDevice device);
typedef WGPUStatus (*WGPUProcDeviceGetAdapterInfo)(WGPUDevice device, WGPUAdapterInfo * adapterInfo);
typedef void (*WGPUProcDeviceGetFeatures)(WGPUDevice device, WGPUSupportedFeatures * features);
typedef WGPUStatus (*WGPUProcDeviceGetLimits)(WGPUDevice device, WGPULimits * limits);
typedef WGPUFuture (*WGPUProcDeviceGetLostFuture)(WGPUDevice device);
typedef WGPUQueue (*WGPUProcDeviceGetQueue)(WGPUDevice device);
typedef WGPUBool (*WGPUProcDeviceHasFeature)(WGPUDevice device, WGPUFeatureName feature);
typedef WGPUFuture (*WGPUProcDevicePopErrorScope)(WGPUDevice device, WGPUPopErrorScopeCallbackInfo callbackInfo);
typedef void (*WGPUProcDevicePushErrorScope)(WGPUDevice device, WGPUErrorFilter filter);
typedef void (*WGPUProcDeviceSetLabel)(WGPUDevice device, WGPUStringView label);
typedef void (*WGPUProcDeviceAddRef)(WGPUDevice device);
typedef void (*WGPUProcDeviceRelease)(WGPUDevice device);
typedef void (*WGPUProcExternalTextureSetLabel)(WGPUExternalTexture externalTexture, WGPUStringView label);
typedef void (*WGPUProcExternalTextureAddRef)(WGPUExternalTexture externalTexture);
typedef void (*WGPUProcExternalTextureRelease)(WGPUExternalTexture externalTexture);
typedef WGPUSurface (*WGPUProcInstanceCreateSurface)(WGPUInstance instance, WGPUSurfaceDescriptor const * descriptor);
typedef void (*WGPUProcInstanceGetWGSLLanguageFeatures)(WGPUInstance instance, WGPUSupportedWGSLLanguageFeatures * features);
typedef WGPUBool (*WGPUProcInstanceHasWGSLLanguageFeature)(WGPUInstance instance, WGPUWGSLLanguageFeatureName feature);
typedef void (*WGPUProcInstanceProcessEvents)(WGPUInstance instance);
typedef WGPUFuture (*WGPUProcInstanceRequestAdapter)(WGPUInstance instance, WGPURequestAdapterOptions const * options, WGPURequestAdapterCallbackInfo callbackInfo);
typedef WGPUWaitStatus (*WGPUProcInstanceWaitAny)(WGPUInstance instance, size_t futureCount, WGPUFutureWaitInfo * futures, uint64_t timeoutNS);
typedef void (*WGPUProcInstanceAddRef)(WGPUInstance instance);
typedef void (*WGPUProcInstanceRelease)(WGPUInstance instance);
typedef void (*WGPUProcPipelineLayoutSetLabel)(WGPUPipelineLayout pipelineLayout, WGPUStringView label);
typedef void (*WGPUProcPipelineLayoutAddRef)(WGPUPipelineLayout pipelineLayout);
typedef void (*WGPUProcPipelineLayoutRelease)(WGPUPipelineLayout pipelineLayout);
typedef void (*WGPUProcQuerySetDestroy)(WGPUQuerySet querySet);
typedef uint32_t (*WGPUProcQuerySetGetCount)(WGPUQuerySet querySet);
typedef WGPUQueryType (*WGPUProcQuerySetGetType)(WGPUQuerySet querySet);
typedef void (*WGPUProcQuerySetSetLabel)(WGPUQuerySet querySet, WGPUStringView label);
typedef void (*WGPUProcQuerySetAddRef)(WGPUQuerySet querySet);
typedef void (*WGPUProcQuerySetRelease)(WGPUQuerySet querySet);
typedef WGPUFuture (*WGPUProcQueueOnSubmittedWorkDone)(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo);
typedef void (*WGPUProcQueueSetLabel)(WGPUQueue queue, WGPUStringView label);
typedef void (*WGPUProcQueueSubmit)(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const * commands);
typedef void (*WGPUProcQueueWriteBuffer)(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, void const * data, size_t size);
typedef void (*WGPUProcQueueWriteTexture)(WGPUQueue queue, WGPUTexelCopyTextureInfo const * destination, void const * data, size_t dataSize, WGPUTexelCopyBufferLayout const * dataLayout, WGPUExtent3D const * writeSize);
typedef void (*WGPUProcQueueAddRef)(WGPUQueue queue);
typedef void (*WGPUProcQueueRelease)(WGPUQueue queue);
typedef void (*WGPUProcRenderBundleSetLabel)(WGPURenderBundle renderBundle, WGPUStringView label);
typedef void (*WGPUProcRenderBundleAddRef)(WGPURenderBundle renderBundle);
typedef void (*WGPUProcRenderBundleRelease)(WGPURenderBundle renderBundle);
typedef void (*WGPUProcRenderBundleEncoderDraw)(WGPURenderBundleEncoder renderBundleEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
typedef void (*WGPUProcRenderBundleEncoderDrawIndexed)(WGPURenderBundleEncoder renderBundleEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
typedef void (*WGPUProcRenderBundleEncoderDrawIndexedIndirect)(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
typedef void (*WGPUProcRenderBundleEncoderDrawIndirect)(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
typedef WGPURenderBundle (*WGPUProcRenderBundleEncoderFinish)(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderBundleDescriptor const * descriptor);
typedef void (*WGPUProcRenderBundleEncoderInsertDebugMarker)(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView markerLabel);
typedef void (*WGPUProcRenderBundleEncoderPopDebugGroup)(WGPURenderBundleEncoder renderBundleEncoder);
typedef void (*WGPUProcRenderBundleEncoderPushDebugGroup)(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView groupLabel);
typedef void (*WGPUProcRenderBundleEncoderSetBindGroup)(WGPURenderBundleEncoder renderBundleEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets);
typedef void (*WGPUProcRenderBundleEncoderSetIndexBuffer)(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);
typedef void (*WGPUProcRenderBundleEncoderSetLabel)(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView label);
typedef void (*WGPUProcRenderBundleEncoderSetPipeline)(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderPipeline pipeline);
typedef void (*WGPUProcRenderBundleEncoderSetVertexBuffer)(WGPURenderBundleEncoder renderBundleEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size);
typedef void (*WGPUProcRenderBundleEncoderAddRef)(WGPURenderBundleEncoder renderBundleEncoder);
typedef void (*WGPUProcRenderBundleEncoderRelease)(WGPURenderBundleEncoder renderBundleEncoder);
typedef void (*WGPUProcRenderPassEncoderBeginOcclusionQuery)(WGPURenderPassEncoder renderPassEncoder, uint32_t queryIndex);
typedef void (*WGPUProcRenderPassEncoderDraw)(WGPURenderPassEncoder renderPassEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
typedef void (*WGPUProcRenderPassEncoderDrawIndexed)(WGPURenderPassEncoder renderPassEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
typedef void (*WGPUProcRenderPassEncoderDrawIndexedIndirect)(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
typedef void (*WGPUProcRenderPassEncoderDrawIndirect)(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
typedef void (*WGPUProcRenderPassEncoderEnd)(WGPURenderPassEncoder renderPassEncoder);
typedef void (*WGPUProcRenderPassEncoderEndOcclusionQuery)(WGPURenderPassEncoder renderPassEncoder);
typedef void (*WGPUProcRenderPassEncoderExecuteBundles)(WGPURenderPassEncoder renderPassEncoder, size_t bundleCount, WGPURenderBundle const * bundles);
typedef void (*WGPUProcRenderPassEncoderInsertDebugMarker)(WGPURenderPassEncoder renderPassEncoder, WGPUStringView markerLabel);
typedef void (*WGPUProcRenderPassEncoderPopDebugGroup)(WGPURenderPassEncoder renderPassEncoder);
typedef void (*WGPUProcRenderPassEncoderPushDebugGroup)(WGPURenderPassEncoder renderPassEncoder, WGPUStringView groupLabel);
typedef void (*WGPUProcRenderPassEncoderSetBindGroup)(WGPURenderPassEncoder renderPassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets);
typedef void (*WGPUProcRenderPassEncoderSetBlendConstant)(WGPURenderPassEncoder renderPassEncoder, WGPUColor const * color);
typedef void (*WGPUProcRenderPassEncoderSetIndexBuffer)(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);
typedef void (*WGPUProcRenderPassEncoderSetLabel)(WGPURenderPassEncoder renderPassEncoder, WGPUStringView label);
typedef void (*WGPUProcRenderPassEncoderSetPipeline)(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline);
typedef void (*WGPUProcRenderPassEncoderSetScissorRect)(WGPURenderPassEncoder renderPassEncoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height);
typedef void (*WGPUProcRenderPassEncoderSetStencilReference)(WGPURenderPassEncoder renderPassEncoder, uint32_t reference);
typedef void (*WGPUProcRenderPassEncoderSetVertexBuffer)(WGPURenderPassEncoder renderPassEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size);
typedef void (*WGPUProcRenderPassEncoderSetViewport)(WGPURenderPassEncoder renderPassEncoder, float x, float y, float width, float height, float minDepth, float maxDepth);
typedef void (*WGPUProcRenderPassEncoderAddRef)(WGPURenderPassEncoder renderPassEncoder);
typedef void (*WGPUProcRenderPassEncoderRelease)(WGPURenderPassEncoder renderPassEncoder);
typedef WGPUBindGroupLayout (*WGPUProcRenderPipelineGetBindGroupLayout)(WGPURenderPipeline renderPipeline, uint32_t groupIndex);
typedef void (*WGPUProcRenderPipelineSetLabel)(WGPURenderPipeline renderPipeline, WGPUStringView label);
typedef void (*WGPUProcRenderPipelineAddRef)(WGPURenderPipeline renderPipeline);
typedef void (*WGPUProcRenderPipelineRelease)(WGPURenderPipeline renderPipeline);
typedef void (*WGPUProcSamplerSetLabel)(WGPUSampler sampler, WGPUStringView label);
typedef void (*WGPUProcSamplerAddRef)(WGPUSampler sampler);
typedef void (*WGPUProcSamplerRelease)(WGPUSampler sampler);
typedef WGPUFuture (*WGPUProcShaderModuleGetCompilationInfo)(WGPUShaderModule shaderModule, WGPUCompilationInfoCallbackInfo callbackInfo);
typedef void (*WGPUProcShaderModuleSetLabel)(WGPUShaderModule shaderModule, WGPUStringView label);
typedef void (*WGPUProcShaderModuleAddRef)(WGPUShaderModule shaderModule);
typedef void (*WGPUProcShaderModuleRelease)(WGPUShaderModule shaderModule);
typedef void (*WGPUProcSupportedFeaturesFreeMembers)(WGPUSupportedFeatures supportedFeatures);
typedef void (*WGPUProcSupportedInstanceFeaturesFreeMembers)(WGPUSupportedInstanceFeatures supportedInstanceFeatures);
typedef void (*WGPUProcSupportedWGSLLanguageFeaturesFreeMembers)(WGPUSupportedWGSLLanguageFeatures supportedWGSLLanguageFeatures);
typedef void (*WGPUProcSurfaceConfigure)(WGPUSurface surface, WGPUSurfaceConfiguration const * config);
typedef WGPUStatus (*WGPUProcSurfaceGetCapabilities)(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities * capabilities);
typedef void (*WGPUProcSurfaceGetCurrentTexture)(WGPUSurface surface, WGPUSurfaceTexture * surfaceTexture);
typedef WGPUStatus (*WGPUProcSurfacePresent)(WGPUSurface surface);
typedef void (*WGPUProcSurfaceSetLabel)(WGPUSurface surface, WGPUStringView label);
typedef void (*WGPUProcSurfaceUnconfigure)(WGPUSurface surface);
typedef void (*WGPUProcSurfaceAddRef)(WGPUSurface surface);
typedef void (*WGPUProcSurfaceRelease)(WGPUSurface surface);
typedef void (*WGPUProcSurfaceCapabilitiesFreeMembers)(WGPUSurfaceCapabilities surfaceCapabilities);
typedef WGPUTextureView (*WGPUProcTextureCreateView)(WGPUTexture texture, WGPUTextureViewDescriptor const * descriptor);
typedef void (*WGPUProcTextureDestroy)(WGPUTexture texture);
typedef uint32_t (*WGPUProcTextureGetDepthOrArrayLayers)(WGPUTexture texture);
typedef WGPUTextureDimension (*WGPUProcTextureGetDimension)(WGPUTexture texture);
typedef WGPUTextureFormat (*WGPUProcTextureGetFormat)(WGPUTexture texture);
typedef uint32_t (*WGPUProcTextureGetHeight)(WGPUTexture texture);
typedef uint32_t (*WGPUProcTextureGetMipLevelCount)(WGPUTexture texture);
typedef uint32_t (*WGPUProcTextureGetSampleCount)(WGPUTexture texture);
typedef WGPUTextureViewDimension (*WGPUProcTextureGetTextureBindingViewDimension)(WGPUTexture texture);
typedef WGPUTextureUsage (*WGPUProcTextureGetUsage)(WGPUTexture texture);
typedef uint32_t (*WGPUProcTextureGetWidth)(WGPUTexture texture);
typedef void (*WGPUProcTextureSetLabel)(WGPUTexture texture, WGPUStringView label);
typedef void (*WGPUProcTextureAddRef)(WGPUTexture texture);
typedef void (*WGPUProcTextureRelease)(WGPUTexture texture);
typedef void (*WGPUProcTextureViewSetLabel)(WGPUTextureView textureView, WGPUStringView label);
typedef void (*WGPUProcTextureViewAddRef)(WGPUTextureView textureView);
typedef void (*WGPUProcTextureViewRelease)(WGPUTextureView textureView);
WGPUInstance wgpuCreateInstance(WGPUInstanceDescriptor const * descriptor);
void wgpuGetInstanceFeatures(WGPUSupportedInstanceFeatures * features);
WGPUStatus wgpuGetInstanceLimits(WGPUInstanceLimits * limits);
WGPUBool wgpuHasInstanceFeature(WGPUInstanceFeatureName feature);
WGPUProc wgpuGetProcAddress(WGPUStringView procName);
void wgpuAdapterGetFeatures(WGPUAdapter adapter, WGPUSupportedFeatures * features);
WGPUStatus wgpuAdapterGetInfo(WGPUAdapter adapter, WGPUAdapterInfo * info);
WGPUStatus wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits * limits);
WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature);
WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor, WGPURequestDeviceCallbackInfo callbackInfo);
void wgpuAdapterAddRef(WGPUAdapter adapter);
void wgpuAdapterRelease(WGPUAdapter adapter);
void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo adapterInfo);
void wgpuBindGroupSetLabel(WGPUBindGroup bindGroup, WGPUStringView label);
void wgpuBindGroupAddRef(WGPUBindGroup bindGroup);
void wgpuBindGroupRelease(WGPUBindGroup bindGroup);
void wgpuBindGroupLayoutSetLabel(WGPUBindGroupLayout bindGroupLayout, WGPUStringView label);
void wgpuBindGroupLayoutAddRef(WGPUBindGroupLayout bindGroupLayout);
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout bindGroupLayout);
void wgpuBufferDestroy(WGPUBuffer buffer);
void const * wgpuBufferGetConstMappedRange(WGPUBuffer buffer, size_t offset, size_t size);
void * wgpuBufferGetMappedRange(WGPUBuffer buffer, size_t offset, size_t size);
WGPUBufferMapState wgpuBufferGetMapState(WGPUBuffer buffer);
uint64_t wgpuBufferGetSize(WGPUBuffer buffer);
WGPUBufferUsage wgpuBufferGetUsage(WGPUBuffer buffer);
WGPUFuture wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo);
WGPUStatus wgpuBufferReadMappedRange(WGPUBuffer buffer, size_t offset, void * data, size_t size);
void wgpuBufferSetLabel(WGPUBuffer buffer, WGPUStringView label);
void wgpuBufferUnmap(WGPUBuffer buffer);
WGPUStatus wgpuBufferWriteMappedRange(WGPUBuffer buffer, size_t offset, void const * data, size_t size);
void wgpuBufferAddRef(WGPUBuffer buffer);
void wgpuBufferRelease(WGPUBuffer buffer);
void wgpuCommandBufferSetLabel(WGPUCommandBuffer commandBuffer, WGPUStringView label);
void wgpuCommandBufferAddRef(WGPUCommandBuffer commandBuffer);
void wgpuCommandBufferRelease(WGPUCommandBuffer commandBuffer);
WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder commandEncoder, WGPUComputePassDescriptor const * descriptor);
WGPURenderPassEncoder wgpuCommandEncoderBeginRenderPass(WGPUCommandEncoder commandEncoder, WGPURenderPassDescriptor const * descriptor);
void wgpuCommandEncoderClearBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t offset, uint64_t size);
void wgpuCommandEncoderCopyBufferToBuffer(WGPUCommandEncoder commandEncoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size);
void wgpuCommandEncoderCopyBufferToTexture(WGPUCommandEncoder commandEncoder, WGPUTexelCopyBufferInfo const * source, WGPUTexelCopyTextureInfo const * destination, WGPUExtent3D const * copySize);
void wgpuCommandEncoderCopyTextureToBuffer(WGPUCommandEncoder commandEncoder, WGPUTexelCopyTextureInfo const * source, WGPUTexelCopyBufferInfo const * destination, WGPUExtent3D const * copySize);
void wgpuCommandEncoderCopyTextureToTexture(WGPUCommandEncoder commandEncoder, WGPUTexelCopyTextureInfo const * source, WGPUTexelCopyTextureInfo const * destination, WGPUExtent3D const * copySize);
WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder commandEncoder, WGPUCommandBufferDescriptor const * descriptor);
void wgpuCommandEncoderInsertDebugMarker(WGPUCommandEncoder commandEncoder, WGPUStringView markerLabel);
void wgpuCommandEncoderPopDebugGroup(WGPUCommandEncoder commandEncoder);
void wgpuCommandEncoderPushDebugGroup(WGPUCommandEncoder commandEncoder, WGPUStringView groupLabel);
void wgpuCommandEncoderResolveQuerySet(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer destination, uint64_t destinationOffset);
void wgpuCommandEncoderSetLabel(WGPUCommandEncoder commandEncoder, WGPUStringView label);
void wgpuCommandEncoderWriteTimestamp(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t queryIndex);
void wgpuCommandEncoderAddRef(WGPUCommandEncoder commandEncoder);
void wgpuCommandEncoderRelease(WGPUCommandEncoder commandEncoder);
void wgpuComputePassEncoderDispatchWorkgroups(WGPUComputePassEncoder computePassEncoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ);
void wgpuComputePassEncoderDispatchWorkgroupsIndirect(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
void wgpuComputePassEncoderEnd(WGPUComputePassEncoder computePassEncoder);
void wgpuComputePassEncoderInsertDebugMarker(WGPUComputePassEncoder computePassEncoder, WGPUStringView markerLabel);
void wgpuComputePassEncoderPopDebugGroup(WGPUComputePassEncoder computePassEncoder);
void wgpuComputePassEncoderPushDebugGroup(WGPUComputePassEncoder computePassEncoder, WGPUStringView groupLabel);
void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets);
void wgpuComputePassEncoderSetLabel(WGPUComputePassEncoder computePassEncoder, WGPUStringView label);
void wgpuComputePassEncoderSetPipeline(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline);
void wgpuComputePassEncoderAddRef(WGPUComputePassEncoder computePassEncoder);
void wgpuComputePassEncoderRelease(WGPUComputePassEncoder computePassEncoder);
WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline computePipeline, uint32_t groupIndex);
void wgpuComputePipelineSetLabel(WGPUComputePipeline computePipeline, WGPUStringView label);
void wgpuComputePipelineAddRef(WGPUComputePipeline computePipeline);
void wgpuComputePipelineRelease(WGPUComputePipeline computePipeline);
WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice device, WGPUBindGroupDescriptor const * descriptor);
WGPUBindGroupLayout wgpuDeviceCreateBindGroupLayout(WGPUDevice device, WGPUBindGroupLayoutDescriptor const * descriptor);
WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, WGPUBufferDescriptor const * descriptor);
WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice device, WGPUCommandEncoderDescriptor const * descriptor);
WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor);
WGPUFuture wgpuDeviceCreateComputePipelineAsync(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor, WGPUCreateComputePipelineAsyncCallbackInfo callbackInfo);
WGPUPipelineLayout wgpuDeviceCreatePipelineLayout(WGPUDevice device, WGPUPipelineLayoutDescriptor const * descriptor);
WGPUQuerySet wgpuDeviceCreateQuerySet(WGPUDevice device, WGPUQuerySetDescriptor const * descriptor);
WGPURenderBundleEncoder wgpuDeviceCreateRenderBundleEncoder(WGPUDevice device, WGPURenderBundleEncoderDescriptor const * descriptor);
WGPURenderPipeline wgpuDeviceCreateRenderPipeline(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor);
WGPUFuture wgpuDeviceCreateRenderPipelineAsync(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor, WGPUCreateRenderPipelineAsyncCallbackInfo callbackInfo);
WGPUSampler wgpuDeviceCreateSampler(WGPUDevice device, WGPUSamplerDescriptor const * descriptor);
WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice device, WGPUShaderModuleDescriptor const * descriptor);
WGPUTexture wgpuDeviceCreateTexture(WGPUDevice device, WGPUTextureDescriptor const * descriptor);
void wgpuDeviceDestroy(WGPUDevice device);
WGPUStatus wgpuDeviceGetAdapterInfo(WGPUDevice device, WGPUAdapterInfo * adapterInfo);
void wgpuDeviceGetFeatures(WGPUDevice device, WGPUSupportedFeatures * features);
WGPUStatus wgpuDeviceGetLimits(WGPUDevice device, WGPULimits * limits);
WGPUFuture wgpuDeviceGetLostFuture(WGPUDevice device);
WGPUQueue wgpuDeviceGetQueue(WGPUDevice device);
WGPUBool wgpuDeviceHasFeature(WGPUDevice device, WGPUFeatureName feature);
WGPUFuture wgpuDevicePopErrorScope(WGPUDevice device, WGPUPopErrorScopeCallbackInfo callbackInfo);
void wgpuDevicePushErrorScope(WGPUDevice device, WGPUErrorFilter filter);
void wgpuDeviceSetLabel(WGPUDevice device, WGPUStringView label);
void wgpuDeviceAddRef(WGPUDevice device);
void wgpuDeviceRelease(WGPUDevice device);
void wgpuExternalTextureSetLabel(WGPUExternalTexture externalTexture, WGPUStringView label);
void wgpuExternalTextureAddRef(WGPUExternalTexture externalTexture);
void wgpuExternalTextureRelease(WGPUExternalTexture externalTexture);
WGPUSurface wgpuInstanceCreateSurface(WGPUInstance instance, WGPUSurfaceDescriptor const * descriptor);
void wgpuInstanceGetWGSLLanguageFeatures(WGPUInstance instance, WGPUSupportedWGSLLanguageFeatures * features);
WGPUBool wgpuInstanceHasWGSLLanguageFeature(WGPUInstance instance, WGPUWGSLLanguageFeatureName feature);
void wgpuInstanceProcessEvents(WGPUInstance instance);
WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance instance, WGPURequestAdapterOptions const * options, WGPURequestAdapterCallbackInfo callbackInfo);
WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance instance, size_t futureCount, WGPUFutureWaitInfo * futures, uint64_t timeoutNS);
void wgpuInstanceAddRef(WGPUInstance instance);
void wgpuInstanceRelease(WGPUInstance instance);
void wgpuPipelineLayoutSetLabel(WGPUPipelineLayout pipelineLayout, WGPUStringView label);
void wgpuPipelineLayoutAddRef(WGPUPipelineLayout pipelineLayout);
void wgpuPipelineLayoutRelease(WGPUPipelineLayout pipelineLayout);
void wgpuQuerySetDestroy(WGPUQuerySet querySet);
uint32_t wgpuQuerySetGetCount(WGPUQuerySet querySet);
WGPUQueryType wgpuQuerySetGetType(WGPUQuerySet querySet);
void wgpuQuerySetSetLabel(WGPUQuerySet querySet, WGPUStringView label);
void wgpuQuerySetAddRef(WGPUQuerySet querySet);
void wgpuQuerySetRelease(WGPUQuerySet querySet);
WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo);
void wgpuQueueSetLabel(WGPUQueue queue, WGPUStringView label);
void wgpuQueueSubmit(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const * commands);
void wgpuQueueWriteBuffer(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, void const * data, size_t size);
void wgpuQueueWriteTexture(WGPUQueue queue, WGPUTexelCopyTextureInfo const * destination, void const * data, size_t dataSize, WGPUTexelCopyBufferLayout const * dataLayout, WGPUExtent3D const * writeSize);
void wgpuQueueAddRef(WGPUQueue queue);
void wgpuQueueRelease(WGPUQueue queue);
void wgpuRenderBundleSetLabel(WGPURenderBundle renderBundle, WGPUStringView label);
void wgpuRenderBundleAddRef(WGPURenderBundle renderBundle);
void wgpuRenderBundleRelease(WGPURenderBundle renderBundle);
void wgpuRenderBundleEncoderDraw(WGPURenderBundleEncoder renderBundleEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
void wgpuRenderBundleEncoderDrawIndexed(WGPURenderBundleEncoder renderBundleEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
void wgpuRenderBundleEncoderDrawIndexedIndirect(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
void wgpuRenderBundleEncoderDrawIndirect(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
WGPURenderBundle wgpuRenderBundleEncoderFinish(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderBundleDescriptor const * descriptor);
void wgpuRenderBundleEncoderInsertDebugMarker(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView markerLabel);
void wgpuRenderBundleEncoderPopDebugGroup(WGPURenderBundleEncoder renderBundleEncoder);
void wgpuRenderBundleEncoderPushDebugGroup(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView groupLabel);
void wgpuRenderBundleEncoderSetBindGroup(WGPURenderBundleEncoder renderBundleEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets);
void wgpuRenderBundleEncoderSetIndexBuffer(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);
void wgpuRenderBundleEncoderSetLabel(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView label);
void wgpuRenderBundleEncoderSetPipeline(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderPipeline pipeline);
void wgpuRenderBundleEncoderSetVertexBuffer(WGPURenderBundleEncoder renderBundleEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size);
void wgpuRenderBundleEncoderAddRef(WGPURenderBundleEncoder renderBundleEncoder);
void wgpuRenderBundleEncoderRelease(WGPURenderBundleEncoder renderBundleEncoder);
void wgpuRenderPassEncoderBeginOcclusionQuery(WGPURenderPassEncoder renderPassEncoder, uint32_t queryIndex);
void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder renderPassEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
void wgpuRenderPassEncoderDrawIndexed(WGPURenderPassEncoder renderPassEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
void wgpuRenderPassEncoderDrawIndexedIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
void wgpuRenderPassEncoderDrawIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset);
void wgpuRenderPassEncoderEnd(WGPURenderPassEncoder renderPassEncoder);
void wgpuRenderPassEncoderEndOcclusionQuery(WGPURenderPassEncoder renderPassEncoder);
void wgpuRenderPassEncoderExecuteBundles(WGPURenderPassEncoder renderPassEncoder, size_t bundleCount, WGPURenderBundle const * bundles);
void wgpuRenderPassEncoderInsertDebugMarker(WGPURenderPassEncoder renderPassEncoder, WGPUStringView markerLabel);
void wgpuRenderPassEncoderPopDebugGroup(WGPURenderPassEncoder renderPassEncoder);
void wgpuRenderPassEncoderPushDebugGroup(WGPURenderPassEncoder renderPassEncoder, WGPUStringView groupLabel);
void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder renderPassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets);
void wgpuRenderPassEncoderSetBlendConstant(WGPURenderPassEncoder renderPassEncoder, WGPUColor const * color);
void wgpuRenderPassEncoderSetIndexBuffer(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);
void wgpuRenderPassEncoderSetLabel(WGPURenderPassEncoder renderPassEncoder, WGPUStringView label);
void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline);
void wgpuRenderPassEncoderSetScissorRect(WGPURenderPassEncoder renderPassEncoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height);
void wgpuRenderPassEncoderSetStencilReference(WGPURenderPassEncoder renderPassEncoder, uint32_t reference);
void wgpuRenderPassEncoderSetVertexBuffer(WGPURenderPassEncoder renderPassEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size);
void wgpuRenderPassEncoderSetViewport(WGPURenderPassEncoder renderPassEncoder, float x, float y, float width, float height, float minDepth, float maxDepth);
void wgpuRenderPassEncoderAddRef(WGPURenderPassEncoder renderPassEncoder);
void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder renderPassEncoder);
WGPUBindGroupLayout wgpuRenderPipelineGetBindGroupLayout(WGPURenderPipeline renderPipeline, uint32_t groupIndex);
void wgpuRenderPipelineSetLabel(WGPURenderPipeline renderPipeline, WGPUStringView label);
void wgpuRenderPipelineAddRef(WGPURenderPipeline renderPipeline);
void wgpuRenderPipelineRelease(WGPURenderPipeline renderPipeline);
void wgpuSamplerSetLabel(WGPUSampler sampler, WGPUStringView label);
void wgpuSamplerAddRef(WGPUSampler sampler);
void wgpuSamplerRelease(WGPUSampler sampler);
WGPUFuture wgpuShaderModuleGetCompilationInfo(WGPUShaderModule shaderModule, WGPUCompilationInfoCallbackInfo callbackInfo);
void wgpuShaderModuleSetLabel(WGPUShaderModule shaderModule, WGPUStringView label);
void wgpuShaderModuleAddRef(WGPUShaderModule shaderModule);
void wgpuShaderModuleRelease(WGPUShaderModule shaderModule);
void wgpuSupportedFeaturesFreeMembers(WGPUSupportedFeatures supportedFeatures);
void wgpuSupportedInstanceFeaturesFreeMembers(WGPUSupportedInstanceFeatures supportedInstanceFeatures);
void wgpuSupportedWGSLLanguageFeaturesFreeMembers(WGPUSupportedWGSLLanguageFeatures supportedWGSLLanguageFeatures);
void wgpuSurfaceConfigure(WGPUSurface surface, WGPUSurfaceConfiguration const * config);
WGPUStatus wgpuSurfaceGetCapabilities(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities * capabilities);
void wgpuSurfaceGetCurrentTexture(WGPUSurface surface, WGPUSurfaceTexture * surfaceTexture);
WGPUStatus wgpuSurfacePresent(WGPUSurface surface);
void wgpuSurfaceSetLabel(WGPUSurface surface, WGPUStringView label);
void wgpuSurfaceUnconfigure(WGPUSurface surface);
void wgpuSurfaceAddRef(WGPUSurface surface);
void wgpuSurfaceRelease(WGPUSurface surface);
void wgpuSurfaceCapabilitiesFreeMembers(WGPUSurfaceCapabilities surfaceCapabilities);
WGPUTextureView wgpuTextureCreateView(WGPUTexture texture, WGPUTextureViewDescriptor const * descriptor);
void wgpuTextureDestroy(WGPUTexture texture);
uint32_t wgpuTextureGetDepthOrArrayLayers(WGPUTexture texture);
WGPUTextureDimension wgpuTextureGetDimension(WGPUTexture texture);
WGPUTextureFormat wgpuTextureGetFormat(WGPUTexture texture);
uint32_t wgpuTextureGetHeight(WGPUTexture texture);
uint32_t wgpuTextureGetMipLevelCount(WGPUTexture texture);
uint32_t wgpuTextureGetSampleCount(WGPUTexture texture);
WGPUTextureViewDimension wgpuTextureGetTextureBindingViewDimension(WGPUTexture texture);
WGPUTextureUsage wgpuTextureGetUsage(WGPUTexture texture);
uint32_t wgpuTextureGetWidth(WGPUTexture texture);
void wgpuTextureSetLabel(WGPUTexture texture, WGPUStringView label);
void wgpuTextureAddRef(WGPUTexture texture);
void wgpuTextureRelease(WGPUTexture texture);
void wgpuTextureViewSetLabel(WGPUTextureView textureView, WGPUStringView label);
void wgpuTextureViewAddRef(WGPUTextureView textureView);
void wgpuTextureViewRelease(WGPUTextureView textureView);


// Cleaned version of wgpu.h --------------------------------------------------
typedef enum WGPUNativeSType
{
WGPUSType_DeviceExtras = 0x00030001,
WGPUSType_NativeLimits = 0x00030002,
WGPUSType_PipelineLayoutExtras = 0x00030003,
WGPUSType_ShaderSourceGLSL = 0x00030004,
WGPUSType_InstanceExtras = 0x00030006,
WGPUSType_BindGroupEntryExtras = 0x00030007,
WGPUSType_BindGroupLayoutEntryExtras = 0x00030008,
WGPUSType_QuerySetDescriptorExtras = 0x00030009,
WGPUSType_SurfaceConfigurationExtras = 0x0003000A,
WGPUSType_SurfaceSourceSwapChainPanel = 0x0003000B,
WGPUSType_PrimitiveStateExtras = 0x0003000C,
WGPUNativeSType_Force32 = 0x7FFFFFFF
} WGPUNativeSType;
typedef enum WGPUNativeSurfaceGetCurrentTextureStatus
{
WGPUSurfaceGetCurrentTextureStatus_Occluded = 0x00030001,
WGPUNativeSurfaceGetCurrentTextureStatus_Force32 = 0x7FFFFFFF
} WGPUNativeSurfaceGetCurrentTextureStatus;
typedef enum WGPUNativeFeature
{
WGPUNativeFeature_Immediates = 0x00030001,
WGPUNativeFeature_TextureAdapterSpecificFormatFeatures = 0x00030002,
WGPUNativeFeature_MultiDrawIndirectCount = 0x00030004,
WGPUNativeFeature_VertexWritableStorage = 0x00030005,
WGPUNativeFeature_TextureBindingArray = 0x00030006,
WGPUNativeFeature_SampledTextureAndStorageBufferArrayNonUniformIndexing = 0x00030007,
WGPUNativeFeature_PipelineStatisticsQuery = 0x00030008,
WGPUNativeFeature_StorageResourceBindingArray = 0x00030009,
WGPUNativeFeature_PartiallyBoundBindingArray = 0x0003000A,
WGPUNativeFeature_TextureFormat16bitNorm = 0x0003000B,
WGPUNativeFeature_TextureCompressionAstcHdr = 0x0003000C,
WGPUNativeFeature_MappablePrimaryBuffers = 0x0003000E,
WGPUNativeFeature_BufferBindingArray = 0x0003000F,
WGPUNativeFeature_UniformBufferAndStorageTextureArrayNonUniformIndexing = 0x00030010,
WGPUNativeFeature_PolygonModeLine = 0x00030013,
WGPUNativeFeature_PolygonModePoint = 0x00030014,
WGPUNativeFeature_ConservativeRasterization = 0x00030015,
WGPUNativeFeature_SpirvShaderPassthrough = 0x00030017,
WGPUNativeFeature_VertexAttribute64bit = 0x00030019,
WGPUNativeFeature_TextureFormatNv12 = 0x0003001A,
WGPUNativeFeature_RayQuery = 0x0003001C,
WGPUNativeFeature_ShaderF64 = 0x0003001D,
WGPUNativeFeature_ShaderI16 = 0x0003001E,
WGPUNativeFeature_ShaderEarlyDepthTest = 0x00030020,
WGPUNativeFeature_Subgroup = 0x00030021,
WGPUNativeFeature_SubgroupVertex = 0x00030022,
WGPUNativeFeature_SubgroupBarrier = 0x00030023,
WGPUNativeFeature_TimestampQueryInsideEncoders = 0x00030024,
WGPUNativeFeature_TimestampQueryInsidePasses = 0x00030025,
WGPUNativeFeature_ShaderInt64 = 0x00030026,
WGPUNativeFeature_Force32 = 0x7FFFFFFF
} WGPUNativeFeature;
typedef enum WGPULogLevel
{
WGPULogLevel_Off = 0x00000000,
WGPULogLevel_Error = 0x00000001,
WGPULogLevel_Warn = 0x00000002,
WGPULogLevel_Info = 0x00000003,
WGPULogLevel_Debug = 0x00000004,
WGPULogLevel_Trace = 0x00000005,
WGPULogLevel_Force32 = 0x7FFFFFFF
} WGPULogLevel;
typedef WGPUFlags WGPUInstanceBackend;
static const WGPUInstanceBackend WGPUInstanceBackend_All = 0x00000000;
static const WGPUInstanceBackend WGPUInstanceBackend_Vulkan = 1 << 0;
static const WGPUInstanceBackend WGPUInstanceBackend_GL = 1 << 1;
static const WGPUInstanceBackend WGPUInstanceBackend_Metal = 1 << 2;
static const WGPUInstanceBackend WGPUInstanceBackend_DX12 = 1 << 3;
static const WGPUInstanceBackend WGPUInstanceBackend_BrowserWebGPU = 1 << 5;
static const WGPUInstanceBackend WGPUInstanceBackend_Primary = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 5);
static const WGPUInstanceBackend WGPUInstanceBackend_Secondary = (1 << 1);
static const WGPUInstanceBackend WGPUInstanceBackend_Force32 = 0x7FFFFFFF;
typedef WGPUFlags WGPUInstanceFlag;
static const WGPUInstanceFlag WGPUInstanceFlag_Empty = 0x00000000;
static const WGPUInstanceFlag WGPUInstanceFlag_Debug = 1 << 0;
static const WGPUInstanceFlag WGPUInstanceFlag_Validation = 1 << 1;
static const WGPUInstanceFlag WGPUInstanceFlag_DiscardHalLabels = 1 << 2;
static const WGPUInstanceFlag WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter = 1 << 3;
static const WGPUInstanceFlag WGPUInstanceFlag_GPUBasedValidation = 1 << 4;
static const WGPUInstanceFlag WGPUInstanceFlag_ValidationIndirectCall = 1 << 5;
static const WGPUInstanceFlag WGPUInstanceFlag_AutomaticTimestampNormalization = 1 << 6;
static const WGPUInstanceFlag WGPUInstanceFlag_Default = 1 << 24;
static const WGPUInstanceFlag WGPUInstanceFlag_Debugging = 1 << 25;
static const WGPUInstanceFlag WGPUInstanceFlag_AdvancedDebugging = 1 << 26;
static const WGPUInstanceFlag WGPUInstanceFlag_WithEnv = 1 << 27;
static const WGPUInstanceFlag WGPUInstanceFlag_Force32 = 0x7FFFFFFF;
typedef enum WGPUDx12Compiler
{
WGPUDx12Compiler_Undefined = 0x00000000,
WGPUDx12Compiler_Fxc = 0x00000001,
WGPUDx12Compiler_Dxc = 0x00000002,
WGPUDx12Compiler_Force32 = 0x7FFFFFFF
} WGPUDx12Compiler;
typedef enum WGPUGles3MinorVersion
{
WGPUGles3MinorVersion_Automatic = 0x00000000,
WGPUGles3MinorVersion_Version0 = 0x00000001,
WGPUGles3MinorVersion_Version1 = 0x00000002,
WGPUGles3MinorVersion_Version2 = 0x00000003,
WGPUGles3MinorVersion_Force32 = 0x7FFFFFFF
} WGPUGles3MinorVersion;
typedef enum WGPUPipelineStatisticName
{
WGPUPipelineStatisticName_VertexShaderInvocations = 0x00000000,
WGPUPipelineStatisticName_ClipperInvocations = 0x00000001,
WGPUPipelineStatisticName_ClipperPrimitivesOut = 0x00000002,
WGPUPipelineStatisticName_FragmentShaderInvocations = 0x00000003,
WGPUPipelineStatisticName_ComputeShaderInvocations = 0x00000004,
WGPUPipelineStatisticName_Force32 = 0x7FFFFFFF
} WGPUPipelineStatisticName;
typedef enum WGPUNativeQueryType
{
WGPUNativeQueryType_PipelineStatistics = 0x00030000,
WGPUNativeQueryType_Force32 = 0x7FFFFFFF
} WGPUNativeQueryType;
typedef enum WGPUDxcMaxShaderModel
{
WGPUDxcMaxShaderModel_V6_0 = 0x00000000,
WGPUDxcMaxShaderModel_V6_1 = 0x00000001,
WGPUDxcMaxShaderModel_V6_2 = 0x00000002,
WGPUDxcMaxShaderModel_V6_3 = 0x00000003,
WGPUDxcMaxShaderModel_V6_4 = 0x00000004,
WGPUDxcMaxShaderModel_V6_5 = 0x00000005,
WGPUDxcMaxShaderModel_V6_6 = 0x00000006,
WGPUDxcMaxShaderModel_V6_7 = 0x00000007,
WGPUDxcMaxShaderModel_Force32 = 0x7FFFFFFF
} WGPUDxcMaxShaderModel;
typedef enum WGPUGLFenceBehaviour
{
WGPUGLFenceBehaviour_Normal = 0x00000000,
WGPUGLFenceBehaviour_AutoFinish = 0x00000001,
WGPUGLFenceBehaviour_Force32 = 0x7FFFFFFF
} WGPUGLFenceBehaviour;
typedef enum WGPUDx12SwapchainKind
{
WGPUDx12SwapchainKind_Undefined = 0x00000000,
WGPUDx12SwapchainKind_DxgiFromHwnd = 0x00000001,
WGPUDx12SwapchainKind_DxgiFromVisual = 0x00000002,
WGPUDx12SwapchainKind_Force32 = 0x7FFFFFFF
} WGPUDx12SwapchainKind;
typedef enum WGPUNativeDisplayHandleType
{
WGPUNativeDisplayHandleType_None = 0x00000000,
WGPUNativeDisplayHandleType_Xlib = 0x00000001,
WGPUNativeDisplayHandleType_Xcb = 0x00000002,
WGPUNativeDisplayHandleType_Wayland = 0x00000003,
WGPUNativeDisplayHandleType_Force32 = 0x7FFFFFFF
} WGPUNativeDisplayHandleType;
typedef struct WGPUXlibDisplayHandle
{
void *display;
int screen;
} WGPUXlibDisplayHandle;
typedef struct WGPUXcbDisplayHandle
{
void *connection;
int screen;
} WGPUXcbDisplayHandle;
typedef struct WGPUWaylandDisplayHandle
{
void *display;
} WGPUWaylandDisplayHandle;
typedef struct WGPUNativeDisplayHandle
{
WGPUNativeDisplayHandleType type;
union
{
WGPUXlibDisplayHandle xlib;
WGPUXcbDisplayHandle xcb;
WGPUWaylandDisplayHandle wayland;
} data;
} WGPUNativeDisplayHandle;
typedef struct WGPUInstanceExtras
{
WGPUChainedStruct chain;
WGPUInstanceBackend backends;
WGPUInstanceFlag flags;
WGPUDx12Compiler dx12ShaderCompiler;
WGPUGles3MinorVersion gles3MinorVersion;
WGPUGLFenceBehaviour glFenceBehaviour;
WGPUStringView dxcPath;
WGPUDxcMaxShaderModel dxcMaxShaderModel;
WGPUDx12SwapchainKind dx12PresentationSystem;
const uint8_t *budgetForDeviceCreation;
const uint8_t *budgetForDeviceLoss;
WGPUNativeDisplayHandle displayHandle;
} WGPUInstanceExtras;
typedef struct WGPUDeviceExtras
{
WGPUChainedStruct chain;
WGPUStringView tracePath;
} WGPUDeviceExtras;
typedef struct WGPUNativeLimits
{
WGPUChainedStruct chain;
uint32_t maxImmediateSize;
uint32_t maxNonSamplerBindings;
uint32_t maxBindingArrayElementsPerShaderStage;
} WGPUNativeLimits;
typedef struct WGPUPipelineLayoutExtras
{
WGPUChainedStruct chain;
uint32_t immediateDataSize;
} WGPUPipelineLayoutExtras;
typedef uint64_t WGPUSubmissionIndex;
typedef struct WGPUShaderDefine
{
WGPUStringView name;
WGPUStringView value;
} WGPUShaderDefine;
typedef struct WGPUShaderSourceGLSL
{
WGPUChainedStruct chain;
WGPUShaderStage stage;
WGPUStringView code;
uint32_t defineCount;
WGPUShaderDefine const *defines;
} WGPUShaderSourceGLSL;
typedef struct WGPUShaderModuleDescriptorSpirV
{
WGPUStringView label;
uint32_t sourceSize;
uint32_t const *source;
} WGPUShaderModuleDescriptorSpirV;
typedef struct WGPURegistryReport
{
size_t numAllocated;
size_t numKeptFromUser;
size_t numReleasedFromUser;
size_t elementSize;
} WGPURegistryReport;
typedef struct WGPUHubReport
{
WGPURegistryReport adapters;
WGPURegistryReport devices;
WGPURegistryReport queues;
WGPURegistryReport pipelineLayouts;
WGPURegistryReport shaderModules;
WGPURegistryReport bindGroupLayouts;
WGPURegistryReport bindGroups;
WGPURegistryReport commandBuffers;
WGPURegistryReport renderBundles;
WGPURegistryReport renderPipelines;
WGPURegistryReport computePipelines;
WGPURegistryReport pipelineCaches;
WGPURegistryReport querySets;
WGPURegistryReport buffers;
WGPURegistryReport textures;
WGPURegistryReport textureViews;
WGPURegistryReport samplers;
} WGPUHubReport;
typedef struct WGPUGlobalReport
{
WGPURegistryReport surfaces;
WGPUHubReport hub;
} WGPUGlobalReport;
typedef struct WGPUInstanceEnumerateAdapterOptions
{
WGPUChainedStruct const *nextInChain;
WGPUInstanceBackend backends;
} WGPUInstanceEnumerateAdapterOptions;
typedef struct WGPUBindGroupEntryExtras
{
WGPUChainedStruct chain;
WGPUBuffer const *buffers;
size_t bufferCount;
WGPUSampler const *samplers;
size_t samplerCount;
WGPUTextureView const *textureViews;
size_t textureViewCount;
} WGPUBindGroupEntryExtras;
typedef struct WGPUBindGroupLayoutEntryExtras
{
WGPUChainedStruct chain;
uint32_t count;
} WGPUBindGroupLayoutEntryExtras;
typedef struct WGPUQuerySetDescriptorExtras
{
WGPUChainedStruct chain;
WGPUPipelineStatisticName const *pipelineStatistics;
size_t pipelineStatisticCount;
} WGPUQuerySetDescriptorExtras;
typedef struct WGPUSurfaceConfigurationExtras
{
WGPUChainedStruct chain;
uint32_t desiredMaximumFrameLatency;
} WGPUSurfaceConfigurationExtras;
typedef struct WGPUSurfaceSourceSwapChainPanel
{
WGPUChainedStruct chain;
void *panelNative;
} WGPUSurfaceSourceSwapChainPanel;
typedef enum WGPUPolygonMode
{
WGPUPolygonMode_Fill = 0,
WGPUPolygonMode_Line = 1,
WGPUPolygonMode_Point = 2,
} WGPUPolygonMode;
typedef struct WGPUPrimitiveStateExtras
{
WGPUChainedStruct chain;
WGPUPolygonMode polygonMode;
WGPUBool conservative;
} WGPUPrimitiveStateExtras;
typedef void (*WGPULogCallback)(WGPULogLevel level, WGPUStringView message, void *userdata);
typedef enum WGPUNativeTextureFormat
{
WGPUNativeTextureFormat_R16Unorm = 0x00030001,
WGPUNativeTextureFormat_R16Snorm = 0x00030002,
WGPUNativeTextureFormat_Rg16Unorm = 0x00030003,
WGPUNativeTextureFormat_Rg16Snorm = 0x00030004,
WGPUNativeTextureFormat_Rgba16Unorm = 0x00030005,
WGPUNativeTextureFormat_Rgba16Snorm = 0x00030006,
WGPUNativeTextureFormat_NV12 = 0x00030007,
WGPUNativeTextureFormat_P010 = 0x00030008,
} WGPUNativeTextureFormat;
void wgpuGenerateReport(WGPUInstance instance, WGPUGlobalReport *report);
size_t wgpuInstanceEnumerateAdapters(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const *options, WGPUAdapter *adapters);
WGPUSubmissionIndex wgpuQueueSubmitForIndex(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const *commands);
float wgpuQueueGetTimestampPeriod(WGPUQueue queue);
WGPUBool wgpuDevicePoll(WGPUDevice device, WGPUBool wait, WGPUSubmissionIndex const *submissionIndex);
WGPUShaderModule wgpuDeviceCreateShaderModuleSpirV(WGPUDevice device, WGPUShaderModuleDescriptorSpirV const *descriptor);
void wgpuSetLogCallback(WGPULogCallback callback, void *userdata);
void wgpuSetLogLevel(WGPULogLevel level);
uint32_t wgpuGetVersion(void);
void *wgpuDeviceGetNativeMetalDevice(WGPUDevice device);
void *wgpuQueueGetNativeMetalCommandQueue(WGPUQueue queue);
void *wgpuTextureGetNativeMetalTexture(WGPUTexture texture);
void wgpuRenderPassEncoderSetImmediates(WGPURenderPassEncoder encoder, uint32_t offset, uint32_t sizeBytes, void const *data);
void wgpuComputePassEncoderSetImmediates(WGPUComputePassEncoder encoder, uint32_t offset, uint32_t sizeBytes, void const *data);
void wgpuRenderBundleEncoderSetImmediates(WGPURenderBundleEncoder encoder, uint32_t offset, uint32_t sizeBytes, void const *data);
void wgpuRenderPassEncoderMultiDrawIndirect(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint32_t count);
void wgpuRenderPassEncoderMultiDrawIndexedIndirect(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint32_t count);
void wgpuRenderPassEncoderMultiDrawIndirectCount(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, WGPUBuffer count_buffer, uint64_t count_buffer_offset, uint32_t max_count);
void wgpuRenderPassEncoderMultiDrawIndexedIndirectCount(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, WGPUBuffer count_buffer, uint64_t count_buffer_offset, uint32_t max_count);
void wgpuComputePassEncoderBeginPipelineStatisticsQuery(WGPUComputePassEncoder computePassEncoder, WGPUQuerySet querySet, uint32_t queryIndex);
void wgpuComputePassEncoderEndPipelineStatisticsQuery(WGPUComputePassEncoder computePassEncoder);
void wgpuRenderPassEncoderBeginPipelineStatisticsQuery(WGPURenderPassEncoder renderPassEncoder, WGPUQuerySet querySet, uint32_t queryIndex);
void wgpuRenderPassEncoderEndPipelineStatisticsQuery(WGPURenderPassEncoder renderPassEncoder);
void wgpuComputePassEncoderWriteTimestamp(WGPUComputePassEncoder computePassEncoder, WGPUQuerySet querySet, uint32_t queryIndex);
void wgpuRenderPassEncoderWriteTimestamp(WGPURenderPassEncoder renderPassEncoder, WGPUQuerySet querySet, uint32_t queryIndex);
WGPUBool wgpuDeviceStartGraphicsDebuggerCapture(WGPUDevice device);
void wgpuDeviceStopGraphicsDebuggerCapture(WGPUDevice device);
