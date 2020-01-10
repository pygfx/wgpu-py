"""
The classes representing the wgpu API. This module defines the classes,
properties, methods and documentation. The actual methods are implemented
in backend modules.
"""


# wgpu.help('requestadapter', 'RequestAdapterOptions', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
async def requestAdapter(*, powerPreference: "GPUPowerPreference"):
    """ Request an Adapter, the object that represents the implementation of WGPU.
    Before requesting an adapter, a wgpu backend should be selected. At the moment
    there is only one backend. Use ``import wgpu.rs`` to select it.

    Params:
        powerPreference(enum): "high-performance" or "low-power"
    """
    raise RuntimeError(
        "Select a backend (by importing wgpu.rs) before requesting an adapter!"
    )


class GPUObject:
    """ The root class for all GPU classes.
    """

    def __init__(self, label, internal, device):
        self._label = label
        self._internal = internal  # The native/raw/real GPU object
        self._device = device

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.label}' at 0x{hex(id(self))}>"

    @property
    def label(self):
        """ A human-readable name identifying the GPU object.
        """
        return self._label


class DictLike:
    def __getitem__(self, name):
        return self.__dict__[name]


class GPUAdapter:  # Not a GPUObject
    """
    An adapter represents an implementation of WGPU on the system.
    Each adapter identifies both an instance of a hardware accelerator
    (e.g. GPU or CPU) and an instance of a implementation of WGPU on
    top of that accelerator.

    If an adapter becomes unavailable, it becomes invalid. Once invalid,
    it never becomes valid again.
    """

    def __init__(self, name, extensions):
        self._name = name
        self._extensions = tuple(extensions)

    @property
    def name(self):
        """ A human-readable name identifying the adapter.
        """
        return self._name

    @property
    def extensions(self):
        """ A tuple that enumerates the extensions supported by the adapter.
        """
        return self._extensions

    # wgpu.help('adapterrequestdevice', 'DeviceDescriptor', dev=True)
    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    async def requestDevice(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        """ Request a Device object.

        Params:
            extensions (list): the extensions that you need.
            limits (dict): the various limits that you need.
        """
        raise NotImplementedError()

    def foobar(self):
        pass


class GPULimits(DictLike):
    """ A dict-like object representing the device limits.
    """

    def __init__(
        self,
        *,
        maxBindGroups=4,
        maxDynamicUniformBuffersPerPipelineLayout=8,
        maxDynamicStorageBuffersPerPipelineLayout=4,
        maxSampledTexturesPerShaderStage=16,
        maxSamplersPerShaderStage=16,
        maxStorageBuffersPerShaderStage=4,
        maxStorageTexturesPerShaderStage=4,
        maxUniformBuffersPerShaderStage=12,
    ):
        self.maxBindGroups = maxBindGroups
        self.maxDynamicUniformBuffersPerPipelineLayout = (
            maxDynamicUniformBuffersPerPipelineLayout
        )
        self.maxDynamicStorageBuffersPerPipelineLayout = (
            maxDynamicStorageBuffersPerPipelineLayout
        )
        self.maxSampledTexturesPerShaderStage = maxSampledTexturesPerShaderStage
        self.maxSamplersPerShaderStage = maxSamplersPerShaderStage
        self.maxStorageBuffersPerShaderStage = maxStorageBuffersPerShaderStage
        self.maxStorageTexturesPerShaderStage = maxStorageTexturesPerShaderStage
        self.maxUniformBuffersPerShaderStage = (maxUniformBuffersPerShaderStage,)


class GPUDevice(GPUObject):
    """
    A GPUDevice is the logical instantiation of an adapter, through which
    internal objects are created. It can be shared across threads.

    A device is the exclusive owner of all internal objects created
    from it: when the device is lost, all objects created from it become
    invalid.
    """

    def __init__(self, label, internal, adapter, extensions, limits, defaultQueue):
        super().__init__(label, internal, None)
        assert isinstance(adapter, GPUAdapter)
        self._adapter = adapter
        self._extensions = extensions
        self._limits = limits
        self._defaultQueue = defaultQueue

    @property
    def extensions(self):
        """ A GPUExtensions object exposing the extensions with which this device was
        created.
        """
        return self._extensions

    @property
    def limits(self):
        """ A GPULimits object exposing the limits with which this device was created.
        """
        return self._limits

    @property
    def defaultQueue(self):
        """ The default queue for this device.
        """
        return self._defaultQueue

    # wgpu.help('devicecreatebuffer', 'BufferDescriptor', dev=True)
    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
    def createBuffer(
        self, *, label="", size: "GPUBufferSize", usage: "GPUBufferUsageFlags"
    ):
        """ Create a Buffer object.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatebuffermapped', 'BufferDescriptor', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    def createBufferMapped(
        self, *, label="", size: "GPUBufferSize", usage: "GPUBufferUsageFlags"
    ):
        """ Create a mapped buffer object (its memory is accesable to the CPU).
        """
        raise NotImplementedError()

    async def createBufferMappedAsync(self, des: dict):
        """ Asynchronously create a mapped buffer object.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatetexture', 'TextureDescriptor', dev=True)
    # IDL: GPUTexture createTexture(GPUTextureDescriptor descriptor);
    def createTexture(
        self,
        *,
        label="",
        size: "GPUExtent3D",
        arrayLayerCount: int = 1,
        mipLevelCount: int = 1,
        sampleCount: int = 1,
        dimension: "GPUTextureDimension" = "2d",
        format: "GPUTextureFormat",
        usage: "GPUTextureUsageFlags",
    ):
        """ Create a Texture object.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatesampler', 'SamplerDescriptor', dev=True)
    # IDL: GPUSampler createSampler(optional GPUSamplerDescriptor descriptor = {});
    def createSampler(
        self,
        *,
        label="",
        addressModeU: "GPUAddressMode" = "clamp-to-edge",
        addressModeV: "GPUAddressMode" = "clamp-to-edge",
        addressModeW: "GPUAddressMode" = "clamp-to-edge",
        magFilter: "GPUFilterMode" = "nearest",
        minFilter: "GPUFilterMode" = "nearest",
        mipmapFilter: "GPUFilterMode" = "nearest",
        lodMinClamp: float = 0,
        lodMaxClamp: float = 0xFFFFFFFF,
        compare: "GPUCompareFunction" = "never",
    ):
        """ Create a Sampler object. Use des (SamplerDescriptor) to specify its modes.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatebindgrouplayout', 'BindGroupLayoutDescriptor', dev=True)
    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def createBindGroupLayout(
        self, *, label="", bindings: "GPUBindGroupLayoutBinding-list"
    ):
        """ Create a GPUBindGroupLayout.

        * A Binding maps a buffer/texture/sampler/uniform to an integer slot.
        * A BindGroup represents a group of such bindings.
        * A BindGroupLayoutBinding is an abstract definition of a binding.
          It describes a single shader resource binding to be included in a
          GPUBindGroupLayout.
        * A BindGroupLayout represents a list of such abstract bindings.

        Each binding has:

        * binding: interger to tie things together
        * visibility: bitset to show in what shader stages the resource will be
          accessible in.
        * type: A member of BindingType that indicates the intended usage of a resource
          binding.
        * textureDimension: Describes the dimensionality of texture view bindings.
        * multisampled: Indicates whether texture view bindings are multisampled.
        * hasDynamicOffset: For uniform-buffer, storage-buffer, and
          readonly-storage-buffer bindings, indicates that the binding has a
          dynamic offset. One offset must be passed to setBindGroup for each
          dynamic binding in increasing order of binding number.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatebindgroup', 'BindGroupDescriptor', dev=True)
    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
    def createBindGroup(
        self,
        *,
        label="",
        layout: "GPUBindGroupLayout",
        bindings: "GPUBindGroupBinding-list",
    ):
        """ Create a GPUBindGroup. The list of bindings are GPUBindGroupBinding objects,
        representing a concrete binding.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatepipelinelayout', 'PipelineLayoutDescriptor', dev=True)
    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor);
    def createPipelineLayout(
        self, *, label="", bindGroupLayouts: "GPUBindGroupLayout-list"
    ):
        """ Create a GPUPipelineLayout, consisting of a list of GPUBindGroupLayout
        objects.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreateshadermodule', 'ShaderModuleDescriptor', dev=True)
    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor);
    def createShaderModule(self, *, label="", code: "GPUShaderCode"):
        raise NotImplementedError()

    # wgpu.help('devicecreatecomputepipeline', 'ComputePipelineDescriptor', dev=True)
    # IDL: GPUComputePipeline createComputePipeline(GPUComputePipelineDescriptor descriptor);
    def createComputePipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        computeStage: "GPUProgrammableStageDescriptor",
    ):
        raise NotImplementedError()

    # wgpu.help('devicecreaterenderpipeline', 'RenderPipelineDescriptor', dev=True)
    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor);
    def createRenderPipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        vertexStage: "GPUProgrammableStageDescriptor",
        fragmentStage: "GPUProgrammableStageDescriptor",
        primitiveTopology: "GPUPrimitiveTopology",
        rasterizationState: "GPURasterizationStateDescriptor" = {},
        colorStates: "GPUColorStateDescriptor-list",
        depthStencilState: "GPUDepthStencilStateDescriptor",
        vertexState: "GPUVertexStateDescriptor" = {},
        sampleCount: int = 1,
        sampleMask: int = 0xFFFFFFFF,
        alphaToCoverageEnabled: bool = False,
    ):
        """ Create a GPURenderPipeline object describing a render pipeline.

        Params:
            layout (GPUPipelineLayout): The layout (list of binding groups).
            vertexStage: ``{"module": vertex_shader, entry_point="main"}``
            fragmentStage: ``{"module": fragment_shader, entry_point="main"}``
            primitiveTopology (enum): wgpu.PrimitiveTopology
            rasterizationState (dict): see below.
            colorStates (list of dicts): see below.
            depthStencilState: TODO
            vertexState: ``{"indexFormat": wgpu.IndexFormat.uint32, "vertexBuffers": []}``
            sampleCount (int): set higher than one for subsampling.
            sampleMask (int): sample bitmask.
            alphaToCoverageEnabled (bool): wheher to anable alpha coverage.

        RasterizationState example dict:
        ```
        {
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
            "depth_bias": 0,
            "depth_bias_slope_scale": 0.0,
            "depth_bias_clamp": 0.0
        }
        ```

        ColorState example dict:
        ```
        {
            "format":,
            "alpha_blend": (wgpu.BlendFactor.One, wgpu.BlendFactor.zero, wgpu.BlendOperation.add),
            "colorBlend":(wgpu.BlendFactor.One, wgpu.BlendFactor.zero, wgpu.BlendOperation.add),
            "writeMask": wgpu.ColorWrite.ALL
        }
        ```
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatecommandencoder', 'CommandEncoderDescriptor', dev=True)
    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {});
    def createCommandEncoder(self, *, label=""):
        raise NotImplementedError()

    # wgpu.help('devicecreaterenderbundleencoder', 'RenderBundleEncoderDescriptor', dev=True)
    # IDL: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor);
    def createRenderBundleEncoder(
        self,
        *,
        label="",
        colorFormats: "GPUTextureFormat-list",
        depthStencilFormat: "GPUTextureFormat",
        sampleCount: int = 1,
    ):
        raise NotImplementedError()

    def _gui_configureSwapChain(self, canvas, format, usage):
        """ Get a swapchain object from a canvas object. Called by BaseCanvas.
        """
        raise NotImplementedError()


class GPUBuffer(GPUObject):
    """
    A GPUBuffer represents a block of memory that can be used in GPU
    operations. Data is stored in linear layout, meaning that each byte
    of the allocation can be addressed by its offset from the start of
    the buffer, subject to alignment restrictions depending on the
    operation. Some GPUBuffers can be mapped which makes the block of
    memory accessible via a numpy array called its mapping.

    Create a buffer using GPUDevice.createBuffer(), GPUDevice.createBufferMapped()
    or GPUDevice.createBufferAsync().
    """

    def __init__(self, label, internal, device, size, usage, state, mapping):
        super().__init__(label, internal, device)
        self._size = size
        self._usage = usage
        self._state = state
        self._mapping = mapping

    def __del__(self):
        self.destroy()

    @property
    def size(self):
        """ The length of the GPUBuffer allocation in bytes.
        """
        return self._size

    @property
    def usage(self):
        """ The allowed usages (int bitmap) for this GPUBuffer.
        """
        return self._usage

    @property
    def state(self):
        """ The current state of the GPUBuffer: "mapped" where the
        buffer is available for CPU operations, "unmapped" where the
        buffer is available for GPU operations, "destroyed", where the
        buffer is no longer available for any operations except destroy.
        """
        return self._state

    # NOTE: this attribute is not specified by IDL, I think its still undecided how to
    #       expose the memory
    # NOTE: could expose this as a numpy array, but I may like the idea to make
    #       wgpu independent of numpy, and you need to cast dtype and shape anyway ...
    @property
    def mapping(self):
        """ The mapped memory of the buffer, exposed as a ctypes array.
        Use something like ``np.frombuffer(b.mapping, np.float32)`` to map
        it to a numpy array of appropriate dtype and shape.
        """
        return self._mapping

    # wgpu.help('buffermapreadasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapReadAsync();
    async def mapReadAsync(self):
        raise NotImplementedError()

    # wgpu.help('buffermapwriteasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapWriteAsync();
    async def mapWriteAsync(self):
        raise NotImplementedError()

    # wgpu.help('bufferunmap', dev=True)
    # IDL: void unmap();
    def unmap(self):
        raise NotImplementedError()

    # wgpu.help('bufferdestroy', dev=True)
    # IDL: void destroy();
    def destroy(self):
        """ An application that no longer requires a GPUBuffer can choose
        to lose access to it before garbage collection by calling
        destroy().
        """
        raise NotImplementedError()


class GPUTexture(GPUObject):
    """
    """

    # wgpu.help('texturecreateview', 'TextureViewDescriptor', dev=True)
    # IDL: GPUTextureView createView(optional GPUTextureViewDescriptor descriptor = {});
    def createView(
        self,
        *,
        label="",
        format: "GPUTextureFormat",
        dimension: "GPUTextureViewDimension",
        aspect: "GPUTextureAspect" = "all",
        baseMipLevel: int = 0,
        mipLevelCount: int = 0,
        baseArrayLayer: int = 0,
        arrayLayerCount: int = 0,
    ):
        raise NotImplementedError()

    # wgpu.help('texturedestroy', dev=True)
    # IDL: void destroy();
    def destroy(self):
        raise NotImplementedError()


class GPUTextureView(GPUObject):
    """
    """


class GPUSampler(GPUObject):
    """
    """


class GPUBindGroupLayout(GPUObject):
    """
    A GPUBindGroupLayout defines the interface between a set of
    resources bound in a GPUBindGroup and their accessibility in shader
    stages.
    """

    def __init__(self, label, internal, device, bindings):
        super().__init__(label, internal, device)
        self._bindings = tuple(bindings)


class GPUBindGroup(GPUObject):
    """
    A GPUBindGroup represents a group of bindings, a link between a shader slot
    and a resource (sampler, texture-view, buffer).
    """

    def __init__(self, label, internal, device, bindings):
        super().__init__(label, internal, device)
        self._bindings = bindings


class GPUPipelineLayout(GPUObject):
    """
    A GPUPipelineLayout describes the layout of a pipeline.
    """

    def __init__(self, label, internal, device, layouts):
        super().__init__(label, internal, device)
        self._layouts = tuple(layouts)  # GPUBindGroupLayout objects


class GPUShaderModule(GPUObject):
    """ A GPUShaderModule represents a programmable shader.
    """


class GPUComputePipeline(GPUObject):
    """
    """


class GPURenderPipeline(GPUObject):
    """
    A GPURenderPipeline represents a single pipeline to draw something
    to a surface. This is where everything comes together.
    """


class GPUCommandBuffer(GPUObject):
    """
    """


class GPUCommandEncoder(GPUObject):
    """
    """

    # wgpu.help('commandencoderbeginrenderpass', 'RenderPassDescriptor', dev=True)
    # IDL: GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor descriptor);
    def beginRenderPass(
        self,
        *,
        label="",
        colorAttachments: "GPURenderPassColorAttachmentDescriptor-list",
        depthStencilAttachment: "GPURenderPassDepthStencilAttachmentDescriptor",
    ):
        raise NotImplementedError()

    # wgpu.help('commandencoderbegincomputepass', 'ComputePassDescriptor', dev=True)
    # IDL: GPUComputePassEncoder beginComputePass(optional GPUComputePassDescriptor descriptor = {});
    def beginComputePass(self, *, label=""):
        raise NotImplementedError()

    # wgpu.help('commandencodercopybuffertobuffer', 'Buffer', 'BufferSize', 'Buffer', 'BufferSize', 'BufferSize', dev=True)
    # IDL: void copyBufferToBuffer( GPUBuffer source, GPUBufferSize sourceOffset, GPUBuffer destination, GPUBufferSize destinationOffset, GPUBufferSize size);
    def copyBufferToBuffer(
        self, source, sourceOffset, destination, destinationOffset, size
    ):
        raise NotImplementedError()

    # wgpu.help('commandencodercopybuffertotexture', 'BufferCopyView', 'TextureCopyView', 'Extent3D', dev=True)
    # IDL: void copyBufferToTexture( GPUBufferCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copyBufferToTexture(self, source, destination, copySize):
        raise NotImplementedError()

    # wgpu.help('commandencodercopytexturetobuffer', 'TextureCopyView', 'BufferCopyView', 'Extent3D', dev=True)
    # IDL: void copyTextureToBuffer( GPUTextureCopyView source, GPUBufferCopyView destination, GPUExtent3D copySize);
    def copyTextureToBuffer(self, source, destination, copySize):
        raise NotImplementedError()

    # wgpu.help('commandencodercopytexturetotexture', 'TextureCopyView', 'TextureCopyView', 'Extent3D', dev=True)
    # IDL: void copyTextureToTexture( GPUTextureCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copyTextureToTexture(self, source, destination, copySize):
        raise NotImplementedError()

    # wgpu.help('commandencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def pushDebugGroup(self, groupLabel):
        raise NotImplementedError()

    # wgpu.help('commandencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def popDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insertDebugMarker(self, markerLabel):
        raise NotImplementedError()

    # wgpu.help('commandencoderfinish', 'CommandBufferDescriptor', dev=True)
    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {});
    def finish(self, *, label=""):
        raise NotImplementedError()


class GPUProgrammablePassEncoder(GPUObject):

    # wgpu.help('programmablepassencodersetbindgroup', 'BindGroup', dev=True)
    # IDL: void setBindGroup(unsigned long index, GPUBindGroup bindGroup,  Uint32Array dynamicOffsetsData,  unsigned long long dynamicOffsetsDataStart,  unsigned long long dynamicOffsetsDataLength);
    def setBindGroup(
        self,
        index,
        bindGroup,
        dynamicOffsetsData,
        dynamicOffsetsDataStart,
        dynamicOffsetsDataLength,
    ):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def pushDebugGroup(self, groupLabel):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def popDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insertDebugMarker(self, markerLabel):
        raise NotImplementedError()


class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('computepassencodersetpipeline', 'ComputePipeline', dev=True)
    # IDL: void setPipeline(GPUComputePipeline pipeline);
    def setPipeline(self, pipeline):
        raise NotImplementedError()

    # wgpu.help('computepassencoderdispatch', dev=True)
    # IDL: void dispatch(unsigned long x, optional unsigned long y = 1, optional unsigned long z = 1);
    def dispatch(self, x, y, z):
        raise NotImplementedError()

    # wgpu.help('computepassencoderdispatchindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void dispatchIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def dispatchIndirect(self, indirectBuffer, indirectOffset):
        raise NotImplementedError()

    # wgpu.help('computepassencoderendpass', dev=True)
    # IDL: void endPass();
    def endPass(self):
        raise NotImplementedError()


class GPURenderEncoderBase(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('renderencoderbasesetpipeline', 'RenderPipeline', dev=True)
    # IDL: void setPipeline(GPURenderPipeline pipeline);
    def setPipeline(self, pipeline):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasesetindexbuffer', 'Buffer', 'BufferSize', dev=True)
    # IDL: void setIndexBuffer(GPUBuffer buffer, optional GPUBufferSize offset = 0);
    def setIndexBuffer(self, buffer, offset):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasesetvertexbuffer', 'Buffer', 'BufferSize', dev=True)
    # IDL: void setVertexBuffer(unsigned long slot, GPUBuffer buffer, optional GPUBufferSize offset = 0);
    def setVertexBuffer(self, slot, buffer, offset):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedraw', dev=True)
    # IDL: void draw(unsigned long vertexCount, unsigned long instanceCount,  unsigned long firstVertex, unsigned long firstInstance);
    def draw(self, vertexCount, instanceCount, firstVertex, firstInstance):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void drawIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def drawIndirect(self, indirectBuffer, indirectOffset):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindexed', dev=True)
    # IDL: void drawIndexed(unsigned long indexCount, unsigned long instanceCount,  unsigned long firstIndex, long baseVertex, unsigned long firstInstance);
    def drawIndexed(
        self, indexCount, instanceCount, firstIndex, baseVertex, firstInstance
    ):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindexedindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void drawIndexedIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def drawIndexedIndirect(self, indirectBuffer, indirectOffset):
        raise NotImplementedError()


class GPURenderPassEncoder(GPURenderEncoderBase):
    """
    """

    # wgpu.help('renderpassencodersetviewport', dev=True)
    # IDL: void setViewport(float x, float y,  float width, float height,  float minDepth, float maxDepth);
    def setViewport(self, x, y, width, height, minDepth, maxDepth):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetscissorrect', dev=True)
    # IDL: void setScissorRect(unsigned long x, unsigned long y, unsigned long width, unsigned long height);
    def setScissorRect(self, x, y, width, height):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetblendcolor', 'Color', dev=True)
    # IDL: void setBlendColor(GPUColor color);
    def setBlendColor(self, color):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetstencilreference', dev=True)
    # IDL: void setStencilReference(unsigned long reference);
    def setStencilReference(self, reference):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderexecutebundles', dev=True)
    # IDL: void executeBundles(sequence<GPURenderBundle> bundles);
    def executeBundles(self, bundles):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderendpass', dev=True)
    # IDL: void endPass();
    def endPass(self):
        raise NotImplementedError()


class GPURenderBundle(GPUObject):
    """
    """


class GPURenderBundleEncoder(GPURenderEncoderBase):
    """
    """

    # wgpu.help('renderbundleencoderfinish', 'RenderBundleDescriptor', dev=True)
    # IDL: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {});
    def finish(self, *, label=""):
        raise NotImplementedError()


class GPUQueue(GPUObject):

    # wgpu.help('queuesubmit', dev=True)
    # IDL: void submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self, commandBuffers):
        raise NotImplementedError()

    # wgpu.help('queuecopyimagebitmaptotexture', 'ImageBitmapCopyView', 'TextureCopyView', 'Extent3D', dev=True)
    # IDL: void copyImageBitmapToTexture( GPUImageBitmapCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copyImageBitmapToTexture(self, source, destination, copySize):
        raise NotImplementedError()


class GPUSwapChain(GPUObject):
    """
    """

    # wgpu.help('swapchaingetcurrenttexture', dev=True)
    # IDL: GPUTexture getCurrentTexture();
    def getCurrentTexture(self):
        """ For now, use getCurrentTextureView.
        """
        raise NotImplementedError("Use getCurrentTextureView() instead for now")

    def getCurrentTextureView(self):
        """ NOTICE: this function is likely to change or be replaced by
        getCurrentTexture() at some point. An incompatibility between wgpu-native and
        WebGPU requires us to implement this workaround.
        """
        raise NotImplementedError()
