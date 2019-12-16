# wgpu.help('requestadapter', 'RequestAdapterOptions', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
async def requestAdapter(options: dict=None):
    """ Request an Adapter, the object that represents the implementation of WGPU.
    Use options (RequestAdapterOptions) to specify e.g. power preference.

    Before requesting an adapter, a wgpu backend should be selected. At the moment
    there is only one backend. Use ``import wgpu.rs`` to select it.
    """
    raise RuntimeError("Select a backend (by importing wgpu.rs) before requesting an adapter!")


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
        self._extensions = extensions

    @property
    def name(self):
        """ A human-readable name identifying the adapter.
        """
        return self._name

    @property
    def extensions(self):
        """ A GPUExtensions object which enumerates the extensions
        supported by the system, and whether each extension is supported
        by the underlying implementation.
        """
        return self._extensions

    # wgpu.help('adapterrequestdevice', 'DeviceDescriptor', dev=True)
    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    async def requestDevice(self, des: dict=None):
        """ Request a Device object. Use des (DeviceDescriptor) to specify
        a device label.
        """
        raise NotImplementedError()

    def foobar(self):
        pass



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
        """ A GPUExtensions object exposing the extensions with which this device was created.
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
    def createBuffer(self, des: dict):
        """ Create a Buffer object. Use des (BufferDescriptor) to specify
        buffer size and usage.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatebuffermapped', 'BufferDescriptor', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    def createBufferMapped(self, des: dict):
        """ Create a mapped buffer object.  Use des (BufferDescriptor) to specify
        buffer size and usage.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatebuffermappedasync', 'BufferDescriptor', dev=True)
    # IDL: Promise<GPUMappedBuffer> createBufferMappedAsync(GPUBufferDescriptor descriptor);
    async def createBufferMappedAsync(self, des: dict):
        """ Asynchronously create a mapped buffer object. Use des (BufferDescriptor) to specify
        buffer size and usage.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatetexture', 'TextureDescriptor', dev=True)
    # IDL: GPUTexture createTexture(GPUTextureDescriptor descriptor);
    def createTexture(self, des: dict):
        """ Create a Texture object. Use des (TextureDescriptor) to specify size,
        dimensions and more.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatesampler', 'SamplerDescriptor', dev=True)
    # IDL: GPUSampler createSampler(optional GPUSamplerDescriptor descriptor = {});
    def createSampler(self, des: dict):
        """ Create a Sampler object. Use des (SamplerDescriptor) to specify its modes.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatebindgrouplayout', 'BindGroupLayoutDescriptor', dev=True)
    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def createBindGroupLayout(self, des: dict):
        """ Create a GPUBindGroupLayout. Use makeBindGroupLayoutDescriptor() to define a list
        of bindings. Each binding (makeBindGroupLayoutBinding()) has:

        * binding: interger to tie things together
        * visibility: bitset to show in what shader stages the resource will be accessinble in.
        * type: A member of BindingType that indicates the intended usage of a resource binding.
        * textureDimension: Describes the dimensionality of texture view bindings.
        * multisampled: Indicates whether texture view bindings are multisampled.
        * hasDynamicOffset: For uniform-buffer, storage-buffer, and
          readonly-storage-buffer bindings, indicates that the binding has a
          dynamic offset. One offset must be passed to setBindGroup for each
          dynamic binding in increasing order of binding number.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreatepipelinelayout', 'PipelineLayoutDescriptor', dev=True)
    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor);
    def createPipelineLayout(self, des: dict):
        raise NotImplementedError()

    # wgpu.help('devicecreatebindgroup', 'BindGroupDescriptor', dev=True)
    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
    def createBindGroup(self, des: dict):
        """ Create a GPUBindGroup. Use makeBindGroupDescriptor() to define it.
        """
        raise NotImplementedError()

    # wgpu.help('devicecreateshadermodule', 'ShaderModuleDescriptor', dev=True)
    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor);
    def createShaderModule(self, des: dict):
        raise NotImplementedError()

    # wgpu.help('devicecreatecomputepipeline', 'ComputePipelineDescriptor', dev=True)
    # IDL: GPUComputePipeline createComputePipeline(GPUComputePipelineDescriptor descriptor);
    def createComputePipeline(self, des: dict):
        raise NotImplementedError()

    # wgpu.help('devicecreaterenderpipeline', 'RenderPipelineDescriptor', dev=True)
    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor);
    def createRenderPipeline(self, des: dict):
        raise NotImplementedError()

    # wgpu.help('devicecreatecommandencoder', 'CommandEncoderDescriptor', dev=True)
    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {});
    def createCommandEncoder(self, des: dict):
        raise NotImplementedError()

    # wgpu.help('devicecreaterenderbundleencoder', 'RenderBundleEncoderDescriptor', dev=True)
    # IDL: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor);
    def createRenderBundleEncoder(self, des: dict):
        raise NotImplementedError()



class GPUBuffer(GPUObject):
    """
    A GPUBuffer represents a block of memory that can be used in GPU
    operations. Data is stored in linear layout, meaning that each byte
    of the allocation can be addressed by its offset from the start of
    the buffer, subject to alignment restrictions depending on the
    operation. Some GPUBuffers can be mapped which makes the block of
    memory accessible via a numpy array called its mapping.
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

    # NOTE: this attribute is not specified by IDL, I think its still undecided how to expose the memory
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
    def createView(self):
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
        self._bindings = bindings


class GPUBindGroup(GPUObject):
    """
    """

class GPUPipelineLayout(GPUObject):
    """
    A GPUPipelineLayout describes the layout of a pipeline.
    """

    @property
    def layout(self):
        """ The GPUBindGroupLayout for this pipeline.
        """
        return self._layout

    @property
    def bindings(self):
        """
        """
        return self._bindings


class GPUShaderModule(GPUObject):
    """ A GPUShaderModule represents a programmable shader.
    """

class GPUComputePipleline(GPUObject):
    """
    """

class GPURenderPipleline(GPUObject):
    """
    A GPURenderPipleline represents a single pipeline to draw something
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
    def beginRenderPass(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderbegincomputepass', 'ComputePassDescriptor', dev=True)
    # IDL: GPUComputePassEncoder beginComputePass(optional GPUComputePassDescriptor descriptor = {});
    def beginComputePass(self):
        raise NotImplementedError()

    # wgpu.help('commandencodercopybuffertobuffer', 'Buffer', 'BufferSize', 'Buffer', 'BufferSize', 'BufferSize', dev=True)
    # IDL: void copyBufferToBuffer( GPUBuffer source, GPUBufferSize sourceOffset, GPUBuffer destination, GPUBufferSize destinationOffset, GPUBufferSize size);
    def copyBufferToBuffer(self):
        raise NotImplementedError()

    # wgpu.help('commandencodercopybuffertotexture', 'BufferCopyView', 'TextureCopyView', 'Extent3D', dev=True)
    # IDL: void copyBufferToTexture( GPUBufferCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copyBufferToTexture(self):
        raise NotImplementedError()

    # wgpu.help('commandencodercopytexturetobuffer', 'TextureCopyView', 'BufferCopyView', 'Extent3D', dev=True)
    # IDL: void copyTextureToBuffer( GPUTextureCopyView source, GPUBufferCopyView destination, GPUExtent3D copySize);
    def copyTextureToBuffer(self):
        raise NotImplementedError()

    # wgpu.help('commandencodercopytexturetotexture', 'TextureCopyView', 'TextureCopyView', 'Extent3D', dev=True)
    # IDL: void copyTextureToTexture( GPUTextureCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copyTextureToTexture(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def pushDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def popDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insertDebugMarker(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderfinish', 'CommandBufferDescriptor', dev=True)
    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {});
    def finish(self):
        raise NotImplementedError()


class GPUProgrammablePassEncoder(GPUObject):

    # wgpu.help('programmablepassencodersetbindgroup', 'BindGroup', dev=True)
    # IDL: void setBindGroup(unsigned long index, GPUBindGroup bindGroup,  Uint32Array dynamicOffsetsData,  unsigned long long dynamicOffsetsDataStart,  unsigned long long dynamicOffsetsDataLength);
    def setBindGroup(self):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def pushDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def popDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insertDebugMarker(self):
        raise NotImplementedError()



class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('computepassencodersetpipeline', 'ComputePipeline', dev=True)
    # IDL: void setPipeline(GPUComputePipeline pipeline);
    def setPipeline(self):
        raise NotImplementedError()

    # wgpu.help('computepassencoderdispatch', dev=True)
    # IDL: void dispatch(unsigned long x, optional unsigned long y = 1, optional unsigned long z = 1);
    def dispatch(self):
        raise NotImplementedError()

    # wgpu.help('computepassencoderdispatchindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void dispatchIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def dispatchIndirect(self):
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
    def setPipeline(self):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasesetindexbuffer', 'Buffer', 'BufferSize', dev=True)
    # IDL: void setIndexBuffer(GPUBuffer buffer, optional GPUBufferSize offset = 0);
    def setIndexBuffer(self):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasesetvertexbuffer', 'Buffer', 'BufferSize', dev=True)
    # IDL: void setVertexBuffer(unsigned long slot, GPUBuffer buffer, optional GPUBufferSize offset = 0);
    def setVertexBuffer(self):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedraw', dev=True)
    # IDL: void draw(unsigned long vertexCount, unsigned long instanceCount,  unsigned long firstVertex, unsigned long firstInstance);
    def draw(self):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void drawIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def drawIndirect(self):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindexed', dev=True)
    # IDL: void drawIndexed(unsigned long indexCount, unsigned long instanceCount,  unsigned long firstIndex, long baseVertex, unsigned long firstInstance);
    def drawIndexed(self):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindexedindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void drawIndexedIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def drawIndexedIndirect(self):
        raise NotImplementedError()


class GPURenderPassEncoder(GPURenderEncoderBase):
    """
    """

    # wgpu.help('renderpassencodersetviewport', dev=True)
    # IDL: void setViewport(float x, float y,  float width, float height,  float minDepth, float maxDepth);
    def setViewport(self):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetscissorrect', dev=True)
    # IDL: void setScissorRect(unsigned long x, unsigned long y, unsigned long width, unsigned long height);
    def setScissorRect(self):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetblendcolor', 'Color', dev=True)
    # IDL: void setBlendColor(GPUColor color);
    def setBlendColor(self):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetstencilreference', dev=True)
    # IDL: void setStencilReference(unsigned long reference);
    def setStencilReference(self):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderexecutebundles', dev=True)
    # IDL: void executeBundles(sequence<GPURenderBundle> bundles);
    def executeBundles(self):
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
    def finish(self):
        raise NotImplementedError()


class GPUQueue(GPUObject):

    # wgpu.help('queuesubmit', dev=True)
    # IDL: void submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self):
        raise NotImplementedError()

    # wgpu.help('queuecopyimagebitmaptotexture', 'ImageBitmapCopyView', 'TextureCopyView', 'Extent3D', dev=True)
    # IDL: void copyImageBitmapToTexture( GPUImageBitmapCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copyImageBitmapToTexture(self):
        raise NotImplementedError()


class GPUSwapChain(GPUObject):
    """
    """

    # wgpu.help('swapchaingetcurrenttexture', dev=True)
    # IDL: GPUTexture getCurrentTexture();
    def getCurrentTexture(self):
        raise NotImplementedError()
