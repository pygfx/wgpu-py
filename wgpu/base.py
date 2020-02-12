"""
The classes representing the wgpu API. This module defines the classes,
properties, methods and documentation. The actual methods are implemented
in backend modules.

Developer notes and tips:

* We follow the IDL spec, with the exception that where in JS the input args
  are provided via a dict, we use kwargs directly.
* However, some input args have subdicts (and sub-sub-dicts).
* For methods that are async in IDL, we also provide sync methods.
* The Async method names have an "Async" suffix.
* We will try hard not to rely on asyncio.

"""


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
def request_adapter(*, power_preference: "GPUPowerPreference"):
    """ Request a GPUAdapter, the object that represents the implementation of WGPU.
    Before requesting an adapter, a wgpu backend should be selected. At the moment
    there is only one backend. Use ``import wgpu.rs`` to select it.

    Params:
        powerPreference(enum): "high-performance" or "low-power"
    """
    raise RuntimeError(
        "Select a backend (by importing wgpu.rs) before requesting an adapter!"
    )


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
async def request_adapter_async(*, power_preference: "GPUPowerPreference"):
    """ Async version of ``request_adapter()``.
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

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


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

    # wgpu.help('DeviceDescriptor', 'adapterrequestdevice', dev=True)
    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    def request_device(
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

    # wgpu.help('DeviceDescriptor', 'adapterrequestdevice', dev=True)
    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    async def request_device_async(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        """ Async version of request_device().
        """
        raise NotImplementedError()


default_limits = dict(
    max_bind_groups=4,
    max_dynamic_uniform_buffers_per_pipeline_layout=8,
    max_dynamic_storage_buffers_per_pipeline_layout=4,
    max_sampled_textures_per_shader_stage=16,
    max_samplers_per_shader_stage=16,
    max_storage_buffers_per_shader_stage=4,
    max_storage_textures_per_shader_stage=4,
    max_uniform_buffers_per_shader_stage=12,
)


class GPUDevice(GPUObject):
    """
    A GPUDevice is the logical instantiation of an adapter, through which
    internal objects are created. It can be shared across threads.

    A device is the exclusive owner of all internal objects created
    from it: when the device is lost, all objects created from it become
    invalid.
    """

    def __init__(self, label, internal, adapter, extensions, limits, default_queue):
        super().__init__(label, internal, None)
        assert isinstance(adapter, GPUAdapter)
        self._adapter = adapter
        self._extensions = extensions
        self._limits = limits
        self._default_queue = default_queue

    @property
    def extensions(self):
        """ A GPUExtensions object exposing the extensions with which this device was
        created.
        """
        return self._extensions

    @property
    def limits(self):
        """ A dict exposing the limits with which this device was created.
        """
        return self._limits

    @property
    def default_queue(self):
        """ The default queue for this device.
        """
        return self._default_queue

    # wgpu.help('BufferDescriptor', 'devicecreatebuffer', dev=True)
    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
    def create_buffer(
        self, *, label="", size: "GPUSize64", usage: "GPUBufferUsageFlags"
    ):
        """ Create a Buffer object.
        """
        raise NotImplementedError()

    # wgpu.help('BufferDescriptor', 'devicecreatebuffermapped', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    def create_buffer_mapped(
        self, *, label="", size: "GPUSize64", usage: "GPUBufferUsageFlags"
    ):
        """ Create a mapped buffer object (its memory is accesable to the CPU).
        """
        raise NotImplementedError()

    # wgpu.help('BufferDescriptor', 'devicecreatebuffermapped', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    async def create_buffer_mapped_async(
        self, *, label="", size: "GPUSize64", usage: "GPUBufferUsageFlags"
    ):
        """ Asynchronously create a mapped buffer object.
        """
        raise NotImplementedError()

    # wgpu.help('TextureDescriptor', 'devicecreatetexture', dev=True)
    # IDL: GPUTexture createTexture(GPUTextureDescriptor descriptor);
    def create_texture(
        self,
        *,
        label="",
        size: "GPUExtent3D",
        array_layer_count: "GPUIntegerCoordinate" = 1,
        mip_level_count: "GPUIntegerCoordinate" = 1,
        sample_count: "GPUSize32" = 1,
        dimension: "GPUTextureDimension" = "2d",
        format: "GPUTextureFormat",
        usage: "GPUTextureUsageFlags",
    ):
        """ Create a Texture object.
        """
        raise NotImplementedError()

    # wgpu.help('SamplerDescriptor', 'devicecreatesampler', dev=True)
    # IDL: GPUSampler createSampler(optional GPUSamplerDescriptor descriptor = {});
    def create_sampler(
        self,
        *,
        label="",
        address_mode_u: "GPUAddressMode" = "clamp-to-edge",
        address_mode_v: "GPUAddressMode" = "clamp-to-edge",
        address_mode_w: "GPUAddressMode" = "clamp-to-edge",
        mag_filter: "GPUFilterMode" = "nearest",
        min_filter: "GPUFilterMode" = "nearest",
        mipmap_filter: "GPUFilterMode" = "nearest",
        lod_min_clamp: float = 0,
        lod_max_clamp: float = 0xFFFFFFFF,
        compare: "GPUCompareFunction" = "never",
    ):
        """ Create a Sampler object. Use des (SamplerDescriptor) to specify its modes.
        """
        raise NotImplementedError()

    # wgpu.help('BindGroupLayoutDescriptor', 'devicecreatebindgrouplayout', dev=True)
    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def create_bind_group_layout(
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

    # wgpu.help('BindGroupDescriptor', 'devicecreatebindgroup', dev=True)
    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
    def create_bind_group(
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

    # wgpu.help('PipelineLayoutDescriptor', 'devicecreatepipelinelayout', dev=True)
    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor);
    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "GPUBindGroupLayout-list"
    ):
        """ Create a GPUPipelineLayout, consisting of a list of GPUBindGroupLayout
        objects.
        """
        raise NotImplementedError()

    # wgpu.help('ShaderModuleDescriptor', 'devicecreateshadermodule', dev=True)
    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor);
    def create_shader_module(self, *, label="", code: "GPUShaderCode"):
        raise NotImplementedError()

    # wgpu.help('ComputePipelineDescriptor', 'devicecreatecomputepipeline', dev=True)
    # IDL: GPUComputePipeline createComputePipeline(GPUComputePipelineDescriptor descriptor);
    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        compute_stage: "GPUProgrammableStageDescriptor",
    ):
        raise NotImplementedError()

    # wgpu.help('RenderPipelineDescriptor', 'devicecreaterenderpipeline', dev=True)
    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor);
    def create_render_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        vertex_stage: "GPUProgrammableStageDescriptor",
        fragment_stage: "GPUProgrammableStageDescriptor",
        primitive_topology: "GPUPrimitiveTopology",
        rasterization_state: "GPURasterizationStateDescriptor" = {},
        color_states: "GPUColorStateDescriptor-list",
        depth_stencil_state: "GPUDepthStencilStateDescriptor",
        vertex_state: "GPUVertexStateDescriptor" = {},
        sample_count: "GPUSize32" = 1,
        sample_mask: "GPUSampleMask" = 0xFFFFFFFF,
        alpha_to_coverage_enabled: bool = False,
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

    # wgpu.help('CommandEncoderDescriptor', 'devicecreatecommandencoder', dev=True)
    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {});
    def create_command_encoder(self, *, label=""):
        raise NotImplementedError()

    # wgpu.help('RenderBundleEncoderDescriptor', 'devicecreaterenderbundleencoder', dev=True)
    # IDL: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor);
    def create_render_bundle_encoder(
        self,
        *,
        label="",
        color_formats: "GPUTextureFormat-list",
        depth_stencil_format: "GPUTextureFormat",
        sample_count: "GPUSize32" = 1,
    ):
        raise NotImplementedError()

    def _gui_configure_swap_chain(self, canvas, format, usage):
        """ Get a SwapChain object from a canvas object. Called by BaseCanvas.
        """
        raise NotImplementedError()


class GPUBuffer(GPUObject):
    """
    A GPUBuffer represents a block of memory that can be used in GPU
    operations. Data is stored in linear layout, meaning that each byte
    of the allocation can be addressed by its offset from the start of
    the buffer, subject to alignment restrictions depending on the
    operation. Some GPUBuffers can be mapped which makes the block of
    memory accessible via a ctypes array called its mapping.

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
    @property
    def mapping(self):
        """ The mapped memory of the buffer, exposed as a ctypes array.
        Can be cast to a ctypes array of appropriate type using
        ``your_array_type.from_buffer(b.mapping)``. Or use something like
        ``np.frombuffer(b.mapping, np.float32)`` to map it to a numpy array
        of appropriate dtype and shape.
        """
        return self._mapping

    # wgpu.help('buffermapreadasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapReadAsync();
    def map_read(self):
        raise NotImplementedError()

    # wgpu.help('buffermapreadasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapReadAsync();
    async def map_read_async(self):
        raise NotImplementedError()

    # wgpu.help('buffermapwriteasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapWriteAsync();
    def map_write(self):
        raise NotImplementedError()

    # wgpu.help('buffermapwriteasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapWriteAsync();
    async def map_write_async(self):
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

    # wgpu.help('TextureViewDescriptor', 'texturecreateview', dev=True)
    # IDL: GPUTextureView createView(optional GPUTextureViewDescriptor descriptor = {});
    def create_view(
        self,
        *,
        label="",
        format: "GPUTextureFormat",
        dimension: "GPUTextureViewDimension",
        aspect: "GPUTextureAspect" = "all",
        base_mip_level: "GPUIntegerCoordinate" = 0,
        mip_level_count: "GPUIntegerCoordinate" = 0,
        base_array_layer: "GPUIntegerCoordinate" = 0,
        array_layer_count: "GPUIntegerCoordinate" = 0,
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

    # wgpu.help('ComputePassDescriptor', 'commandencoderbegincomputepass', dev=True)
    # IDL: GPUComputePassEncoder beginComputePass(optional GPUComputePassDescriptor descriptor = {});
    def begin_compute_pass(self, *, label=""):
        raise NotImplementedError()

    # wgpu.help('RenderPassDescriptor', 'commandencoderbeginrenderpass', dev=True)
    # IDL: GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor descriptor);
    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "GPURenderPassColorAttachmentDescriptor-list",
        depth_stencil_attachment: "GPURenderPassDepthStencilAttachmentDescriptor",
    ):
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'commandencodercopybuffertobuffer', dev=True)
    # IDL: void copyBufferToBuffer( GPUBuffer source, GPUSize64 sourceOffset, GPUBuffer destination, GPUSize64 destinationOffset, GPUSize64 size);
    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        raise NotImplementedError()

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopybuffertotexture', dev=True)
    # IDL: void copyBufferToTexture( GPUBufferCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copy_buffer_to_texture(self, source, destination, copy_size):
        raise NotImplementedError()

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopytexturetobuffer', dev=True)
    # IDL: void copyTextureToBuffer( GPUTextureCopyView source, GPUBufferCopyView destination, GPUExtent3D copySize);
    def copy_texture_to_buffer(self, source, destination, copy_size):
        raise NotImplementedError()

    # wgpu.help('Extent3D', 'TextureCopyView', 'commandencodercopytexturetotexture', dev=True)
    # IDL: void copyTextureToTexture( GPUTextureCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copy_texture_to_texture(self, source, destination, copy_size):
        raise NotImplementedError()

    # wgpu.help('commandencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def push_debug_group(self, group_label):
        raise NotImplementedError()

    # wgpu.help('commandencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def pop_debug_group(self):
        raise NotImplementedError()

    # wgpu.help('commandencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insert_debug_marker(self, marker_label):
        raise NotImplementedError()

    # wgpu.help('CommandBufferDescriptor', 'commandencoderfinish', dev=True)
    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {});
    def finish(self, *, label=""):
        raise NotImplementedError()


class GPUProgrammablePassEncoder(GPUObject):
    # wgpu.help('BindGroup', 'Index32', 'Size64', 'programmablepassencodersetbindgroup', dev=True)
    # IDL: void setBindGroup(GPUIndex32 index, GPUBindGroup bindGroup,  Uint32Array dynamicOffsetsData,  GPUSize64 dynamicOffsetsDataStart,  GPUSize64 dynamicOffsetsDataLength);
    def set_bind_group(
        self,
        index,
        bind_group,
        dynamic_offsets_data,
        dynamic_offsets_data_start,
        dynamic_offsets_data_length,
    ):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def push_debug_group(self, group_label):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def pop_debug_group(self):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insert_debug_marker(self, marker_label):
        raise NotImplementedError()


class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('ComputePipeline', 'computepassencodersetpipeline', dev=True)
    # IDL: void setPipeline(GPUComputePipeline pipeline);
    def set_pipeline(self, pipeline):
        raise NotImplementedError()

    # wgpu.help('Size32', 'computepassencoderdispatch', dev=True)
    # IDL: void dispatch(GPUSize32 x, optional GPUSize32 y = 1, optional GPUSize32 z = 1);
    def dispatch(self, x, y, z):
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'computepassencoderdispatchindirect', dev=True)
    # IDL: void dispatchIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        raise NotImplementedError()

    # wgpu.help('computepassencoderendpass', dev=True)
    # IDL: void endPass();
    def end_pass(self):
        raise NotImplementedError()


class GPURenderEncoderBase(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('RenderPipeline', 'renderencoderbasesetpipeline', dev=True)
    # IDL: void setPipeline(GPURenderPipeline pipeline);
    def set_pipeline(self, pipeline):
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasesetindexbuffer', dev=True)
    # IDL: void setIndexBuffer(GPUBuffer buffer, optional GPUSize64 offset = 0);
    def set_index_buffer(self, buffer, offset):
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Index32', 'Size64', 'renderencoderbasesetvertexbuffer', dev=True)
    # IDL: void setVertexBuffer(GPUIndex32 slot, GPUBuffer buffer, optional GPUSize64 offset = 0);
    def set_vertex_buffer(self, slot, buffer, offset):
        raise NotImplementedError()

    # wgpu.help('Size32', 'renderencoderbasedraw', dev=True)
    # IDL: void draw(GPUSize32 vertexCount, GPUSize32 instanceCount,  GPUSize32 firstVertex, GPUSize32 firstInstance);
    def draw(self, vertex_count, instance_count, first_vertex, first_instance):
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasedrawindirect', dev=True)
    # IDL: void drawIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indirect(self, indirect_buffer, indirect_offset):
        raise NotImplementedError()

    # wgpu.help('SignedOffset32', 'Size32', 'renderencoderbasedrawindexed', dev=True)
    # IDL: void drawIndexed(GPUSize32 indexCount, GPUSize32 instanceCount,  GPUSize32 firstIndex, GPUSignedOffset32 baseVertex, GPUSize32 firstInstance);
    def draw_indexed(
        self, index_count, instance_count, first_index, base_vertex, first_instance
    ):
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasedrawindexedindirect', dev=True)
    # IDL: void drawIndexedIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        raise NotImplementedError()


class GPURenderPassEncoder(GPURenderEncoderBase):
    """
    """

    # wgpu.help('renderpassencodersetviewport', dev=True)
    # IDL: void setViewport(float x, float y,  float width, float height,  float minDepth, float maxDepth);
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        raise NotImplementedError()

    # wgpu.help('IntegerCoordinate', 'renderpassencodersetscissorrect', dev=True)
    # IDL: void setScissorRect(GPUIntegerCoordinate x, GPUIntegerCoordinate y,  GPUIntegerCoordinate width, GPUIntegerCoordinate height);
    def set_scissor_rect(self, x, y, width, height):
        raise NotImplementedError()

    # wgpu.help('Color', 'renderpassencodersetblendcolor', dev=True)
    # IDL: void setBlendColor(GPUColor color);
    def set_blend_color(self, color):
        raise NotImplementedError()

    # wgpu.help('StencilValue', 'renderpassencodersetstencilreference', dev=True)
    # IDL: void setStencilReference(GPUStencilValue reference);
    def set_stencil_reference(self, reference):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderexecutebundles', dev=True)
    # IDL: void executeBundles(sequence<GPURenderBundle> bundles);
    def execute_bundles(self, bundles):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderendpass', dev=True)
    # IDL: void endPass();
    def end_pass(self):
        raise NotImplementedError()


class GPURenderBundle(GPUObject):
    """
    """


class GPURenderBundleEncoder(GPURenderEncoderBase):
    """
    """

    # wgpu.help('RenderBundleDescriptor', 'renderbundleencoderfinish', dev=True)
    # IDL: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {});
    def finish(self, *, label=""):
        raise NotImplementedError()


class GPUQueue(GPUObject):
    # wgpu.help('queuesubmit', dev=True)
    # IDL: void submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self, command_buffers):
        raise NotImplementedError()

    # wgpu.help('Extent3D', 'ImageBitmapCopyView', 'TextureCopyView', 'queuecopyimagebitmaptotexture', dev=True)
    # IDL: void copyImageBitmapToTexture( GPUImageBitmapCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copy_image_bitmap_to_texture(self, source, destination, copy_size):
        raise NotImplementedError()


class GPUSwapChain(GPUObject):
    """
    """

    # wgpu.help('swapchaingetcurrenttexture', dev=True)
    # IDL: GPUTexture getCurrentTexture();
    def get_current_texture(self):
        """ For now, use get_current_texture_view.
        """
        raise NotImplementedError("Use get_current_texture_view() instead for now")

    def get_current_texture_view(self):
        """ NOTICE: this function is likely to change or be replaced by
        getCurrentTexture() at some point. An incompatibility between wgpu-native and
        WebGPU requires us to implement this workaround.
        """
        raise NotImplementedError()
