"""
The classes representing the wgpu API. This module defines the classes,
properties, methods and documentation. The actual methods are implemented
in backend modules.

Developer notes and tips:

* We follow the IDL spec, with the exception that where in JS the input args
  are provided via a dict, we use kwargs directly.
* However, some input args have subdicts (and sub-sub-dicts).
* For methods that are async in IDL, we also provide sync methods.
* The Async method names have an "_async" suffix.
* We will try hard not to rely on asyncio.

"""

import logging

logger = logging.getLogger("wgpu")


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
def request_adapter(*, canvas, power_preference: "GPUPowerPreference"):
    """ Get a :class:`GPUAdapter`, the object that represents an abstract wgpu
    implementation, from which one can request a :class:`GPUDevice`.

    Arguments:
        canvas (WgpuCanvasInterface): The canvas that the adapter should
            be able to render to (to create a swap chain for, to be precise).
            Can be None if you're not rendering to screen (or if you're
            confident that the returned adapter will work just fine).
        powerPreference(PowerPreference): "high-performance" or "low-power"
    """
    raise RuntimeError(
        "Select a backend (by importing wgpu.rs) before requesting an adapter!"
    )


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
async def request_adapter_async(*, canvas, power_preference: "GPUPowerPreference"):
    """ Async version of ``request_adapter()``.
    """
    raise RuntimeError(
        "Select a backend (by importing wgpu.rs) before requesting an adapter!"
    )  # no-cover


class GPUAdapter:  # Not a GPUObject
    """
    An adapter represents both an instance of a hardware accelerator
    (e.g. GPU or CPU) and an implementation of WGPU on top of that
    accelerator. If an adapter becomes unavailable, it becomes invalid.
    Once invalid, it never becomes valid again.
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
        """ A tuple that represents the extensions supported by the adapter.
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
        """ Request a :class:`GPUDevice` from the adapter.

        Arguments:
            label (str): A human readable label. Optional.
            extensions (list of str): the extensions that you need. Default [].
            limits (dict): the various limits that you need. Default {}.
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
        """ Async version of ``request_device()``.
        """
        raise NotImplementedError()

    def _destroy(self):
        pass

    def __del__(self):
        self._destroy()


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


class GPUObject:
    """ The base class for all GPU objects (the device and all objects
    belonging to a device).
    """

    def __init__(self, label, internal, device):
        self._label = label
        self._internal = internal  # The native/raw/real GPU object
        self._device = device
        logger.info(f"Creating {self.__class__.__name__} {label}")

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.label}' at 0x{hex(id(self))}>"

    @property
    def label(self):
        """ A human-readable name identifying the GPU object.
        """
        return self._label

    def _destroy(self):
        """ Subclasses can implement this to clean up.
        """
        pass

    def __del__(self):
        self._destroy()


class GPUDevice(GPUObject):
    """
    A device is the logical instantiation of an adapter, through which
    internal objects are created. It can be shared across threads.
    A device is the exclusive owner of all internal objects created
    from it: when the device is lost, all objects created from it become
    invalid.

    Create a device using :func:`GPUAdapter.request_device` or
    :func:`GPUAdapter.request_device_async`.
    """

    def __init__(self, label, internal, adapter, extensions, limits, default_queue):
        super().__init__(label, internal, None)
        assert isinstance(adapter, GPUAdapter)
        self._adapter = adapter
        self._extensions = tuple(sorted([str(x) for x in extensions]))
        self._limits = limits.copy()
        self._default_queue = default_queue

    @property
    def extensions(self):
        """ A tuple of strings representing the extensions with which this
        device was created.
        """
        return self._extensions

    @property
    def limits(self):
        """ A dict exposing the limits with which this device was created.
        """
        return self._limits

    @property
    def default_queue(self):
        """ The default :class:`GPUQueue` for this device.
        """
        return self._default_queue

    # wgpu.help('BufferDescriptor', 'devicecreatebuffer', dev=True)
    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
    def create_buffer(self, *, label="", size: int, usage: "GPUBufferUsageFlags"):
        """ Create a :class:`GPUBuffer` object.

        Arguments:
            label (str): A human readable label. Optional.
            size (int): The size of the buffer in bytes.
            usage (BufferUsageFlags): The ways in which this buffer will be used.
        """
        raise NotImplementedError()

    # wgpu.help('BufferDescriptor', 'devicecreatebuffermapped', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    def create_buffer_mapped(
        self, *, label="", size: int, usage: "GPUBufferUsageFlags"
    ):
        """ Create a :class:`GPUBuffer` object that is mapped from the start. It must
        be unmapped before using it in a pipeline.

        Arguments:
            label (str): A human readable label. Optional.
            size (int): The size of the buffer in bytes.
            usage (BufferUsageFlags): The ways in which this buffer will be used.
        """
        raise NotImplementedError()

    # wgpu.help('BufferDescriptor', 'devicecreatebuffermapped', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    async def create_buffer_mapped_async(
        self, *, label="", size: int, usage: "GPUBufferUsageFlags"
    ):
        """ Async version of ``create_buffer_mapped()``.
        """
        raise NotImplementedError()

    # wgpu.help('TextureDescriptor', 'devicecreatetexture', dev=True)
    # IDL: GPUTexture createTexture(GPUTextureDescriptor descriptor);
    def create_texture(
        self,
        *,
        label="",
        size: "GPUExtent3D",
        mip_level_count: "GPUIntegerCoordinate" = 1,
        sample_count: "GPUSize32" = 1,
        dimension: "GPUTextureDimension" = "2d",
        format: "GPUTextureFormat",
        usage: "GPUTextureUsageFlags",
    ):
        """ Create a :class:`GPUTexture` object.

        Arguments:
            label (str): A human readable label. Optional.
            size (tuple or dict): The texture size with fields (width, height, depth).
            mip_level_count (int): The number of mip leveles. Default 1.
            sample_count (int): The number of samples. Default 1.
            dimension (TextureDimension): The dimensionality of the texture. Default 2d.
            format (TextureFormat): What channels it stores and how.
            usage (TextureUsageFlags): The ways in which the texture will be used.
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
        compare: "GPUCompareFunction" = None,
    ):
        """ Create a :class:`GPUSampler` object. Samplers specify how a texture is sampled.

        Arguments:
            label (str): A human readable label. Optional.
            address_mode_u (AddressMode): What happens when sampling beyond the x edge.
                Default "clamp-to-edge".
            address_mode_v (AddressMode): What happens when sampling beyond the y edge.
                Default "clamp-to-edge".
            address_mode_w (AddressMode): What happens when sampling beyond the z edge.
                Default "clamp-to-edge".
            mag_filter (FilterMode): Interpolation when zoomed in. Default 'nearest'.
            min_filter (FilterMode): Interpolation when zoomed out. Default 'nearest'.
            mipmap_filter: (FilterMode): Interpolation between mip levels. Default 'nearest'.
            lod_min_clamp (float): The minimum level of detail. Default 0.
            lod_max_clamp (float): The maxium level of detail. Default inf.
            compare (CompareFunction): The sample compare operation for depth textures.
                Only specify this for depth textures. Default None.
        """
        raise NotImplementedError()

    # wgpu.help('BindGroupLayoutDescriptor', 'devicecreatebindgrouplayout', dev=True)
    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def create_bind_group_layout(
        self, *, label="", entries: "GPUBindGroupLayoutEntry-list"
    ):
        """ Create a :class:`GPUBindGroupLayout` object. One or more
        such objects are passed to :func:`create_pipeline_layout` to
        specify the (abstract) pipeline layout for resources. See the
        docs on bind groups for details.

        Arguments:
            label (str): A human readable label. Optional.
            entries (list of dict): A list of layout entry dicts.

        Example entry dict:

        .. code-block:: py

            # Buffer
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "type": wgpu.BindingType.storage_buffer,
                "has_dynamic_offset": False,  # optional
            },
            # Sampler
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "type": wgpu.BindingType.sampler,
            },
            # Sampled texture
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "type": wgpu.BindingType.sampled_texture,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "texture_component_type": wgpu.TextureComponentType.float,
                "multisampled": False,  # optional
            },
            # Storage texture
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "type": wgpu.BindingType.readonly_storage_texture,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "texture_component_type": wgpu.TextureComponentType.float,
                "storage_texture_format": wgpu.TextureFormat.r32float,
                "multisampled": False,  # optional
            },

        About ``has_dynamic_offset``: For uniform-buffer, storage-buffer, and
        readonly-storage-buffer bindings, it indicates whether the binding has a
        dynamic offset. One offset must be passed to ``set_bind_group`` for each
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
        entries: "GPUBindGroupEntry-list",
    ):
        """ Create a :class:`GPUBindGroup` object, which can be used in
        :func:`pass.set_bind_group() <GPUProgrammablePassEncoder.set_bind_group>`
        to attach a group of resources.

        Arguments:
            label (str): A human readable label. Optional.
            layout (GPUBindGroupLayout): The layout (abstract representation)
                for this bind group.
            entries (list of dict): A list of dicts, see below.

        Example entry dicts:

        .. code-block:: py

            # For a sampler
            {
                "binding" : 0,  # slot
                "resource": a_sampler,
            }
            # For a texture view
            {
                "binding" : 0,  # slot
                "resource": a_texture_view,
            }
            # For a buffer
            {
                "binding" : 0,  # slot
                "resource": {
                    "buffer": a_buffer,
                    "offset": 0,
                    "size": 812,
                }
            }
        """
        raise NotImplementedError()

    # wgpu.help('PipelineLayoutDescriptor', 'devicecreatepipelinelayout', dev=True)
    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor);
    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "GPUBindGroupLayout-list"
    ):
        """ Create a :class:`GPUPipelineLayout` object, which can be
        used in :func:`create_render_pipeline` or :func:`create_compute_pipeline`.

        Arguments:
            label (str): A human readable label. Optional.
            bind_group_layouts (list): A list of :class:`GPUBindGroupLayout` objects.
        """
        raise NotImplementedError()

    # wgpu.help('ShaderModuleDescriptor', 'devicecreateshadermodule', dev=True)
    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor);
    def create_shader_module(self, *, label="", code: str):
        """ Create a :class:`GPUShaderModule` object from shader source.

        Currently, only SpirV is supported. One can compile glsl shaders to
        SpirV ahead of time, or use the python-shader package to write shaders
        in Python.

        Arguments:
            label (str): A human readable label. Optional.
            code (bytes): The shadercode, as binary SpirV, or an object
                implementing ``to_spirv()`` or ``to_bytes()``.
        """
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
        """ Create a :class:`GPUComputePipeline` object.

        Arguments:
            label (str): A human readable label. Optional.
            layout (GPUPipelineLayout): object created with ``create_pipeline_layout()``.
            compute_stage (dict): E.g. ``{"module": shader_module, entry_point="main"}``.
        """
        raise NotImplementedError()

    # wgpu.help('RenderPipelineDescriptor', 'devicecreaterenderpipeline', dev=True)
    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor);
    def create_render_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        vertex_stage: "GPUProgrammableStageDescriptor",
        fragment_stage: "GPUProgrammableStageDescriptor" = None,
        primitive_topology: "GPUPrimitiveTopology",
        rasterization_state: "GPURasterizationStateDescriptor" = {},
        color_states: "GPUColorStateDescriptor-list",
        depth_stencil_state: "GPUDepthStencilStateDescriptor" = None,
        vertex_state: "GPUVertexStateDescriptor" = {},
        sample_count: "GPUSize32" = 1,
        sample_mask: "GPUSampleMask" = 0xFFFFFFFF,
        alpha_to_coverage_enabled: bool = False,
    ):
        """ Create a :class:`GPURenderPipeline` object.

        Arguments:
            label (str): A human readable label. Optional.
            layout (GPUPipelineLayout): A layout created with ``create_pipeline_layout()``.
            vertex_stage (dict): E.g. ``{"module": shader_module, entry_point="main"}``
            fragment_stage (dict): E.g. ``{"module": shader_module, entry_point="main"}``. Default None.
            primitive_topology (PrimitiveTopology): The topology, e.g. triangles or lines.
            rasterization_state (dict): Specify rasterization rules. See below. Default None.
            color_states (list of dict): Specify color blending rules. See below.
            depth_stencil_state (dict): Specify texture for depth and stencil. See below. Default None.
            vertex_state (dict): Specify index and vertex buffer info. See below.
            sample_count (int): Set higher than one for subsampling. Default 1.
            sample_mask (int): Sample bitmask. Default all ones.
            alpha_to_coverage_enabled (bool): Wheher to anable alpha coverage. Default False.

        In the example dicts below, the values that are marked as optional,
        the shown value is the default.

        Example rasterization state dict:

        .. code-block:: py

            {
                "front_face": wgpu.FrontFace.ccw,  # optional
                "cull_mode": wgpu.CullMode.none,  # optional
                "depth_bias": 0,  # optional
                "depth_bias_slope_scale": 0.0,  # optional
                "depth_bias_clamp": 0.0  # optional
            }

        Example color state dict:

        .. code-block:: py

            {
                "format": wgpu.TextureFormat.bgra8unorm_srgb,
                "alpha_blend": (
                    wgpu.BlendFactor.One,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "color_blend": (
                    wgpu.BlendFactor.One,
                    wgpu.BlendFactor.zero,
                    gpu.BlendOperation.add,
                ),
                "write_mask": wgpu.ColorWrite.ALL  # optional
            }

        Example depth-stencil state dict:

        .. code-block:: py

            {
                "format": wgpu.TextureFormat.depth24plus_stencil8,
                "depth_write_enabled": False,  # optional
                "depth_compare": wgpu.CompareFunction.always,  # optional
                "stencil_front": {  # optional
                    "compare": wgpu.CompareFunction.equal,
                    "fail_op": wgpu.StencilOperation.keep,
                    "depth_fail_op": wgpu.StencilOperation.keep,
                    "pass_op": wgpu.StencilOperation.keep,
                },
                "stencil_back": {  # optional
                    "compare": wgpu.CompareFunction.equal,
                    "fail_op": wgpu.StencilOperation.keep,
                    "depth_fail_op": wgpu.StencilOperation.keep,
                    "pass_op": wgpu.StencilOperation.keep,
                },
                "stencil_read_mask": 0xFFFFFFFF,  # optional
                "stencil_write_mask": 0xFFFFFFFF,  # optional
            }

        Example vertex state dict:

        .. code-block:: py

            {
                "indexFormat": wgpu.IndexFormat.uint32,
                "vertexBuffers": [
                    {
                        "array_stride": 8,
                        "step_mode": wgpu.InputStepMode.vertex,  # optional
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float2,
                                "offset": 0,
                                "shader_location": 0,
                            },
                            ...
                        ],
                    },
                    ...
                ]
            }
        """
        raise NotImplementedError()

    # wgpu.help('CommandEncoderDescriptor', 'devicecreatecommandencoder', dev=True)
    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {});
    def create_command_encoder(self, *, label=""):
        """ Create a :class:`GPUCommandEncoder` object. A command
        encoder is used to record commands, which can then be submitted
        at once to the GPU.

        Arguments:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()

    # wgpu.help('RenderBundleEncoderDescriptor', 'devicecreaterenderbundleencoder', dev=True)
    # IDL: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor);
    def create_render_bundle_encoder(
        self,
        *,
        label="",
        color_formats: "GPUTextureFormat-list",
        depth_stencil_format: "GPUTextureFormat" = None,
        sample_count: "GPUSize32" = 1,
    ):
        """ Create a :class:`GPURenderBundle` object.

        TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    def configure_swap_chain(self, canvas, format, usage=None):
        """ Get a :class:`GPUSwapChain` object for the given canvas.
        In the WebGPU spec this is a method of the canvas. In wgpu-py
        it's a method of the device.

        Parameters:
            canvas (WgpuCanvasInterface): An object implementing the canvas interface.
            format (TextureFormat): The texture format, e.g. "bgra8unorm-srgb".
            usage (TextureUsage): Default ``TextureUsage.OUTPUT_ATTACHMENT``.
        """
        # This was made a method of device to help decouple the canvas
        # implementation from the wgpu API.
        raise NotImplementedError()

    def get_swap_chain_preferred_format(self, canvas):
        """ Get the preferred swap chain format. In the WebGPU spec
        this is a method of the canvas. In wgpu-py it's a method of the
        device.
        """
        return "bgra8unorm-srgb"  # seems to be a good default


class GPUBuffer(GPUObject):
    """
    A GPUBuffer represents a block of memory that can be used in GPU
    operations. Data is stored in linear layout, meaning that each byte
    of the allocation can be addressed by its offset from the start of
    the buffer, subject to alignment restrictions depending on the
    operation.

    Create a buffer using :func:`GPUDevice.create_buffer`,
    :func:`GPUDevice.create_buffer_mapped` or :func:`GPUDevice.create_buffer_mapped_async`.

    One can sync data in a buffer by mapping it (or by creating a mapped
    buffer) and then setting/getting the values in the mapped array.
    Alternatively, one can tell the GPU (via the command encoder) to
    copy data between buffers and textures.
    """

    def __init__(self, label, internal, device, size, usage, state, mapping):
        super().__init__(label, internal, device)
        self._size = size
        self._usage = usage
        self._state = state
        self._mapping = mapping

    @property
    def size(self):
        """ The length of the GPUBuffer allocation in bytes.
        """
        return self._size

    @property
    def usage(self):
        """ The allowed usages (int bitmap) for this GPUBuffer, specifying
        e.g. whether the buffer may be used as a vertex buffer, uniform buffer,
        target or source for copying data, etc.
        """
        return self._usage

    @property
    def state(self):
        """ The current state of the GPUBuffer:

        * "mapped" when the buffer is available for CPU operations.
        * "unmapped" when the buffer is available for GPU operations.
        * "destroyed", when the buffer is no longer available for any
          operations except destroy.
        """
        return self._state

    # NOTE: this attribute is not specified by IDL, I think its still undecided how to
    #       expose the memory
    @property
    def mapping(self):
        """ The mapped memory of the buffer, exposed as a ctypes array.
        Is only not ``None`` when the buffer is mapped. Can be cast to a
        ctypes array of appropriate type using
        ``your_array_type.from_buffer(b.mapping)``. Or use something
        like ``np.frombuffer(b.mapping, np.float32)`` to map it to a
        numpy array of appropriate dtype and shape.
        """
        return self._mapping

    # wgpu.help('buffermapreadasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapReadAsync();
    def map_read(self):
        """ Make the buffer memory accessable to the CPU for reading.
        Sets the ``mapping`` property and returns the mapped memory as
        a ctypes array.
        """
        raise NotImplementedError()

    # wgpu.help('buffermapreadasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapReadAsync();
    async def map_read_async(self):
        """ Async version of ``map_read()``.
        """
        raise NotImplementedError()

    # wgpu.help('buffermapwriteasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapWriteAsync();
    def map_write(self):
        """ Make the buffer memory accessable to the CPU for writing.
        Sets the ``mapping`` property and returns the mapped memory as
        a ctypes array.
        """
        raise NotImplementedError()

    # wgpu.help('buffermapwriteasync', dev=True)
    # IDL: Promise<ArrayBuffer> mapWriteAsync();
    async def map_write_async(self):
        """ Async version of ``map_write()``.
        """
        raise NotImplementedError()

    # wgpu.help('bufferunmap', dev=True)
    # IDL: void unmap();
    def unmap(self):
        """ Unmap the buffer so that it can be used in a GPU pipeline.
        """
        raise NotImplementedError()

    # wgpu.help('bufferdestroy', dev=True)
    # IDL: void destroy();
    def destroy(self):
        """ An application that no longer requires a buffer can choose
        to destroy it. Note that this is automatically called when the
        Python object is cleaned up by the garbadge collector.
        """
        raise NotImplementedError()


class GPUTexture(GPUObject):
    """
    A texture represents a 1D, 2D or 3D color image object. It also can have mipmaps
    (different levels of varying detail), and arrays. The texture represents
    the "raw" data. A :class:`GPUTextureView` is used to define how the texture data
    should be interpreted.

    Create a texture using :func:`GPUDevice.create_texture`.
    """

    # wgpu.help('TextureViewDescriptor', 'texturecreateview', dev=True)
    # IDL: GPUTextureView createView(optional GPUTextureViewDescriptor descriptor = {});
    def create_view(
        self,
        *,
        label="",
        format: "GPUTextureFormat" = None,
        dimension: "GPUTextureViewDimension" = None,
        aspect: "GPUTextureAspect" = "all",
        base_mip_level: "GPUIntegerCoordinate" = 0,
        mip_level_count: "GPUIntegerCoordinate" = 0,
        base_array_layer: "GPUIntegerCoordinate" = 0,
        array_layer_count: "GPUIntegerCoordinate" = 0,
    ):
        """ Create a :class:`GPUTextureView` object.

        If no aguments are given, a default view is given, with the
        same format and dimension as the texture.

        Arguments:
            label (str): A human readable label. Optional.
            format (TextureFormat): What channels it stores and how.
            dimension (TextureViewDimension): The dimensionality of the texture view.
            aspect (TextureAspect): Whether this view is used for depth, stencil, or all.
                Default all.
            base_mip_level (int): The starting mip level. Default 0.
            mip_level_count (int): The number of mip levels. Default 0.
            base_array_layer (int): The starting array layer. Default 0.
            array_layer_count (int): The number of array layers. Default 0.
        """
        raise NotImplementedError()

    # wgpu.help('texturedestroy', dev=True)
    # IDL: void destroy();
    def destroy(self):
        """ An application that no longer requires a texture can choose
        to destroy it. Note that this is automatically called when the
        Python object is cleaned up by the garbadge collector.
        """
        raise NotImplementedError()


class GPUTextureView(GPUObject):
    """
    A texture view represents a way to represent a :class:`GPUTexture`.

    Create a texture view using :func:`GPUTexture.create_view`.
    """


class GPUSampler(GPUObject):
    """
    A sampler specifies how a texture (view) must be sampled by the shader,
    in terms of subsampling, sampling between mip levels, and sampling out
    of the image boundaries.

    Create a sampler using :func:`GPUDevice.create_sampler`.
    """


class GPUBindGroupLayout(GPUObject):
    """
    A bind group layout defines the interface between a set of
    resources bound in a :class:`GPUBindGroup` and their accessibility in shader
    stages.

    Create a bind group layout using :func:`GPUDevice.create_bind_group_layout`.
    """

    def __init__(self, label, internal, device, bindings):
        super().__init__(label, internal, device)
        self._bindings = tuple(bindings)


class GPUBindGroup(GPUObject):
    """
    A bind group represents a group of bindings, the shader slot,
    and a resource (sampler, texture-view, buffer).

    Create a bind group using :func:`GPUDevice.create_bind_group`.
    """

    def __init__(self, label, internal, device, bindings):
        super().__init__(label, internal, device)
        self._bindings = bindings


class GPUPipelineLayout(GPUObject):
    """
    A pipeline layout describes the layout of a pipeline, as a list
    of :class:`GPUBindGroupLayout` objects.

    Create a pipeline layout using :func:`GPUDevice.create_pipeline_layout`.
    """

    def __init__(self, label, internal, device, layouts):
        super().__init__(label, internal, device)
        self._layouts = tuple(layouts)  # GPUBindGroupLayout objects


class GPUShaderModule(GPUObject):
    """
    A shader module represents a programmable shader.

    Create a shader module using :func:`GPUDevice.create_shader_module`.
    """


class GPUComputePipeline(GPUObject):
    """
    A compute pipeline represents a single pipeline for computations (no rendering).

    Create a compute pipeline using :func:`GPUDevice.create_compute_pipeline`.
    """


class GPURenderPipeline(GPUObject):
    """
    A render pipeline represents a single pipeline to draw something
    using a vertex and a fragment shader. The render target can come
    from a window on the screen or from an in-memory texture (off-screen
    rendering).

    Create a render pipeline using :func:`GPUDevice.create_render_pipeline`.
    """


class GPUCommandBuffer(GPUObject):
    """
    A command buffer stores a series of commands, generated by a
    :class:`GPUCommandEncoder`, to be submitted to a :class:`GPUQueue`.

    Create a command buffer using :func:`GPUCommandEncoder.finish`.
    """


class GPUCommandEncoder(GPUObject):
    """
    A command encoder is used to record a series of commands. When done,
    call :func:`finish` to obtain a GPUCommandBuffer object.

    Create a command encoder using :func:`GPUDevice.create_command_encoder`.
    """

    # wgpu.help('ComputePassDescriptor', 'commandencoderbegincomputepass', dev=True)
    # IDL: GPUComputePassEncoder beginComputePass(optional GPUComputePassDescriptor descriptor = {});
    def begin_compute_pass(self, *, label=""):
        """ Record the beginning of a compute pass. Returns a
        :class:`GPUComputePassEncoder` object.

        Parameters:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()

    # wgpu.help('RenderPassDescriptor', 'commandencoderbeginrenderpass', dev=True)
    # IDL: GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor descriptor);
    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "GPURenderPassColorAttachmentDescriptor-list",
        depth_stencil_attachment: "GPURenderPassDepthStencilAttachmentDescriptor" = None,
        occlusion_query_set: "GPUQuerySet" = None,
    ):
        """ Record the beginning of a render pass. Returns a
        :class:`GPURenderPassEncoder` object.

        Arguments:
            label (str): A human readable label. Optional.
            color_attachements (list of dict): List of color attachement dicts. See below.
            depth_stencil_attachment (dict): A depth stencil attachement dict. See below. Default None.
            occlusion_query_set: Default None. TODO NOT IMPLEMENTED in wgpu-native.

        Example color attachement:

        .. code-block:: py

            {
                "attachement": texture_view,
                "resolve_target": None,  # optional
                "load_value": (0, 0, 0, 0),  # LoadOp.load or a color
                "store_op": wgpu.StoreOp.store,  # optional
            }

        Example depth stencil attachement:

        .. code-block:: py

            {
                "attachment": texture_view,
                "depth_load_value": 0.0,
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_load_value": wgpu.LoadOp.load,
                "stencil_store_op": wgpu.StoreOp.store,
            }
        """
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'commandencodercopybuffertobuffer', dev=True)
    # IDL: void copyBufferToBuffer( GPUBuffer source, GPUSize64 sourceOffset, GPUBuffer destination, GPUSize64 destinationOffset, GPUSize64 size);
    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        """ Copy the contents of a buffer to another buffer.

        Arguments:
            source (GPUBuffer): The source buffer.
            source_offset (int): The byte offset.
            destination (GPUBuffer): The target buffer.
            destination_offset (int): The byte offset in the destination buffer.
            size (int): The number of bytes to copy.
        """
        raise NotImplementedError()

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopybuffertotexture', dev=True)
    # IDL: void copyBufferToTexture( GPUBufferCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copy_buffer_to_texture(self, source, destination, copy_size):
        """ Copy the contents of a buffer to a texture (view).

        Arguments:
            source (GPUBuffer): A dict with fields: buffer, offset, bytes_per_row, rows_per_image.
            destination (GPUTexture): A dict with fields: texture, mip_level, array_layer, origin.
            copy_size (int): The number of bytes to copy.
        """
        raise NotImplementedError()

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopytexturetobuffer', dev=True)
    # IDL: void copyTextureToBuffer( GPUTextureCopyView source, GPUBufferCopyView destination, GPUExtent3D copySize);
    def copy_texture_to_buffer(self, source, destination, copy_size):
        """ Copy the contents of a texture (view) to a buffer.

        Arguments:
            source (GPUTexture): A dict with fields: texture, mip_level, array_layer, origin.
            destination (GPUBuffer):  A dict with fields: buffer, offset, bytes_per_row, rows_per_image.
            copy_size (int): The number of bytes to copy.
        """
        raise NotImplementedError()

    # wgpu.help('Extent3D', 'TextureCopyView', 'commandencodercopytexturetotexture', dev=True)
    # IDL: void copyTextureToTexture( GPUTextureCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copy_texture_to_texture(self, source, destination, copy_size):
        """ Copy the contents of a texture (view) to another texture (view).

        Arguments:
            source (GPUTexture): A dict with fields: texture, mip_level, array_layer, origin.
            destination (GPUTexture):  A dict with fields: texture, mip_level, array_layer, origin.
            copy_size (int): The number of bytes to copy.
        """
        raise NotImplementedError()

    # wgpu.help('commandencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def push_debug_group(self, group_label):
        """ TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # wgpu.help('commandencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def pop_debug_group(self):
        """ TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # wgpu.help('commandencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insert_debug_marker(self, marker_label):
        """ TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # wgpu.help('CommandBufferDescriptor', 'commandencoderfinish', dev=True)
    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {});
    def finish(self, *, label=""):
        """ Finish recording. Returns a :class:`GPUCommandBuffer` to
        submit to a :class:`GPUQueue`.

        Parameters:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()


class GPUProgrammablePassEncoder(GPUObject):
    """
    Base class for the different pass encoder classes.
    """

    # wgpu.help('BindGroup', 'Index32', 'Size32', 'Size64', 'programmablepassencodersetbindgroup', dev=True)
    # IDL: void setBindGroup(GPUIndex32 index, GPUBindGroup bindGroup,  Uint32Array dynamicOffsetsData,  GPUSize64 dynamicOffsetsDataStart,  GPUSize32 dynamicOffsetsDataLength);
    def set_bind_group(
        self,
        index,
        bind_group,
        dynamic_offsets_data,
        dynamic_offsets_data_start,
        dynamic_offsets_data_length,
    ):
        """ Associate the given bind group (i.e. group or resources) with the
        given slot/index.

        Arguments:
            index (int): The slot to bind at.
            bind_group (GPUBindGroup): The bind group to bind.
            dynamic_offsets_data (list of int): A list of offsets (one for each bind group).
            dynamic_offsets_data_start (int): Not used.
            dynamic_offsets_data_length (int): Not used.
        """
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def push_debug_group(self, group_label):
        """ Push a named debug group into the command stream.
        """
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def pop_debug_group(self):
        """ Pop the active debug group.
        """
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insert_debug_marker(self, marker_label):
        """ Insert the given message into the debug message queue.
        """
        raise NotImplementedError()


class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    A compute-pass encoder records commands related to a compute pass.

    Create a compute pass encoder using :func:`GPUCommandEncoder.begin_compute_pass`.
    """

    # wgpu.help('ComputePipeline', 'computepassencodersetpipeline', dev=True)
    # IDL: void setPipeline(GPUComputePipeline pipeline);
    def set_pipeline(self, pipeline):
        """ Set the pipeline for this compute pass.

        Arguments:
            pipeline (GPUComputePipeline): The pipeline to use.
        """
        raise NotImplementedError()

    # wgpu.help('Size32', 'computepassencoderdispatch', dev=True)
    # IDL: void dispatch(GPUSize32 x, optional GPUSize32 y = 1, optional GPUSize32 z = 1);
    def dispatch(self, x, y=1, z=1):
        """ Run the compute shader.

        Arguments:
            x (int): The number of cycles in index x.
            y (int): The number of cycles in index y. Default 1.
            z (int): The number of cycles in index z. Default 1.
        """
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'computepassencoderdispatchindirect', dev=True)
    # IDL: void dispatchIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        """ Like ``dispatch()``, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()

    # wgpu.help('computepassencoderendpass', dev=True)
    # IDL: void endPass();
    def end_pass(self):
        """ Record the end of the compute pass.
        """
        raise NotImplementedError()


class GPURenderEncoderBase(GPUProgrammablePassEncoder):
    """
    Base class for different render-pass encoder classes.
    """

    # wgpu.help('RenderPipeline', 'renderencoderbasesetpipeline', dev=True)
    # IDL: void setPipeline(GPURenderPipeline pipeline);
    def set_pipeline(self, pipeline):
        """ Set the pipeline for this render pass.

        Arguments:
            pipeline (GPURenderPipeline): The pipeline to use.
        """
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasesetindexbuffer', dev=True)
    # IDL: void setIndexBuffer(GPUBuffer buffer, optional GPUSize64 offset = 0, optional GPUSize64 size = 0);
    def set_index_buffer(self, buffer, offset=0, size=0):
        """ Set the index buffer for this render pass.

        Arguments:
            buffer (GPUBuffer): The buffer that contains the indices.
            offset (int): The byte offset in the buffer. Default 0.
            size (int): The number of bytes to use. Default 0.
        """
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Index32', 'Size64', 'renderencoderbasesetvertexbuffer', dev=True)
    # IDL: void setVertexBuffer(GPUIndex32 slot, GPUBuffer buffer, optional GPUSize64 offset = 0, optional GPUSize64 size = 0);
    def set_vertex_buffer(self, slot, buffer, offset=0, size=0):
        """ Associate a vertex buffer with a bind slot.

        Arguments:
            slot (int): The binding slot for the vertex buffer.
            buffer (GPUBuffer): The buffer that contains the vertex data.
            offset (int): The byte offset in the buffer. Default 0.
            size (int): The number of bytes to use. Default 0.
        """
        raise NotImplementedError()

    # wgpu.help('Size32', 'renderencoderbasedraw', dev=True)
    # IDL: void draw(GPUSize32 vertexCount, optional GPUSize32 instanceCount = 1,  optional GPUSize32 firstVertex = 0, optional GPUSize32 firstInstance = 0);
    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        """ Run the render pipeline without an index buffer.

        Arguments:
            vertex_count (int): The number of vertices to draw.
            instance_count (int):  The number of instances to draw. Default 1.
            first_vertex (int): The vertex offset. Default 0.
            first_instance (int):  The instance offset. Default 0.
        """
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasedrawindirect', dev=True)
    # IDL: void drawIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indirect(self, indirect_buffer, indirect_offset):
        """ Like ``draw()``, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()

    # wgpu.help('SignedOffset32', 'Size32', 'renderencoderbasedrawindexed', dev=True)
    # IDL: void drawIndexed(GPUSize32 indexCount, optional GPUSize32 instanceCount = 1,  optional GPUSize32 firstIndex = 0,  optional GPUSignedOffset32 baseVertex = 0,  optional GPUSize32 firstInstance = 0);
    def draw_indexed(
        self,
        index_count,
        instance_count=1,
        first_index=0,
        base_vertex=0,
        first_instance=0,
    ):
        """ Run the render pipeline using an index buffer.

        Arguments:
            index_count (int): The number of indices to draw.
            instance_count (int): The number of instances to draw. Default 1.
            first_index (int):  The index offset. Default 0.
            base_vertex (int):  A number added to each index in the index buffer. Default 0.
            first_instance (int): The instance offset. Default 0.
        """
        raise NotImplementedError()

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasedrawindexedindirect', dev=True)
    # IDL: void drawIndexedIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        """
        Like ``draw_indexed()``, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()


class GPURenderPassEncoder(GPURenderEncoderBase):
    """
    A render-pass encoder records commands related to a render pass.

    Create a render pass encoder using :func:`GPUCommandEncoder.begin_render_pass`.
    """

    # wgpu.help('renderpassencodersetviewport', dev=True)
    # IDL: void setViewport(float x, float y,  float width, float height,  float minDepth, float maxDepth);
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        """ Set the viewport for this render pass. The whole scene is rendered
        to this sub-rectangle.

        Arguments:
            x (int): Horizontal coordinate.
            y (int): Vertical coordinate.
            width (int): Horizontal size.
            height (int): Vertical size.
            min_depth (int): Clipping in depth.
            max_depth (int): Clipping in depth.

        """
        raise NotImplementedError()

    # wgpu.help('IntegerCoordinate', 'renderpassencodersetscissorrect', dev=True)
    # IDL: void setScissorRect(GPUIntegerCoordinate x, GPUIntegerCoordinate y,  GPUIntegerCoordinate width, GPUIntegerCoordinate height);
    def set_scissor_rect(self, x, y, width, height):
        """ Set the scissor rectangle for this render pass. The scene
        is rendered as usual, but is only applied to this sub-rectangle.

        Arguments:
            x (int): Horizontal coordinate.
            y (int): Vertical coordinate.
            width (int): Horizontal size.
            height (int): Vertical size.
        """
        raise NotImplementedError()

    # wgpu.help('Color', 'renderpassencodersetblendcolor', dev=True)
    # IDL: void setBlendColor(GPUColor color);
    def set_blend_color(self, color):
        """ Set the blend color for the render pass.

        Arguments:
            color (tuple or dict): A color with fields (r, g, b, a).
        """
        raise NotImplementedError()

    # wgpu.help('StencilValue', 'renderpassencodersetstencilreference', dev=True)
    # IDL: void setStencilReference(GPUStencilValue reference);
    def set_stencil_reference(self, reference):
        """ Set the reference stencil value for this render pass.

        Arguments:
            reference (int): The reference value.
        """
        raise NotImplementedError()

    # wgpu.help('renderpassencoderexecutebundles', dev=True)
    # IDL: void executeBundles(sequence<GPURenderBundle> bundles);
    def execute_bundles(self, bundles):
        """
        TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # wgpu.help('renderpassencoderendpass', dev=True)
    # IDL: void endPass();
    def end_pass(self):
        """ Record the end of the render pass.
        """
        raise NotImplementedError()


class GPURenderBundle(GPUObject):
    """
    TODO: not yet available in wgpu-native
    """


class GPURenderBundleEncoder(GPURenderEncoderBase):
    """
    TODO: not yet available in wgpu-native
    """

    # wgpu.help('RenderBundleDescriptor', 'renderbundleencoderfinish', dev=True)
    # IDL: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {});
    def finish(self, *, label=""):
        """ Finish recording and return a :class:`GPURenderBundle`.

        Parameters:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()


class GPUQueue(GPUObject):
    """
    A queue can be used to submit command buffers to.

    You can obtain a queue object via the :attr:`GPUDevice.default_queue` property.
    """

    # wgpu.help('queuesubmit', dev=True)
    # IDL: void submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self, command_buffers):
        """ Submit a :class:`GPUCommandBuffer` to the queue.

        Arguments:
            command_buffers (list): The :class:`GPUCommandBuffer` objects to add.
        """
        raise NotImplementedError()

    # wgpu.help('Extent3D', 'ImageBitmapCopyView', 'TextureCopyView', 'queuecopyimagebitmaptotexture', dev=True)
    # IDL: void copyImageBitmapToTexture( GPUImageBitmapCopyView source, GPUTextureCopyView destination, GPUExtent3D copySize);
    def copy_image_bitmap_to_texture(self, source, destination, copy_size):
        """
        TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # Not implemented, because fences do not yet have a use
    # def createFence(self):
    # def signal(self):
    # Fence.getCompletedValue
    # Fence.onCompletion


class GPUSwapChain(GPUObject):
    """
    A swap chain is a placeholder for a texture to be presented to the screen,
    so that you can provide the corresponding texture view as a color attachement
    to :func:`GPUCommandEncoder.begin_render_pass`. The texture view can be
    obtained by using the swap-chain in a with-statement. The swap-chain is
    presented to the screen when the context exits.

    Example:

    .. code-block:: py

        with swap_chain as texture_view:
            ...
            command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "attachment": texture_view,
                        ...
                    }
                ],
                ...
            )

    You can obtain a swap chain using :func:`device.configure_swap_chain() <GPUDevice.configure_swap_chain>`.
    """

    # wgpu.help('swapchaingetcurrenttexture', dev=True)
    # IDL: GPUTexture getCurrentTexture();
    def get_current_texture(self):
        """ WebGPU defines this method, but we deviate from the spec here:
        you should use the swap-chain object as a context manager to obtain
        a texture view to render to.
        """
        raise NotImplementedError(
            "Use the swap-chain as a context manager to get a texture view."
        )

    def __enter__(self):
        raise NotImplementedError()  # Get the current texture view

    def __exit__(self, type, value, tb):
        raise NotImplementedError()  # Present the current texture
