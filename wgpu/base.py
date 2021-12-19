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

import weakref
import logging
from typing import List, Dict

from ._coreutils import ApiDiff
from . import flags, enums, structs


__all__ = [
    "GPUObjectBase",
    "GPU",
    "GPUAdapter",
    "GPUDevice",
    "GPUBuffer",
    "GPUTexture",
    "GPUTextureView",
    "GPUExternalTexture",
    "GPUSampler",
    "GPUBindGroupLayout",
    "GPUBindGroup",
    "GPUPipelineLayout",
    "GPUShaderModule",
    "GPUCompilationMessage",
    "GPUCompilationInfo",
    "GPUPipelineBase",
    "GPUComputePipeline",
    "GPURenderPipeline",
    "GPUCommandBuffer",
    "GPUCommandEncoder",
    "GPUProgrammablePassEncoder",
    "GPUComputePassEncoder",
    "GPURenderEncoderBase",
    "GPURenderPassEncoder",
    "GPURenderBundle",
    "GPURenderBundleEncoder",
    "GPUQueue",
    "GPUQuerySet",
    "GPUCanvasContext",
    "GPUDeviceLostInfo",
    "GPUOutOfMemoryError",
    "GPUValidationError",
    "GPUUncapturedErrorEvent",
]

logger = logging.getLogger("wgpu")


apidiff = ApiDiff()


DEFAULT_ADAPTER_LIMITS = {
    "max_texture_dimension1d": 8192,
    "max_texture_dimension2d": 8192,
    "max_texture_dimension3d": 2048,
    "max_texture_array_layers": 2048,
    "max_bind_groups": 4,
    "max_dynamic_uniform_buffers_per_pipeline_layout": 8,
    "max_dynamic_storage_buffers_per_pipeline_layout": 4,
    "max_sampled_textures_per_shader_stage": 16,
    "max_samplers_per_shader_stage": 16,
    "max_storage_buffers_per_shader_stage": 4,
    "max_storage_textures_per_shader_stage": 4,
    "max_uniform_buffers_per_shader_stage": 12,
    "max_uniform_buffer_binding_size": 16384,
    "max_storage_buffer_binding_size": 134217728,
    "max_vertex_buffers": 8,
    "max_vertex_attributes": 16,
    "max_vertex_buffer_array_stride": 2048,
}


class GPU:
    """Class that represents the root namespace of the API."""

    # IDL: Promise<GPUAdapter?> requestAdapter(optional GPURequestAdapterOptions options = {});
    @apidiff.change("arguments include a canvas object")
    def request_adapter(self, *, canvas, power_preference=None):
        """Get a :class:`GPUAdapter`, the object that represents an abstract wgpu
        implementation, from which one can request a :class:`GPUDevice`.

        Arguments:
            canvas (WgpuCanvasInterface): The canvas that the adapter should
                be able to render to (to create a swap chain for, to be precise).
                Can be None if you're not rendering to screen (or if you're
                confident that the returned adapter will work just fine).
            powerPreference(PowerPreference): "high-performance" or "low-power"
        """
        # todo: include forceFallbackAdapter arg when this is also correctly applied in wgpu-native (currently its not)
        raise RuntimeError(
            "Select a backend (by importing wgpu.rs) before requesting an adapter!"
        )

    # IDL: Promise<GPUAdapter?> requestAdapter(optional GPURequestAdapterOptions options = {});
    @apidiff.change("arguments include a canvas object")
    async def request_adapter_async(self, *, canvas, power_preference=None):
        """Async version of ``request_adapter()``."""
        raise RuntimeError(
            "Select a backend (by importing wgpu.rs) before requesting an adapter!"
        )  # no-cover


class GPUCanvasContext:
    """A context object associated with a canvas, to present what has been drawn."""

    def __init__(self, canvas):
        self._canvas_ref = weakref.ref(canvas)

    def _get_canvas(self):
        """Getter method for internal use."""
        return self._canvas_ref()

    # IDL: readonly attribute (HTMLCanvasElement or OffscreenCanvas) canvas;
    @property
    def canvas(self):
        """The associated canvas object."""
        return self._canvas_ref()

    # IDL: undefined configure(GPUCanvasConfiguration configuration);
    def configure(
        self,
        *,
        device: "GPUDevice",
        format: "enums.TextureFormat",
        usage: "flags.TextureUsage" = 0x10,
        color_space: "enums.PredefinedColorSpace" = "srgb",
        compositing_alpha_mode: "enums.CanvasCompositingAlphaMode" = "opaque",
        size: "structs.Extent3D" = None,
    ):
        """Configures the presentation context for the associated canvas.
        Destroys any textures produced with a previous configuration.

        Arguments:
            device (WgpuDevice): The GPU device object.
            format (TextureFormat): The texture format, e.g. "bgra8unorm-srgb".
                Default uses the preferred_format.
            usage (TextureUsage): Default ``TextureUsage.OUTPUT_ATTACHMENT``.
            color_space (PredefinedColorSpace): Default "srgb".
            compositing_alpha_mode (CanvasCompositingAlphaMode): Default opaque.
            size: The 3D size of the texture to draw to. Default use canvas' physical size.
        """
        self.unconfigure()
        self._device = device
        self._format = format or self.get_preferred_format(device.adapter)
        self._usage = usage or flags.TextureUsage.RENDER_ATTACHMENT
        self._color_space = color_space
        self._compositing_alpha_mode = compositing_alpha_mode
        self._size = size

    # IDL: undefined unconfigure();
    def unconfigure(self):
        """Removes the presentation context configuration.
        Destroys any textures produced while configured."""
        self._device = None
        self._format = None
        self._usage = None
        self._color_space = None
        self._compositing_alpha_mode = None
        self._size = None

    # IDL: GPUTextureFormat getPreferredFormat(GPUAdapter adapter);
    def get_preferred_format(self, adapter):
        """Get the preferred swap chain format."""
        return "bgra8unorm-srgb"  # seems to be a good default

    # IDL: GPUTexture getCurrentTexture();
    def get_current_texture(self):
        """Get the ``GPUTexture`` that will be composited to the canvas
        by the context next.

        NOTE: for the time being, this could return a ``GPUTextureView`` instead.
        """
        raise NotImplementedError()

    @apidiff.add("Present method is exposed")
    def present(self):
        """Present what has been drawn to the current texture, by compositing it
        to the canvas. Note that a canvas based on ``WgpuCanvasBase`` will call this
        method automatically at the end of each draw event.
        """
        raise NotImplementedError()


class GPUAdapter:
    """
    An adapter represents both an instance of a hardware accelerator
    (e.g. GPU or CPU) and an implementation of WGPU on top of that
    accelerator. If an adapter becomes unavailable, it becomes invalid.
    Once invalid, it never becomes valid again.
    """

    def __init__(self, name, internal, features, limits, properties=None):
        self._name = name
        self._features = tuple(sorted(str(x) for x in features))
        self._internal = internal
        self._limits = DEFAULT_ADAPTER_LIMITS.copy()
        self._limits.update(limits)
        self._properties = properties or {}

    # IDL: readonly attribute DOMString name;
    @property
    def name(self):
        """A human-readable name identifying the adapter."""
        return self._name

    # IDL: [SameObject] readonly attribute GPUSupportedFeatures features;
    @property
    def features(self):
        """A tuple of supported feature names."""
        return self._features

    # IDL: [SameObject] readonly attribute GPUSupportedLimits limits;
    @property
    def limits(self):
        """A dict with the adapter limits."""
        return self._limits

    @apidiff.add("useful for desktop")
    @property
    def properties(self):
        """A dict with the adapter properties (info on device, backend, etc.)"""
        return self._properties

    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    def request_device(
        self,
        *,
        label="",
        required_features: "List[enums.FeatureName]" = [],
        required_limits: "Dict[str, int]" = {},
    ):
        """Request a :class:`GPUDevice` from the adapter.

        Arguments:
            label (str): A human readable label. Optional.
            required_features (list of str): the features (extensions) that you need. Default [].
            required_limits (dict): the various limits that you need. Default {}.
        """
        raise NotImplementedError()

    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    async def request_device_async(
        self,
        *,
        label="",
        required_features: "List[enums.FeatureName]" = [],
        required_limits: "Dict[str, int]" = {},
    ):
        """Async version of ``request_device()``."""
        raise NotImplementedError()

    def _destroy(self):
        pass

    def __del__(self):
        self._destroy()

    # IDL: readonly attribute boolean isFallbackAdapter;
    @property
    def is_fallback_adapter(self):
        """Whether this adapter runs on software (rather than dedicated hardware)."""
        return self._properties.get("adapterType", "").lower() in ("software", "cpu")


class GPUObjectBase:
    """The base class for all GPU objects (the device and all objects
    belonging to a device).
    """

    def __init__(self, label, internal, device):
        self._label = label
        self._internal = internal  # The native/raw/real GPU object
        self._device = device
        logger.info(f"Creating {self.__class__.__name__} {label}")

    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.label}' at 0x{hex(id(self))}>"

    # IDL: attribute USVString? label;
    @property
    def label(self):
        """A human-readable name identifying the GPU object."""
        return self._label

    def _destroy(self):
        """Subclasses can implement this to clean up."""
        pass

    def __del__(self):
        self._destroy()

    # Public destroy() methods are implemented on classes as the WebGPU spec specifies.


class GPUDevice(GPUObjectBase):
    """
    A device is the logical instantiation of an adapter, through which
    internal objects are created. It can be shared across threads.
    A device is the exclusive owner of all internal objects created
    from it: when the device is lost, all objects created from it become
    invalid.

    Create a device using :func:`GPUAdapter.request_device` or
    :func:`GPUAdapter.request_device_async`.
    """

    def __init__(self, label, internal, adapter, features, limits, queue):
        super().__init__(label, internal, None)
        assert isinstance(adapter, GPUAdapter)
        self._adapter = adapter
        self._features = tuple(sorted(str(x) for x in features))
        self._limits = DEFAULT_ADAPTER_LIMITS.copy()
        self._limits.update(limits)
        self._queue = queue
        queue._device = self  # because it could not be set earlier

    # IDL: [SameObject] readonly attribute GPUSupportedFeatures features;
    @property
    def features(self):
        """A tuple of strings representing the features (i.e. extensions) with
        which this device was created.
        """
        return self._features

    # IDL: [SameObject] readonly attribute GPUSupportedLimits limits;
    @property
    def limits(self):
        """A dict exposing the limits with which this device was created."""
        return self._limits

    # IDL: [SameObject] readonly attribute GPUQueue queue;
    @property
    def queue(self):
        """The default :class:`GPUQueue` for this device."""
        return self._queue

    @apidiff.add("Too useful to not-have")
    @property
    def adapter(self):
        """The adapter object corresponding to this device."""
        return self._adapter

    # FIXME: new prop to implement
    # IDL: readonly attribute Promise<GPUDeviceLostInfo> lost;
    @property
    def lost(self):
        """Provides information about why the device is lost."""
        raise NotImplementedError()

    # FIXME: new prop to implement
    # IDL: attribute EventHandler onuncapturederror;
    @property
    def onuncapturederror(self):
        """Method called when an error is capured?"""
        raise NotImplementedError()

    # IDL: undefined destroy();
    def destroy(self):
        """Destroy this device."""
        return self._destroy()

    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
    def create_buffer(
        self,
        *,
        label="",
        size: int,
        usage: "flags.BufferUsage",
        mapped_at_creation: bool = False,
    ):
        """Create a :class:`GPUBuffer` object.

        Arguments:
            label (str): A human readable label. Optional.
            size (int): The size of the buffer in bytes.
            usage (BufferUsageFlags): The ways in which this buffer will be used.
            mapped_at_creation (bool): Must be False, use create_buffer_with_data() instead.
        """
        raise NotImplementedError()

    @apidiff.add("replaces WebGPU's mapping API")
    def create_buffer_with_data(self, *, label="", data, usage: "flags.BufferUsage"):
        """Create a :class:`GPUBuffer` object initialized with the given data.

        Arguments:
            label (str): A human readable label. Optional.
            data: Any object supporting the Python buffer protocol (this
                includes bytes, bytearray, ctypes arrays, numpy arrays, etc.).
            usage (BufferUsageFlags): The ways in which this buffer will be used.

        Also see ``GPUQueue.write_buffer()`` and ``GPUQueue.read_buffer()``.
        """
        raise NotImplementedError()

    # IDL: GPUTexture createTexture(GPUTextureDescriptor descriptor);
    def create_texture(
        self,
        *,
        label="",
        size: "structs.Extent3D",
        mip_level_count: int = 1,
        sample_count: int = 1,
        dimension: "enums.TextureDimension" = "2d",
        format: "enums.TextureFormat",
        usage: "flags.TextureUsage",
    ):
        """Create a :class:`GPUTexture` object.

        Arguments:
            label (str): A human readable label. Optional.
            size (tuple or dict): The texture size as a 3-tuple or a
                dict (width, height, depth_or_array_layers).
            mip_level_count (int): The number of mip leveles. Default 1.
            sample_count (int): The number of samples. Default 1.
            dimension (TextureDimension): The dimensionality of the texture. Default 2d.
            format (TextureFormat): What channels it stores and how.
            usage (TextureUsageFlags): The ways in which the texture will be used.

        See https://gpuweb.github.io/gpuweb/#texture-format-caps for a
        list of available texture formats. Note that less formats are
        available for storage usage.
        """
        raise NotImplementedError()

    # IDL: GPUSampler createSampler(optional GPUSamplerDescriptor descriptor = {});
    def create_sampler(
        self,
        *,
        label="",
        address_mode_u: "enums.AddressMode" = "clamp-to-edge",
        address_mode_v: "enums.AddressMode" = "clamp-to-edge",
        address_mode_w: "enums.AddressMode" = "clamp-to-edge",
        mag_filter: "enums.FilterMode" = "nearest",
        min_filter: "enums.FilterMode" = "nearest",
        mipmap_filter: "enums.FilterMode" = "nearest",
        lod_min_clamp: float = 0,
        lod_max_clamp: float = 32,
        compare: "enums.CompareFunction" = None,
        max_anisotropy: int = 1,
    ):
        """Create a :class:`GPUSampler` object. Samplers specify how a texture is sampled.

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
            lod_max_clamp (float): The maxium level of detail. Default 32.
            compare (CompareFunction): The sample compare operation for depth textures.
                Only specify this for depth textures. Default None.
            max_anisotropy (int): The maximum anisotropy value clamp used by the sample,
                betweet 1 and 16, default 1.
        """
        raise NotImplementedError()

    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def create_bind_group_layout(
        self, *, label="", entries: "List[structs.BindGroupLayoutEntry]"
    ):
        """Create a :class:`GPUBindGroupLayout` object. One or more
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
                "buffer": {
                    "type": wgpu.BufferBindingType.storage_buffer,
                    "has_dynamic_offset": False,  # optional
                    "min_binding_size": 0  # optional
                }
            },
            # Sampler
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "sampler": {
                    "type": wgpu.SamplerBindingType.filtering,
                }
            },
            # Sampled texture
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,  # optional
                    "view_dimension": wgpu.TextureViewDimension.d2,  # optional
                    "multisampled": False,  # optional
                }
            },
            # Storage texture
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.read_only,
                    "format": wgpu.TextureFormat.r32float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                }
            },

        About ``has_dynamic_offset``: For uniform-buffer, storage-buffer, and
        readonly-storage-buffer bindings, it indicates whether the binding has a
        dynamic offset. One offset must be passed to ``set_bind_group`` for each
        dynamic binding in increasing order of binding number.
        """
        raise NotImplementedError()

    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
    def create_bind_group(
        self,
        *,
        label="",
        layout: "GPUBindGroupLayout",
        entries: "List[structs.BindGroupEntry]",
    ):
        """Create a :class:`GPUBindGroup` object, which can be used in
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

    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor);
    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "List[GPUBindGroupLayout]"
    ):
        """Create a :class:`GPUPipelineLayout` object, which can be
        used in :func:`create_render_pipeline` or :func:`create_compute_pipeline`.

        Arguments:
            label (str): A human readable label. Optional.
            bind_group_layouts (list): A list of :class:`GPUBindGroupLayout` objects.
        """
        raise NotImplementedError()

    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor);
    def create_shader_module(
        self,
        *,
        label="",
        code: str,
        source_map: dict = None,
        hints: "Dict[str, structs.ShaderModuleCompilationHint]" = None,
    ):
        """Create a :class:`GPUShaderModule` object from shader source.

        Arguments:
            label (str): A human readable label. Optional.
            code (str | bytes): The shader code, as WGSL text or binary SpirV
                (or an object implementing ``to_spirv()`` or ``to_bytes()``).
            hints: unused.
        """
        raise NotImplementedError()

    # IDL: GPUComputePipeline createComputePipeline(GPUComputePipelineDescriptor descriptor);
    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        compute: "structs.ProgrammableStage",
    ):
        """Create a :class:`GPUComputePipeline` object.

        Arguments:
            label (str): A human readable label. Optional.
            layout (GPUPipelineLayout): object created with ``create_pipeline_layout()``.
            compute (dict): E.g. ``{"module": shader_module, entry_point="main"}``.
        """
        raise NotImplementedError()

    # IDL: Promise<GPUComputePipeline> createComputePipelineAsync(GPUComputePipelineDescriptor descriptor);
    async def create_compute_pipeline_async(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        compute: "structs.ProgrammableStage",
    ):
        """Async version of create_compute_pipeline()."""
        raise NotImplementedError()

    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor);
    def create_render_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        vertex: "structs.VertexState",
        primitive: "structs.PrimitiveState" = {},
        depth_stencil: "structs.DepthStencilState" = None,
        multisample: "structs.MultisampleState" = {},
        fragment: "structs.FragmentState" = None,
    ):
        """Create a :class:`GPURenderPipeline` object.

        Arguments:
            label (str): A human readable label. Optional.
            layout (GPUPipelineLayout): A layout created with ``create_pipeline_layout()``.
            vertex (VertexState): Describes the vertex shader entry point of the pipeline and its input buffer layouts.
            primitive (PrimitiveState): Describes the the primitive-related properties
                of the pipeline. If `strip_index_format` is present (which means the
                primitive topology is a strip), and the drawCall is indexed, the
                vertex index list is split into sub-lists using the maximum value of this
                index format as a separator. Example: a list with values
                `[1, 2, 65535, 4, 5, 6]` of type "uint16" will be split in sub-lists
                `[1, 2]` and `[4, 5, 6]`.
            depth_stencil (DepthStencilState): Describes the optional depth-stencil properties, including the testing, operations, and bias. Optional.
            multisample (MultisampleState): Describes the multi-sampling properties of the pipeline.
            fragment (FragmentState): Describes the fragment shader
                entry point of the pipeline and its output colors. If it’s
                None, the No Color Output mode is enabled: the pipeline
                does not produce any color attachment outputs. It still
                performs rasterization and produces depth values based on
                the vertex position output. The depth testing and stencil
                operations can still be used.

        In the example dicts below, the values that are marked as optional,
        the shown value is the default.

        Example vertex (VertexState) dict:

        .. code-block:: py

            {
                "module": shader_module,
                "entry_point": "main",
                "buffers": [
                    {
                        "array_stride": 8,
                        "step_mode": wgpu.VertexStepMode.vertex,  # optional
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

        Example primitive (GPUPrimitiveState) dict:

        .. code-block:: py

            {
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "strip_index_format": wgpu.IndexFormat.uint32,  # see note
                "front_face": wgpu.FrontFace.ccw,  # optional
                "cull_mode": wgpu.CullMode.none,  # optional
            }

        Example depth_stencil (GPUDepthStencilState) dict:

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
                "depth_bias": 0,  # optional
                "depth_bias_slope_scale": 0.0,  # optional
                "depth_bias_clamp": 0.0,  # optional
            }

        Example multisample (MultisampleState) dict:

        .. code-block:: py

            {
                "count": 1,  # optional
                "mask": 0xFFFFFFFF,  # optional
                "alpha_to_coverage_enabled": False  # optional
            }

        Example fragment (FragmentState) dict. The `blend` parameter can be None
        to disable blending (not all texture formats support blending).

        .. code-block:: py

            {
                "module": shader_module,
                "entry_point": "main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.bgra8unorm_srgb,
                        "blend": {
                            "color": (
                                wgpu.BlendFactor.One,
                                wgpu.BlendFactor.zero,
                                gpu.BlendOperation.add,
                            ),
                            "alpha": (
                                wgpu.BlendFactor.One,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                        }
                        "write_mask": wgpu.ColorWrite.ALL  # optional
                    },
                    ...
                ]
            }

        """
        raise NotImplementedError()

    # IDL: Promise<GPURenderPipeline> createRenderPipelineAsync(GPURenderPipelineDescriptor descriptor);
    async def create_render_pipeline_async(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        vertex: "structs.VertexState",
        primitive: "structs.PrimitiveState" = {},
        depth_stencil: "structs.DepthStencilState" = None,
        multisample: "structs.MultisampleState" = {},
        fragment: "structs.FragmentState" = None,
    ):
        """Async version of create_render_pipeline()."""
        raise NotImplementedError()

    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {});
    def create_command_encoder(self, *, label=""):
        """Create a :class:`GPUCommandEncoder` object. A command
        encoder is used to record commands, which can then be submitted
        at once to the GPU.

        Arguments:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()

    # IDL: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor);
    def create_render_bundle_encoder(
        self,
        *,
        label="",
        color_formats: "List[enums.TextureFormat]",
        depth_stencil_format: "enums.TextureFormat" = None,
        sample_count: int = 1,
        depth_read_only: bool = False,
        stencil_read_only: bool = False,
    ):
        """Create a :class:`GPURenderBundle` object.

        TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: GPUQuerySet createQuerySet(GPUQuerySetDescriptor descriptor);
    def create_query_set(self, *, label="", type: "enums.QueryType", count: int):
        """Create a :class:`GPUQuerySet` object."""
        raise NotImplementedError()

    # IDL: undefined pushErrorScope(GPUErrorFilter filter);
    @apidiff.hide
    def push_error_scope(self, filter):
        raise NotImplementedError()

    # IDL: Promise<GPUError?> popErrorScope();
    @apidiff.hide
    def pop_error_scope(self):
        raise NotImplementedError()

    # IDL: GPUExternalTexture importExternalTexture(GPUExternalTextureDescriptor descriptor);
    @apidiff.hide("Specific to browsers.")
    def import_external_texture(
        self,
        *,
        label="",
        source: object,
        color_space: "enums.PredefinedColorSpace" = "srgb",
    ):
        raise NotImplementedError()


class GPUBuffer(GPUObjectBase):
    """
    A GPUBuffer represents a block of memory that can be used in GPU
    operations. Data is stored in linear layout, meaning that each byte
    of the allocation can be addressed by its offset from the start of
    the buffer, subject to alignment restrictions depending on the
    operation.

    Create a buffer using :func:`GPUDevice.create_buffer`,
    :func:`GPUDevice.create_buffer_mapped` or :func:`GPUDevice.create_buffer_mapped_async`.

    One can sync data in a buffer by mapping it (or by creating a mapped
    buffer) and then setting/getting the values in the mapped memoryview.
    Alternatively, one can tell the GPU (via the command encoder) to
    copy data between buffers and textures.
    """

    def __init__(self, label, internal, device, size, usage):
        super().__init__(label, internal, device)
        self._size = size
        self._usage = usage

    @apidiff.add("Too useful to not-have")
    @property
    def size(self):
        """The length of the GPUBuffer allocation in bytes."""
        return self._size

    @apidiff.add("Too useful to not-have")
    @property
    def usage(self):
        """The allowed usages (int bitmap) for this GPUBuffer, specifying
        e.g. whether the buffer may be used as a vertex buffer, uniform buffer,
        target or source for copying data, etc.
        """
        return self._usage

    # WebGPU specifies an API to sync data with the buffer via mapping.
    # The idea is to (async) request mapped data, read from / write to
    # this memory (using getMappedRange), and then unmap.  A buffer
    # must be unmapped before it can be used in a pipeline.
    #
    # This means that the mapped memory is reclaimed (i.e. invalid)
    # when unmap is called, and that whatever object we expose the
    # memory with to the user, must be set to a state where it can no
    # longer be used. I currently can't think of a good way to do this.
    #
    # So instead, we can use mapping internally to allow reading and
    # writing but not expose it via the public API. The only
    # disadvantage (AFAIK) is that there could be use-cases where a
    # memory copy could be avoided when using mapping.

    # IDL: undefined destroy();
    def destroy(self):
        """An application that no longer requires a buffer can choose
        to destroy it. Note that this is automatically called when the
        Python object is cleaned up by the garbadge collector.
        """
        raise NotImplementedError()

    @apidiff.add("Alternative to mapping API")
    def map_read(self):
        """Map the buffer and read the data from it, then unmap.
        Return a memoryview object. Requires the buffer usage to include MAP_READ.

        See `queue.read_buffer()` for a simpler alternative.
        """
        raise NotImplementedError()

    @apidiff.add("Alternative to mapping API")
    def map_write(self, data):
        """Map the buffer and write the data to it, then unmap.
        Return a memoryview object. Requires the buffer usage to include MAP_WRITE.

        See `queue.write_buffer()` for a simpler alternative.
        """
        raise NotImplementedError()

    # IDL: ArrayBuffer getMappedRange(optional GPUSize64 offset = 0, optional GPUSize64 size);
    @apidiff.hide
    def get_mapped_range(self, offset=0, size=None):
        raise NotImplementedError("The Python API differs from WebGPU here")

    # IDL: undefined unmap();
    @apidiff.hide
    def unmap(self):
        raise NotImplementedError("The Python API differs from WebGPU here")

    # IDL: Promise<undefined> mapAsync(GPUMapModeFlags mode, optional GPUSize64 offset = 0, optional GPUSize64 size);
    @apidiff.hide
    async def map_async(self, mode, offset=0, size=None):
        raise NotImplementedError()


class GPUTexture(GPUObjectBase):
    """
    A texture represents a 1D, 2D or 3D color image object. It also can have mipmaps
    (different levels of varying detail), and arrays. The texture represents
    the "raw" data. A :class:`GPUTextureView` is used to define how the texture data
    should be interpreted.

    Create a texture using :func:`GPUDevice.create_texture`.
    """

    def __init__(self, label, internal, device, tex_info):
        super().__init__(label, internal, device)
        self._tex_info = tex_info

    @apidiff.add("Too useful to not-have")
    @property
    def size(self):
        """The size of the texture in mipmap level 0, as a 3-tuple of ints."""
        return self._tex_info["size"]

    @apidiff.add("Too useful to not-have")
    @property
    def mip_level_count(self):
        """The total number of the mipmap levels of the texture."""
        return self._tex_info["mip_level_count"]

    @apidiff.add("Too useful to not-have")
    @property
    def sample_count(self):
        """The number of samples in each texel of the texture."""
        return self._tex_info["sample_count"]

    @apidiff.add("Too useful to not-have")
    @property
    def dimension(self):
        """The dimension of the texture."""
        return self._tex_info["dimension"]

    @apidiff.add("Too useful to not-have")
    @property
    def format(self):
        """The format of the texture."""
        return self._tex_info["format"]

    @apidiff.add("Too useful to not-have")
    @property
    def usage(self):
        """The allowed usages for this texture."""
        return self._tex_info["usage"]

    # IDL: GPUTextureView createView(optional GPUTextureViewDescriptor descriptor = {});
    def create_view(
        self,
        *,
        label="",
        format: "enums.TextureFormat" = None,
        dimension: "enums.TextureViewDimension" = None,
        aspect: "enums.TextureAspect" = "all",
        base_mip_level: int = 0,
        mip_level_count: int = None,
        base_array_layer: int = 0,
        array_layer_count: int = None,
    ):
        """Create a :class:`GPUTextureView` object.

        If no aguments are given, a default view is given, with the
        same format and dimension as the texture.

        Arguments:
            label (str): A human readable label. Optional.
            format (TextureFormat): What channels it stores and how.
            dimension (TextureViewDimension): The dimensionality of the texture view.
            aspect (TextureAspect): Whether this view is used for depth, stencil, or all.
                Default all.
            base_mip_level (int): The starting mip level. Default 0.
            mip_level_count (int): The number of mip levels. Default None.
            base_array_layer (int): The starting array layer. Default 0.
            array_layer_count (int): The number of array layers. Default None.
        """
        raise NotImplementedError()

    # IDL: undefined destroy();
    def destroy(self):
        """An application that no longer requires a texture can choose
        to destroy it. Note that this is automatically called when the
        Python object is cleaned up by the garbadge collector.
        """
        raise NotImplementedError()


class GPUTextureView(GPUObjectBase):
    """
    A texture view represents a way to represent a :class:`GPUTexture`.

    Create a texture view using :func:`GPUTexture.create_view`.
    """

    def __init__(self, label, internal, device, texture, size):
        super().__init__(label, internal, device)
        self._texture = texture
        self._size = size

    @apidiff.add("Too useful to not-have")
    @property
    def size(self):
        """The texture size (as a 3-tuple)."""
        return self._size

    @apidiff.add("Too useful to not-have")
    @property
    def texture(self):
        """The texture object to which this is a view."""
        return self._texture


class GPUSampler(GPUObjectBase):
    """
    A sampler specifies how a texture (view) must be sampled by the shader,
    in terms of subsampling, sampling between mip levels, and sampling out
    of the image boundaries.

    Create a sampler using :func:`GPUDevice.create_sampler`.
    """


class GPUBindGroupLayout(GPUObjectBase):
    """
    A bind group layout defines the interface between a set of
    resources bound in a :class:`GPUBindGroup` and their accessibility in shader
    stages.

    Create a bind group layout using :func:`GPUDevice.create_bind_group_layout`.
    """

    def __init__(self, label, internal, device, bindings):
        super().__init__(label, internal, device)
        self._bindings = tuple(bindings)


class GPUBindGroup(GPUObjectBase):
    """
    A bind group represents a group of bindings, the shader slot,
    and a resource (sampler, texture-view, buffer).

    Create a bind group using :func:`GPUDevice.create_bind_group`.
    """

    def __init__(self, label, internal, device, bindings):
        super().__init__(label, internal, device)
        self._bindings = bindings


class GPUPipelineLayout(GPUObjectBase):
    """
    A pipeline layout describes the layout of a pipeline, as a list
    of :class:`GPUBindGroupLayout` objects.

    Create a pipeline layout using :func:`GPUDevice.create_pipeline_layout`.
    """

    def __init__(self, label, internal, device, layouts):
        super().__init__(label, internal, device)
        self._layouts = tuple(layouts)  # GPUBindGroupLayout objects


class GPUShaderModule(GPUObjectBase):
    """
    A shader module represents a programmable shader.

    Create a shader module using :func:`GPUDevice.create_shader_module`.
    """

    # IDL: Promise<GPUCompilationInfo> compilationInfo();
    def compilation_info(self):
        """Get shader compilation info. Always returns empty string at the moment."""
        return []

    # IDL: Promise<GPUCompilationInfo> compilationInfo();
    async def compilation_info_async(self):
        """Async version of compilation_info()"""
        return self.compilation_info()  # no-cover


class GPUPipelineBase:
    """A mixin class for render and compute pipelines."""

    def __init__(self, label, internal, device, layout):
        super().__init__(label, internal, device)
        self._layout = layout

    # IDL: GPUBindGroupLayout getBindGroupLayout(unsigned long index);
    def get_bind_group_layout(self, index):
        """Get the bind group layout at the given index."""
        return self._layout._layouts[index]


class GPUComputePipeline(GPUPipelineBase, GPUObjectBase):
    """
    A compute pipeline represents a single pipeline for computations (no rendering).

    Create a compute pipeline using :func:`GPUDevice.create_compute_pipeline`.
    """


class GPURenderPipeline(GPUPipelineBase, GPUObjectBase):
    """
    A render pipeline represents a single pipeline to draw something
    using a vertex and a fragment shader. The render target can come
    from a window on the screen or from an in-memory texture (off-screen
    rendering).

    Create a render pipeline using :func:`GPUDevice.create_render_pipeline`.
    """


class GPUCommandBuffer(GPUObjectBase):
    """
    A command buffer stores a series of commands, generated by a
    :class:`GPUCommandEncoder`, to be submitted to a :class:`GPUQueue`.

    Create a command buffer using :func:`GPUCommandEncoder.finish`.
    """


class GPUCommandEncoder(GPUObjectBase):
    """
    A command encoder is used to record a series of commands. When done,
    call :func:`finish` to obtain a GPUCommandBuffer object.

    Create a command encoder using :func:`GPUDevice.create_command_encoder`.
    """

    # IDL: GPUComputePassEncoder beginComputePass(optional GPUComputePassDescriptor descriptor = {});
    def begin_compute_pass(
        self,
        *,
        label="",
        timestamp_writes: "List[structs.ComputePassTimestampWrite]" = [],
    ):
        """Record the beginning of a compute pass. Returns a
        :class:`GPUComputePassEncoder` object.

        Arguments:
            label (str): A human readable label. Optional.
            timestamp_writes: unused
        """
        raise NotImplementedError()

    # IDL: GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor descriptor);
    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "List[structs.RenderPassColorAttachment]",
        depth_stencil_attachment: "structs.RenderPassDepthStencilAttachment" = None,
        occlusion_query_set: "GPUQuerySet" = None,
        timestamp_writes: "List[structs.RenderPassTimestampWrite]" = [],
    ):
        """Record the beginning of a render pass. Returns a
        :class:`GPURenderPassEncoder` object.

        Arguments:
            label (str): A human readable label. Optional.
            color_attachments (list of dict): List of color attachment dicts. See below.
            depth_stencil_attachment (dict): A depth stencil attachment dict. See below. Default None.
            occlusion_query_set: Default None. TODO NOT IMPLEMENTED in wgpu-native.
            timestamp_writes: unused

        Example color attachment:

        .. code-block:: py

            {
                "view": texture_view,
                "resolve_target": None,  # optional
                "load_value": (0, 0, 0, 0),  # LoadOp.load or a color
                "store_op": wgpu.StoreOp.store,  # optional
            }

        Example depth stencil attachment:

        .. code-block:: py

            {
                "view": texture_view,
                "depth_load_value": 0.0,
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_load_value": wgpu.LoadOp.load,
                "stencil_store_op": wgpu.StoreOp.store,
            }
        """
        raise NotImplementedError()

    # IDL: undefined copyBufferToBuffer( GPUBuffer source, GPUSize64 sourceOffset, GPUBuffer destination, GPUSize64 destinationOffset, GPUSize64 size);
    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        """Copy the contents of a buffer to another buffer.

        Arguments:
            source (GPUBuffer): The source buffer.
            source_offset (int): The byte offset (a multiple of 4).
            destination (GPUBuffer): The target buffer.
            destination_offset (int): The byte offset in the destination buffer (a multiple of 4).
            size (int): The number of bytes to copy (a multiple of 4).
        """
        raise NotImplementedError()

    # IDL: undefined copyBufferToTexture( GPUImageCopyBuffer source, GPUImageCopyTexture destination, GPUExtent3D copySize);
    def copy_buffer_to_texture(self, source, destination, copy_size):
        """Copy the contents of a buffer to a texture (view).

        Arguments:
            source (GPUBuffer): A dict with fields: buffer, offset, bytes_per_row, rows_per_image.
            destination (GPUTexture): A dict with fields: texture, mip_level, origin.
            copy_size (int): The number of bytes to copy.

        Note that the `bytes_per_row` must be a multiple of 256.
        """
        raise NotImplementedError()

    # IDL: undefined copyTextureToBuffer( GPUImageCopyTexture source, GPUImageCopyBuffer destination, GPUExtent3D copySize);
    def copy_texture_to_buffer(self, source, destination, copy_size):
        """Copy the contents of a texture (view) to a buffer.

        Arguments:
            source (GPUTexture): A dict with fields: texture, mip_level, origin.
            destination (GPUBuffer):  A dict with fields: buffer, offset, bytes_per_row, rows_per_image.
            copy_size (int): The number of bytes to copy.

        Note that the `bytes_per_row` must be a multiple of 256.
        """
        raise NotImplementedError()

    # IDL: undefined copyTextureToTexture( GPUImageCopyTexture source, GPUImageCopyTexture destination, GPUExtent3D copySize);
    def copy_texture_to_texture(self, source, destination, copy_size):
        """Copy the contents of a texture (view) to another texture (view).

        Arguments:
            source (GPUTexture): A dict with fields: texture, mip_level, origin.
            destination (GPUTexture):  A dict with fields: texture, mip_level, origin.
            copy_size (int): The number of bytes to copy.
        """
        raise NotImplementedError()

    # IDL: undefined pushDebugGroup(USVString groupLabel);
    def push_debug_group(self, group_label):
        """Push a label on the debug group stack. (todo: docs)"""
        raise NotImplementedError()

    # IDL: undefined popDebugGroup();
    def pop_debug_group(self):
        """Pop a label from the debug group stack."""
        raise NotImplementedError()

    # IDL: undefined insertDebugMarker(USVString markerLabel);
    def insert_debug_marker(self, marker_label):
        """Insert a debug label in stack."""
        raise NotImplementedError()

    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {});
    def finish(self, *, label=""):
        """Finish recording. Returns a :class:`GPUCommandBuffer` to
        submit to a :class:`GPUQueue`.

        Arguments:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: undefined writeTimestamp(GPUQuerySet querySet, GPUSize32 queryIndex);
    def write_timestamp(self, query_set, query_index):
        """TODO"""
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: undefined resolveQuerySet( GPUQuerySet querySet, GPUSize32 firstQuery, GPUSize32 queryCount, GPUBuffer destination, GPUSize64 destinationOffset);
    def resolve_query_set(
        self, query_set, first_query, query_count, destination, destination_offset
    ):
        """TODO"""
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: undefined clearBuffer( GPUBuffer buffer, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def clear_buffer(self, buffer, offset=0, size=None):
        """Set (part of) the given buffer to zeros."""
        raise NotImplementedError()


class GPUProgrammablePassEncoder:
    """
    Base class for the different pass encoder classes.
    """

    # IDL: undefined setBindGroup(GPUIndex32 index, GPUBindGroup bindGroup,  Uint32Array dynamicOffsetsData,  GPUSize64 dynamicOffsetsDataStart,  GPUSize32 dynamicOffsetsDataLength);
    def set_bind_group(
        self,
        index,
        bind_group,
        dynamic_offsets_data,
        dynamic_offsets_data_start,
        dynamic_offsets_data_length,
    ):
        """Associate the given bind group (i.e. group or resources) with the
        given slot/index.

        Arguments:
            index (int): The slot to bind at.
            bind_group (GPUBindGroup): The bind group to bind.
            dynamic_offsets_data (list of int): A list of offsets (one for each bind group).
            dynamic_offsets_data_start (int): Not used.
            dynamic_offsets_data_length (int): Not used.
        """
        raise NotImplementedError()

    # IDL: undefined pushDebugGroup(USVString groupLabel);
    def push_debug_group(self, group_label):
        """Push a named debug group into the command stream."""
        raise NotImplementedError()

    # IDL: undefined popDebugGroup();
    def pop_debug_group(self):
        """Pop the active debug group."""
        raise NotImplementedError()

    # IDL: undefined insertDebugMarker(USVString markerLabel);
    def insert_debug_marker(self, marker_label):
        """Insert the given message into the debug message queue."""
        raise NotImplementedError()


class GPUComputePassEncoder(GPUProgrammablePassEncoder, GPUObjectBase):
    """
    A compute-pass encoder records commands related to a compute pass.

    Create a compute pass encoder using :func:`GPUCommandEncoder.begin_compute_pass`.
    """

    # IDL: undefined setPipeline(GPUComputePipeline pipeline);
    def set_pipeline(self, pipeline):
        """Set the pipeline for this compute pass.

        Arguments:
            pipeline (GPUComputePipeline): The pipeline to use.
        """
        raise NotImplementedError()

    # IDL: undefined dispatch(GPUSize32 x, optional GPUSize32 y = 1, optional GPUSize32 z = 1);
    def dispatch(self, x, y=1, z=1):
        """Run the compute shader.

        Arguments:
            x (int): The number of cycles in index x.
            y (int): The number of cycles in index y. Default 1.
            z (int): The number of cycles in index z. Default 1.
        """
        raise NotImplementedError()

    # IDL: undefined dispatchIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        """Like ``dispatch()``, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()

    # IDL: undefined endPass();
    def end_pass(self):
        """Record the end of the compute pass."""
        raise NotImplementedError()


class GPURenderEncoderBase:
    """
    Base class for different render-pass encoder classes.
    """

    # IDL: undefined setPipeline(GPURenderPipeline pipeline);
    def set_pipeline(self, pipeline):
        """Set the pipeline for this render pass.

        Arguments:
            pipeline (GPURenderPipeline): The pipeline to use.
        """
        raise NotImplementedError()

    # IDL: undefined setIndexBuffer(GPUBuffer buffer, GPUIndexFormat indexFormat, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def set_index_buffer(self, buffer, index_format, offset=0, size=None):
        """Set the index buffer for this render pass.

        Arguments:
            buffer (GPUBuffer): The buffer that contains the indices.
            index_format (GPUIndexFormat): The format of the index data
                contained in buffer. If `strip_index_format` is given in the
                call to `create_render_pipeline()`, it must match.
            offset (int): The byte offset in the buffer. Default 0.
            size (int): The number of bytes to use. If zero, the remaining size
                (after offset) of the buffer is used. Default 0.
        """
        raise NotImplementedError()

    # IDL: undefined setVertexBuffer(GPUIndex32 slot, GPUBuffer buffer, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def set_vertex_buffer(self, slot, buffer, offset=0, size=None):
        """Associate a vertex buffer with a bind slot.

        Arguments:
            slot (int): The binding slot for the vertex buffer.
            buffer (GPUBuffer): The buffer that contains the vertex data.
            offset (int): The byte offset in the buffer. Default 0.
            size (int): The number of bytes to use. If zero, the remaining size
                (after offset) of the buffer is used. Default 0.
        """
        raise NotImplementedError()

    # IDL: undefined draw(GPUSize32 vertexCount, optional GPUSize32 instanceCount = 1,  optional GPUSize32 firstVertex = 0, optional GPUSize32 firstInstance = 0);
    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        """Run the render pipeline without an index buffer.

        Arguments:
            vertex_count (int): The number of vertices to draw.
            instance_count (int):  The number of instances to draw. Default 1.
            first_vertex (int): The vertex offset. Default 0.
            first_instance (int):  The instance offset. Default 0.
        """
        raise NotImplementedError()

    # IDL: undefined drawIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indirect(self, indirect_buffer, indirect_offset):
        """Like ``draw()``, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()

    # IDL: undefined drawIndexed(GPUSize32 indexCount, optional GPUSize32 instanceCount = 1,  optional GPUSize32 firstIndex = 0,  optional GPUSignedOffset32 baseVertex = 0,  optional GPUSize32 firstInstance = 0);
    def draw_indexed(
        self,
        index_count,
        instance_count=1,
        first_index=0,
        base_vertex=0,
        first_instance=0,
    ):
        """Run the render pipeline using an index buffer.

        Arguments:
            index_count (int): The number of indices to draw.
            instance_count (int): The number of instances to draw. Default 1.
            first_index (int):  The index offset. Default 0.
            base_vertex (int):  A number added to each index in the index buffer. Default 0.
            first_instance (int): The instance offset. Default 0.
        """
        raise NotImplementedError()

    # IDL: undefined drawIndexedIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        """
        Like ``draw_indexed()``, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()


class GPURenderPassEncoder(
    GPUProgrammablePassEncoder, GPURenderEncoderBase, GPUObjectBase
):
    """
    A render-pass encoder records commands related to a render pass.

    Create a render pass encoder using :func:`GPUCommandEncoder.begin_render_pass`.
    """

    # IDL: undefined setViewport(float x, float y,  float width, float height,  float minDepth, float maxDepth);
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        """Set the viewport for this render pass. The whole scene is rendered
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

    # IDL: undefined setScissorRect(GPUIntegerCoordinate x, GPUIntegerCoordinate y,  GPUIntegerCoordinate width, GPUIntegerCoordinate height);
    def set_scissor_rect(self, x, y, width, height):
        """Set the scissor rectangle for this render pass. The scene
        is rendered as usual, but is only applied to this sub-rectangle.

        Arguments:
            x (int): Horizontal coordinate.
            y (int): Vertical coordinate.
            width (int): Horizontal size.
            height (int): Vertical size.
        """
        raise NotImplementedError()

    # IDL: undefined setBlendConstant(GPUColor color);
    def set_blend_constant(self, color):
        """Set the blend color for the render pass.

        Arguments:
            color (tuple or dict): A color with fields (r, g, b, a).
        """
        raise NotImplementedError()

    # IDL: undefined setStencilReference(GPUStencilValue reference);
    def set_stencil_reference(self, reference):
        """Set the reference stencil value for this render pass.

        Arguments:
            reference (int): The reference value.
        """
        raise NotImplementedError()

    # IDL: undefined executeBundles(sequence<GPURenderBundle> bundles);
    def execute_bundles(self, bundles):
        """
        TODO: not yet available in wgpu-native
        """
        raise NotImplementedError()

    # IDL: undefined endPass();
    def end_pass(self):
        """Record the end of the render pass."""
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: undefined beginOcclusionQuery(GPUSize32 queryIndex);
    def begin_occlusion_query(self, query_index):
        """TODO"""
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: undefined endOcclusionQuery();
    def end_occlusion_query(self):
        """TODO"""
        raise NotImplementedError()


class GPURenderBundle(GPUObjectBase):
    """
    TODO: not yet available in wgpu-native
    """


class GPURenderBundleEncoder(
    GPUProgrammablePassEncoder, GPURenderEncoderBase, GPUObjectBase
):
    """
    TODO: not yet available in wgpu-native
    """

    # IDL: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {});
    def finish(self, *, label=""):
        """Finish recording and return a :class:`GPURenderBundle`.

        Arguments:
            label (str): A human readable label. Optional.
        """
        raise NotImplementedError()


class GPUQueue(GPUObjectBase):
    """
    A queue can be used to submit command buffers to.

    You can obtain a queue object via the :attr:`GPUDevice.default_queue` property.
    """

    # IDL: undefined submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self, command_buffers):
        """Submit a :class:`GPUCommandBuffer` to the queue.

        Arguments:
            command_buffers (list): The :class:`GPUCommandBuffer` objects to add.
        """
        raise NotImplementedError()

    # IDL: undefined writeBuffer( GPUBuffer buffer, GPUSize64 bufferOffset, [AllowShared] BufferSource data, optional GPUSize64 dataOffset = 0, optional GPUSize64 size);
    def write_buffer(self, buffer, buffer_offset, data, data_offset=0, size=None):
        """Takes the data contents and schedules a write operation of
        these contents to the buffer. A snapshot of the data is taken;
        any changes to the data after this function is called do not
        affect the buffer contents.

        Arguments:
            buffer: The :class:`GPUBuffer` object to write to.
            buffer_offset (int): The offset in the buffer to start writing at.
            data: The data to write. Must be contiguous.
            data_offset: The byte offset in the data. Default 0.
            size: The number of bytes to write. Default all minus offset.

        This maps the data to a temporary buffer and then copies that buffer
        to the given buffer. The given buffer's usage must include COPY_DST.

        Also see ``GPUDevice.create_buffer_with_data()`` and ``GPUBuffer.map_write()``.

        """
        raise NotImplementedError()

    @apidiff.add("replaces WebGPU's mapping API")
    def read_buffer(self, buffer, buffer_offset=0, size=None):
        """Takes the data contents of the buffer and return them as a memoryview.

        Arguments:
            buffer: The :class:`GPUBuffer` object to read from.
            buffer_offset (int): The offset in the buffer to start reading from.
            size: The number of bytes to read. Default all minus offset.

        This copies the data in the given buffer to a temporary buffer
        and then maps that buffer to read the data. The given buffer's
        usage must include COPY_SRC.

        Also see ``GPUBuffer.map_read()``.
        """
        raise NotImplementedError()

    # IDL: undefined writeTexture( GPUImageCopyTexture destination, [AllowShared] BufferSource data, GPUImageDataLayout dataLayout, GPUExtent3D size);
    def write_texture(self, destination, data, data_layout, size):
        """Takes the data contents and schedules a write operation of
        these contents to the destination texture in the queue. A
        snapshot of the data is taken; any changes to the data after
        this function is called do not affect the texture contents.

        Arguments:
            destination: A dict with fields: "texture" (a texture object),
                "origin" (a 3-tuple), "mip_level" (an int, default 0).
            data: The data to write.
            data_layout: A dict with fields: "offset" (an int, default 0),
                "bytes_per_row" (an int), "rows_per_image" (an int, default 0).
            size: A 3-tuple of ints specifying the size to write.

        Unlike ``GPUCommandEncoder.copyBufferToTexture()``, there is
        no alignment requirement on `bytes_per_row`.
        """
        raise NotImplementedError()

    @apidiff.add("For symmetry, and to help work around the bytes_per_row constraint")
    def read_texture(self, source, data_layout, size):
        """Reads the contents of the texture and return them as a memoryview.

        Arguments:
            source: A dict with fields: "texture" (a texture object),
                "origin" (a 3-tuple), "mip_level" (an int, default 0).
            data_layout: A dict with fields: "offset" (an int, default 0),
                "bytes_per_row" (an int), "rows_per_image" (an int, default 0).
            size: A 3-tuple of ints specifying the size to write.

        Unlike ``GPUCommandEncoder.copyBufferToTexture()``, there is
        no alignment requirement on `bytes_per_row`, although in the
        current implementation there will be a performance penalty if
        ``bytes_per_row`` is not a multiple of 256 (because we'll be
        copying data row-by-row in Python).
        """
        raise NotImplementedError()

    # FIXME: new method to implement
    # IDL: Promise<undefined> onSubmittedWorkDone();
    def on_submitted_work_done(self):
        """TODO"""
        raise NotImplementedError()

    # IDL: undefined copyExternalImageToTexture( GPUImageCopyExternalImage source, GPUImageCopyTextureTagged destination, GPUExtent3D copySize);
    @apidiff.hide("Specific to browsers.")
    def copy_external_image_to_texture(self, source, destination, copy_size):
        raise NotImplementedError()


# %% Further non-GPUObject classes


class GPUDeviceLostInfo:
    """An object that contains information about the device being lost."""

    def __init__(self, reason, message):
        self._reason = reason
        self._message = message

    # IDL: readonly attribute DOMString message;
    @property
    def message(self):
        """The error message specifying the reason for the device being lost."""
        return self._message

    # IDL: readonly attribute (GPUDeviceLostReason or undefined) reason;
    @property
    def reason(self):
        """The reason (enums.GPUDeviceLostReason) for the device getting lost. Can be None."""
        return self._reason


class GPUOutOfMemoryError(Exception):
    """An error raised when the GPU is out of memory."""

    # IDL: constructor();
    def __init__(self):
        super().__init__("GPU is out of memory.")


class GPUValidationError(Exception):
    """An error raised when the pipeline could not be validated."""

    # IDL: readonly attribute DOMString message;
    @property
    def message(self):
        """The error message specifying the reason for invalidation."""
        return self._message

    # IDL: constructor(DOMString message);
    def __init__(self, message):
        self._message = message


# %% Not implemented


# FIXME: new class to implement
class GPUCompilationMessage:
    """An object that contains information about a problem with shader compilation."""

    # IDL: readonly attribute DOMString message;
    @property
    def message(self):
        """The warning/error message."""
        raise NotImplementedError()

    # IDL: readonly attribute GPUCompilationMessageType type;
    @property
    def type(self):
        """The type of warning/problem."""
        raise NotImplementedError()

    # IDL: readonly attribute unsigned long long lineNum;
    @property
    def line_num(self):
        """The corresponding line number in the shader source."""
        raise NotImplementedError()

    # IDL: readonly attribute unsigned long long linePos;
    @property
    def line_pos(self):
        """The position on the line in the shader source."""
        raise NotImplementedError()

    # IDL: readonly attribute unsigned long long offset;
    @property
    def offset(self):
        """Offset of ..."""
        raise NotImplementedError()

    # IDL: readonly attribute unsigned long long length;
    @property
    def length(self):
        """The length of the line?"""
        raise NotImplementedError()


# FIXME: new class to implement
class GPUCompilationInfo:
    """TODO"""

    # IDL: readonly attribute FrozenArray<GPUCompilationMessage> messages;
    @property
    def messages(self):
        """A list of ``GPUCompilationMessage`` objects."""
        raise NotImplementedError()


# FIXME: new class to implement
class GPUQuerySet(GPUObjectBase):
    """TODO"""

    # IDL: undefined destroy();
    def destroy(self):
        """Destroy the queryset."""
        raise NotImplementedError()


# FIXME: new class to implement
class GPUUncapturedErrorEvent:
    """TODO"""

    # IDL: [SameObject] readonly attribute GPUError error;
    @property
    def error(self):
        """The error object."""
        raise NotImplementedError()

    # IDL: constructor( DOMString type, GPUUncapturedErrorEventInit gpuUncapturedErrorEventInitDict );
    def __init__(self, type, gpu_uncaptured_error_event_init_dict):
        pass


class GPUExternalTexture(GPUObjectBase):
    """Ignore this - specific to browsers."""


# %%%%% Post processing

apidiff.remove_hidden_methods(globals())
