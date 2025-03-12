"""
The classes representing the wgpu API. This module defines the classes,
properties, methods and documentation. The majority of methods are
implemented in backend modules.

This module is maintained using a combination of manual code and
automatically inserted code. Read the codegen/readme.md for more
information.
"""

# Allow using class names in type annotations, before the class is defined. Py3.7+
from __future__ import annotations

import weakref
import logging
from typing import List, Dict, Union, Optional

from ._coreutils import ApiDiff, str_flag_to_int
from ._diagnostics import diagnostics, texture_format_to_bpp
from . import flags, enums, structs


__all__ = [
    "GPU",
    "GPUAdapter",
    "GPUAdapterInfo",
    "GPUBindGroup",
    "GPUBindGroupLayout",
    "GPUBindingCommandsMixin",
    "GPUBuffer",
    "GPUCanvasContext",
    "GPUCommandBuffer",
    "GPUCommandEncoder",
    "GPUCommandsMixin",
    "GPUCompilationInfo",
    "GPUCompilationMessage",
    "GPUComputePassEncoder",
    "GPUComputePipeline",
    "GPUDebugCommandsMixin",
    "GPUDevice",
    "GPUDeviceLostInfo",
    "GPUError",
    "GPUInternalError",
    "GPUObjectBase",
    "GPUOutOfMemoryError",
    "GPUPipelineBase",
    "GPUPipelineError",
    "GPUPipelineLayout",
    "GPUQuerySet",
    "GPUQueue",
    "GPURenderBundle",
    "GPURenderBundleEncoder",
    "GPURenderCommandsMixin",
    "GPURenderPassEncoder",
    "GPURenderPipeline",
    "GPUSampler",
    "GPUShaderModule",
    "GPUTexture",
    "GPUTextureView",
    "GPUValidationError",
]

logger = logging.getLogger("wgpu")


apidiff = ApiDiff()


# Obtain the object tracker. Note that we store a ref of
# the latter on all classes that refer to it. Otherwise, on a sys exit,
# the module attributes are None-ified, and the destructors would
# therefore fail and produce warnings.
object_tracker = diagnostics.object_counts.tracker

# The 'optional' value is used as the default value for optional arguments in the following two cases:
# * The method accepts a descriptor that is optional, so we make all arguments (i.e. descriptor fields) optional, and this one does not have a default value.
# * In wgpu-py we decided that this argument should be optional, even though it's currently not according to the WebGPU spec.
optional = None


class GPU:
    """The entrypoint to the wgpu API.

    The starting point of your wgpu-adventure is always to obtain an
    adapter. This is the equivalent to browser's ``navigator.gpu``.
    When a backend is loaded, the ``wgpu.gpu`` object is replaced with
    a backend-specific implementation.
    """

    # IDL: Promise<GPUAdapter?> requestAdapter(optional GPURequestAdapterOptions options = {}); -> GPUPowerPreference powerPreference, boolean forceFallbackAdapter = false
    @apidiff.change("arguments include canvas")
    def request_adapter_sync(
        self,
        *,
        power_preference: enums.PowerPreference = None,
        force_fallback_adapter: bool = False,
        canvas=None,
    ):
        """Sync version of `request_adapter_async()`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        # If this method gets called, no backend has been loaded yet, let's do that now!
        from .backends.auto import gpu

        return gpu.request_adapter_sync(
            power_preference=power_preference,
            force_fallback_adapter=force_fallback_adapter,
            canvas=canvas,
        )

    # IDL: Promise<GPUAdapter?> requestAdapter(optional GPURequestAdapterOptions options = {}); -> GPUPowerPreference powerPreference, boolean forceFallbackAdapter = false
    @apidiff.change("arguments include canvas")
    async def request_adapter_async(
        self,
        *,
        power_preference: enums.PowerPreference = None,
        force_fallback_adapter: bool = False,
        canvas=None,
    ):
        """Create a `GPUAdapter`, the object that represents an abstract wgpu
        implementation, from which one can request a `GPUDevice`.

        Arguments:
            power_preference (PowerPreference): "high-performance" or "low-power".
            force_fallback_adapter (bool): whether to use a (probably CPU-based)
                fallback adapter.
            canvas : The canvas that the adapter should be able to render to. This can typically
                 be left to None. If given, the object must implement ``WgpuCanvasInterface``.
        """
        # If this method gets called, no backend has been loaded yet, let's do that now!
        from .backends.auto import gpu

        return await gpu.request_adapter_async(
            power_preference=power_preference,
            force_fallback_adapter=force_fallback_adapter,
            canvas=canvas,
        )

    @apidiff.add("Method useful for multi-gpu environments")
    def enumerate_adapters_sync(self):
        """Sync version of `enumerate_adapters_async()`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """

        # If this method gets called, no backend has been loaded yet, let's do that now!
        from .backends.auto import gpu

        return gpu.enumerate_adapters_sync()

    @apidiff.add("Method useful for multi-gpu environments")
    async def enumerate_adapters_async(self):
        """Get a list of adapter objects available on the current system.

        An adapter can then be selected (e.g. using its summary), and a device
        then created from it.

        The order of the devices is such that Vulkan adapters go first, then
        Metal, then D3D12, then OpenGL. Within each category, the order as
        provided by the particular backend is maintained. Note that the same
        device may be present via multiple backends (e.g. vulkan/opengl).

        We cannot make guarantees about whether the order of the adapters
        matches the order as reported by e.g. ``nvidia-smi``. We have found that
        on a Linux multi-gpu cluster, the order does match, but we cannot
        promise that this is always the case. If you want to make sure, do some
        testing by allocating big buffers and checking memory usage using
        ``nvidia-smi``.

        See https://github.com/pygfx/wgpu-py/issues/482 for more details.
        """
        # Note that backends that cannot enumerate adapters, like WebGL,
        # can at least request two adapters with different power preference,
        # and then return both or one (if they represent the same adapter).

        # If this method gets called, no backend has been loaded yet, let's do that now!
        from .backends.auto import gpu

        return await gpu.enumerate_adapters_async()

    # IDL: GPUTextureFormat getPreferredCanvasFormat();
    @apidiff.change("Disabled because we put it on the canvas context")
    def get_preferred_canvas_format(self):
        """Not implemented in wgpu-py; use `GPUCanvasContext.get_preferred_format()` instead.
        The WebGPU spec defines this function, but in wgpu there are different
        kinds of canvases which may each prefer/support a different format.
        """
        raise RuntimeError("Use canvas.get_preferred_format() instead.")

    # IDL: [SameObject] readonly attribute WGSLLanguageFeatures wgslLanguageFeatures;
    @property
    def wgsl_language_features(self):
        """A set of strings representing the WGSL language extensions supported by all adapters.
        Returns an empty set for now."""
        # Looks like at the time of writing there are no definitions for extensions yet
        return set()


# Instantiate API entrypoint
gpu = GPU()


class GPUCanvasContext:
    """Represents a context to configure a canvas and render to it.

    Can be obtained via `gui.WgpuCanvasInterface.get_context("wgpu")`.

    The canvas-context plays a crucial role in connecting the wgpu API to the
    GUI layer, in a way that allows the GUI to be agnostic about wgpu. It
    combines (and checks) the user's preferences with the capabilities and
    preferences of the canvas.
    """

    _ot = object_tracker

    def __init__(self, canvas, present_methods):
        self._ot.increase(self.__class__.__name__)
        self._canvas_ref = weakref.ref(canvas)

        # Surface capabilities. Stored the first time it is obtained
        self._capabilities = None

        # Configuration dict from the user, set via self.configure()
        self._config = None

        # The last used texture
        self._texture = None

        # Determine the present method
        self._present_methods = present_methods
        self._present_method = "screen" if "screen" in present_methods else "bitmap"

    def _get_canvas(self):
        """Getter method for internal use."""
        return self._canvas_ref()

    # IDL: readonly attribute (HTMLCanvasElement or OffscreenCanvas) canvas;
    @property
    def canvas(self):
        """The associated canvas object."""
        return self._canvas_ref()

    def _get_capabilities(self, adapter):
        """Get dict of capabilities and cache the result."""
        if self._capabilities is None:
            self._capabilities = {}
            if self._present_method == "screen":
                # Query capabilities from the surface
                self._capabilities.update(self._get_capabilities_screen(adapter))
            else:
                # Default image capabilities
                self._capabilities = {
                    "formats": ["rgba8unorm-srgb", "rgba8unorm"],
                    "usages": 0xFF,
                    "alpha_modes": [enums.CanvasAlphaMode.opaque],
                }
            # Derived defaults
            if "view_formats" not in self._capabilities:
                self._capabilities["view_formats"] = self._capabilities["formats"]

        return self._capabilities

    def _get_capabilities_screen(self, adapter):
        """Get capabilities for a native surface."""
        raise NotImplementedError()

    @apidiff.add("Better place to define the preferred format")
    def get_preferred_format(self, adapter):
        """Get the preferred surface texture format."""
        capabilities = self._get_capabilities(adapter)
        formats = capabilities["formats"]
        return formats[0] if formats else "bgra8-unorm"

    # IDL: undefined configure(GPUCanvasConfiguration configuration); -> required GPUDevice device, required GPUTextureFormat format, GPUTextureUsageFlags usage = 0x10, sequence<GPUTextureFormat> viewFormats = [], PredefinedColorSpace colorSpace = "srgb", GPUCanvasToneMapping toneMapping = {}, GPUCanvasAlphaMode alphaMode = "opaque"
    def configure(
        self,
        *,
        device: GPUDevice,
        format: enums.TextureFormat,
        usage: flags.TextureUsage = 0x10,
        view_formats: List[enums.TextureFormat] = [],
        color_space: str = "srgb",
        tone_mapping: structs.CanvasToneMapping = {},
        alpha_mode: enums.CanvasAlphaMode = "opaque",
    ):
        """Configures the presentation context for the associated canvas.
        Destroys any textures produced with a previous configuration.
        This clears the drawing buffer to transparent black.

        Arguments:
            device (WgpuDevice): The GPU device object to create compatible textures for.
            format (enums.TextureFormat): The format that textures returned by
                ``get_current_texture()`` will have. Must be one of the supported context
                formats. Can be ``None`` to use the canvas' preferred format.
            usage (flags.TextureUsage): Default ``TextureUsage.OUTPUT_ATTACHMENT``.
            view_formats (List[enums.TextureFormat]): The formats that views created
                from textures returned by ``get_current_texture()`` may use.
            color_space (PredefinedColorSpace): The color space that values written
                into textures returned by ``get_current_texture()`` should be displayed with.
                Default "srgb". Not yet supported.
            tone_mapping (enums.CanvasToneMappingMode): Not yet supported.
            alpha_mode (structs.CanvasAlphaMode): Determines the effect that alpha values
                will have on the content of textures returned by ``get_current_texture()``
                when read, displayed, or used as an image source. Default "opaque".
        """
        # Check types

        if not isinstance(device, GPUDevice):
            raise TypeError("Given device is not a device.")

        if format is None:
            format = self.get_preferred_format(device.adapter)
        if format not in enums.TextureFormat:
            raise ValueError(f"Configure: format {format} not in {enums.TextureFormat}")

        if not isinstance(usage, int):
            usage = str_flag_to_int(flags.TextureUsage, usage)

        color_space  # noqa - not really supported, just assume srgb for now
        tone_mapping  # noqa - not supported yet

        if alpha_mode not in enums.CanvasAlphaMode:
            raise ValueError(
                f"Configure: alpha_mode {alpha_mode} not in {enums.CanvasAlphaMode}"
            )

        # Check against capabilities

        capabilities = self._get_capabilities(device.adapter)

        if format not in capabilities["formats"]:
            raise ValueError(
                f"Configure: unsupported texture format: {format} not in {capabilities['formats']}"
            )

        if not usage & capabilities["usages"]:
            raise ValueError(
                f"Configure: unsupported texture usage: {usage} not in {capabilities['usages']}"
            )

        for view_format in view_formats:
            if view_format not in capabilities["view_formats"]:
                raise ValueError(
                    f"Configure: unsupported view format: {view_format} not in {capabilities['view_formats']}"
                )

        if alpha_mode not in capabilities["alpha_modes"]:
            raise ValueError(
                f"Configure: unsupported alpha-mode: {alpha_mode} not in {capabilities['alpha_modes']}"
            )

        # Store

        self._config = {
            "device": device,
            "format": format,
            "usage": usage,
            "view_formats": view_formats,
            "color_space": color_space,
            "tone_mapping": tone_mapping,
            "alpha_mode": alpha_mode,
        }

        if self._present_method == "screen":
            self._configure_screen(**self._config)

    def _configure_screen(
        self,
        *,
        device,
        format,
        usage,
        view_formats,
        color_space,
        tone_mapping,
        alpha_mode,
    ):
        raise NotImplementedError()

    # IDL: undefined unconfigure();
    def unconfigure(self):
        """Removes the presentation context configuration.
        Destroys any textures produced while configured.
        """
        if self._present_method == "screen":
            self._unconfigure_screen()
        self._config = None
        self._drop_texture()

    def _unconfigure_screen(self):
        raise NotImplementedError()

    # IDL: GPUTexture getCurrentTexture();
    def get_current_texture(self):
        """Get the `GPUTexture` that will be composited to the canvas next."""
        if not self._config:
            raise RuntimeError(
                "Canvas context must be configured before calling get_current_texture()."
            )

        # When the texture is active right now, we could either:
        # * return the existing texture
        # * warn about it, and create a new one
        # * raise an error
        # Right now we return the existing texture, so user can retrieve it in different render passes that write to the same frame.

        if self._texture is None:
            if self._present_method == "screen":
                self._texture = self._create_texture_screen()
            else:
                self._texture = self._create_texture_bitmap()

        return self._texture

    def _create_texture_bitmap(self):
        canvas = self._get_canvas()
        width, height = canvas.get_physical_size()
        width, height = max(width, 1), max(height, 1)

        # Note that the label 'present' is used by read_texture() to determine
        # that it can use a shared copy buffer.
        device = self._config["device"]
        self._texture = device.create_texture(
            label="present",
            size=(width, height, 1),
            format=self._config["format"],
            usage=self._config["usage"] | flags.TextureUsage.COPY_SRC,
        )
        return self._texture

    def _create_texture_screen(self):
        raise NotImplementedError()

    def _drop_texture(self):
        if self._texture:
            self._texture._release()  # not destroy, because it may be in use.
            self._texture = None

    @apidiff.add("The present method is used by the canvas")
    def present(self):
        """Hook for the canvas to present the rendered result.

        Present what has been drawn to the current texture, by compositing it to the
        canvas. Don't call this yourself; this is called automatically by the canvas.
        """

        if not self._texture:
            result = {"method": "skip"}
        elif self._present_method == "screen":
            self._present_screen()
            result = {"method": "screen"}
        elif self._present_method == "bitmap":
            bitmap = self._present_bitmap()
            result = {"method": "bitmap", "format": "rgba-u8", "data": bitmap}
        else:
            result = {"method": "fail", "message": "incompatible present methods"}

        self._drop_texture()
        return result

    def _present_bitmap(self):
        texture = self._texture
        device = texture._device

        size = texture.size
        format = texture.format
        nchannels = 4  # we expect rgba or bgra
        if not format.startswith(("rgba", "bgra")):
            raise RuntimeError(f"Image present unsupported texture format {format}.")
        if "8" in format:
            bytes_per_pixel = nchannels
        elif "16" in format:
            bytes_per_pixel = nchannels * 2
        elif "32" in format:
            bytes_per_pixel = nchannels * 4
        else:
            raise RuntimeError(
                f"Image present unsupported texture format bitdepth {format}."
            )

        data = device.queue.read_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )

        # Represent as memory object to avoid numpy dependency
        # Equivalent: np.frombuffer(data, np.uint8).reshape(size[1], size[0], nchannels)
        return data.cast("B", (size[1], size[0], nchannels))

    def _present_screen(self):
        raise NotImplementedError()

    def __del__(self):
        self._ot.decrease(self.__class__.__name__)
        self._release()

    def _release(self):
        self._drop_texture()


class GPUAdapterInfo:
    """Represents information about an adapter."""

    def __init__(self, info):
        self._info = info

    # IDL: readonly attribute DOMString vendor;
    @property
    def vendor(self):
        """The vendor that built this adaptor."""
        return self._info["vendor"]

    # IDL: readonly attribute DOMString architecture;
    @property
    def architecture(self):
        """The adapters architecrure."""
        return self._info["architecture"]

    # IDL: readonly attribute DOMString device;
    @property
    def device(self):
        """The kind of device that this adapter represents."""
        return self._info["device"]

    # IDL: readonly attribute DOMString description;
    @property
    def description(self):
        """A textual description of the adapter."""
        return self._info["description"]


class GPUAdapter:
    """Represents an abstract wgpu implementation.

    An adapter represents both an instance of a hardware accelerator
    (e.g. GPU or CPU) and an implementation of WGPU on top of that
    accelerator.

    The adapter is used to request a device object. The adapter object
    enumerates its capabilities (features) and limits.

    If an adapter becomes unavailable, it becomes invalid.
    Once invalid, it never becomes valid again.
    """

    _ot = object_tracker

    def __init__(self, internal, features, limits, adapter_info):
        self._ot.increase(self.__class__.__name__)
        self._internal = internal

        assert isinstance(features, set)
        assert isinstance(limits, dict)
        assert isinstance(adapter_info, dict)

        self._features = features
        self._limits = limits
        self._adapter_info = adapter_info

    # IDL: [SameObject] readonly attribute GPUSupportedFeatures features;
    @property
    def features(self):
        """A set of feature names supported by the adapter."""
        return self._features

    # IDL: [SameObject] readonly attribute GPUSupportedLimits limits;
    @property
    def limits(self):
        """A dict with limits for the adapter."""
        return self._limits

    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {}); -> USVString label = "", sequence<GPUFeatureName> requiredFeatures = [], record<DOMString, GPUSize64> requiredLimits = {}, GPUQueueDescriptor defaultQueue = {}
    def request_device_sync(
        self,
        *,
        label: str = "",
        required_features: List[enums.FeatureName] = [],
        required_limits: Dict[str, int] = {},
        default_queue: structs.QueueDescriptor = {},
    ):
        """Sync version of `request_device_async()`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        raise NotImplementedError()

    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {}); -> USVString label = "", sequence<GPUFeatureName> requiredFeatures = [], record<DOMString, GPUSize64> requiredLimits = {}, GPUQueueDescriptor defaultQueue = {}
    async def request_device_async(
        self,
        *,
        label: str = "",
        required_features: List[enums.FeatureName] = [],
        required_limits: Dict[str, int] = {},
        default_queue: structs.QueueDescriptor = {},
    ):
        """Request a `GPUDevice` from the adapter.

        Arguments:
            label (str): A human-readable label. Optional.
            required_features (list of str): the features (extensions) that you need. Default [].
            required_limits (dict): the various limits that you need. Default {}.
            default_queue (structs.QueueDescriptor): Descriptor for the default queue. Optional.
        """
        raise NotImplementedError()

    def _release(self):
        pass

    def __del__(self):
        self._ot.decrease(self.__class__.__name__)
        self._release()

    # IDL: readonly attribute boolean isFallbackAdapter;
    @property
    def is_fallback_adapter(self):
        """Whether this adapter runs on software (rather than dedicated hardware)."""
        return self._adapter_info.get("adapter_type", "").lower() in ("software", "cpu")

    # IDL: [SameObject] readonly attribute GPUAdapterInfo info;
    @property
    def info(self):
        """A dict with information about this adapter, such as the vendor and device name."""
        # Note: returns a dict rather than an GPUAdapterInfo instance.
        return self._adapter_info

    @apidiff.add("Useful in multi-gpu environments")
    @property
    def summary(self):
        """A one-line summary of the info of this adapter (device, adapter_type, backend_type)."""
        d = self._adapter_info
        return f"{d['device']} ({d['adapter_type']}) via {d['backend_type']}"


class GPUObjectBase:
    """The base class for all GPU objects.

    A GPU object is an object that can be thought of having a representation on
    the GPU; the device and all objects belonging to a device.
    """

    _ot = object_tracker
    _nbytes = 0

    def __init__(self, label, internal, device):
        self._ot.increase(self.__class__.__name__, self._nbytes)
        self._label = label
        self._internal = internal  # The native/raw/real GPU object
        self._device = device
        logger.info(f"Creating {self.__class__.__name__} {label}")

    # IDL: attribute USVString label;
    @property
    def label(self):
        """A human-readable name identifying the GPU object."""
        return self._label

    def __str__(self):
        if self._label:
            return f'<{self.__class__.__name__} "{self._label}">'
        else:
            return f"<{self.__class__.__name__} {id(self)}>"

    def _release(self):
        """Subclasses can implement this to clean up."""
        pass

    def __del__(self):
        self._ot.decrease(self.__class__.__name__, self._nbytes)
        self._release()

    # Public destroy() methods are implemented on classes as the WebGPU spec specifies.


class GPUDevice(GPUObjectBase):
    """The top-level interface through which GPU objects are created.

    A device is the logical instantiation of an adapter, through which
    internal objects are created. It can be shared across threads.
    A device is the exclusive owner of all internal objects created
    from it: when the device is lost, all objects created from it become
    invalid.

    Create a device using `GPUAdapter.request_device_sync()` or
    `GPUAdapter.request_device_async()`.
    """

    def __init__(self, label, internal, adapter, features, limits, queue):
        super().__init__(label, internal, self)

        assert isinstance(adapter, GPUAdapter)
        assert isinstance(features, set)
        assert isinstance(limits, dict)

        self._adapter = adapter
        self._features = features
        self._limits = limits
        self._queue = queue
        queue._device = self  # because it could not be set earlier

    # IDL: [SameObject] readonly attribute GPUSupportedFeatures features;
    @property
    def features(self):
        """A set of feature names supported by this device."""
        return self._features

    # IDL: [SameObject] readonly attribute GPUSupportedLimits limits;
    @property
    def limits(self):
        """A dict with limits for this device."""
        return self._limits

    # IDL: [SameObject] readonly attribute GPUQueue queue;
    @property
    def queue(self):
        """The default `GPUQueue` for this device."""
        return self._queue

    @apidiff.add("Too useful to not-have")
    @property
    def adapter(self):
        """The adapter object corresponding to this device."""
        return self._adapter

    # IDL: readonly attribute Promise<GPUDeviceLostInfo> lost;
    @apidiff.hide("Not a Pythonic API")
    @property
    def lost_sync(self):
        """Sync version of `lost`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        return self._get_lost_sync()

    # IDL: readonly attribute Promise<GPUDeviceLostInfo> lost;
    @apidiff.hide("Not a Pythonic API")
    @property
    async def lost_async(self):
        """Provides information about why the device is lost."""
        # In JS you can device.lost.then ... to handle lost devices.
        # We may want to eventually support something similar async-like?
        # at some point

        # Properties don't get repeated at _api.py, so we use a proxy method.
        return await self._get_lost_async()

    def _get_lost_sync(self):
        raise NotImplementedError()

    async def _get_lost_async(self):
        raise NotImplementedError()

    # IDL: attribute EventHandler onuncapturederror;
    @apidiff.hide("Specific to browsers")
    @property
    def onuncapturederror(self):
        """Event handler.

        In JS you'd do ``gpuDevice.addEventListener('uncapturederror', ...)``. We'd need
        to figure out how to do this in Python.
        """
        raise NotImplementedError()

    # IDL: undefined destroy();
    def destroy(self):
        """Destroy this device.

        This cleans up all its resources and puts it in an unusable state.
        Note that all objects get cleaned up properly automatically; this
        is only intended to support explicit destroying.

        NOTE: not yet implemented; for the moment this does nothing.
        """
        raise NotImplementedError()

    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor); -> USVString label = "", required GPUSize64 size, required GPUBufferUsageFlags usage, boolean mappedAtCreation = false
    def create_buffer(
        self,
        *,
        label: str = "",
        size: int,
        usage: flags.BufferUsage,
        mapped_at_creation: bool = False,
    ):
        """Create a `GPUBuffer` object.

        Arguments:
            label (str): A human-readable label. Optional.
            size (int): The size of the buffer in bytes.
            usage (flags.BufferUsage): The ways in which this buffer will be used.
            mapped_at_creation (bool): Whether the buffer is initially mapped.

        Alignment: the size must be a multiple of 4.

        """
        raise NotImplementedError()

    @apidiff.add("Convenience function")
    def create_buffer_with_data(self, *, label="", data, usage: "flags.BufferUsage"):
        """Create a `GPUBuffer` object initialized with the given data.

        This is a convenience function that creates a mapped buffer,
        writes the given data to it, and then unmaps the buffer.

        Arguments:
            label (str): A human-readable label. Optional.
            data: Any object supporting the Python buffer protocol (this
                includes bytes, bytearray, ctypes arrays, numpy arrays, etc.).
            usage (flags.BufferUsage): The ways in which this buffer will be used.

        Alignment: if the size (in bytes) of data is not a multiple of 4, the buffer
                   size is rounded up to the nearest multiple of 4.

        Also see `GPUBuffer.write_mapped()` and `GPUQueue.write_buffer()`.
        """
        # This function was originally created to support the workflow
        # of initializing a buffer with data when we did not support
        # buffer mapping. Now that we do have buffer mapping it is not
        # strictly necessary, but it's still quite useful and feels
        # more Pythonic than having to write the boilerplate code below.

        # Create a view of known type
        data = memoryview(data).cast("B")
        size = (data.nbytes + 3) & ~3  # round up to a multiple of 4

        # Create the buffer and write data
        buf = self.create_buffer(
            label=label, size=size, usage=usage, mapped_at_creation=True
        )
        buf.write_mapped(data)
        buf.unmap()
        return buf

    # IDL: GPUTexture createTexture(GPUTextureDescriptor descriptor); -> USVString label = "", required GPUExtent3D size, GPUIntegerCoordinate mipLevelCount = 1, GPUSize32 sampleCount = 1, GPUTextureDimension dimension = "2d", required GPUTextureFormat format, required GPUTextureUsageFlags usage, sequence<GPUTextureFormat> viewFormats = []
    def create_texture(
        self,
        *,
        label: str = "",
        size: Union[List[int], structs.Extent3D],
        mip_level_count: int = 1,
        sample_count: int = 1,
        dimension: enums.TextureDimension = "2d",
        format: enums.TextureFormat,
        usage: flags.TextureUsage,
        view_formats: List[enums.TextureFormat] = [],
    ):
        """Create a `GPUTexture` object.

        Arguments:
            label (str): A human-readable label. Optional.
            size (tuple or dict): The texture size as a 3-tuple or a `structs.Extent3D`.
            mip_level_count (int): The number of mip leveles. Default 1.
            sample_count (int): The number of samples. Default 1.
            dimension (enums.TextureDimension): The dimensionality of the texture. Default 2d.
            format (TextureFormat): What channels it stores and how.
            usage (flags.TextureUsage): The ways in which the texture will be used.
            view_formats (optional): A list of formats that views are allowed to have
              in addition to the texture's own view. Using these formats may have
              a performance penalty.

        See https://gpuweb.github.io/gpuweb/#texture-format-caps for a
        list of available texture formats. Note that fewer formats are
        available for storage usage.
        """
        raise NotImplementedError()

    # IDL: GPUSampler createSampler(optional GPUSamplerDescriptor descriptor = {}); -> USVString label = "", GPUAddressMode addressModeU = "clamp-to-edge", GPUAddressMode addressModeV = "clamp-to-edge", GPUAddressMode addressModeW = "clamp-to-edge", GPUFilterMode magFilter = "nearest", GPUFilterMode minFilter = "nearest", GPUMipmapFilterMode mipmapFilter = "nearest", float lodMinClamp = 0, float lodMaxClamp = 32, GPUCompareFunction compare, [Clamp] unsigned short maxAnisotropy = 1
    def create_sampler(
        self,
        *,
        label: str = "",
        address_mode_u: enums.AddressMode = "clamp-to-edge",
        address_mode_v: enums.AddressMode = "clamp-to-edge",
        address_mode_w: enums.AddressMode = "clamp-to-edge",
        mag_filter: enums.FilterMode = "nearest",
        min_filter: enums.FilterMode = "nearest",
        mipmap_filter: enums.MipmapFilterMode = "nearest",
        lod_min_clamp: float = 0,
        lod_max_clamp: float = 32,
        compare: enums.CompareFunction = optional,
        max_anisotropy: int = 1,
    ):
        """Create a `GPUSampler` object. Samplers specify how a texture is sampled.

        Arguments:
            label (str): A human-readable label. Optional.
            address_mode_u (enums.AddressMode): What happens when sampling beyond the x edge.
                Default "clamp-to-edge".
            address_mode_v (enums.AddressMode): What happens when sampling beyond the y edge.
                Default "clamp-to-edge".
            address_mode_w (enums.AddressMode): What happens when sampling beyond the z edge.
                Default "clamp-to-edge".
            mag_filter (enums.FilterMode): Interpolation when zoomed in. Default 'nearest'.
            min_filter (enums.FilterMode): Interpolation when zoomed out. Default 'nearest'.
            mipmap_filter: (enums.MipmapFilterMode): Interpolation between mip levels. Default 'nearest'.
            lod_min_clamp (float): The minimum level of detail. Default 0.
            lod_max_clamp (float): The maximum level of detail. Default 32.
            compare (enums.CompareFunction): The sample compare operation for depth textures.
                Only specify this for depth textures. Default None.
            max_anisotropy (int): The maximum anisotropy value clamp used by the sample,
                betweet 1 and 16, default 1.
        """
        raise NotImplementedError()

    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor); -> USVString label = "", required sequence<GPUBindGroupLayoutEntry> entries
    def create_bind_group_layout(
        self, *, label: str = "", entries: List[structs.BindGroupLayoutEntry]
    ):
        """Create a `GPUBindGroupLayout` object. One or more
        such objects are passed to `create_pipeline_layout()` to
        specify the (abstract) pipeline layout for resources. See the
        docs on bind groups for details.

        Arguments:
            label (str): A human-readable label. Optional.
            entries (list): A list of `structs.BindGroupLayoutEntry` dicts.
                Each contains either a `structs.BufferBindingLayout`,
                `structs.SamplerBindingLayout`, `structs.TextureBindingLayout`,
                or `structs.StorageTextureBindingLayout`.

        Example with `structs.BufferBindingLayout`:

        .. code-block:: py

            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage_buffer,
                    "has_dynamic_offset": False,  # optional
                    "min_binding_size": 0  # optional
                }
            },

        Note on ``has_dynamic_offset``: For uniform-buffer, storage-buffer, and
        readonly-storage-buffer bindings, it indicates whether the binding has a
        dynamic offset. One offset must be passed to `pass.set_bind_group()`
        for each dynamic binding in increasing order of binding number.
        """
        raise NotImplementedError()

    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor); -> USVString label = "", required GPUBindGroupLayout layout, required sequence<GPUBindGroupEntry> entries
    def create_bind_group(
        self,
        *,
        label: str = "",
        layout: GPUBindGroupLayout,
        entries: List[structs.BindGroupEntry],
    ):
        """Create a `GPUBindGroup` object, which can be used in
        `pass.set_bind_group()` to attach a group of resources.

        Arguments:
            label (str): A human-readable label. Optional.
            layout (GPUBindGroupLayout): The layout (abstract representation)
                for this bind group.
            entries (list): A list of `structs.BindGroupEntry` dicts. The ``resource`` field
                is either `GPUSampler`, `GPUTextureView` or `structs.BufferBinding`.

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

    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor); -> USVString label = "", required sequence<GPUBindGroupLayout> bindGroupLayouts
    def create_pipeline_layout(
        self, *, label: str = "", bind_group_layouts: List[GPUBindGroupLayout]
    ):
        """Create a `GPUPipelineLayout` object, which can be
        used in `create_render_pipeline()` or `create_compute_pipeline()`.

        Arguments:
            label (str): A human-readable label. Optional.
            bind_group_layouts (list): A list of `GPUBindGroupLayout` objects.
        """
        raise NotImplementedError()

    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor); -> USVString label = "", required USVString code, object sourceMap, sequence<GPUShaderModuleCompilationHint> compilationHints = []
    def create_shader_module(
        self,
        *,
        label: str = "",
        code: str,
        source_map: dict = optional,
        compilation_hints: List[structs.ShaderModuleCompilationHint] = [],
    ):
        """Create a `GPUShaderModule` object from shader source.

        The primary shader language is WGSL, though SpirV is also supported,
        as well as GLSL (experimental).

        Arguments:
            label (str): A human-readable label. Optional.
            code (str | bytes): The shader code, as WGSL, GLSL or SpirV.
                For GLSL code, the label must be given and contain the word
                'comp', 'vert' or 'frag'. For SpirV the code must be bytes.
            compilation_hints: currently unused.
        """
        raise NotImplementedError()

    # IDL: GPUComputePipeline createComputePipeline(GPUComputePipelineDescriptor descriptor); -> USVString label = "", required (GPUPipelineLayout or GPUAutoLayoutMode) layout, required GPUProgrammableStage compute
    def create_compute_pipeline(
        self,
        *,
        label: str = "",
        layout: Union[GPUPipelineLayout, enums.AutoLayoutMode],
        compute: structs.ProgrammableStage,
    ):
        """Create a `GPUComputePipeline` object.

        Arguments:
            label (str): A human-readable label. Optional.
            layout (GPUPipelineLayout): object created with `create_pipeline_layout()`.
            compute (structs.ProgrammableStage): Binds shader module and entrypoint.
        """
        raise NotImplementedError()

    # IDL: Promise<GPUComputePipeline> createComputePipelineAsync(GPUComputePipelineDescriptor descriptor); -> USVString label = "", required (GPUPipelineLayout or GPUAutoLayoutMode) layout, required GPUProgrammableStage compute
    async def create_compute_pipeline_async(
        self,
        *,
        label: str = "",
        layout: Union[GPUPipelineLayout, enums.AutoLayoutMode],
        compute: structs.ProgrammableStage,
    ):
        """Async version of `create_compute_pipeline()`.

        Both versions are compatible with WebGPU."""
        raise NotImplementedError()

    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor); -> USVString label = "", required (GPUPipelineLayout or GPUAutoLayoutMode) layout, required GPUVertexState vertex, GPUPrimitiveState primitive = {}, GPUDepthStencilState depthStencil, GPUMultisampleState multisample = {}, GPUFragmentState fragment
    def create_render_pipeline(
        self,
        *,
        label: str = "",
        layout: Union[GPUPipelineLayout, enums.AutoLayoutMode],
        vertex: structs.VertexState,
        primitive: structs.PrimitiveState = {},
        depth_stencil: structs.DepthStencilState = optional,
        multisample: structs.MultisampleState = {},
        fragment: structs.FragmentState = optional,
    ):
        """Create a `GPURenderPipeline` object.

        Arguments:
            label (str): A human-readable label. Optional.
            layout (GPUPipelineLayout): The layout for the new pipeline.
            vertex (structs.VertexState): Describes the vertex shader entry point of the
                pipeline and its input buffer layouts.
            primitive (structs.PrimitiveState): Describes the primitive-related properties
                of the pipeline. If `strip_index_format` is present (which means the
                primitive topology is a strip), and the drawCall is indexed, the
                vertex index list is split into sub-lists using the maximum value of this
                index format as a separator. Example: a list with values
                `[1, 2, 65535, 4, 5, 6]` of type "uint16" will be split in sub-lists
                `[1, 2]` and `[4, 5, 6]`.
            depth_stencil (structs.DepthStencilState): Describes the optional depth-stencil
                properties, including the testing, operations, and bias. Optional.
            multisample (structs.MultisampleState): Describes the multi-sampling properties of the pipeline.
            fragment (structs.FragmentState): Describes the fragment shader
                entry point of the pipeline and its output colors. If it's
                None, the No-Color-Output mode is enabled: the pipeline
                does not produce any color attachment outputs. It still
                performs rasterization and produces depth values based on
                the vertex position output. The depth testing and stencil
                operations can still be used.

        In the example dicts below, the values that are marked as optional,
        the shown value is the default.

        Example vertex (structs.VertexState) dict:

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

        Example primitive (structs.PrimitiveState) dict:

        .. code-block:: py

            {
                "topology": wgpu.PrimitiveTopology.triangle_list, # optional
                "strip_index_format": wgpu.IndexFormat.uint32,  # see note
                "front_face": wgpu.FrontFace.ccw,  # optional
                "cull_mode": wgpu.CullMode.none,  # optional
            }

        Example depth_stencil (structs.DepthStencilState) dict:

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

        Example multisample (structs.MultisampleState) dict:

        .. code-block:: py

            {
                "count": 1,  # optional
                "mask": 0xFFFFFFFF,  # optional
                "alpha_to_coverage_enabled": False  # optional
            }

        Example fragment (structs.FragmentState) dict. The `blend` parameter can be None
        to disable blending (not all texture formats support blending).

        .. code-block:: py

            {
                "module": shader_module,
                "entry_point": "main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.bgra8unorm_srgb,
                        "blend": {
                            "color": {
                                "src_target": wgpu.BlendFactor.one,  # optional
                                "dst_target": wgpu.BlendFactor.zero,  # optional
                                "operation": gpu.BlendOperation.add, # optional
                            },
                            "alpha": {
                                "src_target": wgpu.BlendFactor.one, # optional
                                "dst_target": wgpu.BlendFactor.zero, # optional
                                "operation": wgpu.BlendOperation.add, # optional
                            },
                        }
                        "write_mask": wgpu.ColorWrite.ALL  # optional
                    },
                    ...
                ]
            }

        """
        raise NotImplementedError()

    # IDL: Promise<GPURenderPipeline> createRenderPipelineAsync(GPURenderPipelineDescriptor descriptor); -> USVString label = "", required (GPUPipelineLayout or GPUAutoLayoutMode) layout, required GPUVertexState vertex, GPUPrimitiveState primitive = {}, GPUDepthStencilState depthStencil, GPUMultisampleState multisample = {}, GPUFragmentState fragment
    async def create_render_pipeline_async(
        self,
        *,
        label: str = "",
        layout: Union[GPUPipelineLayout, enums.AutoLayoutMode],
        vertex: structs.VertexState,
        primitive: structs.PrimitiveState = {},
        depth_stencil: structs.DepthStencilState = optional,
        multisample: structs.MultisampleState = {},
        fragment: structs.FragmentState = optional,
    ):
        """Async version of `create_render_pipeline()`.

        Both versions are compatible with WebGPU."""
        raise NotImplementedError()

    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {}); -> USVString label = ""
    def create_command_encoder(self, *, label: str = ""):
        """Create a `GPUCommandEncoder` object. A command
        encoder is used to record commands, which can then be submitted
        at once to the GPU.

        Arguments:
            label (str): A human-readable label. Optional.
        """
        raise NotImplementedError()

    # IDL: GPURenderBundleEncoder createRenderBundleEncoder(GPURenderBundleEncoderDescriptor descriptor); -> USVString label = "", required sequence<GPUTextureFormat?> colorFormats, GPUTextureFormat depthStencilFormat, GPUSize32 sampleCount = 1, boolean depthReadOnly = false, boolean stencilReadOnly = false
    def create_render_bundle_encoder(
        self,
        *,
        label: str = "",
        color_formats: List[enums.TextureFormat],
        depth_stencil_format: enums.TextureFormat = optional,
        sample_count: int = 1,
        depth_read_only: bool = False,
        stencil_read_only: bool = False,
    ):
        """Create a `GPURenderBundleEncoder` object.

        Render bundles represent a pre-recorded bundle of commands. In cases where the same
        commands are issued across multiple views or frames, using a rander bundle can improve
        performance by removing the overhead of repeating the commands.

        Arguments:
            label (str): A human-readable label. Optional.
            color_formats (list): A list of the `GPUTextureFormats` of the color attachments for this pass or bundle.
            depth_stencil_format (GPUTextureFormat): The format of the depth/stencil attachment for this pass or bundle.
            sample_count (int): The number of samples per pixel in the attachments for this pass or bundle. Default 1.
            depth_read_only (bool): If true, indicates that the render bundle does not modify the depth component of any depth-stencil attachments. Default False.
            stencil_read_only (bool): If true, indicates that the render bundle does not modify the stencil component of any depth-stencil attachments. Default False.
        """
        raise NotImplementedError()

    # IDL: GPUQuerySet createQuerySet(GPUQuerySetDescriptor descriptor); -> USVString label = "", required GPUQueryType type, required GPUSize32 count
    def create_query_set(self, *, label: str = "", type: enums.QueryType, count: int):
        """Create a `GPUQuerySet` object."""
        raise NotImplementedError()

    # IDL: undefined pushErrorScope(GPUErrorFilter filter);
    @apidiff.hide
    def push_error_scope(self, filter: enums.ErrorFilter):
        """Pushes a new GPU error scope onto the stack."""
        raise NotImplementedError()

    # IDL: Promise<GPUError?> popErrorScope();
    @apidiff.hide
    def pop_error_scope_sync(self):
        """Sync version of `pop_error_scope_async().

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        raise NotImplementedError()

    # IDL: Promise<GPUError?> popErrorScope();
    @apidiff.hide
    async def pop_error_scope_async(self):
        """Pops a GPU error scope from the stack."""
        raise NotImplementedError()

    # IDL: GPUExternalTexture importExternalTexture(GPUExternalTextureDescriptor descriptor); -> USVString label = "", required (HTMLVideoElement or VideoFrame) source, PredefinedColorSpace colorSpace = "srgb"
    @apidiff.hide("Specific to browsers")
    def import_external_texture(
        self,
        *,
        label: str = "",
        source: Union[memoryview, object],
        color_space: str = "srgb",
    ):
        """For browsers only."""
        raise NotImplementedError()


class GPUBuffer(GPUObjectBase):
    """Represents a block of memory that can be used in GPU operations.

    Data is stored in linear layout, meaning that each byte
    of the allocation can be addressed by its offset from the start of
    the buffer, subject to alignment restrictions depending on the
    operation.

    Create a buffer using `GPUDevice.create_buffer()`.

    One can sync data in a buffer by mapping it and then getting and setting data.
    Alternatively, one can tell the GPU (via the command encoder) to
    copy data between buffers and textures.
    """

    def __init__(self, label, internal, device, size, usage, map_state):
        self._nbytes = size
        super().__init__(label, internal, device)
        self._size = size
        self._usage = usage
        self._map_state = map_state

    # IDL: readonly attribute GPUSize64Out size;
    @property
    def size(self):
        """The length of the GPUBuffer allocation in bytes."""
        return self._size

    # IDL: readonly attribute GPUFlagsConstant usage;
    @property
    def usage(self):
        """The allowed usages (int bitmap) for this GPUBuffer, specifying
        e.g. whether the buffer may be used as a vertex buffer, uniform buffer,
        target or source for copying data, etc.
        """
        return self._usage

    # IDL: readonly attribute GPUBufferMapState mapState;
    @property
    def map_state(self):
        """The mapping state of the buffer, see `BufferMapState`."""
        return self._map_state

    # WebGPU specifies an API to sync data with the buffer via mapping.
    # The idea is to (async) request mapped data, read from / write to
    # this memory (using getMappedRange), and then unmap.  A buffer
    # must be unmapped before it can be used in a pipeline.
    #
    # This means that the mapped memory is reclaimed (i.e. invalid)
    # when unmap is called, and that whatever object we expose the
    # memory with to the user, must be set to a state where it can no
    # longer be used. There does not seem to be a good way to do this.
    #
    # In our Python API we do make use of the same map/unmap mechanism,
    # but reading and writing data goes via method calls instead of via
    # an array-like object that exposes the shared memory.

    # IDL: Promise<undefined> mapAsync(GPUMapModeFlags mode, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def map_sync(
        self, mode: flags.MapMode, offset: int = 0, size: Optional[int] = None
    ):
        """Sync version of `map_async()`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        raise NotImplementedError()

    # IDL: Promise<undefined> mapAsync(GPUMapModeFlags mode, optional GPUSize64 offset = 0, optional GPUSize64 size);
    async def map_async(
        self, mode: flags.MapMode, offset: int = 0, size: Optional[int] = None
    ):
        """Maps the given range of the GPUBuffer.

        When this call returns, the buffer content is ready to be
        accessed with ``read_mapped`` or ``write_mapped``. Don't forget
        to ``unmap()`` when done.

        Arguments:
            mode (enum): The mapping mode, either wgpu.MapMode.READ or
                wgpu.MapMode.WRITE, can also be a string.
            offset (str): the buffer offset in bytes. Default 0.
            size (int): the size to read. Default until the end.

        Alignment: the offset must be a multiple of 8, the size must be a multiple of 4.
        """
        raise NotImplementedError()

    # IDL: undefined unmap();
    def unmap(self):
        """Unmaps the buffer.

        Unmaps the mapped range of the GPUBuffer and makes its contents
        available for use by the GPU again.
        """
        raise NotImplementedError()

    @apidiff.add("Replacement for get_mapped_range")
    def read_mapped(self, buffer_offset=None, size=None, *, copy=True):
        """Read mapped buffer data.

        This method must only be called when the buffer is in a mapped state.
        This is the Python alternative to WebGPU's ``getMappedRange``.
        Returns a memoryview that is a copy of the mapped data (it won't
        become invalid when the buffer is ummapped).

        Arguments:
            buffer_offset (int): the buffer offset in bytes. Must be at
                least as large as the offset specified in ``map()``. The default
                is the offset of the mapped range.
            size (int): the size to read (in bytes). The resulting range must fit into the range
                specified in ``map()``. The default is as large as the mapped range allows.
            copy (bool): whether a copy of the data is given. Default True.
                If False, the returned memoryview represents the mapped data
                directly, and is released when the buffer is unmapped.
                WARNING: views of the returned data (e.g. memoryview objects or
                numpy arrays) can still be used after the base memory is released,
                which can result in corrupted data and segfaults. Therefore, when
                setting copy to False, make *very* sure the memory is not accessed
                after the buffer is unmapped.

        Alignment: the buffer offset must be a multiple of 8, the size must be a multiple of 4.

        Also see `GPUBuffer.write_mapped()`, `GPUQueue.read_buffer()` and `GPUQueue.write_buffer()`.
        """
        raise NotImplementedError()

    @apidiff.add("Replacement for get_mapped_range")
    def write_mapped(self, data, buffer_offset=None):
        """Write mapped buffer data.

        This method must only be called when the buffer is in a mapped state.
        This is the Python alternative to WebGPU's ``getMappedRange``.
        Since the data can also be a view into a larger array, this method
        allows updating the buffer with minimal data copying.

        Arguments:
            data (buffer-like): The data to write to the buffer.
                Must be an object that supports the buffer protocol,
                e.g. bytes, memoryview, numpy array, etc. Does not have to
                be contiguous, since the data is copied anyway.
            buffer_offset (int): the buffer offset in bytes. Must be at least
                as large as the offset specified in ``map()``. The default
                is the offset of the mapped range.

        Alignment: the buffer offset must be a multiple of 8.


        Also see `GPUBuffer.read_mapped, `GPUQueue.read_buffer()` and `GPUQueue.write_buffer()`.
        """
        raise NotImplementedError()

    # IDL: ArrayBuffer getMappedRange(optional GPUSize64 offset = 0, optional GPUSize64 size);
    @apidiff.hide
    def get_mapped_range(self, offset: int = 0, size: Optional[int] = None):
        raise NotImplementedError("The Python API differs from WebGPU here")

    # IDL: undefined destroy();
    def destroy(self):
        """Destroy the buffer.

        Explicitly destroys the buffer, freeing its memory and putting
        the object in an unusable state. In general its easier (and
        safer) to just let the garbage collector do its thing.
        """
        raise NotImplementedError()


class GPUTexture(GPUObjectBase):
    """Represents a 1D, 2D or 3D color image object.

    A texture also can have mipmaps (different levels of varying
    detail), and arrays. The texture represents the "raw" data. A
    `GPUTextureView` is used to define how the texture data
    should be interpreted.

    Create a texture using `GPUDevice.create_texture()`.
    """

    def __init__(self, label, internal, device, tex_info):
        self._nbytes = self._estimate_nbytes(tex_info)
        super().__init__(label, internal, device)
        self._tex_info = tex_info

    def _estimate_nbytes(self, tex_info):
        format = tex_info["format"]
        size = tex_info["size"]
        sample_count = tex_info["sample_count"] or 1
        mip_level_count = tex_info["mip_level_count"] or 1

        bpp = texture_format_to_bpp.get(format, 0)
        npixels = size[0] * size[1] * size[2]
        nbytes_at_mip_level = sample_count * npixels * bpp / 8

        nbytes = 0
        for i in range(mip_level_count):
            nbytes += nbytes_at_mip_level
            nbytes_at_mip_level /= 2

        # Return rounded to nearest integer
        return int(nbytes + 0.5)

    @apidiff.add("Too useful to not-have")
    @property
    def size(self):
        """The size of the texture in mipmap level 0, as a 3-tuple of ints."""
        return self._tex_info["size"]

    # IDL: readonly attribute GPUIntegerCoordinateOut width;
    @property
    def width(self):
        """The texture's width. Also see ``.size``."""
        return self._tex_info["size"][0]

    # IDL: readonly attribute GPUIntegerCoordinateOut height;
    @property
    def height(self):
        """The texture's height. Also see ``.size``."""
        return self._tex_info["size"][1]

    # IDL: readonly attribute GPUIntegerCoordinateOut depthOrArrayLayers;
    @property
    def depth_or_array_layers(self):
        """The texture's depth or number of layers. Also see ``.size``."""
        return self._tex_info["size"][2]

    # IDL: readonly attribute GPUIntegerCoordinateOut mipLevelCount;
    @property
    def mip_level_count(self):
        """The total number of the mipmap levels of the texture."""
        return self._tex_info["mip_level_count"]

    # IDL: readonly attribute GPUSize32Out sampleCount;
    @property
    def sample_count(self):
        """The number of samples in each texel of the texture."""
        return self._tex_info["sample_count"]

    # IDL: readonly attribute GPUTextureDimension dimension;
    @property
    def dimension(self):
        """The dimension of the texture."""
        return self._tex_info["dimension"]

    # IDL: readonly attribute GPUTextureFormat format;
    @property
    def format(self):
        """The format of the texture."""
        return self._tex_info["format"]

    # IDL: readonly attribute GPUFlagsConstant usage;
    @property
    def usage(self):
        """The allowed usages for this texture."""
        return self._tex_info["usage"]

    # IDL: GPUTextureView createView(optional GPUTextureViewDescriptor descriptor = {}); -> USVString label = "", GPUTextureFormat format, GPUTextureViewDimension dimension, GPUTextureAspect aspect = "all", GPUIntegerCoordinate baseMipLevel = 0, GPUIntegerCoordinate mipLevelCount, GPUIntegerCoordinate baseArrayLayer = 0, GPUIntegerCoordinate arrayLayerCount
    def create_view(
        self,
        *,
        label: str = "",
        format: enums.TextureFormat = optional,
        dimension: enums.TextureViewDimension = optional,
        aspect: enums.TextureAspect = "all",
        base_mip_level: int = 0,
        mip_level_count: int = optional,
        base_array_layer: int = 0,
        array_layer_count: int = optional,
    ):
        """Create a `GPUTextureView` object.

        If no arguments are given, a default view is given, with the
        same format and dimension as the texture.

        Arguments:
            label (str): A human-readable label. Optional.
            format (enums.TextureFormat): What channels it stores and how.
            dimension (enums.TextureViewDimension): The dimensionality of the texture view.
            aspect (enums.TextureAspect): Whether this view is used for depth, stencil, or all.
                Default all.
            base_mip_level (int): The starting mip level. Default 0.
            mip_level_count (int): The number of mip levels. Default None.
            base_array_layer (int): The starting array layer. Default 0.
            array_layer_count (int): The number of array layers. Default None.
        """
        raise NotImplementedError()

    # IDL: undefined destroy();
    def destroy(self):
        """Destroy the texture.

        Explicitly destroys the texture, freeing its memory and putting
        the object in an unusable state. In general its easier (and
        safer) to just let the garbage collector do its thing.
        """
        raise NotImplementedError()


class GPUTextureView(GPUObjectBase):
    """Represents a way to represent a `GPUTexture`.

    Create a texture view using `GPUTexture.create_view()`.
    """

    def __init__(self, label, internal, device, texture, size):
        super().__init__(label, internal, device)
        self._texture = texture
        self._size = size

    @apidiff.add("Need to know size e.g. for texture view provided by canvas")
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
    """Defines how a texture (view) must be sampled by the shader.

    It defines the subsampling, sampling between mip levels, and sampling out
    of the image boundaries.

    Create a sampler using `GPUDevice.create_sampler()`.
    """


class GPUBindGroupLayout(GPUObjectBase):
    """Defines the interface between a set of resources bound in a `GPUBindGroup`.

    It also defines their accessibility in shader stages.

    Create a bind group layout using `GPUDevice.create_bind_group_layout()`.
    """

    pass


class GPUBindGroup(GPUObjectBase):
    """Represents a group of resource bindings (buffer, sampler, texture-view).

    It holds the shader slot and a reference to the resource (sampler,
    texture-view, buffer).

    Create a bind group using `GPUDevice.create_bind_group()`.
    """

    pass


class GPUPipelineLayout(GPUObjectBase):
    """Describes the layout of a pipeline, as a list of `GPUBindGroupLayout` objects.

    Create a pipeline layout using `GPUDevice.create_pipeline_layout()`.
    """

    pass


class GPUShaderModule(GPUObjectBase):
    """Represents a programmable shader.

    Create a shader module using `GPUDevice.create_shader_module()`.
    """

    # IDL: Promise<GPUCompilationInfo> getCompilationInfo();
    def get_compilation_info_sync(self):
        """Sync version of `get_compilation_info_async()`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        raise NotImplementedError()

    # IDL: Promise<GPUCompilationInfo> getCompilationInfo();
    async def get_compilation_info_async(self):
        """Get shader compilation info. Always returns empty list at the moment."""
        # How can this return shader errors if one cannot create a
        # shader module when the shader source has errors?
        raise NotImplementedError()


class GPUPipelineBase:
    """A mixin class for render and compute pipelines."""

    # IDL: [NewObject] GPUBindGroupLayout getBindGroupLayout(unsigned long index);
    def get_bind_group_layout(self, index: int):
        """Get the bind group layout at the given index."""
        raise NotImplementedError()


class GPUComputePipeline(GPUPipelineBase, GPUObjectBase):
    """Represents a single pipeline for computations (no rendering).

    Create a compute pipeline using `GPUDevice.create_compute_pipeline()`.
    """


class GPURenderPipeline(GPUPipelineBase, GPUObjectBase):
    """Represents a single pipeline to draw something.

    The rendering typically involves a vertex and fragment stage, though
    the latter is optional.
    The render target can come from a window on the screen or from an
    in-memory texture (off-screen rendering).

    Create a render pipeline using `GPUDevice.create_render_pipeline()`.
    """


class GPUCommandBuffer(GPUObjectBase):
    """Stores a series of commands generated by a `GPUCommandEncoder`.

    The buffered commands can subsequently be submitted to a `GPUQueue`.

    Command buffers are single use, you must only submit them once and
    submitting them destroys them. Use render bundles to re-use commands.

    Create a command buffer using `GPUCommandEncoder.finish()`.
    """


class GPUCommandsMixin:
    """Mixin for classes that encode commands."""

    pass


class GPUBindingCommandsMixin:
    """Mixin for classes that defines bindings."""

    # IDL: undefined setBindGroup(GPUIndex32 index, GPUBindGroup? bindGroup, Uint32Array dynamicOffsetsData, GPUSize64 dynamicOffsetsDataStart, GPUSize32 dynamicOffsetsDataLength);
    @apidiff.change(
        "In the WebGPU specification, this method has two different signatures."
    )
    def set_bind_group(
        self,
        index,
        bind_group,
        dynamic_offsets_data=[],
        dynamic_offsets_data_start=None,
        dynamic_offsets_data_length=None,
    ):
        """Associate the given bind group (i.e. group or resources) with the
        given slot/index.

        Arguments:
            index (int): The slot to bind at.
            bind_group (GPUBindGroup): The bind group to bind.
            dynamic_offsets_data (list of int): A list of offsets (one for each entry in bind group marked as ``buffer.has_dynamic_offset``). Default ``[]``.
            dynamic_offsets_data_start (int): Offset in elements into dynamic_offsets_data where the buffer offset data begins. Default None.
            dynamic_offsets_data_length (int): Number of buffer offsets to read from dynamic_offsets_data. Default None.
        """
        raise NotImplementedError()


class GPUDebugCommandsMixin:
    """Mixin for classes that support debug groups and markers."""

    # IDL: undefined pushDebugGroup(USVString groupLabel);
    def push_debug_group(self, group_label: str):
        """Push a named debug group into the command stream."""
        raise NotImplementedError()

    # IDL: undefined popDebugGroup();
    def pop_debug_group(self):
        """Pop the active debug group."""
        raise NotImplementedError()

    # IDL: undefined insertDebugMarker(USVString markerLabel);
    def insert_debug_marker(self, marker_label: str):
        """Insert the given message into the debug message queue."""
        raise NotImplementedError()


class GPURenderCommandsMixin:
    """Mixin for classes that provide rendering commands."""

    # IDL: undefined setPipeline(GPURenderPipeline pipeline);
    def set_pipeline(self, pipeline: GPURenderPipeline):
        """Set the pipeline for this render pass.

        Arguments:
            pipeline (GPURenderPipeline): The pipeline to use.
        """
        raise NotImplementedError()

    # IDL: undefined setIndexBuffer(GPUBuffer buffer, GPUIndexFormat indexFormat, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def set_index_buffer(
        self,
        buffer: GPUBuffer,
        index_format: enums.IndexFormat,
        offset: int = 0,
        size: Optional[int] = None,
    ):
        """Set the index buffer for this render pass.

        Arguments:
            buffer (GPUBuffer): The buffer that contains the indices.
            index_format (GPUIndexFormat): The format of the index data
                contained in buffer. If `strip_index_format` is given in the
                call to `GPUDevice.create_render_pipeline()`, it must match.
            offset (int): The byte offset in the buffer. Default 0.
            size (int): The number of bytes to use. If zero, the remaining size
                (after offset) of the buffer is used. Default 0.
        """
        raise NotImplementedError()

    # IDL: undefined setVertexBuffer(GPUIndex32 slot, GPUBuffer? buffer, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def set_vertex_buffer(
        self, slot: int, buffer: GPUBuffer, offset: int = 0, size: Optional[int] = None
    ):
        """Associate a vertex buffer with a bind slot.

        Arguments:
            slot (int): The binding slot for the vertex buffer.
            buffer (GPUBuffer): The buffer that contains the vertex data.
            offset (int): The byte offset in the buffer. Default 0.
            size (int): The number of bytes to use. If zero, the remaining size
                (after offset) of the buffer is used. Default 0.

        Alignment: the offset must be a multiple of 4.
        """
        raise NotImplementedError()

    # IDL: undefined draw(GPUSize32 vertexCount, optional GPUSize32 instanceCount = 1, optional GPUSize32 firstVertex = 0, optional GPUSize32 firstInstance = 0);
    def draw(
        self,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ):
        """Run the render pipeline without an index buffer.

        Arguments:
            vertex_count (int): The number of vertices to draw.
            instance_count (int):  The number of instances to draw. Default 1.
            first_vertex (int): The vertex offset. Default 0.
            first_instance (int):  The instance offset. Default 0.
        """
        raise NotImplementedError()

    # IDL: undefined drawIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indirect(self, indirect_buffer: GPUBuffer, indirect_offset: int):
        """Like `draw()`, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.

        Alignment: the indirect offset must be a multiple of 4.
        """
        raise NotImplementedError()

    # IDL: undefined drawIndexed(GPUSize32 indexCount, optional GPUSize32 instanceCount = 1, optional GPUSize32 firstIndex = 0, optional GPUSignedOffset32 baseVertex = 0, optional GPUSize32 firstInstance = 0);
    def draw_indexed(
        self,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ):
        """Run the render pipeline using an index buffer.

        Arguments:
            index_count (int): The number of indices to draw.
            instance_count (int): The number of instances to draw. Default 1.
            first_index (int):  The index offset. Default 0.
            base_vertex (int):  A number added to each index in the index buffer. Default 0.
            first_instance (int): The instance offset. Default 0.

        Alignment: the indirect offset must be a multiple of 4.
        """
        raise NotImplementedError()

    # IDL: undefined drawIndexedIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def draw_indexed_indirect(self, indirect_buffer: GPUBuffer, indirect_offset: int):
        """
        Like `draw_indexed()`, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()


class GPUCommandEncoder(GPUCommandsMixin, GPUDebugCommandsMixin, GPUObjectBase):
    """Object to record a series of commands.

    When done, call `finish()` to obtain a `GPUCommandBuffer` object.

    Create a command encoder using `GPUDevice.create_command_encoder()`.
    """

    # IDL: GPUComputePassEncoder beginComputePass(optional GPUComputePassDescriptor descriptor = {}); -> USVString label = "", GPUComputePassTimestampWrites timestampWrites
    def begin_compute_pass(
        self,
        *,
        label: str = "",
        timestamp_writes: structs.ComputePassTimestampWrites = optional,
    ):
        """Record the beginning of a compute pass. Returns a
        `GPUComputePassEncoder` object.

        Arguments:
            label (str): A human-readable label. Optional.
            timestamp_writes: unused
        """
        raise NotImplementedError()

    # IDL: GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor descriptor); -> USVString label = "", required sequence<GPURenderPassColorAttachment?> colorAttachments, GPURenderPassDepthStencilAttachment depthStencilAttachment, GPUQuerySet occlusionQuerySet, GPURenderPassTimestampWrites timestampWrites, GPUSize64 maxDrawCount = 50000000
    def begin_render_pass(
        self,
        *,
        label: str = "",
        color_attachments: List[structs.RenderPassColorAttachment],
        depth_stencil_attachment: structs.RenderPassDepthStencilAttachment = optional,
        occlusion_query_set: GPUQuerySet = optional,
        timestamp_writes: structs.RenderPassTimestampWrites = optional,
        max_draw_count: int = 50000000,
    ):
        """Record the beginning of a render pass. Returns a
        `GPURenderPassEncoder` object.

        Arguments:
            label (str): A human-readable label. Optional.
            color_attachments (list): List of `structs.RenderPassColorAttachment` dicts.
            depth_stencil_attachment (structs.RenderPassDepthStencilAttachment): Describes the depth stencil attachment. Default None.
            occlusion_query_set (GPUQuerySet): Default None.
            timestamp_writes: unused
        """
        raise NotImplementedError()

    # IDL: undefined clearBuffer( GPUBuffer buffer, optional GPUSize64 offset = 0, optional GPUSize64 size);
    def clear_buffer(
        self, buffer: GPUBuffer, offset: int = 0, size: Optional[int] = None
    ):
        """Set (part of) the given buffer to zeros.

        Arguments:
            buffer (GPUBuffer): The buffer to clear.
            offset (int): The byte offset.
            size (int, optional): The size to clear in bytes. If None, the effective size is the full size minus the offset.

        Alignment: both offset and size must be a multiple of 4.
        """
        raise NotImplementedError()

    # IDL: undefined copyBufferToBuffer( GPUBuffer source, GPUSize64 sourceOffset, GPUBuffer destination, GPUSize64 destinationOffset, GPUSize64 size);
    def copy_buffer_to_buffer(
        self,
        source: GPUBuffer,
        source_offset: int,
        destination: GPUBuffer,
        destination_offset: int,
        size: int,
    ):
        """Copy the contents of a buffer to another buffer.

        Arguments:
            source (GPUBuffer): The source buffer.
            source_offset (int): The byte offset.
            destination (GPUBuffer): The target buffer.
            destination_offset (int): The byte offset in the destination buffer.
            size (int): The number of bytes to copy.

        Alignment: the size, source offset, and destination offset must all be a multiple of 4.
        """
        raise NotImplementedError()

    # IDL: undefined copyBufferToTexture( GPUImageCopyBuffer source, GPUImageCopyTexture destination, GPUExtent3D copySize);
    def copy_buffer_to_texture(
        self,
        source: structs.ImageCopyBuffer,
        destination: structs.ImageCopyTexture,
        copy_size: Union[List[int], structs.Extent3D],
    ):
        """Copy the contents of a buffer to a texture (view).

        Arguments:
            source (GPUBuffer): A dict with fields: buffer, offset, bytes_per_row, rows_per_image.
            destination (GPUTexture): A dict with fields: texture, mip_level, origin.
            copy_size (int): The number of bytes to copy.

        Alignment: the ``bytes_per_row`` must be a multiple of 256.
        """
        raise NotImplementedError()

    # IDL: undefined copyTextureToBuffer( GPUImageCopyTexture source, GPUImageCopyBuffer destination, GPUExtent3D copySize);
    def copy_texture_to_buffer(
        self,
        source: structs.ImageCopyTexture,
        destination: structs.ImageCopyBuffer,
        copy_size: Union[List[int], structs.Extent3D],
    ):
        """Copy the contents of a texture (view) to a buffer.

        Arguments:
            source (GPUTexture): A dict with fields: texture, mip_level, origin.
            destination (GPUBuffer):  A dict with fields: buffer, offset, bytes_per_row, rows_per_image.
            copy_size (int): The number of bytes to copy.

        Alignment: the ``bytes_per_row`` must be a multiple of 256.
        """
        raise NotImplementedError()

    # IDL: undefined copyTextureToTexture( GPUImageCopyTexture source, GPUImageCopyTexture destination, GPUExtent3D copySize);
    def copy_texture_to_texture(
        self,
        source: structs.ImageCopyTexture,
        destination: structs.ImageCopyTexture,
        copy_size: Union[List[int], structs.Extent3D],
    ):
        """Copy the contents of a texture (view) to another texture (view).

        Arguments:
            source (GPUTexture): A dict with fields: texture, mip_level, origin.
            destination (GPUTexture):  A dict with fields: texture, mip_level, origin.
            copy_size (int): The number of bytes to copy.
        """
        raise NotImplementedError()

    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {}); -> USVString label = ""
    def finish(self, *, label: str = ""):
        """Finish recording. Returns a `GPUCommandBuffer` to
        submit to a `GPUQueue`.

        Arguments:
            label (str): A human-readable label. Optional.
        """
        raise NotImplementedError()

    # IDL: undefined resolveQuerySet( GPUQuerySet querySet, GPUSize32 firstQuery, GPUSize32 queryCount, GPUBuffer destination, GPUSize64 destinationOffset);
    def resolve_query_set(
        self,
        query_set: GPUQuerySet,
        first_query: int,
        query_count: int,
        destination: GPUBuffer,
        destination_offset: int,
    ):
        """
        Resolves query results from a ``GPUQuerySet`` out into a range of a ``GPUBuffer``.

        Arguments:
            query_set (GPUQuerySet): The source query set.
            first_query (int): The first query to resolve.
            query_count (int): The amount of queries to resolve.
            destination (GPUBuffer): The buffer to write the results to.
            destination_offset (int): The byte offset in the buffer.

        Alignment: the destination offset must be a multiple of 256.
        """
        raise NotImplementedError()


class GPUComputePassEncoder(
    GPUCommandsMixin, GPUDebugCommandsMixin, GPUBindingCommandsMixin, GPUObjectBase
):
    """Object to records commands for a compute pass.

    Create a compute pass encoder using `GPUCommandEncoder.begin_compute_pass()`.
    """

    # IDL: undefined setPipeline(GPUComputePipeline pipeline);
    def set_pipeline(self, pipeline: GPUComputePipeline):
        """Set the pipeline for this compute pass.

        Arguments:
            pipeline (GPUComputePipeline): The pipeline to use.
        """
        raise NotImplementedError()

    # IDL: undefined dispatchWorkgroups(GPUSize32 workgroupCountX, optional GPUSize32 workgroupCountY = 1, optional GPUSize32 workgroupCountZ = 1);
    def dispatch_workgroups(
        self,
        workgroup_count_x: int,
        workgroup_count_y: int = 1,
        workgroup_count_z: int = 1,
    ):
        """Run the compute shader.

        Arguments:
            x (int): The number of cycles in index x.
            y (int): The number of cycles in index y. Default 1.
            z (int): The number of cycles in index z. Default 1.
        """
        raise NotImplementedError()

    # IDL: undefined dispatchWorkgroupsIndirect(GPUBuffer indirectBuffer, GPUSize64 indirectOffset);
    def dispatch_workgroups_indirect(
        self, indirect_buffer: GPUBuffer, indirect_offset: int
    ):
        """Like `dispatch_workgroups()`, but the function arguments are in a buffer.

        Arguments:
            indirect_buffer (GPUBuffer): The buffer that contains the arguments.
            indirect_offset (int): The byte offset at which the arguments are.
        """
        raise NotImplementedError()

    # IDL: undefined end();
    def end(self):
        """Record the end of the compute pass."""
        raise NotImplementedError()


class GPURenderPassEncoder(
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPURenderCommandsMixin,
    GPUObjectBase,
):
    """Object to records commands for a render pass.

    Create a render pass encoder using `GPUCommandEncoder.begin_render_pass`.
    """

    # IDL: undefined setViewport(float x, float y, float width, float height, float minDepth, float maxDepth);
    def set_viewport(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        min_depth: float,
        max_depth: float,
    ):
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
    def set_scissor_rect(self, x: int, y: int, width: int, height: int):
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
    def set_blend_constant(self, color: Union[List[float], structs.Color]):
        """Set the blend color for the render pass.

        Arguments:
            color (tuple or dict): A color with fields (r, g, b, a).
        """
        raise NotImplementedError()

    # IDL: undefined setStencilReference(GPUStencilValue reference);
    def set_stencil_reference(self, reference: int):
        """Set the reference stencil value for this render pass.

        Arguments:
            reference (int): The reference value.
        """
        raise NotImplementedError()

    # IDL: undefined executeBundles(sequence<GPURenderBundle> bundles);
    def execute_bundles(self, bundles: List[GPURenderBundle]):
        """Executes commands previously recorded into the render bundles
          as part of this render pass.

        Arguments:
            bundles (Sequence[GPURenderBundle]): A sequence of Render Bundle objects.
        """
        raise NotImplementedError()

    # IDL: undefined end();
    def end(self):
        """Record the end of the render pass."""
        raise NotImplementedError()

    # IDL: undefined beginOcclusionQuery(GPUSize32 queryIndex);
    def begin_occlusion_query(self, query_index: int):
        """Begins an occlusion query.

        Arguments:
            query_index (int): The index in the GPUQuerySet at which to write the
                result of the occlusion query. The Query Set is specified as the
                occlusion_query_set argument in begin_render_pass().
        """

        raise NotImplementedError()

    # IDL: undefined endOcclusionQuery();
    def end_occlusion_query(self):
        """Ends an occlusion query."""
        raise NotImplementedError()


class GPURenderBundle(GPUObjectBase):
    """A reusable bundle of render commands."""

    pass


class GPURenderBundleEncoder(
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPURenderCommandsMixin,
    GPUObjectBase,
):
    """Encodes a series of render commands into a reusable render bundle."""

    # IDL: GPURenderBundle finish(optional GPURenderBundleDescriptor descriptor = {}); -> USVString label = ""
    def finish(self, *, label: str = ""):
        """Finish recording and return a `GPURenderBundle`.

        Arguments:
            label (str): A human-readable label. Optional.
        """
        raise NotImplementedError()


class GPUQueue(GPUObjectBase):
    """Object to submit command buffers to.

    You can obtain a queue object via the :attr:`GPUDevice.queue` property.
    """

    # IDL: undefined submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self, command_buffers: List[GPUCommandBuffer]):
        """Submit a `GPUCommandBuffer` to the queue.

        Arguments:
            command_buffers (list): The `GPUCommandBuffer` objects to add.
        """
        raise NotImplementedError()

    # IDL: undefined writeBuffer( GPUBuffer buffer, GPUSize64 bufferOffset, AllowSharedBufferSource data, optional GPUSize64 dataOffset = 0, optional GPUSize64 size);
    def write_buffer(
        self,
        buffer: GPUBuffer,
        buffer_offset: int,
        data: memoryview,
        data_offset: int = 0,
        size: Optional[int] = None,
    ):
        """Takes the data contents and schedules a write operation to the buffer.

        Changes to the data after this function is called don't affect
        the buffer contents.

        Arguments:
            buffer: The `GPUBuffer` object to write to.
            buffer_offset (int): The offset in the buffer to start writing at.
            data: The data to write to the buffer. Must be an object that supports
                the buffer protocol, e.g. bytes, memoryview, numpy array, etc.
                Must be contiguous.
            data_offset: The offset in the data, in elements. Default 0.
            size: The number of bytes to write. Default all minus offset.

        This maps the data to a temporary buffer and then copies that buffer
        to the given buffer. The given buffer's usage must include COPY_DST.

        Alignment: the buffer offset must be a multiple of 4, the total size to write must be a multiple of 4 bytes.

        Also see `GPUBuffer.map_sync()` and `GPUBuffer.map_async()`.

        """
        raise NotImplementedError()

    @apidiff.add("For symmetry with queue.write_buffer")
    def read_buffer(self, buffer, buffer_offset=0, size=None):
        """Takes the data contents of the buffer and return them as a memoryview.

        Arguments:
            buffer: The `GPUBuffer` object to read from.
            buffer_offset (int): The offset in the buffer to start reading from.
            size: The number of bytes to read. Default all minus offset.

        This copies the data in the given buffer to a temporary buffer
        and then maps that buffer to read the data. The given buffer's
        usage must include COPY_SRC.

        Also see `GPUBuffer._sync()` and `GPUBuffer._async()`.
        """
        raise NotImplementedError()

    # IDL: undefined writeTexture( GPUImageCopyTexture destination, AllowSharedBufferSource data, GPUImageDataLayout dataLayout, GPUExtent3D size);
    def write_texture(
        self,
        destination: structs.ImageCopyTexture,
        data: memoryview,
        data_layout: structs.ImageDataLayout,
        size: Union[List[int], structs.Extent3D],
    ):
        """Takes the data contents and schedules a write operation of
        these contents to the destination texture in the queue. A
        snapshot of the data is taken; any changes to the data after
        this function is called do not affect the texture contents.

        Arguments:
            destination: A dict with fields: "texture" (a texture object),
                "origin" (a 3-tuple), "mip_level" (an int, default 0).
            data: The data to write to the texture. Must be an object that supports
                the buffer protocol, e.g. bytes, memoryview, numpy array, etc.
                Must be contiguous.
            data_layout: A dict with fields: "offset" (an int, default 0),
                "bytes_per_row" (an int), "rows_per_image" (an int, default 0).
            size: A 3-tuple of ints specifying the size to write.

        Unlike `GPUCommandEncoder.copyBufferToTexture()`, there is
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

        Unlike `GPUCommandEncoder.copyBufferToTexture()`, there is
        no alignment requirement on `bytes_per_row`, although in the
        current implementation there will be a performance penalty if
        ``bytes_per_row`` is not a multiple of 256 (because we'll be
        copying data row-by-row in Python).
        """
        raise NotImplementedError()

    # IDL: undefined copyExternalImageToTexture( GPUImageCopyExternalImage source, GPUImageCopyTextureTagged destination, GPUExtent3D copySize);
    @apidiff.hide("Specific to browsers")
    def copy_external_image_to_texture(
        self,
        source: structs.ImageCopyExternalImage,
        destination: structs.ImageCopyTextureTagged,
        copy_size: Union[List[int], structs.Extent3D],
    ):
        raise NotImplementedError()

    # IDL: Promise<undefined> onSubmittedWorkDone();
    def on_submitted_work_done_sync(self):
        """Sync version of `on_submitted_work_done_async()`.

        Provided by wgpu-py, but not compatible with WebGPU.
        """
        raise NotImplementedError()

    # IDL: Promise<undefined> onSubmittedWorkDone();
    async def on_submitted_work_done_async(self):
        """TODO"""
        raise NotImplementedError()


# %% Further non-GPUObject classes


class GPUDeviceLostInfo:
    """An object that contains information about the device being lost."""

    # Not used at the moment, see device.lost prop

    def __init__(self, reason, message):
        self._reason = reason
        self._message = message

    # IDL: readonly attribute DOMString message;
    @property
    def message(self):
        """The error message specifying the reason for the device being lost."""
        return self._message

    # IDL: readonly attribute GPUDeviceLostReason reason;
    @property
    def reason(self):
        """The reason (enums.GPUDeviceLostReason) for the device getting lost. Can be None."""
        return self._reason


class GPUError(Exception):
    """A generic GPU error."""

    def __init__(self, message):
        super().__init__(message)

    # IDL: readonly attribute DOMString message;
    @property
    def message(self):
        """The error message."""
        return self.args[0]


class GPUOutOfMemoryError(GPUError, MemoryError):
    """An error raised when the GPU is out of memory."""

    # IDL: constructor(DOMString message);
    def __init__(self, message: str):
        super().__init__(message or "GPU is out of memory.")


class GPUValidationError(GPUError):
    """An error raised when the pipeline could not be validated."""

    # IDL: constructor(DOMString message);
    def __init__(self, message: str):
        super().__init__(message)


class GPUPipelineError(Exception):
    """An error raised when a pipeline could not be created."""

    # IDL: constructor(optional DOMString message = "", GPUPipelineErrorInit options);
    def __init__(self, message: str, options: structs.PipelineErrorInit):
        super().__init__(message or "")
        self._options = options

    # IDL: readonly attribute GPUPipelineErrorReason reason;
    @property
    def reason(self):
        """The reason for the failure."""
        return self.args[0]


class GPUInternalError(GPUError):
    """An error raised for implementation-specific reasons.

    An operation failed for a system or implementation-specific
    reason even when all validation requirements have been satisfied.
    """

    # IDL: constructor(DOMString message);
    def __init__(self, message: str):
        super().__init__(message)


# %% Not implemented


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


class GPUCompilationInfo:
    """TODO"""

    # IDL: readonly attribute FrozenArray<GPUCompilationMessage> messages;
    @property
    def messages(self):
        """A list of `GPUCompilationMessage` objects."""
        raise NotImplementedError()


class GPUQuerySet(GPUObjectBase):
    """An object to store the results of queries on passes.

    You can obtain a query set object via :attr:`GPUDevice.create_query_set`.
    """

    def __init__(self, label, internal, device, type, count):
        super().__init__(label, internal, device)
        self._type = type
        self._count = count

    # IDL: readonly attribute GPUQueryType type;
    @property
    def type(self):
        """The type of the queries managed by this queryset."""
        return self._type

    # IDL: readonly attribute GPUSize32Out count;
    @property
    def count(self):
        """The number of the queries managed by this queryset."""
        return self._count

    # IDL: undefined destroy();
    def destroy(self):
        """Destroy the QuerySet.

        This cleans up all its resources and puts it in an unusable state.
        Note that all objects get cleaned up properly automatically; this
        is only intended to support explicit destroying.

        NOTE: not yet implemented; for the moment this does nothing.
        """
        raise NotImplementedError()


# %%%%% Post processing

# Note that some toplevel classes are already filtered out by the codegen,
# like GPUExternalTexture and GPUUncapturedErrorEvent, and more.

apidiff.remove_hidden_methods(globals())


def _seed_object_counts():
    m = globals()
    for class_name in __all__:
        cls = m[class_name]
        if not class_name.endswith(("Base", "Mixin")):
            if hasattr(cls, "_ot"):
                object_tracker.counts[class_name] = 0


def generic_repr(self):
    try:
        module_name = self.__module__
        if module_name.startswith("wgpu"):
            if module_name == "wgpu._classes":
                module_name = "wgpu"
            elif "backends." in module_name:
                backend_name = self.__module__.split("backends")[-1].split(".")[1]
                module_name = f"wgpu.backends.{backend_name}"
        object_str = "object"
        if isinstance(self, GPUObjectBase):
            object_str = f"object '{self.label}'"
        return (
            f"<{module_name}.{self.__class__.__name__} {object_str} at {hex(id(self))}>"
        )
    except Exception:  # easy fallback
        return object.__repr__(self)


def _set_repr_methods():
    m = globals()
    for class_name in __all__:
        cls = m[class_name]
        if len(cls.mro()) == 2:  # class itself and object
            cls.__repr__ = generic_repr


_async_warnings = {}


def _set_compat_methods_for_async_methods():
    def create_new_method(name):
        def proxy_method(self, *args, **kwargs):
            warning = _async_warnings.pop(name, None)
            if warning:
                logger.warning(warning)
            return getattr(self, name)(*args, **kwargs)

        proxy_method.__name__ = name + "_backwards_compat_proxy"
        proxy_method.__doc__ = f"Backwards compatible method for {name}()"
        return proxy_method

    m = globals()
    for class_name in __all__:
        cls = m[class_name]
        for name, func in list(cls.__dict__.items()):
            if name.endswith("_sync") and callable(func):
                old_name = name[:-5]
                setattr(cls, old_name, create_new_method(name))
                _async_warnings[name] = (
                    f"WGPU: {old_name}() is deprecated, use {name}() instead."
                )


_seed_object_counts()
_set_repr_methods()
_set_compat_methods_for_async_methods()
