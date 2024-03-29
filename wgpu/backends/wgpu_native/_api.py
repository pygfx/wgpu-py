"""
WGPU backend implementation based on wgpu-native.

The wgpu-native project (https://github.com/gfx-rs/wgpu-native) is a Rust
library based on wgpu-core, which wraps Metal, Vulkan, DX12, and more.
It compiles to a dynamic library exposing a C-API, accompanied by a C
header file. We wrap this using cffi, which uses the header file to do
most type conversions for us.

This module is maintained using a combination of manual code and
automatically inserted code. In short, the codegen utility inserts
new methods and checks plus annotates all structs and C api calls.

Read the codegen/readme.md for more information.
"""

import os
import ctypes
import logging
import ctypes.util
from weakref import WeakKeyDictionary
from typing import List, Dict, Union

from ... import classes, flags, enums, structs
from ..._coreutils import str_flag_to_int

from ._ffi import ffi, lib
from ._mappings import cstructfield2enum, enummap, enum_str2int, enum_int2str
from ._helpers import (
    get_wgpu_instance,
    get_surface_id_from_canvas,
    get_memoryview_from_address,
    get_memoryview_and_address,
    to_snake_case,
    to_camel_case,
    ErrorHandler,
    SafeLibCalls,
)


logger = logging.getLogger("wgpu")  # noqa


# The API is prettu well defined
__all__ = classes.__all__.copy()


# %% Helper functions and objects


# Features that wgpu-native supports that are not part of WebGPU
NATIVE_FEATURES = (
    "PushConstants",
    "TextureAdapterSpecificFormatFeatures",
    "MultiDrawIndirect",
    "MultiDrawIndirectCount",
    "VertexWritableStorage",
)

# Object to be able to bind the lifetime of objects to other objects
_refs_per_struct = WeakKeyDictionary()

# Some enum keys need a shortcut
_cstructfield2enum_alt = {
    "load_op": "LoadOp",
    "store_op": "StoreOp",
    "depth_store_op": "StoreOp",
    "stencil_store_op": "StoreOp",
}


def new_struct_p(ctype, **kwargs):
    """Create a pointer to an ffi struct. Provides a flatter syntax
    and converts our string enums to int enums needed in C. The passed
    kwargs are also bound to the lifetime of the new struct.
    """
    assert ctype.endswith(" *")
    struct_p = _new_struct_p(ctype, **kwargs)
    _refs_per_struct[struct_p] = kwargs
    return struct_p
    # Some kwargs may be other ffi objects, and some may represent
    # pointers. These need special care because them "being in" the
    # current struct does not prevent them from being cleaned up by
    # Python's garbage collector. Keeping hold of these objects in the
    # calling code is painful and prone to missing cases, so we solve
    # the issue here. We cannot attach an attribute to the struct directly,
    # so we use a global WeakKeyDictionary. Also see issue #52.


def new_struct(ctype, **kwargs):
    """Create an ffi value struct. The passed kwargs are also bound
    to the lifetime of the new struct.
    """
    assert not ctype.endswith("*")
    struct_p = _new_struct_p(ctype + " *", **kwargs)
    struct = struct_p[0]
    _refs_per_struct[struct] = kwargs
    return struct


def _new_struct_p(ctype, **kwargs):
    struct_p = ffi.new(ctype)
    for key, val in kwargs.items():
        if isinstance(val, str) and isinstance(getattr(struct_p, key), int):
            # An enum - these are ints in C, but str in our public API
            if key in _cstructfield2enum_alt:
                structname = _cstructfield2enum_alt[key]
            else:
                structname = cstructfield2enum[ctype.strip(" *")[4:] + "." + key]
            ival = enummap[structname + "." + val]
            setattr(struct_p, key, ival)
        else:
            setattr(struct_p, key, val)
    return struct_p


def _tuple_from_tuple_or_dict(ob, fields):
    """Given a tuple/list/dict, return a tuple. Also checks tuple size.

    >> # E.g.
    >> _tuple_from_tuple_or_dict({"x": 1, "y": 2}, ("x", "y"))
    (1, 2)
    >> _tuple_from_tuple_or_dict([1, 2], ("x", "y"))
    (1, 2)
    """
    error_msg = "Expected tuple/key/dict with fields: {}"
    if isinstance(ob, (list, tuple)):
        if len(ob) != len(fields):
            raise ValueError(error_msg.format(", ".join(fields)))
        return tuple(ob)
    elif isinstance(ob, dict):
        try:
            return tuple(ob[key] for key in fields)
        except KeyError:
            raise ValueError(error_msg.format(", ".join(fields)))
    else:
        raise TypeError(error_msg.format(", ".join(fields)))


_empty_label = ffi.new("char []", b"")


def to_c_label(label):
    """Get the C representation of a label."""
    if not label:
        return _empty_label
    else:
        return ffi.new("char []", label.encode())


def feature_flag_to_feature_names(flag):
    """Convert a feature flags into a tuple of names."""
    feature_names = {}  # import this from mappings?
    features = []
    for i in range(32):
        val = int(2**i)
        if flag & val:
            features.append(feature_names.get(val, val))
    return tuple(sorted(features))


def check_struct(struct_name, d):
    """Check that all keys in the given dict exist in the corresponding struct."""
    valid_keys = set(getattr(structs, struct_name))
    invalid_keys = set(d.keys()).difference(valid_keys)
    if invalid_keys:
        raise ValueError(f"Invalid keys in {struct_name}: {invalid_keys}")


error_handler = ErrorHandler(logger)
libf = SafeLibCalls(lib, error_handler)


# %% The API


class GPU(classes.GPU):
    def request_adapter(
        self, *, power_preference=None, force_fallback_adapter=False, canvas=None
    ):
        """Create a `GPUAdapter`, the object that represents an abstract wgpu
        implementation, from which one can request a `GPUDevice`.

        This is the implementation based on wgpu-native.

        Arguments:
            power_preference (PowerPreference): "high-performance" or "low-power".
            force_fallback_adapter (bool): whether to use a (probably CPU-based)
                fallback adapter.
            canvas (WgpuCanvasInterface): The canvas that the adapter should
                be able to render to. This can typically be left to None.
        """

        # ----- Surface ID

        # Get surface id that the adapter must be compatible with. If we
        # don't pass a valid surface id, there is no guarantee we'll be
        # able to create a surface texture for it (from this adapter).
        surface_id = ffi.NULL
        if canvas is not None:
            if canvas.get_surface_info():  # e.g. could be an off-screen canvas
                surface_id = canvas.get_context()._get_surface_id()

        # ----- Select backend

        # Try to read the WGPU_BACKEND_TYPE environment variable to see
        # if a backend should be forced.
        force_backend = os.getenv("WGPU_BACKEND_TYPE", "").strip()
        backend = enum_str2int["BackendType"]["Undefined"]
        if force_backend:
            try:
                backend = enum_str2int["BackendType"][force_backend]
            except KeyError:
                logger.warning(
                    f"Invalid value for WGPU_BACKEND_TYPE: '{force_backend}'.\n"
                    f"Valid values are: {list(enum_str2int['BackendType'].keys())}"
                )
            else:
                logger.warning(f"Forcing backend: {force_backend} ({backend})")

        # ----- Request adapter

        # H: nextInChain: WGPUChainedStruct *, compatibleSurface: WGPUSurface, powerPreference: WGPUPowerPreference, backendType: WGPUBackendType, forceFallbackAdapter: WGPUBool/int
        struct = new_struct_p(
            "WGPURequestAdapterOptions *",
            compatibleSurface=surface_id,
            powerPreference=power_preference or "high-performance",
            forceFallbackAdapter=bool(force_fallback_adapter),
            backendType=backend,
            # not used: nextInChain
        )

        adapter_id = None
        error_msg = None

        @ffi.callback("void(WGPURequestAdapterStatus, WGPUAdapter, char *, void *)")
        def callback(status, result, message, userdata):
            if status != 0:
                nonlocal error_msg
                msg = "-" if message == ffi.NULL else ffi.string(message).decode()
                error_msg = f"Request adapter failed ({status}): {msg}"
            else:
                nonlocal adapter_id
                adapter_id = result

        # H: void f(WGPUInstance instance, WGPURequestAdapterOptions const * options, WGPURequestAdapterCallback callback, void * userdata)
        libf.wgpuInstanceRequestAdapter(get_wgpu_instance(), struct, callback, ffi.NULL)

        # For now, Rust will call the callback immediately
        # todo: when wgpu gets an event loop -> while run wgpu event loop or something
        if adapter_id is None:  # pragma: no cover
            error_msg = error_msg or "Could not obtain new adapter id."
            raise RuntimeError(error_msg)

        return self._create_adapter(adapter_id)

    async def request_adapter_async(
        self, *, power_preference=None, force_fallback_adapter=False, canvas=None
    ):
        """Async version of ``request_adapter()``.
        This is the implementation based on wgpu-native.
        """
        return self.request_adapter(
            power_preference=power_preference,
            force_fallback_adapter=force_fallback_adapter,
            canvas=canvas,
        )  # no-cover

    def enumerate_adapters(self):
        """Get a list of adapter objects available on the current system.
        This is the implementation based on wgpu-native.
        """
        # The first call is to get the number of adapters, and the second call
        # is to get the actual adapters. Note that the second arg (now NULL) can
        # be a `WGPUInstanceEnumerateAdapterOptions` to filter by backend.
        instance = get_wgpu_instance()
        # H: size_t f(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const * options, WGPUAdapter * adapters)
        count = libf.wgpuInstanceEnumerateAdapters(instance, ffi.NULL, ffi.NULL)
        adapters = ffi.new("WGPUAdapter[]", count)
        # H: size_t f(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const * options, WGPUAdapter * adapters)
        libf.wgpuInstanceEnumerateAdapters(instance, ffi.NULL, adapters)
        return [self._create_adapter(adapter) for adapter in adapters]

    def _create_adapter(self, adapter_id):
        # ----- Get adapter info

        # H: nextInChain: WGPUChainedStructOut *, vendorID: int, vendorName: char *, architecture: char *, deviceID: int, name: char *, driverDescription: char *, adapterType: WGPUAdapterType, backendType: WGPUBackendType
        c_properties = new_struct_p(
            "WGPUAdapterProperties *",
            # not used: nextInChain
            # not used: deviceID
            # not used: vendorID
            # not used: name
            # not used: driverDescription
            # not used: adapterType
            # not used: backendType
            # not used: vendorName
            # not used: architecture
        )

        # H: void f(WGPUAdapter adapter, WGPUAdapterProperties * properties)
        libf.wgpuAdapterGetProperties(adapter_id, c_properties)

        def to_py_str(key):
            char_p = getattr(c_properties, key)
            if char_p:
                return ffi.string(char_p).decode(errors="ignore")
            return ""

        # Populate a dict according to the WebGPU spec: https://gpuweb.github.io/gpuweb/#gpuadapterinfo
        # And add all other info we get from wgpu-native too.
        adapter_info = {
            # Spec
            "vendor": to_py_str("vendorName"),
            "architecture": to_py_str("architecture"),
            "device": to_py_str("name"),
            "description": to_py_str("driverDescription"),
            # Extra
            "vendor_id": c_properties.vendorID,
            "device_id": c_properties.deviceID,
            "adapter_type": enum_int2str["AdapterType"].get(
                c_properties.adapterType, "unknown"
            ),
            "backend_type": enum_int2str["BackendType"].get(
                c_properties.backendType, "unknown"
            ),
        }

        # ----- Get adapter limits

        # H: nextInChain: WGPUChainedStructOut *, limits: WGPULimits
        c_supported_limits = new_struct_p(
            "WGPUSupportedLimits *",
            # not used: nextInChain
            # not used: limits
        )
        c_limits = c_supported_limits.limits
        # H: WGPUBool f(WGPUAdapter adapter, WGPUSupportedLimits * limits)
        libf.wgpuAdapterGetLimits(adapter_id, c_supported_limits)
        limits = {to_snake_case(k): getattr(c_limits, k) for k in sorted(dir(c_limits))}

        # ----- Get adapter features

        # WebGPU features
        features = set()
        for f in sorted(enums.FeatureName):
            key = f"FeatureName.{f}"
            i = enummap[key]
            # H: WGPUBool f(WGPUAdapter adapter, WGPUFeatureName feature)
            if libf.wgpuAdapterHasFeature(adapter_id, i):
                features.add(f)

        # Native features
        for f in NATIVE_FEATURES:
            i = getattr(lib, f"WGPUNativeFeature_{f}")
            # H: WGPUBool f(WGPUAdapter adapter, WGPUFeatureName feature)
            if libf.wgpuAdapterHasFeature(adapter_id, i):
                features.add(f)

        # ----- Done

        return GPUAdapter(adapter_id, features, limits, adapter_info)


# Instantiate API entrypoint
gpu = GPU()


class GPUCanvasContext(classes.GPUCanvasContext):
    # The way this works, is that the context must first be configured.
    # Then a texture can be obtained, which can be written to, and then it
    # can be presented. The lifetime of the texture is between
    # get_current_texture() and present(). We keep track of the texture so
    # we can give meaningful errors/warnings on invalid use, rather than
    # the more cryptic Rust panics.

    def __init__(self, canvas):
        super().__init__(canvas)
        self._device = None  # set in configure()
        self._surface_id = None
        self._config = None
        self._texture = None

    def _get_surface_id(self):
        if self._surface_id is None:
            # get_surface_id_from_canvas calls wgpuInstanceCreateSurface
            self._surface_id = get_surface_id_from_canvas(self._get_canvas())
        return self._surface_id

    def configure(
        self,
        *,
        device: "GPUDevice",
        format: "enums.TextureFormat",
        usage: "flags.TextureUsage" = 0x10,
        view_formats: "List[enums.TextureFormat]" = [],
        color_space: str = "srgb",
        alpha_mode: "enums.CanvasAlphaMode" = "opaque",
    ):
        # Handle inputs

        # Store for later
        self._device = device
        # Handle usage
        if isinstance(usage, str):
            usage = str_flag_to_int(flags.TextureUsage, usage)
        # View formats
        c_view_formats = ffi.NULL
        if view_formats:
            view_formats_list = [enummap["TextureFormat." + x] for x in view_formats]
            c_view_formats = ffi.new("WGPUTextureFormat []", view_formats_list)
        # Lookup alpha mode, needs explicit conversion because enum names mismatch
        c_alpha_mode = getattr(lib, f"WGPUCompositeAlphaMode_{alpha_mode.capitalize()}")
        # The format is used as-is
        if format is None:
            format = self.get_preferred_format(device.adapter)
        # The color_space is not used for now
        color_space

        # Get what's supported

        # H: nextInChain: WGPUChainedStructOut *, formatCount: int, formats: WGPUTextureFormat *, presentModeCount: int, presentModes: WGPUPresentMode *, alphaModeCount: int, alphaModes: WGPUCompositeAlphaMode *
        capabilities = new_struct_p(
            "WGPUSurfaceCapabilities *",
            # not used: formatCount
            # not used: formats
            # not used: presentModeCount
            # not used: presentModes
            # not used: alphaModeCount
            # not used: alphaModes
            # not used: nextInChain
        )
        # H: void f(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities * capabilities)
        libf.wgpuSurfaceGetCapabilities(
            self._get_surface_id(), self._device.adapter._internal, capabilities
        )

        capable_formats = []
        for i in range(capabilities.formatCount):
            int_val = capabilities.formats[i]
            capable_formats.append(enum_int2str["TextureFormat"][int_val])

        capable_present_modes = []
        for i in range(capabilities.presentModeCount):
            int_val = capabilities.presentModes[i]
            str_val = enum_int2str["PresentMode"][int_val]
            capable_present_modes.append(str_val.lower())

        capable_alpha_modes = []
        for i in range(capabilities.alphaModeCount):
            int_val = capabilities.alphaModes[i]
            str_val = enum_int2str["CompositeAlphaMode"][int_val]
            capable_alpha_modes.append(str_val.lower())

        # H: void f(WGPUSurfaceCapabilities capabilities)
        libf.wgpuSurfaceCapabilitiesFreeMembers(capabilities[0])

        # Check if input is supported

        if format not in capable_formats:
            raise ValueError(
                f"Given format '{format}' is not in supported formats {capable_formats}"
            )
        if alpha_mode not in capable_alpha_modes:
            raise ValueError(
                f"Given format '{alpha_mode}' is not in supported formats {capable_alpha_modes}"
            )

        # Select the present mode to determine vsync behavior.
        # * https://docs.rs/wgpu/latest/wgpu/enum.PresentMode.html
        # * https://github.com/pygfx/wgpu-py/issues/256
        #
        # Fifo: Wait for vsync, with a queue of ± 3 frames.
        # FifoRelaxed: Like fifo but less lag and more tearing? aka adaptive vsync.
        # Mailbox: submit without queue, but present on vsync. Not always available.
        # Immediate: no queue, no waiting, with risk of tearing, vsync off.
        #
        # In general Fifo gives the best result, but sometimes people want to
        # benchmark something and get the highest FPS possible. Note
        # that we've observed rate limiting regardless of setting this
        # to Immediate, depending on OS or being on battery power.
        if getattr(self._get_canvas(), "_vsync", True):
            present_mode_pref = ["fifo", "mailbox"]
        else:
            present_mode_pref = ["immediate", "mailbox", "fifo"]
        present_modes = [p for p in present_mode_pref if p in capable_present_modes]
        present_mode = (present_modes or capable_present_modes)[0]
        c_present_mode = getattr(lib, f"WGPUPresentMode_{present_mode.capitalize()}")

        # Prepare config object

        # H: nextInChain: WGPUChainedStruct *, device: WGPUDevice, format: WGPUTextureFormat, usage: WGPUTextureUsageFlags/int, viewFormatCount: int, viewFormats: WGPUTextureFormat *, alphaMode: WGPUCompositeAlphaMode, width: int, height: int, presentMode: WGPUPresentMode
        config = new_struct_p(
            "WGPUSurfaceConfiguration *",
            device=device._internal,
            format=format,
            usage=usage,
            viewFormatCount=len(view_formats),
            viewFormats=c_view_formats,
            alphaMode=c_alpha_mode,
            width=0,
            height=0,
            presentMode=c_present_mode,
            # not used: nextInChain
        )

        # Configure
        self._configure(config)

    def _configure(self, config):
        # If a texture is still active, better destroy it first
        self._destroy_texture()
        # Set the size
        width, height = self._get_canvas().get_physical_size()
        config.width = width
        config.height = height
        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Cannot configure canvas that has no pixels ({width}x{height})."
            )
        # Configure, and store the config if we did not error out
        # H: void f(WGPUSurface surface, WGPUSurfaceConfiguration const * config)
        libf.wgpuSurfaceConfigure(self._get_surface_id(), config)
        self._config = config

    def unconfigure(self):
        self._destroy_texture()
        self._config = None
        # H: void f(WGPUSurface surface)
        libf.wgpuSurfaceUnconfigure(self._get_surface_id())

    def _destroy_texture(self):
        if self._texture:
            self._texture.destroy()
            self._texture = None

    def get_current_texture(self):
        # If the canvas has changed since the last configure, we need to re-configure it
        if not self._config:
            raise RuntimeError(
                "Canvas context must be configured before calling get_current_texture()."
            )

        # When the texture is active right now, we could either:
        # * return the existing texture
        # * warn about it, and create a new one
        # * raise an error
        # Right now we do the warning, so things still (kinda) keep working
        if self._texture:
            self._destroy_texture()
            logger.warning(
                "get_current_texture() is called multiple times before pesent()."
            )

        # Reconfigure when the canvas has resized.
        # On some systems (Windows+Qt) this is not necessary, because
        # the texture status would be Outdated below, resulting in a
        # reconfigure. But on others (e.g. glfwf) the texture size does
        # not have to match the window size, apparently. The downside
        # for doing this check on the former systems, is that errors
        # get logged, which would not be there if we did not
        # pre-emptively reconfigure. These log entries are harmless but
        # anoying, and I currently don't know how to prevent them
        # elegantly. See issue #352
        old_size = (self._config.width, self._config.height)
        new_size = tuple(self._get_canvas().get_physical_size())
        if old_size != new_size:
            self._configure(self._config)

        # Try to obtain a texture.
        # `If it fails, depending on status, we reconfure and try again.

        # H: texture: WGPUTexture, suboptimal: WGPUBool/int, status: WGPUSurfaceGetCurrentTextureStatus
        surface_texture = new_struct_p(
            "WGPUSurfaceTexture *",
            # not used: texture
            # not used: suboptimal
            # not used: status
        )

        for attempt in [1, 2]:
            # H: void f(WGPUSurface surface, WGPUSurfaceTexture * surfaceTexture)
            libf.wgpuSurfaceGetCurrentTexture(self._get_surface_id(), surface_texture)
            status = surface_texture.status
            texture_id = surface_texture.texture
            if status == lib.WGPUSurfaceGetCurrentTextureStatus_Success:
                break  # success
            if texture_id:
                # H: void f(WGPUTexture texture)
                libf.wgpuTextureRelease(texture_id)
            if attempt == 1 and status in [
                lib.WGPUSurfaceGetCurrentTextureStatus_Timeout,
                lib.WGPUSurfaceGetCurrentTextureStatus_Outdated,
                lib.WGPUSurfaceGetCurrentTextureStatus_Lost,
            ]:
                # Configure and try again.
                # On Window+Qt this happens e.g. when the window has resized
                # (status==Outdated), but also when moving the window from one
                # monitor to another with different scale-factor.
                logger.info(f"Re-configuring canvas context ({status}).")
                self._configure(self._config)
            else:
                # WGPUSurfaceGetCurrentTextureStatus_OutOfMemory
                # WGPUSurfaceGetCurrentTextureStatus_DeviceLost
                # Or if this is the second attempt.
                raise RuntimeError(f"Cannot get surface texture ({status}).")

        # I don't expect this to happen, but lets check just in case.
        if not texture_id:
            raise RuntimeError("Cannot get surface texture (no texture)")

        # Things look good, but texture may still be suboptimal, whatever that means
        if surface_texture.suboptimal:
            logger.warning("The surface texture is suboptimal.")

        return self._create_python_texture(texture_id)

    def _create_python_texture(self, texture_id):
        # Create the Python wrapper

        # We can derive texture props from the config and common sense:
        # width = self._config.width
        # height = self._config.height
        # depth = 1
        # mip_level_count = 1
        # sample_count = 1
        # dimension = enums.TextureDimension.d2
        # format = enum_int2str["TextureFormat"][self._config.format]
        # usage = self._config.usage

        # But we can also read them from the texture
        # H: uint32_t f(WGPUTexture texture)
        width = libf.wgpuTextureGetWidth(texture_id)
        # H: uint32_t f(WGPUTexture texture)
        height = libf.wgpuTextureGetHeight(texture_id)
        # H: uint32_t f(WGPUTexture texture)
        depth = libf.wgpuTextureGetDepthOrArrayLayers(texture_id)
        # H: uint32_t f(WGPUTexture texture)
        mip_level_count = libf.wgpuTextureGetMipLevelCount(texture_id)
        # H: uint32_t f(WGPUTexture texture)
        sample_count = libf.wgpuTextureGetSampleCount(texture_id)
        # H: WGPUTextureDimension f(WGPUTexture texture)
        c_dim = libf.wgpuTextureGetDimension(texture_id)  # -> to string
        dimension = enum_int2str["TextureDimension"][c_dim]
        # H: WGPUTextureFormat f(WGPUTexture texture)
        c_format = libf.wgpuTextureGetFormat(texture_id)
        format = enum_int2str["TextureFormat"][c_format]
        # H: WGPUTextureUsageFlags f(WGPUTexture texture)
        usage = libf.wgpuTextureGetUsage(texture_id)

        label = ""
        # Cannot yet set label, because it's not implemented in wgpu-native
        # label = "surface-texture"
        # H: void f(WGPUTexture texture, char const * label)
        # libf.wgpuTextureSetLabel(texture_id, to_c_label(label))

        tex_info = {
            "size": (width, height, depth),
            "mip_level_count": mip_level_count,
            "sample_count": sample_count,
            "dimension": dimension,
            "format": format,
            "usage": usage,
        }

        self._texture = GPUTexture(label, texture_id, self._device, tex_info)
        return self._texture

    def present(self):
        if not self._texture:
            # This can happen when a user somehow forgot to call
            # get_current_texture(). But then what was this person rendering to
            # then? The thing is that this also happens when there is an
            # exception in the draw function before the call to
            # get_current_texture(). In this scenario our warning may
            # add confusion, so provide context and make it a debug level warning.
            msg = "Warning in present(): No texture to present, missing call to get_current_texture()?"
            logger.debug(msg)
        else:
            # Present the texture, then destroy it
            # H: void f(WGPUSurface surface)
            libf.wgpuSurfacePresent(self._get_surface_id())
            self._destroy_texture()

    def get_preferred_format(self, adapter):
        # H: WGPUTextureFormat f(WGPUSurface surface, WGPUAdapter adapter)
        format = libf.wgpuSurfaceGetPreferredFormat(
            self._get_surface_id(), adapter._internal
        )
        return enum_int2str["TextureFormat"][format]

    def _destroy(self):
        self._destroy_texture()
        if self._surface_id is not None and libf is not None:
            self._surface_id, surface_id = None, self._surface_id
            # H: void f(WGPUSurface surface)
            libf.wgpuSurfaceRelease(surface_id)


class GPUObjectBase(classes.GPUObjectBase):
    pass


class GPUAdapterInfo(classes.GPUAdapterInfo):
    pass


class GPUAdapter(classes.GPUAdapter):
    def request_device(
        self,
        *,
        label="",
        required_features: "List[enums.FeatureName]" = [],
        required_limits: "Dict[str, int]" = {},
        default_queue: "structs.QueueDescriptor" = {},
    ):
        if default_queue:
            check_struct("QueueDescriptor", default_queue)
        return self._request_device(
            label, required_features, required_limits, default_queue, ""
        )

    def _request_device(
        self, label, required_features, required_limits, default_queue, trace_path
    ):
        # ---- Handle features

        assert isinstance(required_features, (tuple, list, set))

        c_features = set()
        for f in required_features:
            if isinstance(f, str):
                if "_" in f:
                    f = "".join(x.title() for x in f.split("_"))
                i1 = enummap.get(f"FeatureName.{f}", None)
                i2 = getattr(lib, f"WGPUNativeFeature_{f}", None)
                i = i2 if i1 is None else i1
                if i is None:  # pragma: no cover
                    raise KeyError(f"Unknown feature: '{f}'")
                c_features.add(i)
            else:
                raise TypeError("Features must be given as str.")

        c_features = sorted(c_features)  # makes it a list

        # ----- Set limits

        # H: nextInChain: WGPUChainedStruct *, limits: WGPULimits
        c_required_limits = new_struct_p(
            "WGPURequiredLimits *",
            # not used: nextInChain
            # not used: limits
        )
        c_limits = c_required_limits.limits

        # Set all limits to the adapter default
        # This is important, because zero does NOT mean default, and a limit of zero
        # for a specific limit may break a lot of applications.
        for key, val in self.limits.items():
            setattr(c_limits, to_camel_case(key), val)

        # Overload with any set limits
        required_limits = required_limits or {}
        for key, val in required_limits.items():
            setattr(c_limits, to_camel_case(key), val)

        # ---- Set queue descriptor

        # Note that the default_queue arg is a descriptor (dict for QueueDescriptor), but is currently empty :)
        # H: nextInChain: WGPUChainedStruct *, label: char *
        queue_struct = new_struct(
            "WGPUQueueDescriptor",
            label=to_c_label("default_queue"),
            # not used: nextInChain
        )

        # ----- Compose device descriptor extras

        c_trace_path = ffi.NULL
        if trace_path:  # no-cover
            c_trace_path = ffi.new("char []", trace_path.encode())

        # H: chain: WGPUChainedStruct, tracePath: char *
        extras = new_struct_p(
            "WGPUDeviceExtras *",
            tracePath=c_trace_path,
            # not used: chain
        )
        extras.chain.sType = lib.WGPUSType_DeviceExtras

        # ----- Device lost

        @ffi.callback("void(WGPUDeviceLostReason, char *, void *)")
        def device_lost_callback(c_reason, c_message, userdata):
            reason = enum_int2str["DeviceLostReason"].get(c_reason, "Unknown")
            message = ffi.string(c_message).decode(errors="ignore")
            error_handler.log_error(f"The WGPU device was lost ({reason}):\n{message}")

        # Keep the ref alive
        self._device_lost_callback = device_lost_callback

        # ----- Request device

        # H: nextInChain: WGPUChainedStruct *, label: char *, requiredFeatureCount: int, requiredFeatures: WGPUFeatureName *, requiredLimits: WGPURequiredLimits *, defaultQueue: WGPUQueueDescriptor, deviceLostCallback: WGPUDeviceLostCallback, deviceLostUserdata: void *
        struct = new_struct_p(
            "WGPUDeviceDescriptor *",
            label=to_c_label(label),
            nextInChain=ffi.cast("WGPUChainedStruct * ", extras),
            requiredFeatureCount=len(c_features),
            requiredFeatures=ffi.new("WGPUFeatureName []", c_features),
            requiredLimits=c_required_limits,
            defaultQueue=queue_struct,
            deviceLostCallback=device_lost_callback,
            # not used: deviceLostUserdata
        )

        device_id = None
        error_msg = None

        @ffi.callback("void(WGPURequestDeviceStatus, WGPUDevice, char *, void *)")
        def callback(status, result, message, userdata):
            if status != 0:
                nonlocal error_msg
                msg = "-" if message == ffi.NULL else ffi.string(message).decode()
                error_msg = f"Request device failed ({status}): {msg}"
            else:
                nonlocal device_id
                device_id = result

        # H: void f(WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor, WGPURequestDeviceCallback callback, void * userdata)
        libf.wgpuAdapterRequestDevice(self._internal, struct, callback, ffi.NULL)

        if device_id is None:  # pragma: no cover
            error_msg = error_msg or "Could not obtain new device id."
            raise RuntimeError(error_msg)

        # ----- Get device limits

        # H: nextInChain: WGPUChainedStructOut *, limits: WGPULimits
        c_supported_limits = new_struct_p(
            "WGPUSupportedLimits *",
            # not used: nextInChain
            # not used: limits
        )
        c_limits = c_supported_limits.limits
        # H: WGPUBool f(WGPUDevice device, WGPUSupportedLimits * limits)
        libf.wgpuDeviceGetLimits(device_id, c_supported_limits)
        limits = {to_snake_case(k): getattr(c_limits, k) for k in dir(c_limits)}

        # ----- Get device features

        # WebGPU features
        features = set()
        for f in sorted(enums.FeatureName):
            key = f"FeatureName.{f}"
            i = enummap[key]
            # H: WGPUBool f(WGPUDevice device, WGPUFeatureName feature)
            if libf.wgpuDeviceHasFeature(device_id, i):
                features.add(f)

        # Native features
        for f in NATIVE_FEATURES:
            i = getattr(lib, f"WGPUNativeFeature_{f}")
            # H: WGPUBool f(WGPUDevice device, WGPUFeatureName feature)
            if libf.wgpuDeviceHasFeature(device_id, i):
                features.add(f)

        # ---- Get queue

        # H: WGPUQueue f(WGPUDevice device)
        queue_id = libf.wgpuDeviceGetQueue(device_id)
        queue = GPUQueue("", queue_id, None)

        # ----- Done

        return GPUDevice(label, device_id, self, features, limits, queue)

    async def request_device_async(
        self,
        *,
        label="",
        required_features: "List[enums.FeatureName]" = [],
        required_limits: "Dict[str, int]" = {},
        default_queue: "structs.QueueDescriptor" = {},
    ):
        if default_queue:
            check_struct("QueueDescriptor", default_queue)
        return self._request_device(
            label, required_features, required_limits, default_queue, ""
        )  # no-cover

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUAdapter adapter)
            libf.wgpuAdapterRelease(internal)


class GPUDevice(classes.GPUDevice, GPUObjectBase):
    def __init__(self, label, internal, adapter, features, limits, queue):
        super().__init__(label, internal, adapter, features, limits, queue)

        @ffi.callback("void(WGPUErrorType, char *, void *)")
        def uncaptured_error_callback(c_type, c_message, userdata):
            error_type = enum_int2str["ErrorType"].get(c_type, "Unknown")
            message = ffi.string(c_message).decode(errors="ignore")
            message = "\n".join(line.rstrip() for line in message.splitlines())
            error_handler.handle_error(error_type, message)

        # Keep the ref alive
        self._uncaptured_error_callback = uncaptured_error_callback

        # H: void f(WGPUDevice device, WGPUErrorCallback callback, void * userdata)
        libf.wgpuDeviceSetUncapturedErrorCallback(
            self._internal, uncaptured_error_callback, ffi.NULL
        )

    def _poll(self):
        # Internal function
        if self._internal:
            # H: WGPUBool f(WGPUDevice device, WGPUBool wait, WGPUWrappedSubmissionIndex const * wrappedSubmissionIndex)
            libf.wgpuDevicePoll(self._internal, True, ffi.NULL)

    def create_buffer(
        self,
        *,
        label="",
        size: int,
        usage: "flags.BufferUsage",
        mapped_at_creation: bool = False,
    ):
        return self._create_buffer(label, int(size), usage, bool(mapped_at_creation))

    def _create_buffer(self, label, size, usage, mapped_at_creation):
        # Create a buffer object
        if isinstance(usage, str):
            usage = str_flag_to_int(flags.BufferUsage, usage)
        # H: nextInChain: WGPUChainedStruct *, label: char *, usage: WGPUBufferUsageFlags/int, size: int, mappedAtCreation: WGPUBool/int
        struct = new_struct_p(
            "WGPUBufferDescriptor *",
            label=to_c_label(label),
            size=size,
            usage=int(usage),
            mappedAtCreation=mapped_at_creation,
            # not used: nextInChain
        )
        map_state = (
            enums.BufferMapState.mapped
            if mapped_at_creation
            else enums.BufferMapState.unmapped
        )
        # H: WGPUBuffer f(WGPUDevice device, WGPUBufferDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateBuffer(self._internal, struct)
        # Note that there is wgpuBufferGetSize and wgpuBufferGetUsage,
        # but we already know these, so they are kindof useless?
        # Return wrapped buffer
        return GPUBuffer(label, id, self, size, usage, map_state)

    def create_texture(
        self,
        *,
        label="",
        size: "Union[List[int], structs.Extent3D]",
        mip_level_count: int = 1,
        sample_count: int = 1,
        dimension: "enums.TextureDimension" = "2d",
        format: "enums.TextureFormat",
        usage: "flags.TextureUsage",
        view_formats: "List[enums.TextureFormat]" = [],
    ):
        if isinstance(usage, str):
            usage = str_flag_to_int(flags.TextureUsage, usage)
        usage = int(usage)
        size = _tuple_from_tuple_or_dict(
            size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depthOrArrayLayers: int
        c_size = new_struct(
            "WGPUExtent3D",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        if view_formats:
            raise NotImplementedError(
                "create_texture(.. view_formats is not yet supported."
            )

        if not mip_level_count:
            mip_level_count = 1  # or lib.WGPU_MIP_LEVEL_COUNT_UNDEFINED ?
        mip_level_count = int(mip_level_count)

        if not sample_count:
            sample_count = 1
        sample_count = int(sample_count)

        # H: nextInChain: WGPUChainedStruct *, label: char *, usage: WGPUTextureUsageFlags/int, dimension: WGPUTextureDimension, size: WGPUExtent3D, format: WGPUTextureFormat, mipLevelCount: int, sampleCount: int, viewFormatCount: int, viewFormats: WGPUTextureFormat *
        struct = new_struct_p(
            "WGPUTextureDescriptor *",
            label=to_c_label(label),
            size=c_size,
            mipLevelCount=mip_level_count,
            sampleCount=sample_count,
            dimension=dimension,
            format=format,
            usage=usage,
            # not used: nextInChain
            # not used: viewFormatCount
            # not used: viewFormats
        )
        # H: WGPUTexture f(WGPUDevice device, WGPUTextureDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateTexture(self._internal, struct)

        # Note that there are methods (e.g. wgpuTextureGetHeight) to get
        # the below props, but we know them now, so why bother?
        tex_info = {
            "size": size,
            "mip_level_count": mip_level_count,
            "sample_count": sample_count,
            "dimension": dimension,
            "format": format,
            "usage": usage,
        }
        return GPUTexture(label, id, self, tex_info)

    def create_sampler(
        self,
        *,
        label="",
        address_mode_u: "enums.AddressMode" = "clamp-to-edge",
        address_mode_v: "enums.AddressMode" = "clamp-to-edge",
        address_mode_w: "enums.AddressMode" = "clamp-to-edge",
        mag_filter: "enums.FilterMode" = "nearest",
        min_filter: "enums.FilterMode" = "nearest",
        mipmap_filter: "enums.MipmapFilterMode" = "nearest",
        lod_min_clamp: float = 0,
        lod_max_clamp: float = 32,
        compare: "enums.CompareFunction" = None,
        max_anisotropy: int = 1,
    ):
        # H: nextInChain: WGPUChainedStruct *, label: char *, addressModeU: WGPUAddressMode, addressModeV: WGPUAddressMode, addressModeW: WGPUAddressMode, magFilter: WGPUFilterMode, minFilter: WGPUFilterMode, mipmapFilter: WGPUMipmapFilterMode, lodMinClamp: float, lodMaxClamp: float, compare: WGPUCompareFunction, maxAnisotropy: int
        struct = new_struct_p(
            "WGPUSamplerDescriptor *",
            label=to_c_label(label),
            addressModeU=address_mode_u,
            addressModeV=address_mode_v,
            addressModeW=address_mode_w,
            magFilter=mag_filter,
            minFilter=min_filter,
            mipmapFilter=mipmap_filter,
            lodMinClamp=lod_min_clamp,
            lodMaxClamp=lod_max_clamp,
            compare=0 if compare is None else compare,
            maxAnisotropy=max_anisotropy,
            # not used: nextInChain
        )

        # H: WGPUSampler f(WGPUDevice device, WGPUSamplerDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateSampler(self._internal, struct)
        return GPUSampler(label, id, self)

    def create_bind_group_layout(
        self, *, label="", entries: "List[structs.BindGroupLayoutEntry]"
    ):
        c_entries_list = []
        for entry in entries:
            check_struct("BindGroupLayoutEntry", entry)
            buffer = {}
            sampler = {}
            texture = {}
            storage_texture = {}
            if entry.get("buffer"):
                info = entry["buffer"]
                check_struct("BufferBindingLayout", info)
                min_binding_size = info.get("min_binding_size", None)
                if min_binding_size is None:
                    min_binding_size = 0  # lib.WGPU_LIMIT_U64_UNDEFINED
                # H: nextInChain: WGPUChainedStruct *, type: WGPUBufferBindingType, hasDynamicOffset: WGPUBool/int, minBindingSize: int
                buffer = new_struct(
                    "WGPUBufferBindingLayout",
                    type=info["type"],
                    hasDynamicOffset=info.get("has_dynamic_offset", False),
                    minBindingSize=min_binding_size,
                    # not used: nextInChain
                )
            elif entry.get("sampler"):
                info = entry["sampler"]
                check_struct("SamplerBindingLayout", info)
                # H: nextInChain: WGPUChainedStruct *, type: WGPUSamplerBindingType
                sampler = new_struct(
                    "WGPUSamplerBindingLayout",
                    type=info["type"],
                    # not used: nextInChain
                )
            elif entry.get("texture"):
                info = entry["texture"]
                check_struct("TextureBindingLayout", info)
                # H: nextInChain: WGPUChainedStruct *, sampleType: WGPUTextureSampleType, viewDimension: WGPUTextureViewDimension, multisampled: WGPUBool/int
                texture = new_struct(
                    "WGPUTextureBindingLayout",
                    sampleType=info.get("sample_type", "float"),
                    viewDimension=info.get("view_dimension", "2d"),
                    multisampled=info.get("multisampled", False),
                    # not used: nextInChain
                )
            elif entry.get("storage_texture"):
                info = entry["storage_texture"]
                check_struct("StorageTextureBindingLayout", info)
                # H: nextInChain: WGPUChainedStruct *, access: WGPUStorageTextureAccess, format: WGPUTextureFormat, viewDimension: WGPUTextureViewDimension
                storage_texture = new_struct(
                    "WGPUStorageTextureBindingLayout",
                    access=info["access"],
                    viewDimension=info.get("view_dimension", "2d"),
                    format=info["format"],
                    # not used: nextInChain
                )
            else:
                raise ValueError(
                    "Bind group layout entry did not contain field 'buffer', 'sampler', 'texture', nor 'storage_texture'"
                )
                # Unreachable - fool the codegen
                check_struct("ExternalTextureBindingLayout", info)
            visibility = entry["visibility"]
            if isinstance(visibility, str):
                visibility = str_flag_to_int(flags.ShaderStage, visibility)
            # H: nextInChain: WGPUChainedStruct *, binding: int, visibility: WGPUShaderStageFlags/int, buffer: WGPUBufferBindingLayout, sampler: WGPUSamplerBindingLayout, texture: WGPUTextureBindingLayout, storageTexture: WGPUStorageTextureBindingLayout
            c_entry = new_struct(
                "WGPUBindGroupLayoutEntry",
                binding=int(entry["binding"]),
                visibility=int(visibility),
                buffer=buffer,
                sampler=sampler,
                texture=texture,
                storageTexture=storage_texture,
                # not used: nextInChain
            )
            c_entries_list.append(c_entry)

        c_entries_array = ffi.NULL
        if c_entries_list:
            c_entries_array = ffi.new("WGPUBindGroupLayoutEntry []", c_entries_list)

        # H: nextInChain: WGPUChainedStruct *, label: char *, entryCount: int, entries: WGPUBindGroupLayoutEntry *
        struct = new_struct_p(
            "WGPUBindGroupLayoutDescriptor *",
            label=to_c_label(label),
            entries=c_entries_array,
            entryCount=len(c_entries_list),
            # not used: nextInChain
        )

        # Note: wgpu-core re-uses BindGroupLayouts with the same (or similar
        # enough) descriptor. You would think that this means that the id is
        # the same when you call wgpuDeviceCreateBindGroupLayout with the same
        # input, but it's not. So we cannot let wgpu-native/core decide when
        # to re-use a BindGroupLayout. I don't feel confident checking here
        # whether a BindGroupLayout can be re-used, so we simply don't. Higher
        # level code can sometimes make this decision because it knows the app
        # logic.

        # H: WGPUBindGroupLayout f(WGPUDevice device, WGPUBindGroupLayoutDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateBindGroupLayout(self._internal, struct)
        return GPUBindGroupLayout(label, id, self, entries)

    def create_bind_group(
        self,
        *,
        label="",
        layout: "GPUBindGroupLayout",
        entries: "List[structs.BindGroupEntry]",
    ):
        c_entries_list = []
        for entry in entries:
            check_struct("BindGroupEntry", entry)
            # The resource can be a sampler, texture view, or buffer descriptor
            resource = entry["resource"]
            if isinstance(resource, GPUSampler):
                # H: nextInChain: WGPUChainedStruct *, binding: int, buffer: WGPUBuffer, offset: int, size: int, sampler: WGPUSampler, textureView: WGPUTextureView
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    binding=int(entry["binding"]),
                    buffer=ffi.NULL,
                    offset=0,
                    size=0,
                    sampler=resource._internal,
                    textureView=ffi.NULL,
                    # not used: nextInChain
                )
            elif isinstance(resource, GPUTextureView):
                # H: nextInChain: WGPUChainedStruct *, binding: int, buffer: WGPUBuffer, offset: int, size: int, sampler: WGPUSampler, textureView: WGPUTextureView
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    binding=int(entry["binding"]),
                    buffer=ffi.NULL,
                    offset=0,
                    size=0,
                    sampler=ffi.NULL,
                    textureView=resource._internal,
                    # not used: nextInChain
                )
            elif isinstance(resource, dict):  # Buffer binding
                # H: nextInChain: WGPUChainedStruct *, binding: int, buffer: WGPUBuffer, offset: int, size: int, sampler: WGPUSampler, textureView: WGPUTextureView
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    binding=int(entry["binding"]),
                    buffer=resource["buffer"]._internal,
                    offset=resource["offset"],
                    size=resource["size"],
                    sampler=ffi.NULL,
                    textureView=ffi.NULL,
                    # not used: nextInChain
                )
            else:
                raise TypeError(f"Unexpected resource type {type(resource)}")
            c_entries_list.append(c_entry)

        c_entries_array = ffi.NULL
        if c_entries_list:
            c_entries_array = ffi.new("WGPUBindGroupEntry []", c_entries_list)

        # H: nextInChain: WGPUChainedStruct *, label: char *, layout: WGPUBindGroupLayout, entryCount: int, entries: WGPUBindGroupEntry *
        struct = new_struct_p(
            "WGPUBindGroupDescriptor *",
            label=to_c_label(label),
            layout=layout._internal,
            entries=c_entries_array,
            entryCount=len(c_entries_list),
            # not used: nextInChain
        )

        # H: WGPUBindGroup f(WGPUDevice device, WGPUBindGroupDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateBindGroup(self._internal, struct)
        return GPUBindGroup(label, id, self, entries)

    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "List[GPUBindGroupLayout]"
    ):
        bind_group_layouts_ids = [x._internal for x in bind_group_layouts]

        c_layout_array = ffi.new("WGPUBindGroupLayout []", bind_group_layouts_ids)
        # H: nextInChain: WGPUChainedStruct *, label: char *, bindGroupLayoutCount: int, bindGroupLayouts: WGPUBindGroupLayout *
        struct = new_struct_p(
            "WGPUPipelineLayoutDescriptor *",
            label=to_c_label(label),
            bindGroupLayouts=c_layout_array,
            bindGroupLayoutCount=len(bind_group_layouts),
            # not used: nextInChain
        )

        # H: WGPUPipelineLayout f(WGPUDevice device, WGPUPipelineLayoutDescriptor const * descriptor)
        id = libf.wgpuDeviceCreatePipelineLayout(self._internal, struct)
        return GPUPipelineLayout(label, id, self, bind_group_layouts)

    def create_shader_module(
        self,
        *,
        label="",
        code: str,
        source_map: dict = None,
        compilation_hints: "List[structs.ShaderModuleCompilationHint]" = [],
    ):
        if compilation_hints:
            for hint in compilation_hints.values():
                check_struct("ShaderModuleCompilationHint", hint)
        if isinstance(code, str):
            looks_like_wgsl = any(
                x in code for x in ("@compute", "@vertex", "@fragment")
            )
            looks_like_glsl = code.lstrip().startswith("#version ")
            if looks_like_glsl and not looks_like_wgsl:
                # === GLSL
                if "comp" in label.lower():
                    c_stage = flags.ShaderStage.COMPUTE
                elif "vert" in label.lower():
                    c_stage = flags.ShaderStage.VERTEX
                elif "frag" in label.lower():
                    c_stage = flags.ShaderStage.FRAGMENT
                else:
                    raise ValueError(
                        "GLSL shader needs to use the label to specify compute/vertex/fragment stage."
                    )
                defines = []
                if c_stage == flags.ShaderStage.VERTEX:
                    defines.append(
                        # H: name: char *, value: char *
                        new_struct(
                            "WGPUShaderDefine",
                            name=ffi.new("char []", "gl_VertexID".encode()),
                            value=ffi.new("char []", "gl_VertexIndex".encode()),
                        )
                    )
                c_defines = ffi.new("WGPUShaderDefine []", defines)
                # H: chain: WGPUChainedStruct, stage: WGPUShaderStage, code: char *, defineCount: int, defines: WGPUShaderDefine *
                source_struct = new_struct_p(
                    "WGPUShaderModuleGLSLDescriptor *",
                    code=ffi.new("char []", code.encode()),
                    stage=c_stage,
                    defineCount=len(defines),
                    defines=c_defines,
                    # not used: chain
                )
                source_struct[0].chain.next = ffi.NULL
                source_struct[0].chain.sType = lib.WGPUSType_ShaderModuleGLSLDescriptor
            else:
                # === WGSL
                # H: chain: WGPUChainedStruct, code: char *
                source_struct = new_struct_p(
                    "WGPUShaderModuleWGSLDescriptor *",
                    code=ffi.new("char []", code.encode()),
                    # not used: chain
                )
                source_struct[0].chain.next = ffi.NULL
                source_struct[0].chain.sType = lib.WGPUSType_ShaderModuleWGSLDescriptor
        elif isinstance(code, bytes):
            # === Spirv
            data = code
            # Validate
            magic_nr = b"\x03\x02#\x07"  # 0x7230203
            if data[:4] != magic_nr:
                raise ValueError("Given shader data does not look like a SpirV module")
            # From bytes to WGPUU32Array
            data_u8 = ffi.new("uint8_t[]", data)
            data_u32 = ffi.cast("uint32_t *", data_u8)
            # H: chain: WGPUChainedStruct, codeSize: int, code: uint32_t *
            source_struct = new_struct_p(
                "WGPUShaderModuleSPIRVDescriptor *",
                code=data_u32,
                codeSize=len(data) // 4,
                # not used: chain
            )
            source_struct[0].chain.next = ffi.NULL
            source_struct[0].chain.sType = lib.WGPUSType_ShaderModuleSPIRVDescriptor
        else:
            raise TypeError(
                "Shader code must be str for WGSL or GLSL, or bytes for SpirV."
            )

        # Note, we could give hints here that specify entrypoint and pipelinelayout before compiling
        # H: nextInChain: WGPUChainedStruct *, label: char *, hintCount: int, hints: WGPUShaderModuleCompilationHint *
        struct = new_struct_p(
            "WGPUShaderModuleDescriptor *",
            label=to_c_label(label),
            nextInChain=ffi.cast("WGPUChainedStruct *", source_struct),
            hintCount=0,
            hints=ffi.NULL,
        )
        # H: WGPUShaderModule f(WGPUDevice device, WGPUShaderModuleDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateShaderModule(self._internal, struct)
        if id == ffi.NULL:
            raise RuntimeError("Shader module creation failed")
        return GPUShaderModule(label, id, self)

    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "Union[GPUPipelineLayout, enums.AutoLayoutMode]",
        compute: "structs.ProgrammableStage",
    ):
        check_struct("ProgrammableStage", compute)
        # H: nextInChain: WGPUChainedStruct *, module: WGPUShaderModule, entryPoint: char *, constantCount: int, constants: WGPUConstantEntry *
        c_compute_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=compute["module"]._internal,
            entryPoint=ffi.new("char []", compute["entry_point"].encode()),
            # not used: nextInChain
            # not used: constantCount
            # not used: constants
        )

        if isinstance(layout, GPUPipelineLayout):
            layout_id = layout._internal
        elif layout == enums.AutoLayoutMode.auto:
            layout_id = ffi.NULL
        else:
            raise TypeError(
                "create_compute_pipeline() 'layout' arg must be a GPUPipelineLayout or 'auto'"
            )

        # H: nextInChain: WGPUChainedStruct *, label: char *, layout: WGPUPipelineLayout, compute: WGPUProgrammableStageDescriptor
        struct = new_struct_p(
            "WGPUComputePipelineDescriptor *",
            label=to_c_label(label),
            layout=layout_id,
            compute=c_compute_stage,
            # not used: nextInChain
        )
        # H: WGPUComputePipeline f(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateComputePipeline(self._internal, struct)
        return GPUComputePipeline(label, id, self)

    async def create_compute_pipeline_async(
        self,
        *,
        label="",
        layout: "Union[GPUPipelineLayout, enums.AutoLayoutMode]",
        compute: "structs.ProgrammableStage",
    ):
        return self.create_compute_pipeline(label=label, layout=layout, compute=compute)

    def create_render_pipeline(
        self,
        *,
        label="",
        layout: "Union[GPUPipelineLayout, enums.AutoLayoutMode]",
        vertex: "structs.VertexState",
        primitive: "structs.PrimitiveState" = {},
        depth_stencil: "structs.DepthStencilState" = None,
        multisample: "structs.MultisampleState" = {},
        fragment: "structs.FragmentState" = None,
    ):
        depth_stencil = depth_stencil or {}
        multisample = multisample or {}
        primitive = primitive or {}

        check_struct("VertexState", vertex)
        check_struct("DepthStencilState", depth_stencil)
        check_struct("MultisampleState", multisample)
        check_struct("PrimitiveState", primitive)

        c_vertex_buffer_layout_list = []
        for buffer_des in vertex["buffers"]:
            c_attributes_list = []
            for attribute in buffer_des["attributes"]:
                # H: format: WGPUVertexFormat, offset: int, shaderLocation: int
                c_attribute = new_struct(
                    "WGPUVertexAttribute",
                    format=attribute["format"],
                    offset=attribute["offset"],
                    shaderLocation=attribute["shader_location"],
                )
                c_attributes_list.append(c_attribute)
            c_attributes_array = ffi.new("WGPUVertexAttribute []", c_attributes_list)
            # H: arrayStride: int, stepMode: WGPUVertexStepMode, attributeCount: int, attributes: WGPUVertexAttribute *
            c_vertex_buffer_descriptor = new_struct(
                "WGPUVertexBufferLayout",
                arrayStride=buffer_des["array_stride"],
                stepMode=buffer_des.get("step_mode", "vertex"),
                attributes=c_attributes_array,
                attributeCount=len(c_attributes_list),
            )
            c_vertex_buffer_layout_list.append(c_vertex_buffer_descriptor)
        c_vertex_buffer_descriptors_array = ffi.new(
            "WGPUVertexBufferLayout []", c_vertex_buffer_layout_list
        )
        # H: nextInChain: WGPUChainedStruct *, module: WGPUShaderModule, entryPoint: char *, constantCount: int, constants: WGPUConstantEntry *, bufferCount: int, buffers: WGPUVertexBufferLayout *
        c_vertex_state = new_struct(
            "WGPUVertexState",
            module=vertex["module"]._internal,
            entryPoint=ffi.new("char []", vertex["entry_point"].encode()),
            buffers=c_vertex_buffer_descriptors_array,
            bufferCount=len(c_vertex_buffer_layout_list),
            # not used: nextInChain
            # not used: constantCount
            # not used: constants
        )

        # H: nextInChain: WGPUChainedStruct *, topology: WGPUPrimitiveTopology, stripIndexFormat: WGPUIndexFormat, frontFace: WGPUFrontFace, cullMode: WGPUCullMode
        c_primitive_state = new_struct(
            "WGPUPrimitiveState",
            topology=primitive["topology"],
            stripIndexFormat=primitive.get("strip_index_format", 0),
            frontFace=primitive.get("front_face", "ccw"),
            cullMode=primitive.get("cull_mode", "none"),
            # not used: nextInChain
        )

        c_depth_stencil_state = ffi.NULL
        if depth_stencil:
            if depth_stencil.get("format", None) is None:
                raise ValueError("depth_stencil needs format")
            stencil_front = depth_stencil.get("stencil_front", {})
            check_struct("StencilFaceState", stencil_front)
            # H: compare: WGPUCompareFunction, failOp: WGPUStencilOperation, depthFailOp: WGPUStencilOperation, passOp: WGPUStencilOperation
            c_stencil_front = new_struct(
                "WGPUStencilFaceState",
                compare=stencil_front.get("compare", "always"),
                failOp=stencil_front.get("fail_op", "keep"),
                depthFailOp=stencil_front.get("depth_fail_op", "keep"),
                passOp=stencil_front.get("pass_op", "keep"),
            )
            stencil_back = depth_stencil.get("stencil_back", {})
            check_struct("StencilFaceState", stencil_back)
            # H: compare: WGPUCompareFunction, failOp: WGPUStencilOperation, depthFailOp: WGPUStencilOperation, passOp: WGPUStencilOperation
            c_stencil_back = new_struct(
                "WGPUStencilFaceState",
                compare=stencil_back.get("compare", "always"),
                failOp=stencil_back.get("fail_op", "keep"),
                depthFailOp=stencil_back.get("depth_fail_op", "keep"),
                passOp=stencil_back.get("pass_op", "keep"),
            )
            # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, depthWriteEnabled: WGPUBool/int, depthCompare: WGPUCompareFunction, stencilFront: WGPUStencilFaceState, stencilBack: WGPUStencilFaceState, stencilReadMask: int, stencilWriteMask: int, depthBias: int, depthBiasSlopeScale: float, depthBiasClamp: float
            c_depth_stencil_state = new_struct_p(
                "WGPUDepthStencilState *",
                format=depth_stencil["format"],
                depthWriteEnabled=bool(depth_stencil.get("depth_write_enabled", False)),
                depthCompare=depth_stencil.get("depth_compare", "always"),
                stencilFront=c_stencil_front,
                stencilBack=c_stencil_back,
                stencilReadMask=depth_stencil.get("stencil_read_mask", 0xFFFFFFFF),
                stencilWriteMask=depth_stencil.get("stencil_write_mask", 0xFFFFFFFF),
                depthBias=depth_stencil.get("depth_bias", 0),
                depthBiasSlopeScale=depth_stencil.get("depth_bias_slope_scale", 0),
                depthBiasClamp=depth_stencil.get("depth_bias_clamp", 0),
                # not used: nextInChain
            )

        # H: nextInChain: WGPUChainedStruct *, count: int, mask: int, alphaToCoverageEnabled: WGPUBool/int
        c_multisample_state = new_struct(
            "WGPUMultisampleState",
            count=multisample.get("count", 1),
            mask=multisample.get("mask", 0xFFFFFFFF),
            alphaToCoverageEnabled=multisample.get("alpha_to_coverage_enabled", False),
            # not used: nextInChain
        )

        c_fragment_state = ffi.NULL
        if fragment is not None:
            c_color_targets_list = []
            for target in fragment["targets"]:
                if not target.get("blend", None):
                    c_blend = ffi.NULL
                else:
                    alpha_blend = _tuple_from_tuple_or_dict(
                        target["blend"]["alpha"],
                        ("src_factor", "dst_factor", "operation"),
                    )
                    # H: operation: WGPUBlendOperation, srcFactor: WGPUBlendFactor, dstFactor: WGPUBlendFactor
                    c_alpha_blend = new_struct(
                        "WGPUBlendComponent",
                        srcFactor=alpha_blend[0],
                        dstFactor=alpha_blend[1],
                        operation=alpha_blend[2],
                    )
                    color_blend = _tuple_from_tuple_or_dict(
                        target["blend"]["color"],
                        ("src_factor", "dst_factor", "operation"),
                    )
                    # H: operation: WGPUBlendOperation, srcFactor: WGPUBlendFactor, dstFactor: WGPUBlendFactor
                    c_color_blend = new_struct(
                        "WGPUBlendComponent",
                        srcFactor=color_blend[0],
                        dstFactor=color_blend[1],
                        operation=color_blend[2],
                    )
                    # H: color: WGPUBlendComponent, alpha: WGPUBlendComponent
                    c_blend = new_struct_p(
                        "WGPUBlendState *",
                        color=c_color_blend,
                        alpha=c_alpha_blend,
                    )
                # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, blend: WGPUBlendState *, writeMask: WGPUColorWriteMaskFlags/int
                c_color_state = new_struct(
                    "WGPUColorTargetState",
                    format=target["format"],
                    blend=c_blend,
                    writeMask=target.get("write_mask", 0xF),
                    # not used: nextInChain
                )
                c_color_targets_list.append(c_color_state)
            c_color_targets_array = ffi.new(
                "WGPUColorTargetState []", c_color_targets_list
            )
            check_struct("FragmentState", fragment)
            # H: nextInChain: WGPUChainedStruct *, module: WGPUShaderModule, entryPoint: char *, constantCount: int, constants: WGPUConstantEntry *, targetCount: int, targets: WGPUColorTargetState *
            c_fragment_state = new_struct_p(
                "WGPUFragmentState *",
                module=fragment["module"]._internal,
                entryPoint=ffi.new("char []", fragment["entry_point"].encode()),
                targets=c_color_targets_array,
                targetCount=len(c_color_targets_list),
                # not used: nextInChain
                # not used: constantCount
                # not used: constants
            )

        # H: nextInChain: WGPUChainedStruct *, label: char *, layout: WGPUPipelineLayout, vertex: WGPUVertexState, primitive: WGPUPrimitiveState, depthStencil: WGPUDepthStencilState *, multisample: WGPUMultisampleState, fragment: WGPUFragmentState *
        struct = new_struct_p(
            "WGPURenderPipelineDescriptor *",
            label=to_c_label(label),
            layout=layout._internal,
            vertex=c_vertex_state,
            primitive=c_primitive_state,
            depthStencil=c_depth_stencil_state,
            multisample=c_multisample_state,
            fragment=c_fragment_state,
            # not used: nextInChain
        )

        # H: WGPURenderPipeline f(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateRenderPipeline(self._internal, struct)
        return GPURenderPipeline(label, id, self)

    async def create_render_pipeline_async(
        self,
        *,
        label="",
        layout: "Union[GPUPipelineLayout, enums.AutoLayoutMode]",
        vertex: "structs.VertexState",
        primitive: "structs.PrimitiveState" = {},
        depth_stencil: "structs.DepthStencilState" = None,
        multisample: "structs.MultisampleState" = {},
        fragment: "structs.FragmentState" = None,
    ):
        return self.create_render_pipeline(
            label=label,
            layout=layout,
            vertex=vertex,
            primitive=primitive,
            depth_stencil=depth_stencil,
            multisample=multisample,
            fragment=fragment,
        )

    def create_command_encoder(self, *, label=""):
        # H: nextInChain: WGPUChainedStruct *, label: char *
        struct = new_struct_p(
            "WGPUCommandEncoderDescriptor *",
            label=to_c_label(label),
            # not used: nextInChain
        )

        # H: WGPUCommandEncoder f(WGPUDevice device, WGPUCommandEncoderDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateCommandEncoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

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
        raise NotImplementedError()
        # Note: also enable the coresponing memtest when implementing this!

    def create_query_set(self, *, label="", type: "enums.QueryType", count: int):
        # H: nextInChain: WGPUChainedStruct *, label: char *, type: WGPUQueryType, count: int
        query_set_descriptor = new_struct_p(
            "WGPUQuerySetDescriptor *",
            label=to_c_label(label),
            type=type,
            count=count,
            # not used: nextInChain
        )

        # H: WGPUQuerySet f(WGPUDevice device, WGPUQuerySetDescriptor const * descriptor)
        query_id = libf.wgpuDeviceCreateQuerySet(self._internal, query_set_descriptor)
        return GPUQuerySet(label, query_id, self._internal, type, count)

    def _destroy(self):
        if self._queue is not None:
            self._queue._destroy()
            self._queue = None
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUDevice device)
            libf.wgpuDeviceRelease(internal)
            # wgpuDeviceDestroy(internal) is also an option


class GPUBuffer(classes.GPUBuffer, GPUObjectBase):
    def __init__(self, label, internal, device, size, usage, map_state):
        super().__init__(label, internal, device, size, usage, map_state)

        self._mapped_status = 0, 0, 0
        self._mapped_memoryviews = []
        # If mapped at creation, set to write mode (no point in reading zeros)
        if self._map_state == enums.BufferMapState.mapped:
            self._mapped_status = 0, self.size, flags.MapMode.WRITE

    def _check_range(self, offset, size):
        # Apply defaults
        if offset is None:
            offset = 0
            if self._mapped_status[2] != 0:
                offset = self._mapped_status[0]
        else:
            offset = int(offset)
        if size is None:
            size = self.size - offset
            if self._mapped_status[2] != 0:
                size = self._mapped_status[1] - offset
        else:
            size = int(size)
        # Checks
        if offset < 0:
            raise ValueError("Mapped offset must not be smaller than zero.")
        if offset % 8:
            raise ValueError("Mapped offset must be a multiple of 8.")
        if size < 1:
            raise ValueError("Mapped size must be larger than zero.")
        if size % 4:
            raise ValueError("Mapped offset must be a multiple of 4.")
        if offset + size > self.size:
            raise ValueError("Mapped range must not extend beyond total buffer size.")
        return offset, size

    def map(self, mode, offset=0, size=None):
        sync_on_read = True

        # Check mode
        if isinstance(mode, str):
            if mode == "READ_NOSYNC":  # for internal use
                sync_on_read = False
                mode = "READ"
            mode = str_flag_to_int(flags.MapMode, mode)
        map_mode = int(mode)

        # Check offset and size
        offset, size = self._check_range(offset, size)

        # Can we even map?
        if self._map_state != enums.BufferMapState.unmapped:
            raise RuntimeError("Can only map a buffer if its currently unmapped.")

        # Sync up when reading, otherwise the memory may be all zeros.
        # See https://github.com/gfx-rs/wgpu-native/issues/305
        if sync_on_read and map_mode & lib.WGPUMapMode_Read:
            if self._mapped_status[2] == 0 and self._usage & flags.BufferUsage.MAP_READ:
                encoder = self._device.create_command_encoder()
                command_buffer = encoder.finish()
                self._device.queue.submit([command_buffer])

        status = 999

        @ffi.callback("void(WGPUBufferMapAsyncStatus, void*)")
        def callback(status_, user_data_p):
            nonlocal status
            status = status_

        # Map it
        self._map_state = enums.BufferMapState.pending
        # H: void f(WGPUBuffer buffer, WGPUMapModeFlags mode, size_t offset, size_t size, WGPUBufferMapCallback callback, void * userdata)
        libf.wgpuBufferMapAsync(
            self._internal, map_mode, offset, size, callback, ffi.NULL
        )

        # Let it do some cycles
        self._device._poll()

        if status != 0:  # no-cover
            raise RuntimeError(f"Could not map buffer ({status}).")
        self._map_state = enums.BufferMapState.mapped
        self._mapped_status = offset, offset + size, mode
        self._mapped_memoryviews = []

    async def map_async(self, mode, offset=0, size=None):
        return self.map(mode, offset, size)  # for now

    def unmap(self):
        if self._map_state != enums.BufferMapState.mapped:
            raise RuntimeError("Can only unmap a buffer if its currently mapped.")
        # H: void f(WGPUBuffer buffer)
        libf.wgpuBufferUnmap(self._internal)
        self._map_state = enums.BufferMapState.unmapped
        self._mapped_status = 0, 0, 0
        self._release_memoryviews()

    def _release_memoryviews(self):
        # Release the mapped memoryview objects. These objects
        # themselves become unusable, but any views on them do not.
        for m in self._mapped_memoryviews:
            try:
                m.release()
            except Exception:  # no-cover
                pass
        self._mapped_memoryviews = []

    def read_mapped(self, buffer_offset=None, size=None, *, copy=True):
        # Can we even read?
        if self._map_state != enums.BufferMapState.mapped:
            raise RuntimeError("Can only read from a buffer if its mapped.")
        elif not (self._mapped_status[2] & flags.MapMode.READ):
            raise RuntimeError(
                "Can only read from a buffer if its mapped in read mode."
            )

        # Check offset and size
        offset, size = self._check_range(buffer_offset, size)
        if offset < self._mapped_status[0] or (offset + size) > self._mapped_status[1]:
            raise ValueError(
                "The range for buffer reading is not contained in the currently mapped range."
            )

        # Get mapped memoryview.
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        src_ptr = libf.wgpuBufferGetMappedRange(self._internal, offset, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, size)

        if copy:
            # Copy the data. The memoryview created above becomes invalid when the buffer
            # is unmapped, so we don't want to pass that memory to the user.
            data = memoryview((ctypes.c_uint8 * size)()).cast("B")
            data[:] = src_m
            return data
        else:
            # Return view on the actually mapped data
            data = src_m
            if hasattr(data, "toreadonly"):  # Py 3.8+
                data = data.toreadonly()
            self._mapped_memoryviews.append(data)
            return data

    def write_mapped(self, data, buffer_offset=None, size=None):
        # Can we even write?
        if self._map_state != enums.BufferMapState.mapped:
            raise RuntimeError("Can only write to a buffer if its mapped.")
        elif not (self._mapped_status[2] & flags.MapMode.WRITE):
            raise RuntimeError(
                "Can only write from a buffer if its mapped in write mode."
            )

        # Cast data to a memoryview. This also works for e.g. numpy arrays,
        # and the resulting memoryview will be a view on the data.
        data = memoryview(data).cast("B")

        # Check offset and size
        if size is None:
            size = data.nbytes
        offset, size = self._check_range(buffer_offset, size)
        if offset < self._mapped_status[0] or (offset + size) > self._mapped_status[1]:
            raise ValueError(
                "The range for buffer writing is not contained in the currently mapped range."
            )

        # Check data size and given size. If the latter was given, it should match!
        if data.nbytes != size:  # no-cover
            raise ValueError(
                "Data passed to GPUBuffer.write_mapped() does not match the given size."
            )

        # Get mapped memoryview
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        src_ptr = libf.wgpuBufferGetMappedRange(self._internal, offset, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, size)

        # Copy data
        src_m[:] = data

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        self._release_memoryviews()
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBuffer buffer)
            libf.wgpuBufferRelease(internal)


class GPUTexture(classes.GPUTexture, GPUObjectBase):
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
        # Resolve defaults
        if not format:
            format = self._tex_info["format"]
        if not dimension:
            dimension = self._tex_info["dimension"]  # from create_texture
        if not aspect:
            aspect = "all"
        if not mip_level_count:
            mip_level_count = self._tex_info["mip_level_count"] - base_mip_level
        if not array_layer_count:
            if dimension in ("1d", "2d", "3d"):
                array_layer_count = 1  # or WGPU_ARRAY_LAYER_COUNT_UNDEFINED ?
            elif dimension == "cube":
                array_layer_count = 6
            elif dimension in ("2d-array", "cube-array"):
                array_layer_count = self._tex_info["size"][2] - base_array_layer

        # H: nextInChain: WGPUChainedStruct *, label: char *, format: WGPUTextureFormat, dimension: WGPUTextureViewDimension, baseMipLevel: int, mipLevelCount: int, baseArrayLayer: int, arrayLayerCount: int, aspect: WGPUTextureAspect
        struct = new_struct_p(
            "WGPUTextureViewDescriptor *",
            label=to_c_label(label),
            format=format,
            dimension=dimension,
            aspect=aspect,
            baseMipLevel=base_mip_level,
            mipLevelCount=mip_level_count,
            baseArrayLayer=base_array_layer,
            arrayLayerCount=array_layer_count,
            # not used: nextInChain
        )
        # H: WGPUTextureView f(WGPUTexture texture, WGPUTextureViewDescriptor const * descriptor)
        id = libf.wgpuTextureCreateView(self._internal, struct)
        return GPUTextureView(label, id, self._device, self, self.size)

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTexture texture)
            libf.wgpuTextureRelease(internal)


class GPUTextureView(classes.GPUTextureView, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTextureView textureView)
            libf.wgpuTextureViewRelease(internal)


class GPUSampler(classes.GPUSampler, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUSampler sampler)
            libf.wgpuSamplerRelease(internal)


class GPUBindGroupLayout(classes.GPUBindGroupLayout, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroupLayout bindGroupLayout)
            libf.wgpuBindGroupLayoutRelease(internal)


class GPUBindGroup(classes.GPUBindGroup, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroup bindGroup)
            libf.wgpuBindGroupRelease(internal)


class GPUPipelineLayout(classes.GPUPipelineLayout, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUPipelineLayout pipelineLayout)
            libf.wgpuPipelineLayoutRelease(internal)


class GPUShaderModule(classes.GPUShaderModule, GPUObjectBase):
    def get_compilation_info(self):
        # Here's a little setup to implement this method. Unfortunately,
        # this is not yet implemented in wgpu-native. Another problem
        # is that if there is an error in the shader source, we raise
        # an exception, so the user never gets a GPUShaderModule object
        # that can be used to call this method :/ So perhaps we should
        # do this stuff in device.create_shader_module() and attach it
        # to the exception that we raise?

        # info = None
        #
        # @ffi.callback("void(WGPUCompilationInfoRequestStatus, WGPUCompilationInfo*, void*)")
        # def callback(status_, info_, userdata):
        #     if status_ == 0:
        #         nonlocal info
        #         info = info_
        #     else:
        #         pass
        #
        # H: void f(WGPUShaderModule shaderModule, WGPUCompilationInfoCallback callback, void * userdata)
        # libf.wgpuShaderModuleGetCompilationInfo(self._internal, callback, ffi.NULL)
        #
        # self._device._poll()
        #
        # if info is None:
        #     raise RuntimeError("Could not obtain shader compilation info.")
        #
        #  ... and then turn these WGPUCompilationInfoRequestStatus objects into Python objects ...

        return []

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUShaderModule shaderModule)
            libf.wgpuShaderModuleRelease(internal)


class GPUPipelineBase(classes.GPUPipelineBase):
    def get_bind_group_layout(self, index):
        """Get the bind group layout at the given index."""
        if isinstance(self, GPUComputePipeline):
            # H: WGPUBindGroupLayout f(WGPUComputePipeline computePipeline, uint32_t groupIndex)
            layout_id = libf.wgpuComputePipelineGetBindGroupLayout(
                self._internal, index
            )
        else:
            # H: WGPUBindGroupLayout f(WGPURenderPipeline renderPipeline, uint32_t groupIndex)
            layout_id = libf.wgpuRenderPipelineGetBindGroupLayout(self._internal, index)
        return GPUBindGroupLayout("", layout_id, self._device, [])


class GPUComputePipeline(classes.GPUComputePipeline, GPUPipelineBase, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUComputePipeline computePipeline)
            libf.wgpuComputePipelineRelease(internal)


class GPURenderPipeline(classes.GPURenderPipeline, GPUPipelineBase, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderPipeline renderPipeline)
            libf.wgpuRenderPipelineRelease(internal)


class GPUCommandBuffer(classes.GPUCommandBuffer, GPUObjectBase):
    def _destroy(self):
        # Note that command buffers get destroyed when they are submitted.
        # In earlier versions we had to take this into account by setting
        # _internal to None. That seems not necessary anymore.
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUCommandBuffer commandBuffer)
            libf.wgpuCommandBufferRelease(internal)


class GPUCommandsMixin(classes.GPUCommandsMixin):
    pass


class GPUBindingCommandsMixin(classes.GPUBindingCommandsMixin):
    def set_bind_group(
        self,
        index,
        bind_group,
        dynamic_offsets_data,
        dynamic_offsets_data_start,
        dynamic_offsets_data_length,
    ):
        offsets = list(dynamic_offsets_data)
        c_offsets = ffi.new("uint32_t []", offsets)
        bind_group_id = bind_group._internal
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
            libf.wgpuComputePassEncoderSetBindGroup(
                self._internal, index, bind_group_id, len(offsets), c_offsets
            )
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
            libf.wgpuRenderPassEncoderSetBindGroup(
                self._internal,
                index,
                bind_group_id,
                len(offsets),
                c_offsets,
            )


class GPUDebugCommandsMixin(classes.GPUDebugCommandsMixin):
    def push_debug_group(self, group_label):
        c_group_label = ffi.new("char []", group_label.encode())
        color = 0
        # todo: these functions are temporarily not available in wgpu-native
        return  # noqa
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder, char const * groupLabel)
            libf.wgpuComputePassEncoderPushDebugGroup(
                self._internal, c_group_label, color
            )
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder, char const * groupLabel)
            libf.wgpuRenderPassEncoderPushDebugGroup(
                self._internal, c_group_label, color
            )

    def pop_debug_group(self):
        # todo: these functions are temporarily not available in wgpu-native
        return  # noqa
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder)
            libf.wgpuComputePassEncoderPopDebugGroup(self._internal)
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder)
            libf.wgpuRenderPassEncoderPopDebugGroup(self._internal)

    def insert_debug_marker(self, marker_label):
        c_marker_label = ffi.new("char []", marker_label.encode())
        color = 0
        # todo: these functions are temporarily not available in wgpu-native
        return  # noqa
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder, char const * markerLabel)
            libf.wgpuComputePassEncoderInsertDebugMarker(
                self._internal, c_marker_label, color
            )
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder, char const * markerLabel)
            libf.wgpuRenderPassEncoderInsertDebugMarker(
                self._internal, c_marker_label, color
            )


class GPURenderCommandsMixin(classes.GPURenderCommandsMixin):
    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline)
        libf.wgpuRenderPassEncoderSetPipeline(self._internal, pipeline_id)

    def set_index_buffer(self, buffer, index_format, offset=0, size=None):
        if not size:
            size = buffer.size - offset
        c_index_format = enummap[f"IndexFormat.{index_format}"]
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
        libf.wgpuRenderPassEncoderSetIndexBuffer(
            self._internal, buffer._internal, c_index_format, int(offset), int(size)
        )

    def set_vertex_buffer(self, slot, buffer, offset=0, size=None):
        if not size:
            size = buffer.size - offset
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size)
        libf.wgpuRenderPassEncoderSetVertexBuffer(
            self._internal, int(slot), buffer._internal, int(offset), int(size)
        )

    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
        libf.wgpuRenderPassEncoderDraw(
            self._internal, vertex_count, instance_count, first_vertex, first_instance
        )

    def draw_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        libf.wgpuRenderPassEncoderDrawIndirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def draw_indexed(
        self,
        index_count,
        instance_count=1,
        first_index=0,
        base_vertex=0,
        first_instance=0,
    ):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
        libf.wgpuRenderPassEncoderDrawIndexed(
            self._internal,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )

    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        libf.wgpuRenderPassEncoderDrawIndexedIndirect(
            self._internal, buffer_id, int(indirect_offset)
        )


class GPUCommandEncoder(
    classes.GPUCommandEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUObjectBase
):
    def begin_compute_pass(
        self, *, label="", timestamp_writes: "structs.ComputePassTimestampWrites" = None
    ):
        timestamp_writes_struct = ffi.NULL
        if timestamp_writes is not None:
            check_struct("ComputePassTimestampWrites", timestamp_writes)
            # H: querySet: WGPUQuerySet, beginningOfPassWriteIndex: int, endOfPassWriteIndex: int
            timestamp_writes_struct = new_struct_p(
                "WGPUComputePassTimestampWrites *",
                querySet=timestamp_writes["query_set"]._internal,
                beginningOfPassWriteIndex=timestamp_writes[
                    "beginning_of_pass_write_index"
                ],
                endOfPassWriteIndex=timestamp_writes["end_of_pass_write_index"],
            )
        # H: nextInChain: WGPUChainedStruct *, label: char *, timestampWrites: WGPUComputePassTimestampWrites *
        struct = new_struct_p(
            "WGPUComputePassDescriptor *",
            label=to_c_label(label),
            timestampWrites=timestamp_writes_struct,
            # not used: nextInChain
        )
        # H: WGPUComputePassEncoder f(WGPUCommandEncoder commandEncoder, WGPUComputePassDescriptor const * descriptor)
        raw_pass = libf.wgpuCommandEncoderBeginComputePass(self._internal, struct)
        return GPUComputePassEncoder(label, raw_pass, self)

    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "List[structs.RenderPassColorAttachment]",
        depth_stencil_attachment: "structs.RenderPassDepthStencilAttachment" = None,
        occlusion_query_set: "GPUQuerySet" = None,
        timestamp_writes: "structs.RenderPassTimestampWrites" = None,
        max_draw_count: int = 50000000,
    ):
        # Note that occlusion_query_set is ignored because wgpu-native does not have it.
        if timestamp_writes is not None:
            check_struct("RenderPassTimestampWrites", timestamp_writes)

        objects_to_keep_alive = {}

        c_color_attachments_list = []
        for color_attachment in color_attachments:
            check_struct("RenderPassColorAttachment", color_attachment)
            texture_view = color_attachment["view"]
            if not isinstance(texture_view, GPUTextureView):
                raise TypeError("Color attachement view must be a GPUTextureView.")
            texture_view_id = texture_view._internal
            objects_to_keep_alive[texture_view_id] = texture_view
            c_resolve_target = (
                ffi.NULL
                if color_attachment.get("resolve_target", None) is None
                else color_attachment["resolve_target"]._internal
            )  # this is a TextureViewId or null
            clear_value = color_attachment.get("clear_value", (0, 0, 0, 0))
            if isinstance(clear_value, dict):
                check_struct("Color", clear_value)
                clear_value = _tuple_from_tuple_or_dict(clear_value, "rgba")
            # H: r: float, g: float, b: float, a: float
            c_clear_value = new_struct(
                "WGPUColor",
                r=clear_value[0],
                g=clear_value[1],
                b=clear_value[2],
                a=clear_value[3],
            )
            # H: nextInChain: WGPUChainedStruct *, view: WGPUTextureView, resolveTarget: WGPUTextureView, loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearValue: WGPUColor
            c_attachment = new_struct(
                "WGPURenderPassColorAttachment",
                view=texture_view_id,
                resolveTarget=c_resolve_target,
                loadOp=color_attachment["load_op"],
                storeOp=color_attachment["store_op"],
                clearValue=c_clear_value,
                # not used: resolveTarget
                # not used: nextInChain
            )
            c_color_attachments_list.append(c_attachment)
        c_color_attachments_array = ffi.new(
            "WGPURenderPassColorAttachment []", c_color_attachments_list
        )

        c_depth_stencil_attachment = ffi.NULL
        if depth_stencil_attachment is not None:
            check_struct("RenderPassDepthStencilAttachment", depth_stencil_attachment)
            depth_clear_value = depth_stencil_attachment.get("depth_clear_value", 0)
            stencil_clear_value = depth_stencil_attachment.get("stencil_clear_value", 0)
            # H: view: WGPUTextureView, depthLoadOp: WGPULoadOp, depthStoreOp: WGPUStoreOp, depthClearValue: float, depthReadOnly: WGPUBool/int, stencilLoadOp: WGPULoadOp, stencilStoreOp: WGPUStoreOp, stencilClearValue: int, stencilReadOnly: WGPUBool/int
            c_depth_stencil_attachment = new_struct_p(
                "WGPURenderPassDepthStencilAttachment *",
                view=depth_stencil_attachment["view"]._internal,
                depthLoadOp=depth_stencil_attachment["depth_load_op"],
                depthStoreOp=depth_stencil_attachment["depth_store_op"],
                depthClearValue=float(depth_clear_value),
                depthReadOnly=depth_stencil_attachment.get("depth_read_only", False),
                stencilLoadOp=depth_stencil_attachment["stencil_load_op"],
                stencilStoreOp=depth_stencil_attachment["stencil_store_op"],
                stencilClearValue=int(stencil_clear_value),
                stencilReadOnly=depth_stencil_attachment.get(
                    "stencil_read_only", False
                ),
            )

        # H: nextInChain: WGPUChainedStruct *, label: char *, colorAttachmentCount: int, colorAttachments: WGPURenderPassColorAttachment *, depthStencilAttachment: WGPURenderPassDepthStencilAttachment *, occlusionQuerySet: WGPUQuerySet, timestampWrites: WGPURenderPassTimestampWrites *
        struct = new_struct_p(
            "WGPURenderPassDescriptor *",
            label=to_c_label(label),
            colorAttachments=c_color_attachments_array,
            colorAttachmentCount=len(c_color_attachments_list),
            depthStencilAttachment=c_depth_stencil_attachment,
            # not used: occlusionQuerySet
            # not used: timestampWrites
            # not used: nextInChain
        )

        # H: WGPURenderPassEncoder f(WGPUCommandEncoder commandEncoder, WGPURenderPassDescriptor const * descriptor)
        raw_pass = libf.wgpuCommandEncoderBeginRenderPass(self._internal, struct)
        encoder = GPURenderPassEncoder(label, raw_pass, self)
        encoder._objects_to_keep_alive = objects_to_keep_alive
        return encoder

    def clear_buffer(self, buffer, offset=0, size=None):
        offset = int(offset)
        if offset % 4 != 0:  # pragma: no cover
            raise ValueError("offset must be a multiple of 4")
        if size is None:  # pragma: no cover
            size = buffer.size - offset
        size = int(size)
        if size <= 0:  # pragma: no cover
            raise ValueError("clear_buffer size must be > 0")
        if size % 4 != 0:  # pragma: no cover
            raise ValueError("size must be a multiple of 4")
        if offset + size > buffer.size:  # pragma: no cover
            raise ValueError("buffer size out of range")
        # H: void f(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t offset, uint64_t size)
        libf.wgpuCommandEncoderClearBuffer(
            self._internal, buffer._internal, int(offset), size
        )

    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        if source_offset % 4 != 0:  # pragma: no cover
            raise ValueError("source_offset must be a multiple of 4")
        if destination_offset % 4 != 0:  # pragma: no cover
            raise ValueError("destination_offset must be a multiple of 4")
        if size % 4 != 0:  # pragma: no cover
            raise ValueError("size must be a multiple of 4")

        if not isinstance(source, GPUBuffer):  # pragma: no cover
            raise TypeError("copy_buffer_to_buffer() source must be a GPUBuffer.")
        if not isinstance(destination, GPUBuffer):  # pragma: no cover
            raise TypeError("copy_buffer_to_buffer() destination must be a GPUBuffer.")
        # H: void f(WGPUCommandEncoder commandEncoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size)
        libf.wgpuCommandEncoderCopyBufferToBuffer(
            self._internal,
            source._internal,
            int(source_offset),
            destination._internal,
            int(destination_offset),
            int(size),
        )

    def copy_buffer_to_texture(self, source, destination, copy_size):
        row_alignment = 256
        bytes_per_row = int(source["bytes_per_row"])
        if (bytes_per_row % row_alignment) != 0:
            raise ValueError(
                f"bytes_per_row must ({bytes_per_row}) be a multiple of {row_alignment}"
            )
        if isinstance(destination["texture"], GPUTextureView):
            raise ValueError("copy destination texture must be a texture, not a view")

        size = _tuple_from_tuple_or_dict(
            copy_size, ("width", "height", "depth_or_array_layers")
        )

        c_source = new_struct_p(
            "WGPUImageCopyBuffer *",
            buffer=source["buffer"]._internal,
            # H: nextInChain: WGPUChainedStruct *, offset: int, bytesPerRow: int, rowsPerImage: int
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(source.get("offset", 0)),
                bytesPerRow=bytes_per_row,
                rowsPerImage=int(source.get("rows_per_image", size[1])),
                # not used: nextInChain
            ),
        )

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: nextInChain: WGPUChainedStruct *, texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_destination = new_struct_p(
            "WGPUImageCopyTexture *",
            texture=destination["texture"]._internal,
            mipLevel=int(destination.get("mip_level", 0)),
            origin=c_origin,
            aspect=enums.TextureAspect.all,
            # not used: nextInChain
        )

        # H: width: int, height: int, depthOrArrayLayers: int
        c_copy_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUImageCopyBuffer const * source, WGPUImageCopyTexture const * destination, WGPUExtent3D const * copySize)
        libf.wgpuCommandEncoderCopyBufferToTexture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_buffer(self, source, destination, copy_size):
        row_alignment = 256
        bytes_per_row = int(destination["bytes_per_row"])
        if (bytes_per_row % row_alignment) != 0:
            raise ValueError(
                f"bytes_per_row must ({bytes_per_row}) be a multiple of {row_alignment}"
            )
        if isinstance(source["texture"], GPUTextureView):
            raise ValueError("copy source texture must be a texture, not a view")

        size = _tuple_from_tuple_or_dict(
            copy_size, ("width", "height", "depth_or_array_layers")
        )

        ori = _tuple_from_tuple_or_dict(source.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: nextInChain: WGPUChainedStruct *, texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_source = new_struct_p(
            "WGPUImageCopyTexture *",
            texture=source["texture"]._internal,
            mipLevel=int(source.get("mip_level", 0)),
            origin=c_origin,
            aspect=0,
            # not used: nextInChain
        )

        c_destination = new_struct_p(
            "WGPUImageCopyBuffer *",
            buffer=destination["buffer"]._internal,
            # H: nextInChain: WGPUChainedStruct *, offset: int, bytesPerRow: int, rowsPerImage: int
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(destination.get("offset", 0)),
                bytesPerRow=bytes_per_row,
                rowsPerImage=int(destination.get("rows_per_image", size[1])),
                # not used: nextInChain
            ),
        )

        # H: width: int, height: int, depthOrArrayLayers: int
        c_copy_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUImageCopyTexture const * source, WGPUImageCopyBuffer const * destination, WGPUExtent3D const * copySize)
        libf.wgpuCommandEncoderCopyTextureToBuffer(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_texture(self, source, destination, copy_size):
        if isinstance(source["texture"], GPUTextureView):
            raise ValueError("copy source texture must be a texture, not a view")
        if isinstance(destination["texture"], GPUTextureView):
            raise ValueError("copy destination texture must be a texture, not a view")

        ori = _tuple_from_tuple_or_dict(source.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin1 = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: nextInChain: WGPUChainedStruct *, texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_source = new_struct_p(
            "WGPUImageCopyTexture *",
            texture=source["texture"]._internal,
            mipLevel=int(source.get("mip_level", 0)),
            origin=c_origin1,
            # not used: nextInChain
            # not used: aspect
        )

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin2 = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: nextInChain: WGPUChainedStruct *, texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_destination = new_struct_p(
            "WGPUImageCopyTexture *",
            texture=destination["texture"]._internal,
            mipLevel=int(destination.get("mip_level", 0)),
            origin=c_origin2,
            # not used: nextInChain
            # not used: aspect
        )

        size = _tuple_from_tuple_or_dict(
            copy_size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depthOrArrayLayers: int
        c_copy_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUImageCopyTexture const * source, WGPUImageCopyTexture const * destination, WGPUExtent3D const * copySize)
        libf.wgpuCommandEncoderCopyTextureToTexture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def finish(self, *, label=""):
        # H: nextInChain: WGPUChainedStruct *, label: char *
        struct = new_struct_p(
            "WGPUCommandBufferDescriptor *",
            label=to_c_label(label),
            # not used: nextInChain
        )
        # H: WGPUCommandBuffer f(WGPUCommandEncoder commandEncoder, WGPUCommandBufferDescriptor const * descriptor)
        id = libf.wgpuCommandEncoderFinish(self._internal, struct)
        return GPUCommandBuffer(label, id, self)

    def resolve_query_set(
        self, query_set, first_query, query_count, destination, destination_offset
    ):
        # H: void f(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer destination, uint64_t destinationOffset)
        libf.wgpuCommandEncoderResolveQuerySet(
            self._internal,
            query_set._internal,
            int(first_query),
            int(query_count),
            destination._internal,
            int(destination_offset),
        )

    def _destroy(self):
        # Note that the native object gets destroyed on finish.
        # Also see GPUCommandBuffer._destroy()
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUCommandEncoder commandEncoder)
            libf.wgpuCommandEncoderRelease(internal)


class GPUComputePassEncoder(
    classes.GPUComputePassEncoder,
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPUObjectBase,
):
    """ """

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline)
        libf.wgpuComputePassEncoderSetPipeline(self._internal, pipeline_id)

    def dispatch_workgroups(
        self, workgroup_count_x, workgroup_count_y=1, workgroup_count_z=1
    ):
        # H: void f(WGPUComputePassEncoder computePassEncoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ)
        libf.wgpuComputePassEncoderDispatchWorkgroups(
            self._internal, workgroup_count_x, workgroup_count_y, workgroup_count_z
        )

    def dispatch_workgroups_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        libf.wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def end(self):
        # H: void f(WGPUComputePassEncoder computePassEncoder)
        libf.wgpuComputePassEncoderEnd(self._internal)

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUComputePassEncoder computePassEncoder)
            libf.wgpuComputePassEncoderRelease(internal)


class GPURenderPassEncoder(
    classes.GPURenderPassEncoder,
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPURenderCommandsMixin,
    GPUObjectBase,
):
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, float x, float y, float width, float height, float minDepth, float maxDepth)
        libf.wgpuRenderPassEncoderSetViewport(
            self._internal,
            float(x),
            float(y),
            float(width),
            float(height),
            float(min_depth),
            float(max_depth),
        )

    def set_scissor_rect(self, x, y, width, height):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height)
        libf.wgpuRenderPassEncoderSetScissorRect(
            self._internal, int(x), int(y), int(width), int(height)
        )

    def set_blend_constant(self, color):
        color = _tuple_from_tuple_or_dict(color, "rgba")
        # H: r: float, g: float, b: float, a: float
        c_color = new_struct_p(
            "WGPUColor *",
            r=color[0],
            g=color[1],
            b=color[2],
            a=color[3],
        )
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPUColor const * color)
        libf.wgpuRenderPassEncoderSetBlendConstant(self._internal, c_color)

    def set_stencil_reference(self, reference):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t reference)
        libf.wgpuRenderPassEncoderSetStencilReference(self._internal, int(reference))

    def end(self):
        # H: void f(WGPURenderPassEncoder renderPassEncoder)
        libf.wgpuRenderPassEncoderEnd(self._internal)

    def execute_bundles(self, bundles):
        raise NotImplementedError()

    def begin_occlusion_query(self, query_index):
        raise NotImplementedError()

    def end_occlusion_query(self):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderPassEncoder renderPassEncoder)
            libf.wgpuRenderPassEncoderRelease(internal)


class GPURenderBundleEncoder(
    classes.GPURenderBundleEncoder,
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPURenderCommandsMixin,
    GPUObjectBase,
):
    def finish(self, *, label=""):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderBundleEncoder renderBundleEncoder)
            libf.wgpuRenderBundleEncoderRelease(internal)


class GPUQueue(classes.GPUQueue, GPUObjectBase):
    def submit(self, command_buffers):
        command_buffer_ids = [cb._internal for cb in command_buffers]
        c_command_buffers = ffi.new("WGPUCommandBuffer []", command_buffer_ids)
        # H: void f(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const * commands)
        libf.wgpuQueueSubmit(self._internal, len(command_buffer_ids), c_command_buffers)

    def write_buffer(self, buffer, buffer_offset, data, data_offset=0, size=None):
        # We support anything that memoryview supports, i.e. anything
        # that implements the buffer protocol, including, bytes,
        # bytearray, ctypes arrays, numpy arrays, etc.
        m, address = get_memoryview_and_address(data)
        nbytes = m.nbytes

        # Deal with offset and size
        buffer_offset = int(buffer_offset)
        data_offset = int(data_offset)
        if not size:
            data_length = nbytes - data_offset
        else:
            data_length = int(size)

        if not (0 <= buffer_offset < buffer.size):  # pragma: no cover
            raise ValueError("Invalid buffer_offset")
        if not (0 <= data_offset < nbytes):  # pragma: no cover
            raise ValueError("Invalid data_offset")
        if not (0 <= data_length <= (nbytes - data_offset)):  # pragma: no cover
            raise ValueError("Invalid data_length")
        if not (data_length <= buffer.size - buffer_offset):  # pragma: no cover
            raise ValueError("Invalid data_length")

        # Make the call. Note that this call copies the data - it's ok
        # if we lose our reference to the data once we leave this function.
        c_data = ffi.cast("uint8_t *", address + data_offset)
        # H: void f(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, void const * data, size_t size)
        libf.wgpuQueueWriteBuffer(
            self._internal, buffer._internal, buffer_offset, c_data, data_length
        )

    def read_buffer(self, buffer, buffer_offset=0, size=None):
        # Note that write_buffer probably does a very similar thing
        # using a temporary buffer. But write_buffer is official API
        # so it's a single call, while here we must create the temporary
        # buffer and do the copying ourselves.

        if not size:
            data_length = buffer.size - buffer_offset
        else:
            data_length = int(size)
        if not (0 <= buffer_offset < buffer.size):  # pragma: no cover
            raise ValueError("Invalid buffer_offset")
        if not (data_length <= buffer.size - buffer_offset):  # pragma: no cover
            raise ValueError("Invalid data_length")

        device = buffer._device

        # Create temporary buffer
        tmp_usage = flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ
        tmp_buffer = device._create_buffer("", data_length, tmp_usage, False)

        # Copy data to temp buffer
        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(buffer, buffer_offset, tmp_buffer, 0, data_length)
        command_buffer = encoder.finish()
        self.submit([command_buffer])

        # Download from mappable buffer
        tmp_buffer.map("READ_NOSYNC")
        data = tmp_buffer.read_mapped()
        tmp_buffer.destroy()

        return data

    def write_texture(self, destination, data, data_layout, size):
        # Note that the bytes_per_row restriction does not apply for
        # this function; wgpu-native deals with it.

        if isinstance(destination["texture"], GPUTextureView):
            raise ValueError("copy destination texture must be a texture, not a view")

        m, address = get_memoryview_and_address(data)

        c_data = ffi.cast("uint8_t *", address)
        data_length = m.nbytes

        # We could allow size=None in this method, and derive the size from the data.
        # Or compare size with the data size if it is given. However, the data
        # could be a bit raw, being 1D and/or the shape expressed in bytes, so
        # this gets a bit muddy. Also methods like copy_buffer_to_texture have the
        # same size arg, so let's just leave it like this.
        #
        # data_size = list(reversed(m.shape)) + [1, 1, 1]
        # data_size = data_size[:3]

        size = _tuple_from_tuple_or_dict(
            size, ("width", "height", "depth_or_array_layers")
        )

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: nextInChain: WGPUChainedStruct *, texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_destination = new_struct_p(
            "WGPUImageCopyTexture *",
            texture=destination["texture"]._internal,
            mipLevel=destination.get("mip_level", 0),
            origin=c_origin,
            aspect=enums.TextureAspect.all,
            # not used: nextInChain
        )

        # H: nextInChain: WGPUChainedStruct *, offset: int, bytesPerRow: int, rowsPerImage: int
        c_data_layout = new_struct_p(
            "WGPUTextureDataLayout *",
            offset=data_layout.get("offset", 0),
            bytesPerRow=data_layout["bytes_per_row"],
            rowsPerImage=data_layout.get("rows_per_image", size[1]),
            # not used: nextInChain
        )

        # H: width: int, height: int, depthOrArrayLayers: int
        c_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUQueue queue, WGPUImageCopyTexture const * destination, void const * data, size_t dataSize, WGPUTextureDataLayout const * dataLayout, WGPUExtent3D const * writeSize)
        libf.wgpuQueueWriteTexture(
            self._internal, c_destination, c_data, data_length, c_data_layout, c_size
        )

    def read_texture(self, source, data_layout, size):
        # Note that the bytes_per_row restriction does not apply for
        # this function; we have to deal with it.

        device = source["texture"]._device

        # Get and calculate striding info
        ori_offset = data_layout.get("offset", 0)
        ori_stride = data_layout["bytes_per_row"]
        extra_stride = (256 - ori_stride % 256) % 256
        full_stride = ori_stride + extra_stride

        size = _tuple_from_tuple_or_dict(
            size, ("width", "height", "depth_or_array_layers")
        )

        # Create temporary buffer
        data_length = full_stride * size[1] * size[2]
        tmp_usage = flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ
        tmp_buffer = device._create_buffer("", data_length, tmp_usage, False)

        destination = {
            "buffer": tmp_buffer,
            "offset": 0,
            "bytes_per_row": full_stride,  # or WGPU_COPY_STRIDE_UNDEFINED ?
            "rows_per_image": data_layout.get("rows_per_image", size[1]),
        }

        # Copy data to temp buffer
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(source, destination, size)
        command_buffer = encoder.finish()
        self.submit([command_buffer])

        # Download from mappable buffer
        tmp_buffer.map("READ_NOSYNC")
        data = tmp_buffer.read_mapped()
        tmp_buffer.destroy()

        # Fix data strides if necessary
        # Ugh, cannot do striding with memoryviews (yet: https://bugs.python.org/issue41226)
        # and Numpy is not a dependency.
        if extra_stride or ori_offset:
            data_length2 = ori_stride * size[1] * size[2] + ori_offset
            data2 = memoryview((ctypes.c_uint8 * data_length2)()).cast(data.format)
            for i in range(size[1] * size[2]):
                row = data[i * full_stride : i * full_stride + ori_stride]
                data2[
                    ori_offset
                    + i * ori_stride : ori_offset
                    + i * ori_stride
                    + ori_stride
                ] = row
            data = data2

        return data

    def on_submitted_work_done(self):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUQueue queue)
            libf.wgpuQueueRelease(internal)


class GPURenderBundle(classes.GPURenderBundle, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderBundle renderBundle)
            libf.wgpuRenderBundleRelease(internal)


class GPUQuerySet(classes.GPUQuerySet, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUQuerySet querySet)
            libf.wgpuQuerySetRelease(internal)

    def destroy(self):
        self._destroy()


# %% Subclasses that don't need anything else


class GPUCompilationMessage(classes.GPUCompilationMessage):
    pass


class GPUCompilationInfo(classes.GPUCompilationInfo):
    pass


class GPUDeviceLostInfo(classes.GPUDeviceLostInfo):
    pass


class GPUError(classes.GPUError):
    pass


class GPUOutOfMemoryError(classes.GPUOutOfMemoryError, GPUError):
    pass


class GPUValidationError(classes.GPUValidationError, GPUError):
    pass


class GPUPipelineError(classes.GPUPipelineError):
    pass


class GPUInternalError(classes.GPUInternalError, GPUError):
    pass


# %%


def _copy_docstrings():
    base_classes = GPUObjectBase, GPUCanvasContext, GPUAdapter
    for ob in globals().values():
        if not (isinstance(ob, type) and issubclass(ob, base_classes)):
            continue
        elif ob.__module__ != __name__:
            continue  # no-cover
        base_cls = ob.mro()[1]
        ob.__doc__ = base_cls.__doc__
        for name, attr in ob.__dict__.items():
            if name.startswith("_") or not hasattr(attr, "__doc__"):
                continue  # no-cover
            base_attr = getattr(base_cls, name, None)
            if base_attr is not None:
                attr.__doc__ = base_attr.__doc__


_copy_docstrings()
