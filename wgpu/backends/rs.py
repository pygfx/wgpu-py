"""
WGPU backend implementation based on wgpu-native

The wgpu-native project (https://github.com/gfx-rs/wgpu) is a Rust
library based on gfx-hal, which wraps Metal, Vulkan, DX12 and more in
the future. It compiles to a dynamic library exposing a C-API,
accompanied by a C header file. We wrap this using cffi, which uses the
header file to do most type conversions for us.

Developer notes and tips:

* The purpose of this module is to tie our Pythonic API, which closely
  resembles the WebGPU spec, to the C API of wgpu-native.
* Most of it is converting dicts to ffi structs. You may think that
  this can be automated, and this would indeed be possible for 80-90%
  of the methods. However, the API's do not always line up, and there's
  async stuff to take into account too. Therefore we do it manually.
  In the end, I think that this will make the code easier to maintain.
* Use new_struct() and new_struct_p() to create a C structure with
  minimal boilerplate. It also converts string enum values to their
  corresponding integers.
* The codegen will validate and annotate all struct creations and
  function calls. So you could instantiate it without any fields and
  then run the codegen to fill it in.
* You may also need wgpu.h as a reference.
* To update to the latest wgpu.h, run codegen to validate all structs
  and function calls. Then check the diffs where changes are needed.
  See the codegen's readme for details.
"""


import os
import sys
import ctypes
import logging
import ctypes.util
from weakref import WeakKeyDictionary
from typing import List, Dict

from .. import base, flags, enums, structs
from .. import _register_backend
from .._coreutils import ApiDiff

from .rs_ffi import ffi, lib, check_expected_version
from .rs_mappings import cstructfield2enum, enummap, enum_str2int, enum_int2str
from .rs_helpers import (
    get_surface_id_from_canvas,
    get_memoryview_from_address,
    get_memoryview_and_address,
    to_snake_case,
    to_camel_case,
)


logger = logging.getLogger("wgpu")  # noqa
apidiff = ApiDiff()

# The wgpu-native version that we target/expect
__version__ = "0.11.0.1"
__commit_sha__ = "9d962ef667ef6006cca7bac7489d5bf303a2a244"
version_info = tuple(map(int, __version__.split(".")))
check_expected_version(version_info)  # produces a warning on mismatch


# %% Helper functions and objects


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


def _loadop_and_clear_from_value(value):
    """In WebGPU, the load op, can be given either as "load" or a value.
    The latter translates to "clear" plus that value in wgpu-native.
    The value can be float/int/color, but we don't deal with that here.
    """
    if isinstance(value, str):
        assert value == "load"
        return 1, 0  # WGPULoadOp_Load and a stub value
    else:
        return 0, value  # WGPULoadOp_Clear and the value


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
        val = int(2 ** i)
        if flag & val:
            features.append(feature_names.get(val, val))
    return tuple(sorted(features))


def check_struct(struct_name, d):
    """Check that all keys in the given dict exist in the corresponding struct."""
    valid_keys = set(getattr(structs, struct_name))
    invalid_keys = set(d.keys()).difference(valid_keys)
    if invalid_keys:
        raise ValueError(f"Invalid keys in {struct_name}: {invalid_keys}")


# %% The API


@_register_backend
class GPU(base.GPU):
    def request_adapter(self, *, canvas, power_preference=None):
        """Get a :class:`GPUAdapter`, the object that represents an abstract wgpu
        implementation, from which one can request a :class:`GPUDevice`.

        This is the implementation based on the Rust wgpu-native library.

        Arguments:
            canvas (WgpuCanvas): The canvas that the adapter should be able to
                render to (to create a swap chain for, to be precise). Can be None
                if you're not rendering to screen (or if you're confident that the
                returned adapter will work just fine).
            powerPreference(PowerPreference): "high-performance" or "low-power"
        """

        # Get surface id that the adapter must be compatible with. If we
        # don't pass a valid surface id, there is no guarantee we'll be
        # able to create a swapchain for it (from this adapter).
        surface_id = ffi.NULL
        if canvas is not None:
            window_id = canvas.get_window_id()
            if window_id is not None:  # e.g. could be an off-screen canvas
                surface_id = get_surface_id_from_canvas(canvas)

        # Try to read the WGPU_BACKEND_TYPE environment variable to see
        # if a backend should be forced. When you run into trouble with
        # the automatic selection of wgpu, you can use this variable
        # to force a specific backend. For instance, on Windows you
        # might want to force Vulkan, to avoid DX12 which seems to ignore
        # the NVidia control panel settings.
        # See https://github.com/gfx-rs/wgpu/issues/1416
        # todo: for the moment we default to forcing Vulkan on Windows
        force_backend = os.getenv("WGPU_BACKEND_TYPE", None)
        backend = enum_str2int["BackendType"]["Null"]
        if force_backend is None:  # Allow OUR defaults
            if sys.platform.startswith("win"):
                backend = enum_str2int["BackendType"]["Vulkan"]
        elif force_backend:
            try:
                backend = enum_str2int["BackendType"][force_backend]
            except KeyError:
                logger.warning(
                    f"Invalid value for WGPU_BACKEND_TYPE: '{force_backend}'.\n"
                    f"Valid values are: {list(enum_str2int['BackendType'].keys())}"
                )
            else:
                logger.warning(f"Forcing backend: {force_backend} ({backend})")

        # H: chain: WGPUChainedStruct, backend: WGPUBackendType
        extras = new_struct_p(
            "WGPUAdapterExtras *",
            backend=backend,
            # not used: chain
        )
        extras.chain.sType = lib.WGPUSType_AdapterExtras

        # Convert the descriptor
        # H: nextInChain: WGPUChainedStruct *, compatibleSurface: WGPUSurface, powerPreference: WGPUPowerPreference, forceFallbackAdapter: bool
        struct = new_struct_p(
            "WGPURequestAdapterOptions *",
            compatibleSurface=surface_id,
            nextInChain=ffi.cast("WGPUChainedStruct * ", extras),
            powerPreference=power_preference or "high-performance",
            forceFallbackAdapter=False,
        )

        # Do the API call and get the adapter id

        adapter_id = None

        @ffi.callback("void(WGPURequestAdapterStatus, WGPUAdapter, char *, void *)")
        def callback(status, result, message, userdata):
            assert status == 0, "Request adapter not successful"
            # message is not used yet in wgpu-native
            nonlocal adapter_id
            adapter_id = result

        # H: void f(WGPUInstance instance, WGPURequestAdapterOptions const * options, WGPURequestAdapterCallback callback, void * userdata)
        lib.wgpuInstanceRequestAdapter(
            ffi.NULL, struct, callback, ffi.NULL
        )  # userdata, stub

        # For now, Rust will call the callback immediately
        # todo: when wgpu gets an event loop -> while run wgpu event loop or something
        assert adapter_id is not None

        # Get info on the adapter
        # H: nextInChain: WGPUChainedStructOut *, vendorID: int, deviceID: int, name: char *, driverDescription: char *, adapterType: WGPUAdapterType, backendType: WGPUBackendType
        c_properties = new_struct_p(
            "WGPUAdapterProperties *",
            # not used: nextInChain
            # not used: deviceID
            # not used: vendorID
            # not used: name
            # not used: driverDescription
            # not used: adapterType
            # not used: backendType
        )

        # H: void f(WGPUAdapter adapter, WGPUAdapterProperties * properties)
        lib.wgpuAdapterGetProperties(adapter_id, c_properties)
        properties = {
            "name": "",
            "vendorID": c_properties.vendorID,
            "deviceID": c_properties.deviceID,
            "driverDescription": "",
            "adapterType": enum_int2str["AdapterType"].get(
                c_properties.adapterType, "unknown"
            ),
            "backendType": enum_int2str["BackendType"].get(
                c_properties.backendType, "unknown"
            ),
        }
        if c_properties.name:
            properties["name"] = ffi.string(c_properties.name).decode(errors="ignore")

        # Get limits
        # H: nextInChain: WGPUChainedStructOut *, limits: WGPULimits
        c_supported_limits = new_struct_p(
            "WGPUSupportedLimits *",
            # not used: nextInChain
            # not used: limits
        )
        c_limits = c_supported_limits.limits
        # H: bool f(WGPUAdapter adapter, WGPUSupportedLimits * limits)
        lib.wgpuAdapterGetLimits(adapter_id, c_supported_limits)
        limits = {to_snake_case(key): getattr(c_limits, key) for key in dir(c_limits)}
        limits = {key: val for key, val in limits.items() if val}  # filter zeros

        # Features are (temporarily? not supported)
        # c_features_flag = xx.wgpu_adapter_features(adapter_id)  # noqa
        # features = feature_flag_to_feature_names(c_features_flag)
        features = ()

        return GPUAdapter("WGPU", adapter_id, features, limits, properties)

    async def request_adapter_async(self, *, canvas, power_preference=None):
        """Async version of ``request_adapter()``.
        This function uses the Rust WGPU library.
        """
        return self.request_adapter(
            canvas=canvas, power_preference=power_preference
        )  # no-cover


class GPUCanvasContext(base.GPUCanvasContext):
    def __init__(self, canvas):
        super().__init__(canvas)
        self._surface_size = (-1, -1)
        self._surface_id = None
        self._internal = None
        self._current_texture = None

    def get_current_texture(self):
        if self._device is None:
            raise RuntimeError(
                "Preset context must be configured before get_current_texture()."
            )
        if self._current_texture is None:
            self._create_native_swap_chain_if_needed()
            # H: WGPUTextureView f(WGPUSwapChain swapChain)
            view_id = lib.wgpuSwapChainGetCurrentTextureView(self._internal)
            size = self._surface_size[0], self._surface_size[1], 1
            self._current_texture = GPUTextureView(
                "swap_chain", view_id, self._device, None, size
            )
        return self._current_texture

    def present(self):
        if self._internal is not None and lib is not None:
            # H: void f(WGPUSwapChain swapChain)
            lib.wgpuSwapChainPresent(self._internal)
        # Reset - always ask for a fresh texture (exactly once) on each draw
        self._current_texture = None

    def _create_native_swap_chain_if_needed(self):
        canvas = self._get_canvas()
        psize = canvas.get_physical_size()
        if psize == self._surface_size:
            return
        self._surface_size = psize

        # logger.info(str((psize, canvas.get_logical_size(), canvas.get_pixel_ratio())))

        # H: nextInChain: WGPUChainedStruct *, label: char *, usage: WGPUTextureUsageFlags/int, format: WGPUTextureFormat, width: int, height: int, presentMode: WGPUPresentMode
        struct = new_struct_p(
            "WGPUSwapChainDescriptor *",
            usage=self._usage,
            format=self._format,
            width=max(1, psize[0]),
            height=max(1, psize[1]),
            presentMode=2,
            # not used: nextInChain
            # not used: label
        )
        # present_mode -> 0: Immediate, 1: Mailbox, 2: Fifo

        if self._surface_id is None:
            self._surface_id = get_surface_id_from_canvas(canvas)

        # H: WGPUSwapChain f(WGPUDevice device, WGPUSurface surface, WGPUSwapChainDescriptor const * descriptor)
        self._internal = lib.wgpuDeviceCreateSwapChain(
            self._device._internal, self._surface_id, struct
        )


class GPUObjectBase(base.GPUObjectBase):
    pass


class GPUAdapter(base.GPUAdapter):
    def request_device(
        self,
        *,
        label="",
        required_features: "List[enums.FeatureName]" = [],
        required_limits: "Dict[str, int]" = {},
    ):
        return self._request_device(label, required_features, required_limits, "")

    @apidiff.add("a sweet bonus feature from wgpu-native")
    def request_device_tracing(
        self,
        trace_path,
        *,
        label="",
        required_features: "list(enums.FeatureName)" = [],
        required_limits: "Dict[str, int]" = {},
    ):
        """Write a trace of all commands to a file so it can be reproduced
        elsewhere. The trace is cross-platform!
        """
        if not os.path.isdir(trace_path):
            os.makedirs(trace_path, exist_ok=True)
        elif os.listdir(trace_path):
            logger.warning(f"Trace directory not empty: {trace_path}")
        return self._request_device(
            label, required_features, required_limits, trace_path
        )

    def _request_device(self, label, features, required_limits, trace_path):
        c_trace_path = ffi.NULL
        if trace_path:  # no-cover
            c_trace_path = ffi.new("char []", trace_path.encode())

        # Vanilla WGPU does not support interpolating samplers for float32 textures,
        # which is sad for scientific data in particular. We can enable it
        # (on the hardware were wgpu-py likely runs) using the feature:
        # WGPUNativeFeature_TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES

        # H: chain: WGPUChainedStruct, nativeFeatures: WGPUNativeFeature, label: char*, tracePath: char*
        extras = new_struct_p(
            "WGPUDeviceExtras *",
            tracePath=c_trace_path,
            nativeFeatures=lib.WGPUNativeFeature_TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            # not used: chain
            # not used: label
        )
        extras.chain.sType = lib.WGPUSType_DeviceExtras

        # Set limits
        # H: nextInChain: WGPUChainedStruct *, limits: WGPULimits
        c_required_limits = new_struct_p(
            "WGPURequiredLimits *",
            # not used: nextInChain
            # not used: limits
        )
        c_limits = c_required_limits.limits
        required_limits = required_limits or {}
        for key, val in required_limits.items():
            setattr(c_limits, to_camel_case(key), val)

        # H: nextInChain: WGPUChainedStruct *, label: char *, requiredFeaturesCount: int, requiredFeatures: WGPUFeatureName *, requiredLimits: WGPURequiredLimits *
        struct = new_struct_p(
            "WGPUDeviceDescriptor *",
            label=to_c_label(label),
            nextInChain=ffi.cast("WGPUChainedStruct * ", extras),
            requiredFeaturesCount=0,
            requiredFeatures=ffi.new("WGPUFeatureName []", []),
            requiredLimits=c_required_limits,
        )

        device_id = None

        @ffi.callback("void(WGPURequestDeviceStatus, WGPUDevice, char *, void *)")
        def callback(status, result, message, userdata):
            assert status == 0, "Request device not successful"
            # message is not used yet in wgpu-native
            nonlocal device_id
            device_id = result

        # H: void f(WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor, WGPURequestDeviceCallback callback, void * userdata)
        lib.wgpuAdapterRequestDevice(self._internal, struct, callback, ffi.NULL)

        assert device_id is not None

        # Get the actual limits reported by the device
        # H: nextInChain: WGPUChainedStructOut *, limits: WGPULimits
        c_supported_limits = new_struct_p(
            "WGPUSupportedLimits *",
            # not used: nextInChain
            # not used: limits
        )
        c_limits = c_supported_limits.limits
        # H: bool f(WGPUDevice device, WGPUSupportedLimits * limits)
        lib.wgpuDeviceGetLimits(device_id, c_supported_limits)
        limits = {to_snake_case(key): getattr(c_limits, key) for key in dir(c_limits)}
        limits = {key: val for key, val in limits.items() if val}  # filter zeros

        # Get actual features reported by the device
        # c_features_flag = xx.wgpu_device_features(device_id)
        # features = feature_flag_to_feature_names(c_features_flag)
        features = ()

        # Get the queue to which commands can be submitted
        # H: WGPUQueue f(WGPUDevice device)
        queue_id = lib.wgpuDeviceGetQueue(device_id)
        queue = GPUQueue("", queue_id, None)

        return GPUDevice(label, device_id, self, features, limits, queue)

    async def request_device_async(
        self,
        *,
        label="",
        required_features: "List[enums.FeatureName]" = [],
        required_limits: "Dict[str, int]" = {},
    ):
        return self._request_device(
            label, required_features, required_limits, ""
        )  # no-cover

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            internal  # doesnotexist.wgpuAdapterDrop(internal)


class GPUDevice(base.GPUDevice, GPUObjectBase):
    def create_buffer(
        self,
        *,
        label="",
        size: int,
        usage: "flags.BufferUsage",
        mapped_at_creation: bool = False,
    ):
        size = int(size)
        if mapped_at_creation:
            raise ValueError(
                "In wgpu-py, mapped_at_creation must be False. Use create_buffer_with_data() instead."
            )
        return self._create_buffer(label, size, usage, False)

    def create_buffer_with_data(self, *, label="", data, usage: "flags.BufferUsage"):

        # Get a memoryview of the data
        m, src_address = get_memoryview_and_address(data)
        m = m.cast("B", shape=(m.nbytes,))
        size = m.nbytes

        # Create the buffer (usage does not have to be MAP_READ or MAP_WRITE)
        buffer = self._create_buffer(label, size, usage, True)

        # Copy the data to the mapped memory
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        dst_ptr = lib.wgpuBufferGetMappedRange(buffer._internal, 0, size)
        dst_address = int(ffi.cast("intptr_t", dst_ptr))
        dst_m = get_memoryview_from_address(dst_address, size)
        dst_m[:] = m  # nicer than ctypes.memmove(dst_address, src_address, m.nbytes)

        buffer._unmap()
        return buffer

    def _create_buffer(self, label, size, usage, mapped_at_creation):

        # Create a buffer object
        # H: nextInChain: WGPUChainedStruct *, label: char *, usage: WGPUBufferUsageFlags/int, size: int, mappedAtCreation: bool
        struct = new_struct_p(
            "WGPUBufferDescriptor *",
            label=to_c_label(label),
            size=size,
            usage=usage,
            mappedAtCreation=mapped_at_creation,
            # not used: nextInChain
        )
        # H: WGPUBuffer f(WGPUDevice device, WGPUBufferDescriptor const * descriptor)
        id = lib.wgpuDeviceCreateBuffer(self._internal, struct)
        # Return wrapped buffer
        return GPUBuffer(label, id, self, size, usage)

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
        # H: nextInChain: WGPUChainedStruct *, label: char *, usage: WGPUTextureUsageFlags/int, dimension: WGPUTextureDimension, size: WGPUExtent3D, format: WGPUTextureFormat, mipLevelCount: int, sampleCount: int
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
        )
        # H: WGPUTexture f(WGPUDevice device, WGPUTextureDescriptor const * descriptor)
        id = lib.wgpuDeviceCreateTexture(self._internal, struct)

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
        mipmap_filter: "enums.FilterMode" = "nearest",
        lod_min_clamp: float = 0,
        lod_max_clamp: float = 32,
        compare: "enums.CompareFunction" = None,
        max_anisotropy: int = 1,
    ):
        # H: nextInChain: WGPUChainedStruct *, label: char *, addressModeU: WGPUAddressMode, addressModeV: WGPUAddressMode, addressModeW: WGPUAddressMode, magFilter: WGPUFilterMode, minFilter: WGPUFilterMode, mipmapFilter: WGPUFilterMode, lodMinClamp: float, lodMaxClamp: float, compare: WGPUCompareFunction, maxAnisotropy: int
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
        id = lib.wgpuDeviceCreateSampler(self._internal, struct)
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
                # H: nextInChain: WGPUChainedStruct *, type: WGPUBufferBindingType, hasDynamicOffset: bool, minBindingSize: int
                buffer = new_struct(
                    "WGPUBufferBindingLayout",
                    type=info["type"],
                    hasDynamicOffset=info.get("has_dynamic_offset", False),
                    minBindingSize=info.get("min_binding_size", 0),
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
                # H: nextInChain: WGPUChainedStruct *, sampleType: WGPUTextureSampleType, viewDimension: WGPUTextureViewDimension, multisampled: bool
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
            # H: nextInChain: WGPUChainedStruct *, binding: int, visibility: WGPUShaderStageFlags/int, buffer: WGPUBufferBindingLayout, sampler: WGPUSamplerBindingLayout, texture: WGPUTextureBindingLayout, storageTexture: WGPUStorageTextureBindingLayout
            c_entry = new_struct(
                "WGPUBindGroupLayoutEntry",
                binding=int(entry["binding"]),
                visibility=int(entry["visibility"]),
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

        # H: WGPUBindGroupLayout f(WGPUDevice device, WGPUBindGroupLayoutDescriptor const * descriptor)
        id = lib.wgpuDeviceCreateBindGroupLayout(self._internal, struct)

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
        id = lib.wgpuDeviceCreateBindGroup(self._internal, struct)
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
        id = lib.wgpuDeviceCreatePipelineLayout(self._internal, struct)
        return GPUPipelineLayout(label, id, self, bind_group_layouts)

    def create_shader_module(
        self,
        *,
        label="",
        code: str,
        source_map: dict = None,
        hints: "Dict[str, structs.ShaderModuleCompilationHint]" = None,
    ):
        if hints:
            for val in hints.values():
                check_struct("ShaderModuleCompilationHint", val)
        if isinstance(code, str):
            # WGSL
            # H: chain: WGPUChainedStruct, code: char *
            source_struct = new_struct_p(
                "WGPUShaderModuleWGSLDescriptor *",
                code=ffi.new("char []", code.encode()),
                # not used: chain
            )
            source_struct[0].chain.next = ffi.NULL
            source_struct[0].chain.sType = lib.WGPUSType_ShaderModuleWGSLDescriptor
        else:
            # Must be Spirv then
            if isinstance(code, bytes):
                data = code
            elif hasattr(code, "to_bytes"):
                data = code.to_bytes()
            elif hasattr(code, "to_spirv"):
                data = code.to_spirv()
            else:
                raise TypeError("Shader code must be str for WGSL, or bytes for SpirV.")
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

        # H: nextInChain: WGPUChainedStruct *, label: char *
        struct = new_struct_p(
            "WGPUShaderModuleDescriptor *",
            label=to_c_label(label),
            nextInChain=ffi.cast("WGPUChainedStruct *", source_struct),
            # not used: nextInChain
        )

        # H: WGPUShaderModule f(WGPUDevice device, WGPUShaderModuleDescriptor const * descriptor)
        id = lib.wgpuDeviceCreateShaderModule(self._internal, struct)
        return GPUShaderModule(label, id, self)

    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
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

        # H: nextInChain: WGPUChainedStruct *, label: char *, layout: WGPUPipelineLayout, compute: WGPUProgrammableStageDescriptor
        struct = new_struct_p(
            "WGPUComputePipelineDescriptor *",
            label=to_c_label(label),
            layout=layout._internal,
            compute=c_compute_stage,
            # not used: nextInChain
            # not used: compute
        )

        # H: WGPUComputePipeline f(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor)
        id = lib.wgpuDeviceCreateComputePipeline(self._internal, struct)
        return GPUComputePipeline(label, id, self, layout)

    async def create_compute_pipeline_async(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        compute: "structs.ProgrammableStage",
    ):
        return self.create_compute_pipeline(label=label, layout=layout, compute=compute)

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
            assert (
                depth_stencil.get("format", None) is not None
            ), "depth_stencil needs format"
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
            check_struct("StencilFaceState", stencil_front)
            # H: compare: WGPUCompareFunction, failOp: WGPUStencilOperation, depthFailOp: WGPUStencilOperation, passOp: WGPUStencilOperation
            c_stencil_back = new_struct(
                "WGPUStencilFaceState",
                compare=stencil_back.get("compare", "always"),
                failOp=stencil_back.get("fail_op", "keep"),
                depthFailOp=stencil_back.get("depth_fail_op", "keep"),
                passOp=stencil_back.get("pass_op", "keep"),
            )
            # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, depthWriteEnabled: bool, depthCompare: WGPUCompareFunction, stencilFront: WGPUStencilFaceState, stencilBack: WGPUStencilFaceState, stencilReadMask: int, stencilWriteMask: int, depthBias: int, depthBiasSlopeScale: float, depthBiasClamp: float
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

        # H: nextInChain: WGPUChainedStruct *, count: int, mask: int, alphaToCoverageEnabled: bool
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
        id = lib.wgpuDeviceCreateRenderPipeline(self._internal, struct)
        return GPURenderPipeline(label, id, self, layout)

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
        id = lib.wgpuDeviceCreateCommandEncoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

    # FIXME: new method to implement
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

    # FIXME: new method to implement
    def create_query_set(self, *, label="", type: "enums.QueryType", count: int):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUDevice device)
            internal  # lib.wgpuDeviceDrop(internal)  # Causes a hang


class GPUBuffer(base.GPUBuffer, GPUObjectBase):
    def map_read(self):
        size = self.size

        # Prepare
        status = 99
        data = memoryview((ctypes.c_uint8 * size)()).cast("B")

        @ffi.callback("void(WGPUBufferMapAsyncStatus, void*)")
        def callback(status_, user_data_p):
            nonlocal status
            status = status_

        # Map it
        # H: void f(WGPUBuffer buffer, WGPUMapModeFlags mode, size_t offset, size_t size, WGPUBufferMapCallback callback, void * userdata)
        lib.wgpuBufferMapAsync(
            self._internal, lib.WGPUMapMode_Read, 0, size, callback, ffi.NULL
        )

        # Let it do some cycles
        # H: void f(WGPUInstance instance)
        # lib.wgpuInstanceProcessEvents(ffi.NULL)
        # H: void f(WGPUDevice device, bool force_wait)
        lib.wgpuDevicePoll(self._device._internal, True)

        if status != 0:  # no-cover
            raise RuntimeError(f"Could not read buffer data ({status}).")

        # Copy data
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        src_ptr = lib.wgpuBufferGetMappedRange(self._internal, 0, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, size)
        data[:] = src_m

        self._unmap()
        return data

    def map_write(self, data):
        size = self.size

        data = memoryview(data).cast("B")
        assert data.nbytes == self.size

        # Prepare
        status = 99

        @ffi.callback("void(WGPUBufferMapAsyncStatus, void*)")
        def callback(status_, user_data_p):
            nonlocal status
            status = status_

        # Map it
        # H: void f(WGPUBuffer buffer, WGPUMapModeFlags mode, size_t offset, size_t size, WGPUBufferMapCallback callback, void * userdata)
        lib.wgpuBufferMapAsync(
            self._internal, lib.WGPUMapMode_Write, 0, size, callback, ffi.NULL
        )

        # Let it do some cycles
        # H: void f(WGPUInstance instance)
        # lib.wgpuInstanceProcessEvents(ffi.NULL)
        # H: void f(WGPUDevice device, bool force_wait)
        lib.wgpuDevicePoll(self._device._internal, True)

        if status != 0:  # no-cover
            raise RuntimeError(f"Could not read buffer data ({status}).")

        # Copy data
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        src_ptr = lib.wgpuBufferGetMappedRange(self._internal, 0, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, size)
        src_m[:] = data

        self._unmap()

    def _unmap(self):
        # H: void f(WGPUBuffer buffer)
        lib.wgpuBufferUnmap(self._internal)

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBuffer buffer)
            lib.wgpuBufferDrop(internal)


class GPUTexture(base.GPUTexture, GPUObjectBase):
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
                array_layer_count = 1
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
        id = lib.wgpuTextureCreateView(self._internal, struct)
        return GPUTextureView(label, id, self._device, self, self.size)

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTexture texture)
            lib.wgpuTextureDrop(internal)


class GPUTextureView(base.GPUTextureView, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTextureView textureView)
            lib.wgpuTextureViewDrop(internal)


class GPUSampler(base.GPUSampler, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUSampler sampler)
            lib.wgpuSamplerDrop(internal)


class GPUBindGroupLayout(base.GPUBindGroupLayout, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroupLayout bindGroupLayout)
            lib.wgpuBindGroupLayoutDrop(internal)


class GPUBindGroup(base.GPUBindGroup, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroup bindGroup)
            lib.wgpuBindGroupDrop(internal)


class GPUPipelineLayout(base.GPUPipelineLayout, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUPipelineLayout pipelineLayout)
            lib.wgpuPipelineLayoutDrop(internal)


class GPUShaderModule(base.GPUShaderModule, GPUObjectBase):
    def compilation_info(self):
        return super().compilation_info()

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUShaderModule shaderModule)
            lib.wgpuShaderModuleDrop(internal)


class GPUPipelineBase(base.GPUPipelineBase):
    pass


class GPUComputePipeline(base.GPUComputePipeline, GPUPipelineBase, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUComputePipeline computePipeline)
            lib.wgpuComputePipelineDrop(internal)


class GPURenderPipeline(base.GPURenderPipeline, GPUPipelineBase, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderPipeline renderPipeline)
            lib.wgpuRenderPipelineDrop(internal)


class GPUCommandBuffer(base.GPUCommandBuffer, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUCommandBuffer commandBuffer)
            internal  # lib.wgpuCommandBufferDrop(internal)  # Causes 'Cannot remove a vacant resource'


class GPUCommandEncoder(base.GPUCommandEncoder, GPUObjectBase):
    def begin_compute_pass(
        self,
        *,
        label="",
        timestamp_writes: "List[structs.ComputePassTimestampWrite]" = [],
    ):
        for val in timestamp_writes:
            check_struct("ComputePassTimestampWrite", val)
        # H: nextInChain: WGPUChainedStruct *, label: char *, timestampWriteCount: int, timestampWrites: WGPUComputePassTimestampWrite *
        struct = new_struct_p(
            "WGPUComputePassDescriptor *",
            label=to_c_label(label),
            # not used: nextInChain
            # not used: timestampWriteCount
            # not used: timestampWrites
        )
        # H: WGPUComputePassEncoder f(WGPUCommandEncoder commandEncoder, WGPUComputePassDescriptor const * descriptor)
        raw_pass = lib.wgpuCommandEncoderBeginComputePass(self._internal, struct)
        return GPUComputePassEncoder(label, raw_pass, self)

    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "List[structs.RenderPassColorAttachment]",
        depth_stencil_attachment: "structs.RenderPassDepthStencilAttachment" = None,
        occlusion_query_set: "GPUQuerySet" = None,
        timestamp_writes: "List[structs.RenderPassTimestampWrite]" = [],
    ):
        # Note that occlusion_query_set is ignored because wgpu-native does not have it.
        for val in timestamp_writes:
            check_struct("RenderPassTimestampWrite", val)

        c_color_attachments_list = []
        for color_attachment in color_attachments:
            check_struct("RenderPassColorAttachment", color_attachment)
            assert isinstance(color_attachment["view"], GPUTextureView)
            texture_view_id = color_attachment["view"]._internal
            c_resolve_target = (
                ffi.NULL
                if color_attachment.get("resolve_target", None) is None
                else color_attachment["resolve_target"]._internal
            )  # this is a TextureViewId or null
            c_load_op, clear_color = _loadop_and_clear_from_value(
                color_attachment["load_value"]
            )
            clr = (
                (0.0, 0.0, 0.0, 0.0)
                if clear_color == 0
                else _tuple_from_tuple_or_dict(clear_color, "rgba")
            )
            # H: r: float, g: float, b: float, a: float
            c_clear_color = new_struct(
                "WGPUColor",
                r=clr[0],
                g=clr[1],
                b=clr[2],
                a=clr[3],
            )
            # H: view: WGPUTextureView, resolveTarget: WGPUTextureView, loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearColor: WGPUColor
            c_attachment = new_struct(
                "WGPURenderPassColorAttachment",
                view=texture_view_id,
                resolveTarget=c_resolve_target,
                loadOp=c_load_op,
                storeOp=color_attachment.get("store_op", "store"),
                clearColor=c_clear_color,
                # not used: resolveTarget
            )
            c_color_attachments_list.append(c_attachment)
        c_color_attachments_array = ffi.new(
            "WGPURenderPassColorAttachment []", c_color_attachments_list
        )

        c_depth_stencil_attachment = ffi.NULL
        if depth_stencil_attachment is not None:
            check_struct("RenderPassDepthStencilAttachment", depth_stencil_attachment)
            c_depth_load_op, c_depth_clear = _loadop_and_clear_from_value(
                depth_stencil_attachment["depth_load_value"]
            )
            c_stencil_load_op, c_stencil_clear = _loadop_and_clear_from_value(
                depth_stencil_attachment["stencil_load_value"]
            )
            # H: view: WGPUTextureView, depthLoadOp: WGPULoadOp, depthStoreOp: WGPUStoreOp, clearDepth: float, depthReadOnly: bool, stencilLoadOp: WGPULoadOp, stencilStoreOp: WGPUStoreOp, clearStencil: int, stencilReadOnly: bool
            c_depth_stencil_attachment = new_struct_p(
                "WGPURenderPassDepthStencilAttachment *",
                view=depth_stencil_attachment["view"]._internal,
                depthLoadOp=c_depth_load_op,
                depthStoreOp=depth_stencil_attachment["depth_store_op"],
                clearDepth=float(c_depth_clear),
                depthReadOnly=depth_stencil_attachment.get("depth_read_only", False),
                stencilLoadOp=c_stencil_load_op,
                stencilStoreOp=depth_stencil_attachment["stencil_store_op"],
                clearStencil=int(c_stencil_clear),
                stencilReadOnly=depth_stencil_attachment.get(
                    "stencil_read_only", False
                ),
            )

        # H: nextInChain: WGPUChainedStruct *, label: char *, colorAttachmentCount: int, colorAttachments: WGPURenderPassColorAttachment *, depthStencilAttachment: WGPURenderPassDepthStencilAttachment *, occlusionQuerySet: WGPUQuerySet, timestampWriteCount: int, timestampWrites: WGPURenderPassTimestampWrite *
        struct = new_struct_p(
            "WGPURenderPassDescriptor *",
            label=to_c_label(label),
            colorAttachments=c_color_attachments_array,
            colorAttachmentCount=len(c_color_attachments_list),
            depthStencilAttachment=c_depth_stencil_attachment,
            # not used: occlusionQuerySet
            # not used: nextInChain
            # not used: timestampWriteCount
            # not used: timestampWrites
        )

        # H: WGPURenderPassEncoder f(WGPUCommandEncoder commandEncoder, WGPURenderPassDescriptor const * descriptor)
        raw_pass = lib.wgpuCommandEncoderBeginRenderPass(self._internal, struct)
        return GPURenderPassEncoder(label, raw_pass, self)

    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        assert source_offset % 4 == 0, "source_offsetmust be a multiple of 4"
        assert destination_offset % 4 == 0, "destination_offset must be a multiple of 4"
        assert size % 4 == 0, "size must be a multiple of 4"

        assert isinstance(source, GPUBuffer)
        assert isinstance(destination, GPUBuffer)
        # H: void f(WGPUCommandEncoder commandEncoder, WGPUBuffer source, uint64_t sourceOffset, WGPUBuffer destination, uint64_t destinationOffset, uint64_t size)
        lib.wgpuCommandEncoderCopyBufferToBuffer(
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

        c_source = new_struct_p(
            "WGPUImageCopyBuffer *",
            buffer=source["buffer"]._internal,
            # H: nextInChain: WGPUChainedStruct *, offset: int, bytesPerRow: int, rowsPerImage: int
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(source.get("offset", 0)),
                bytesPerRow=bytes_per_row,
                rowsPerImage=int(source.get("rows_per_image", 0)),
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
            aspect=0,  # all, stencilonly, depthonly, force32
            # not used: nextInChain
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

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUImageCopyBuffer const * source, WGPUImageCopyTexture const * destination, WGPUExtent3D const * copySize)
        lib.wgpuCommandEncoderCopyBufferToTexture(
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
                rowsPerImage=int(destination.get("rows_per_image", 0)),
                # not used: nextInChain
            ),
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

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUImageCopyTexture const * source, WGPUImageCopyBuffer const * destination, WGPUExtent3D const * copySize)
        lib.wgpuCommandEncoderCopyTextureToBuffer(
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
        lib.wgpuCommandEncoderCopyTextureToTexture(
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
        id = lib.wgpuCommandEncoderFinish(self._internal, struct)
        return GPUCommandBuffer(label, id, self)

    # FIXME: new method to implement
    def push_debug_group(self, group_label):
        raise NotImplementedError()

    # FIXME: new method to implement
    def pop_debug_group(self):
        raise NotImplementedError()

    # FIXME: new method to implement
    def insert_debug_marker(self, marker_label):
        raise NotImplementedError()

    # FIXME: new method to implement
    def write_timestamp(self, query_set, query_index):
        raise NotImplementedError()

    # FIXME: new method to implement
    def resolve_query_set(
        self, query_set, first_query, query_count, destination, destination_offset
    ):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUCommandEncoder commandEncoder)
            internal  # lib.wgpuCommandEncoderDrop(internal)  # Causes 'Cannot remove a vacant resource'

    # FIXME: new method to implement
    def clear_buffer(self, buffer, offset=0, size=None):
        raise NotImplementedError()


class GPUProgrammablePassEncoder(base.GPUProgrammablePassEncoder):
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
            # H: void f(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, uint32_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
            lib.wgpuComputePassEncoderSetBindGroup(
                self._internal, index, bind_group_id, len(offsets), c_offsets
            )
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t groupIndex, WGPUBindGroup group, uint32_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
            lib.wgpuRenderPassEncoderSetBindGroup(
                self._internal,
                index,
                bind_group_id,
                len(offsets),
                c_offsets,
            )

    def push_debug_group(self, group_label):
        c_group_label = ffi.new("char []", group_label.encode())
        color = 0
        # todo: these functions are temporarily not available in wgpu-native
        return  # noqa
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder, char const * groupLabel)
            lib.wgpuComputePassEncoderPushDebugGroup(
                self._internal, c_group_label, color
            )
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder, char const * groupLabel)
            lib.wgpuRenderPassEncoderPushDebugGroup(
                self._internal, c_group_label, color
            )

    def pop_debug_group(self):
        # todo: these functions are temporarily not available in wgpu-native
        return  # noqa
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder)
            lib.wgpuComputePassEncoderPopDebugGroup(self._internal)
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder)
            lib.wgpuRenderPassEncoderPopDebugGroup(self._internal)

    def insert_debug_marker(self, marker_label):
        c_marker_label = ffi.new("char []", marker_label.encode())
        color = 0
        # todo: these functions are temporarily not available in wgpu-native
        return  # noqa
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPUComputePassEncoder computePassEncoder, char const * markerLabel)
            lib.wgpuComputePassEncoderInsertDebugMarker(
                self._internal, c_marker_label, color
            )
        else:
            # H: void f(WGPURenderPassEncoder renderPassEncoder, char const * markerLabel)
            lib.wgpuRenderPassEncoderInsertDebugMarker(
                self._internal, c_marker_label, color
            )


class GPUComputePassEncoder(
    base.GPUComputePassEncoder, GPUProgrammablePassEncoder, GPUObjectBase
):
    """ """

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline)
        lib.wgpuComputePassEncoderSetPipeline(self._internal, pipeline_id)

    def dispatch(self, x, y=1, z=1):
        # H: void f(WGPUComputePassEncoder computePassEncoder, uint32_t x, uint32_t y, uint32_t z)
        lib.wgpuComputePassEncoderDispatch(self._internal, x, y, z)

    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        lib.wgpuComputePassEncoderDispatchIndirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def end_pass(self):
        # H: void f(WGPUComputePassEncoder computePassEncoder)
        lib.wgpuComputePassEncoderEndPass(self._internal)

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            internal  # doesnotexist.wgpuComputePassDrop(internal)


class GPURenderEncoderBase(base.GPURenderEncoderBase):
    """ """

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline)
        lib.wgpuRenderPassEncoderSetPipeline(self._internal, pipeline_id)

    def set_index_buffer(self, buffer, index_format, offset=0, size=None):
        if not size:
            size = buffer.size - offset
        c_index_format = enummap[f"IndexFormat.{index_format}"]
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
        lib.wgpuRenderPassEncoderSetIndexBuffer(
            self._internal, buffer._internal, c_index_format, int(offset), int(size)
        )

    def set_vertex_buffer(self, slot, buffer, offset=0, size=None):
        if not size:
            size = buffer.size - offset
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size)
        lib.wgpuRenderPassEncoderSetVertexBuffer(
            self._internal, int(slot), buffer._internal, int(offset), int(size)
        )

    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
        lib.wgpuRenderPassEncoderDraw(
            self._internal, vertex_count, instance_count, first_vertex, first_instance
        )

    def draw_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        lib.wgpuRenderPassEncoderDrawIndirect(
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
        lib.wgpuRenderPassEncoderDrawIndexed(
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
        lib.wgpuRenderPassEncoderDrawIndexedIndirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            internal  # doesnotexist.wgpuRenderPassDrop(internal)


class GPURenderPassEncoder(
    base.GPURenderPassEncoder,
    GPUProgrammablePassEncoder,
    GPURenderEncoderBase,
    GPUObjectBase,
):
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, float x, float y, float width, float height, float minDepth, float maxDepth)
        lib.wgpuRenderPassEncoderSetViewport(
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
        lib.wgpuRenderPassEncoderSetScissorRect(
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
        lib.wgpuRenderPassEncoderSetBlendConstant(self._internal, c_color)

    def set_stencil_reference(self, reference):
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t reference)
        lib.wgpuRenderPassEncoderSetStencilReference(self._internal, int(reference))

    def end_pass(self):
        # H: void f(WGPURenderPassEncoder renderPassEncoder)
        lib.wgpuRenderPassEncoderEndPass(self._internal)

    # FIXME: new method to implement
    def execute_bundles(self, bundles):
        raise NotImplementedError()

    # FIXME: new method to implement
    def begin_occlusion_query(self, query_index):
        raise NotImplementedError()

    # FIXME: new method to implement
    def end_occlusion_query(self):
        raise NotImplementedError()


class GPURenderBundleEncoder(
    base.GPURenderBundleEncoder,
    GPUProgrammablePassEncoder,
    GPURenderEncoderBase,
    GPUObjectBase,
):

    # FIXME: new method to implement
    def finish(self, *, label=""):
        raise NotImplementedError()


class GPUQueue(base.GPUQueue, GPUObjectBase):
    def submit(self, command_buffers):
        command_buffer_ids = [cb._internal for cb in command_buffers]
        c_command_buffers = ffi.new("WGPUCommandBuffer []", command_buffer_ids)
        # H: void f(WGPUQueue queue, uint32_t commandCount, WGPUCommandBuffer const * commands)
        lib.wgpuQueueSubmit(self._internal, len(command_buffer_ids), c_command_buffers)

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

        assert 0 <= buffer_offset < buffer.size
        assert 0 <= data_offset < nbytes
        assert 0 <= data_length <= (nbytes - data_offset)
        assert data_length <= buffer.size - buffer_offset

        # Make the call. Note that this call copies the data - it's ok
        # if we lose our reference to the data once we leave this function.
        c_data = ffi.cast("uint8_t *", address + data_offset)
        # H: void f(WGPUQueue queue, WGPUBuffer buffer, uint64_t bufferOffset, void const * data, size_t size)
        lib.wgpuQueueWriteBuffer(
            self._internal, buffer._internal, buffer_offset, c_data, data_length
        )

    def read_buffer(self, buffer, buffer_offset=0, size=None):

        # Note that write_buffer probably does a very similar thing
        # using a temporaty buffer. But write_buffer is official API
        # so it's a single call, while here we must create the temporary
        # buffer and do the copying ourselves.

        if not size:
            data_length = buffer.size - buffer_offset
        else:
            data_length = int(size)
        assert 0 <= buffer_offset < buffer.size
        assert data_length <= buffer.size - buffer_offset

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
        data = tmp_buffer.map_read()
        tmp_buffer.destroy()

        return data

    def write_texture(self, destination, data, data_layout, size):

        # Note that the bytes_per_row restriction does not apply for
        # this function; wgpu-native deals with it.

        if isinstance(destination["texture"], GPUTextureView):
            raise ValueError("copy destination texture must be a texture, not a view")

        m, address = get_memoryview_and_address(data)
        # todo: could we not derive the size from the shape of m?

        c_data = ffi.cast("uint8_t *", address)
        data_length = m.nbytes

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
            # not used: nextInChain
            # not used: aspect
        )

        # H: nextInChain: WGPUChainedStruct *, offset: int, bytesPerRow: int, rowsPerImage: int
        c_data_layout = new_struct_p(
            "WGPUTextureDataLayout *",
            offset=data_layout.get("offset", 0),
            bytesPerRow=data_layout["bytes_per_row"],
            rowsPerImage=data_layout.get("rows_per_image", 0),
            # not used: nextInChain
        )

        size = _tuple_from_tuple_or_dict(
            size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depthOrArrayLayers: int
        c_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUQueue queue, WGPUImageCopyTexture const * destination, void const * data, size_t dataSize, WGPUTextureDataLayout const * dataLayout, WGPUExtent3D const * writeSize)
        lib.wgpuQueueWriteTexture(
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
            "bytes_per_row": full_stride,
            "rows_per_image": data_layout.get("rows_per_image", 0),
        }

        # Copy data to temp buffer
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(source, destination, size)
        command_buffer = encoder.finish()
        self.submit([command_buffer])

        # Download from mappable buffer
        data = tmp_buffer.map_read()
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

    # FIXME: new method to implement
    def on_submitted_work_done(self):
        raise NotImplementedError()


class GPURenderBundle(base.GPURenderBundle, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderBundle renderBundle)
            lib.wgpuRenderBundleDrop(internal)


class GPUDeviceLostInfo(base.GPUDeviceLostInfo):
    pass


class GPUOutOfMemoryError(base.GPUOutOfMemoryError, Exception):
    pass


class GPUValidationError(base.GPUValidationError, Exception):
    pass


class GPUCompilationMessage(base.GPUCompilationMessage):
    pass


class GPUCompilationInfo(base.GPUCompilationInfo):
    pass


class GPUQuerySet(base.GPUQuerySet, GPUObjectBase):
    pass

    def destroy(self):
        if self._internal is not None and lib is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUQuerySet querySet)
            lib.wgpuQuerySetDrop(internal)


class GPUUncapturedErrorEvent(base.GPUUncapturedErrorEvent):
    pass


class GPUExternalTexture(base.GPUExternalTexture, GPUObjectBase):
    pass


# %%


def _copy_docstrings():
    base_classes = GPUObjectBase, GPUCanvasContext
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
