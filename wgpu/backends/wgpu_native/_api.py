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

# Allow using class names in type annotations, without Ruff triggering F821
from __future__ import annotations

import os
import time
import logging
from weakref import WeakKeyDictionary
from typing import NoReturn, Sequence

from ..._async import LoopInterface
from ..._coreutils import str_flag_to_int, ArrayLike, CanvasLike
from ... import classes, flags, enums, structs

from ._ffi import ffi, lib
from ._mappings import cstructfield2enum, enummap, enum_str2int, enum_int2str
from ._helpers import (
    get_wgpu_instance,
    get_surface_id_from_info,
    get_memoryview_from_address,
    get_memoryview_and_address,
    to_snake_case,
    ErrorHandler,
    SafeLibCalls,
)

logger = logging.getLogger("wgpu")


# The API is pretty well defined
__all__ = classes.__all__.copy()


# %% Helper functions and objects

# The 'optional' value is used as the default value for certain optional arguments, see the comment in _classes.py for details.
optional = None


# Object to be able to bind the lifetime of objects to other objects
_refs_per_struct = WeakKeyDictionary()

# Some enum keys need a shortcut
_cstructfield2enum_alt = {
    "load_op": "LoadOp",
    "store_op": "StoreOp",
    "depth_store_op": "StoreOp",
    "stencil_store_op": "StoreOp",
}


def print_struct(s, indent=""):
    """Tool to pretty-print struct contents during debugging."""
    for key in dir(s):
        val = getattr(s, key)
        if repr(val).startswith("<cdata "):
            if "NULL" in repr(val):
                print(indent + key + ": null")
            elif "'char *'" in repr(val):
                print(indent + key + ":", ffi.string(val).decode())
            elif "WGPUStringView" in repr(val):
                print(indent + key + ":", from_c_string_view(val))
            elif " *'" in repr(val):
                print(indent + key + ": pointer")
            elif "struct WGPU" in repr(val):
                print(indent + key + ":", repr(val))
            else:
                print(indent + key + ":")
                print(indent + "{")
                print_struct(val, indent + "  ")
                print(indent + "}")
        else:
            print(indent + key + ":", val)


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
    _refs_per_struct[struct] = tuple(kwargs.values())
    return struct


def _new_struct_p(ctype, **kwargs):
    struct_p = ffi.new(ctype)
    for key, val in kwargs.items():
        if val is None:
            pass  # None means not-given / null in C
        elif isinstance(val, str) and isinstance(getattr(struct_p, key), int):
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


def new_array(ctype, elements):
    assert ctype.endswith("[]")
    if isinstance(elements, int):
        # elements == count
        return ffi.new(ctype, elements)
    elif elements:
        array = ffi.new(ctype, elements)
        # The array is a contiguous copy of the element structs. We don't need
        # to keep a reference to the elements, but we do to sub-structs and
        # sub-arrays of these elements.
        _refs_per_struct[array] = [
            _refs_per_struct.get(el, None)
            for el in elements
            if isinstance(el, ffi.CData)
        ]
        return array
    else:
        return ffi.NULL


def _tuple_from_tuple_or_dict(ob, fields, defaults=()):
    """Given a tuple/list/dict, return a tuple. Also checks tuple size.

    >> # E.g.
    >> _tuple_from_tuple_or_dict({"x": 1, "y": 2}, ("x", "y"))
    (1, 2)
    >> _tuple_from_tuple_or_dict([1, 2], ("x", "y"))
    (1, 2)

    If defaults are given, it will be the default values.  If the length of defaults is
    shorter than the length of fields, it gives the default values for the final args.
    Other arguments are still required.

    >> _tuple_from_tuple_or_dict({"x": 1}, ("x", "y"), (2,))
    (1, 2)
    >> _tuple_from_tuple_or_dict([], ("x", "y"), (1, 2))
    (1, 2)

    """
    error_msg = "Expected tuple/key/dict with fields: {}"
    required = len(fields) - len(defaults)
    if isinstance(ob, (list, tuple)):
        fields_len = len(fields)
        ob_len = len(ob)
        if ob_len == fields_len:
            # Optimize for this fast case
            return tuple(ob)
        elif required <= ob_len < fields_len:
            defaults_needed = fields_len - ob_len
            return tuple((*ob, *defaults[-defaults_needed:]))
        else:
            raise ValueError(error_msg.format(", ".join(fields)))
    elif isinstance(ob, dict):
        if any(key not in fields for key in ob):
            raise ValueError("Unexpected key in {}".format(ob))
        try:
            return tuple(
                (
                    ob.get(key, defaults[index - required])
                    if index >= required
                    else ob[key]
                )
                for index, key in enumerate(fields)
            )
        except KeyError:
            raise ValueError(error_msg.format(", ".join(fields))) from None
    else:
        raise TypeError(error_msg.format(", ".join(fields)))


def _tuple_from_extent3d(size):
    return _tuple_from_tuple_or_dict(
        # required, 1, 1
        size,
        ("width", "height", "depth_or_array_layers"),
        (1, 1),
    )


def _tuple_from_origin3d(destination):
    fields = destination.get("origin", (0, 0, 0))
    # Each field individually is 0 if not specified
    return _tuple_from_tuple_or_dict(fields, "xyz", (0, 0, 0))


def _tuple_from_color(rgba):
    return _tuple_from_tuple_or_dict(rgba, "rgba")


def _get_override_constant_entries(field):
    constants = field.get("constants")
    if not constants:
        return ffi.NULL, []
    c_constant_entries = []
    for key, value in constants.items():
        assert isinstance(key, (str, int))
        assert isinstance(value, (int, float, bool))
        # H: nextInChain: WGPUChainedStruct *, key: WGPUStringView, value: float
        c_constant_entry = new_struct(
            "WGPUConstantEntry",
            # not used: nextInChain
            key=to_c_string_view(str(key)),
            value=float(value),
        )
        c_constant_entries.append(c_constant_entry)
    # We need to return and hold onto c_constant_entries in order to prevent the C
    # strings from being GC'ed.
    c_constants = new_array("WGPUConstantEntry[]", c_constant_entries)
    return c_constants, c_constant_entries


# H: data: char *, length: int
_null_string = new_struct(
    "WGPUStringView",
    data=ffi.NULL,
    length=lib.WGPU_STRLEN,
)
# H: data: char *, length: int
_empty_string = new_struct(
    "WGPUStringView",
    data=ffi.NULL,
    length=0,
)


def to_c_string_view(string: str):
    """Turn a string into a "WGPUStringView. None becomes the null-string."""
    if string is None:
        # The null-string. wgpu-core interprets this different from the empty sting,
        # e.g. when not-setting the trace path, it should be the null-string.
        return _null_string
    elif not string:
        # The empty string
        return _empty_string
    else:
        # A string with nonzero length
        data = ffi.new("char []", string.encode())  # includes null terminator!
        # length = len(data) - 1  # explicit length (minus null terminator)
        length = lib.WGPU_STRLEN  # Zero-terminated string
        # H: data: char *, length: int
        return new_struct(
            "WGPUStringView",
            data=data,
            length=length,
        )


def from_c_string_view(struct):
    if not struct or struct.data == ffi.NULL or struct.length == 0:
        return ""
    elif struct.length == lib.WGPU_STRLEN:
        # null-terminated
        return ffi.string(struct.data).decode(errors="ignore")
    else:
        # explicit length
        return ffi.string(struct.data, struct.length).decode(errors="ignore")


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
    struct_class = getattr(structs, struct_name)
    if isinstance(d, struct_class):
        pass  # nice job using a dataclass!
    elif isinstance(d, dict):
        valid_keys = set(struct_class.__annotations__.keys())
        invalid_keys = set(d.keys()).difference(valid_keys)
        if invalid_keys:
            raise ValueError(f"Invalid keys in {struct_name}: {invalid_keys}")
    else:
        raise TypeError(f"Expecting {struct_name} or dict, but got {d!r}")


def _get_limits(id: int, device: bool = False, adapter: bool = False):
    """Gets the limits for a device or an adapter"""
    assert device + adapter == 1  # exactly one is set

    # H: chain: WGPUChainedStructOut, maxPushConstantSize: int, maxNonSamplerBindings: int
    c_limits_native = new_struct(
        "WGPUNativeLimits",
        # H: next: WGPUChainedStructOut *, sType: WGPUSType
        chain=new_struct(
            "WGPUChainedStructOut",
            # not used: next
            sType=lib.WGPUSType_NativeLimits,
        ),
        # not used: maxPushConstantSize
        # not used: maxNonSamplerBindings
    )

    # Note that the object returned by ffi.cast() does not own the memory, so we must keep a ref to the uncast object, until wgpu-native has consumed it.
    c_limit_next_in_chain = ffi.addressof(c_limits_native, "chain")

    # H: nextInChain: WGPUChainedStructOut *, maxTextureDimension1D: int, maxTextureDimension2D: int, maxTextureDimension3D: int, maxTextureArrayLayers: int, maxBindGroups: int, maxBindGroupsPlusVertexBuffers: int, maxBindingsPerBindGroup: int, maxDynamicUniformBuffersPerPipelineLayout: int, maxDynamicStorageBuffersPerPipelineLayout: int, maxSampledTexturesPerShaderStage: int, maxSamplersPerShaderStage: int, maxStorageBuffersPerShaderStage: int, maxStorageTexturesPerShaderStage: int, maxUniformBuffersPerShaderStage: int, maxUniformBufferBindingSize: int, maxStorageBufferBindingSize: int, minUniformBufferOffsetAlignment: int, minStorageBufferOffsetAlignment: int, maxVertexBuffers: int, maxBufferSize: int, maxVertexAttributes: int, maxVertexBufferArrayStride: int, maxInterStageShaderVariables: int, maxColorAttachments: int, maxColorAttachmentBytesPerSample: int, maxComputeWorkgroupStorageSize: int, maxComputeInvocationsPerWorkgroup: int, maxComputeWorkgroupSizeX: int, maxComputeWorkgroupSizeY: int, maxComputeWorkgroupSizeZ: int, maxComputeWorkgroupsPerDimension: int
    c_limits = new_struct_p(
        "WGPULimits *",
        nextInChain=c_limit_next_in_chain,
        # not used: maxTextureDimension1D
        # not used: maxTextureDimension2D
        # not used: maxTextureDimension3D
        # not used: maxTextureArrayLayers
        # not used: maxBindGroups
        # not used: maxBindGroupsPlusVertexBuffers
        # not used: maxBindingsPerBindGroup
        # not used: maxDynamicUniformBuffersPerPipelineLayout
        # not used: maxDynamicStorageBuffersPerPipelineLayout
        # not used: maxSampledTexturesPerShaderStage
        # not used: maxSamplersPerShaderStage
        # not used: maxStorageBuffersPerShaderStage
        # not used: maxStorageTexturesPerShaderStage
        # not used: maxUniformBuffersPerShaderStage
        # not used: maxUniformBufferBindingSize
        # not used: maxStorageBufferBindingSize
        # not used: minUniformBufferOffsetAlignment
        # not used: minStorageBufferOffsetAlignment
        # not used: maxVertexBuffers
        # not used: maxBufferSize
        # not used: maxVertexAttributes
        # not used: maxVertexBufferArrayStride
        # not used: maxInterStageShaderVariables
        # not used: maxColorAttachments
        # not used: maxColorAttachmentBytesPerSample
        # not used: maxComputeWorkgroupStorageSize
        # not used: maxComputeInvocationsPerWorkgroup
        # not used: maxComputeWorkgroupSizeX
        # not used: maxComputeWorkgroupSizeY
        # not used: maxComputeWorkgroupSizeZ
        # not used: maxComputeWorkgroupsPerDimension
    )
    if adapter:
        # H: WGPUStatus f(WGPUAdapter adapter, WGPULimits * limits)
        status = libf.wgpuAdapterGetLimits(id, c_limits)
        if status != lib.WGPUStatus_Success:
            raise RuntimeError("Error calling wgpuAdapterGetLimits")
    else:
        # H: WGPUStatus f(WGPUDevice device, WGPULimits * limits)
        status = libf.wgpuDeviceGetLimits(id, c_limits)
        if status != lib.WGPUStatus_Success:
            raise RuntimeError("Error calling wgpuDeviceGetLimits")

    key_value_pairs = [
        (to_snake_case(name, "-"), getattr(limits, name))
        for limits in (c_limits, c_limits_native)
        for name in dir(limits)
        if "chain" not in name.lower()  # Skip the pointers
    ]
    limits = dict(sorted(key_value_pairs))
    return limits


def _get_features(id: int, device: bool = False, adapter: bool = False):
    """Gets the features for a device or an adapter"""
    assert device + adapter == 1  # exactly one of them is set

    if adapter:
        # H: WGPUBool f(WGPUAdapter adapter, WGPUFeatureName feature)
        has_feature = lambda feature: libf.wgpuAdapterHasFeature(id, feature)
    else:
        # H: WGPUBool f(WGPUDevice device, WGPUFeatureName feature)
        has_feature = lambda feature: libf.wgpuDeviceHasFeature(id, feature)

    features = set()

    # Standard features
    not_supported_by_wgpu_native = {
        "subgroups",
        "core-features-and-limits",
        "texture-formats-tier1",
        "texture-formats-tier2",
        "primitive-index",
    }
    for f in sorted(enums.FeatureName):
        if f in not_supported_by_wgpu_native:
            continue
        if has_feature(enummap[f"FeatureName.{f}"]):
            features.add(f)

    # Native features
    for name, feature_id in enum_str2int["NativeFeature"].items():
        if has_feature(feature_id):
            features.add(name)
    return features


error_handler = ErrorHandler(logger)
libf = SafeLibCalls(lib, error_handler)


# %% The API


class GPU(classes.GPU):
    def request_adapter_async(
        self,
        *,
        feature_level: str = "core",
        power_preference: enums.PowerPreferenceEnum | None = None,
        force_fallback_adapter: bool = False,
        canvas: CanvasLike | None = None,
        loop: LoopInterface | None = None,
    ) -> GPUPromise[GPUAdapter]:
        """Create a `GPUAdapter`, the object that represents an abstract wgpu
        implementation, from which one can request a `GPUDevice`.

        This is the implementation based on wgpu-native.

        Arguments:
            power_preference (PowerPreference): "high-performance" or "low-power".
            force_fallback_adapter (bool): whether to use a (probably CPU-based)
                fallback adapter.
            canvas : The canvas that the adapter should be able to render to. This can typically
                be left to None. If given, the object must implement ``WgpuCanvasInterface``.
        """

        # Similar to https://github.com/gfx-rs/wgpu?tab=readme-ov-file#environment-variables
        # It seems that the environment variables are only respected in their
        # testing environments maybe????
        # In Dec 2024 we couldn't get the use of their environment variables to work
        # This should only be used in testing environments and API users
        # should beware
        # We chose the variable name WGPUPY_WGPU_ADAPTER_NAME instead WGPU_ADAPTER_NAME
        # to avoid a clash
        if adapter_name := os.getenv(("WGPUPY_WGPU_ADAPTER_NAME")):
            adapters = self._enumerate_adapters()
            adapters_llvm = [a for a in adapters if adapter_name in a.summary]
            if not adapters_llvm:
                raise ValueError(f"Adapter with name '{adapter_name}' not found.")
            promise = GPUPromise("llm adapter", None, loop=loop)
            promise._wgpu_set_input(adapters_llvm[0])

            return promise
        # ----- Surface ID

        # Get surface id that the adapter must be compatible with. If we
        # don't pass a valid surface id, there is no guarantee we'll be
        # able to create a surface texture for it (from this adapter).
        surface_id = ffi.NULL
        if canvas is not None:
            surface_id = canvas.get_context("wgpu")._surface_id  # can still be NULL

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
                    f"Invalid value for WGPU_BACKEND_TYPE: '{force_backend}'.\nValid values are: {list(enum_str2int['BackendType'].keys())}"
                )
            else:
                logger.warning(f"Forcing backend: {force_backend} ({backend})")

        # ----- Request adapter

        c_feature_level = {
            "core": lib.WGPUFeatureLevel_Core,
            "compatibility": lib.WGPUFeatureLevel_Compatibility,
        }[feature_level]

        # H: nextInChain: WGPUChainedStruct *, featureLevel: WGPUFeatureLevel, powerPreference: WGPUPowerPreference, forceFallbackAdapter: WGPUBool/int, backendType: WGPUBackendType, compatibleSurface: WGPUSurface
        struct = new_struct_p(
            "WGPURequestAdapterOptions *",
            # not used: nextInChain
            featureLevel=c_feature_level,
            powerPreference=power_preference or "high-performance",
            forceFallbackAdapter=bool(force_fallback_adapter),
            backendType=backend,
            compatibleSurface=surface_id,
        )

        @ffi.callback(
            "void(WGPURequestAdapterStatus, WGPUAdapter, WGPUStringView, void *, void *)"
        )
        def request_adapter_callback(status, result, c_message, _userdata1, _userdata2):
            if status != lib.WGPURequestAdapterStatus_Success:
                msg = from_c_string_view(c_message)
                promise._wgpu_set_error(
                    RuntimeError(f"Request adapter failed ({status}): {msg}")
                )
            else:
                promise._wgpu_set_input(result)

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPURequestAdapterCallback, userdata1: void*, userdata2: void*
        callback_info = new_struct(
            "WGPURequestAdapterCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowProcessEvents,
            callback=request_adapter_callback,
            # not used: userdata1
            # not used: userdata2
        )

        def handler(adapter_id):
            return self._create_adapter(adapter_id, loop)

        instance = get_wgpu_instance()

        def poller():
            # H: void f(WGPUInstance instance)
            libf.wgpuInstanceProcessEvents(instance)

        # Note that although we claim this is an asynchronous method, the callback
        # happens within libf.wgpuInstanceRequestAdapter
        promise = GPUPromise(
            "request_adapter",
            handler,
            loop=loop,
            poller=poller,
            keepalive=request_adapter_callback,
        )

        # H: WGPUFuture f(WGPUInstance instance, WGPURequestAdapterOptions const * options, WGPURequestAdapterCallbackInfo callbackInfo)
        libf.wgpuInstanceRequestAdapter(get_wgpu_instance(), struct, callback_info)

        return promise

    def enumerate_adapters_async(
        self, *, loop: LoopInterface | None = None
    ) -> GPUPromise[list[GPUAdapter]]:
        """Get a list of adapter objects available on the current system.
        This is the implementation based on wgpu-native.
        """
        result = self._enumerate_adapters(loop)
        # We already have the result, so we return a resolved promise.
        # The reason this is async is to allow this to work on backends where we cannot actually enumerate adapters.
        promise = GPUPromise("enumerate_adapters", None, loop=loop)
        promise._wgpu_set_input(result)
        return promise

    def _enumerate_adapters(self, loop) -> list[GPUAdapter]:
        # The first call is to get the number of adapters, and the second call
        # is to get the actual adapters. Note that the second arg (now NULL) can
        # be a `WGPUInstanceEnumerateAdapterOptions` to filter by backend.
        instance = get_wgpu_instance()
        # H: size_t f(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const * options, WGPUAdapter * adapters)
        count = libf.wgpuInstanceEnumerateAdapters(instance, ffi.NULL, ffi.NULL)
        adapters = new_array("WGPUAdapter[]", count)
        # H: size_t f(WGPUInstance instance, WGPUInstanceEnumerateAdapterOptions const * options, WGPUAdapter * adapters)
        libf.wgpuInstanceEnumerateAdapters(instance, ffi.NULL, adapters)
        return [self._create_adapter(adapter, loop) for adapter in adapters]

    def _create_adapter(self, adapter_id, loop):
        # ----- Get adapter info

        # H: nextInChain: WGPUChainedStructOut *, vendor: WGPUStringView, architecture: WGPUStringView, device: WGPUStringView, description: WGPUStringView, backendType: WGPUBackendType, adapterType: WGPUAdapterType, vendorID: int, deviceID: int
        c_info = new_struct_p(
            "WGPUAdapterInfo *",
            # not used: nextInChain
            # not used: vendor
            # not used: architecture
            # not used: device
            # not used: description
            # not used: backendType
            # not used: adapterType
            # not used: vendorID
            # not used: deviceID
        )

        # H: WGPUStatus f(WGPUAdapter adapter, WGPUAdapterInfo * info)
        status = libf.wgpuAdapterGetInfo(adapter_id, c_info)
        if status != lib.WGPUStatus_Success:
            raise RuntimeError("Error calling wgpuAdapterGetInfo")

        def to_py_str(key):
            string_view = getattr(c_info, key)
            return from_c_string_view(string_view)

        # Populate a dict according to the WebGPU spec: https://gpuweb.github.io/gpuweb/#gpuadapterinfo
        # And add all other info we get from wgpu-native too.
        # note: device is human readable. description is driver-description; usually more cryptic, or empty.
        adapter_info_data = {
            # Spec
            "vendor": to_py_str("vendor"),
            "architecture": to_py_str("architecture"),
            "device": to_py_str("device"),
            "description": to_py_str("description"),
            # Defaults for stackgroup info from §3.6.2.4 of webgpu specification
            "stackgroup_min_size": getattr(c_info, "stackgroupMinSize", 4),
            "stackgroup_max_size": getattr(c_info, "stackgroupMaxSize", 128),
            # Extra
            "vendor_id": int(c_info.vendorID),
            "device_id": int(c_info.deviceID),
            "adapter_type": enum_int2str["AdapterType"].get(
                c_info.adapterType, "unknown"
            ),
            "backend_type": enum_int2str["BackendType"].get(
                c_info.backendType, "unknown"
            ),
        }
        adapter_info = GPUAdapterInfo(adapter_info_data)

        # Allow Rust to release its string objects
        # H: void f(WGPUAdapterInfo adapterInfo)
        libf.wgpuAdapterInfoFreeMembers(c_info[0])

        # ----- Get adapter limits and features
        limits = _get_limits(adapter_id, adapter=True)
        features = _get_features(adapter_id, adapter=True)

        # ----- Done
        return GPUAdapter(adapter_id, features, limits, adapter_info, loop)


# Instantiate API entrypoint
gpu = GPU()


class GPUPromise(classes.GPUPromise):
    pass


class GPUCanvasContext(classes.GPUCanvasContext):
    # The way this works, is that the context must first be configured.
    # Then a texture can be obtained, which can be written to, and then it
    # can be presented. The lifetime of the texture is between
    # get_current_texture() and present(). We keep track of the texture so
    # we can give meaningful errors/warnings on invalid use, rather than
    # the more cryptic Rust panics.

    _surface_id = ffi.NULL
    _wgpu_config = None
    _skip_present_screen = False

    def __init__(self, canvas, present_methods):
        super().__init__(canvas, present_methods)

        # Obtain the surface id. The lifetime is of the surface is bound
        # to the lifetime of this context object.
        if self._present_method == "screen":
            self._surface_id = get_surface_id_from_info(self._present_methods["screen"])
        else:  # method == "bitmap"
            self._surface_id = ffi.NULL

        # A stat for get_current_texture
        self._number_of_successive_unsuccesful_textures = 0

    def _get_capabilities_screen(self, adapter):
        adapter_id = adapter._internal
        surface_id = self._surface_id
        assert surface_id

        minimal_capabilities = {
            "usages": flags.TextureUsage.RENDER_ATTACHMENT,
            "formats": [
                enums.TextureFormat.bgra8unorm_srgb,
                enums.TextureFormat.bgra8unorm,
            ],
            "alpha_modes": enums.CanvasAlphaMode.opaque,
            "present_modes": ["fifo"],
        }

        # H: nextInChain: WGPUChainedStructOut *, usages: WGPUTextureUsage/int, formatCount: int, formats: WGPUTextureFormat *, presentModeCount: int, presentModes: WGPUPresentMode *, alphaModeCount: int, alphaModes: WGPUCompositeAlphaMode *
        c_capabilities = new_struct_p(
            "WGPUSurfaceCapabilities *",
            # not used: nextInChain
            # not used: usages
            # not used: formatCount
            # not used: formats
            # not used: presentModeCount
            # not used: presentModes
            # not used: alphaModeCount
            # not used: alphaModes
        )

        # H: WGPUStatus f(WGPUSurface surface, WGPUAdapter adapter, WGPUSurfaceCapabilities * capabilities)
        status = libf.wgpuSurfaceGetCapabilities(surface_id, adapter_id, c_capabilities)
        if status != lib.WGPUStatus_Success:
            raise RuntimeError("Error calling wgpuSurfaceGetCapabilities")

        # Convert to Python.
        capabilities = {}

        # When the surface is found not to be compatible, the fields below may
        # be null pointers. This probably means that the surface won't work,
        # and trying to use it will result in an error (or Rust panic). Since
        # I'm not sure what the best time/place to error would be, we pretend
        # that everything is fine here, and populate the fields with values
        # that wgpu-core claims are guaranteed to exist on any (compatible)
        # surface.

        capabilities["usages"] = c_capabilities.usages

        if c_capabilities.formats:
            capabilities["formats"] = formats = []
            for i in range(c_capabilities.formatCount):
                int_val = c_capabilities.formats[i]
                formats.append(enum_int2str["TextureFormat"][int_val])

        else:
            capabilities["formats"] = minimal_capabilities["formats"]

        if c_capabilities.alphaModes:
            capabilities["alpha_modes"] = alpha_modes = []
            for i in range(c_capabilities.alphaModeCount):
                int_val = c_capabilities.alphaModes[i]
                str_val = enum_int2str["CompositeAlphaMode"][int_val]
                alpha_modes.append(str_val.lower())
        else:
            capabilities["alpha_modes"] = minimal_capabilities["alpha_modes"]

        if c_capabilities.presentModes:
            capabilities["present_modes"] = present_modes = []
            for i in range(c_capabilities.presentModeCount):
                int_val = c_capabilities.presentModes[i]
                str_val = enum_int2str["PresentMode"][int_val]
                present_modes.append(str_val.lower())
        else:
            capabilities["present_modes"] = minimal_capabilities["present_modes"]

        # H: void f(WGPUSurfaceCapabilities surfaceCapabilities)
        libf.wgpuSurfaceCapabilitiesFreeMembers(c_capabilities[0])

        return capabilities

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
        capabilities = self._get_capabilities(device.adapter)

        # Convert to C values

        view_formats_list = [enummap["TextureFormat." + x] for x in view_formats]
        c_view_formats = new_array("WGPUTextureFormat[]", view_formats_list)

        # Lookup alpha mode, needs explicit conversion because enum names mismatch
        c_alpha_mode = getattr(lib, f"WGPUCompositeAlphaMode_{alpha_mode.capitalize()}")

        # The color_space is not used for now
        color_space  # noqa - not used yet
        check_struct("CanvasToneMapping", tone_mapping)
        tone_mapping_mode = tone_mapping.get("mode", "standard")
        tone_mapping_mode  # noqa - not used yet

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

        present_modes = [
            p for p in present_mode_pref if p in capabilities["present_modes"]
        ]
        present_mode = (present_modes or capabilities["present_modes"])[0]
        c_present_mode = getattr(lib, f"WGPUPresentMode_{present_mode.capitalize()}")

        # Prepare config object
        width, height = self._get_canvas().get_physical_size()

        # H: nextInChain: WGPUChainedStruct *, device: WGPUDevice, format: WGPUTextureFormat, usage: WGPUTextureUsage/int, width: int, height: int, viewFormatCount: int, viewFormats: WGPUTextureFormat *, alphaMode: WGPUCompositeAlphaMode, presentMode: WGPUPresentMode
        self._wgpu_config = new_struct_p(
            "WGPUSurfaceConfiguration *",
            # not used: nextInChain
            device=device._internal,
            format=format,
            usage=usage,
            viewFormatCount=len(view_formats),
            viewFormats=c_view_formats,
            alphaMode=c_alpha_mode,
            presentMode=c_present_mode,
            width=width,  # overriden elsewhere in this class
            height=height,  # overriden elsewhere in this class
        )

        # Configure now (if possible)
        self._configure_screen_real()

    def _configure_screen_real(self):
        # If a texture is still active, better release it first
        self._drop_texture()
        # Configure, and store the config if we did not error out
        if (
            self._surface_id
            and self._wgpu_config.width > 0
            and self._wgpu_config.height > 0
        ):
            # H: void f(WGPUSurface surface, WGPUSurfaceConfiguration const * config)
            libf.wgpuSurfaceConfigure(self._surface_id, self._wgpu_config)
        else:
            # Set size to zero, to trigger auto-configure later
            self._wgpu_config.width = 0

    def _unconfigure_screen(self):
        if self._surface_id:
            # H: void f(WGPUSurface surface)
            libf.wgpuSurfaceUnconfigure(self._surface_id)
            self._wgpu_config = None

    def _create_texture_screen(self):
        # Check
        if self._surface_id is None:
            raise RuntimeError("Looks like the CanvasContext is already destroyed.")
        if self._wgpu_config is None:
            raise RuntimeError(
                "Cannot get surface texture because the CanvasContext has not yet been configured."
            )

        # When the window size has changed, we need to reconfigure. If we wouldn't:
        #
        # * On some systems (seen on MacOS with glfw and Qt) the texture status that we get below
        #   will happily report 'SuccessOptimal', even when the the window has resized, and the
        #   texture will simply be stretched to fit the window. I believe this can be considered a bug.
        # * On other systems (seen on Windows and Linux) the texture status would report 'SuccessSuboptimal',
        #   and the texture will either be stretched (Windows) or blitted to the window leaving either
        #   part of the texture invisible, or making part of the window black/transparent (Linux).
        # * On some systems the texture status is 'Outdated' even if we do set the size. We deal with
        #   that by providing a dummy texture, and warn when this happens too often in succession.

        # Get size info
        old_size = (self._wgpu_config.width, self._wgpu_config.height)
        new_size = tuple(self._get_canvas().get_physical_size())
        if new_size[0] <= 0 or new_size[1] <= 0:
            # It's the responsibility of the drawing /scheduling logic to prevent this case.
            raise RuntimeError("Cannot get texture for a canvas with zero pixels.")

        # Re-configure when the size has changed.
        if new_size != old_size:
            self._wgpu_config.width = new_size[0]
            self._wgpu_config.height = new_size[1]
            self._configure_screen_real()

        # Prepare for obtaining a texture.
        status_str_map = enum_int2str["SurfaceGetCurrentTextureStatus"]
        # H: nextInChain: WGPUChainedStructOut *, texture: WGPUTexture, status: WGPUSurfaceGetCurrentTextureStatus
        surface_texture = new_struct_p(
            "WGPUSurfaceTexture *",
            # not used: nextInChain
            # not used: texture
            # not used: status
        )

        # Try to obtain texture
        # H: void f(WGPUSurface surface, WGPUSurfaceTexture * surfaceTexture)
        libf.wgpuSurfaceGetCurrentTexture(self._surface_id, surface_texture)
        status_int = surface_texture.status
        status_str = status_str_map.get(status_int, "Unknown")
        texture_id = surface_texture.texture

        if status_int == lib.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
            # Yay! Everything is good and we can render this frame.
            self._number_of_successive_unsuccesful_textures = 0
        elif status_int in [
            lib.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal,
            lib.WGPUSurfaceGetCurrentTextureStatus_Timeout,
            lib.WGPUSurfaceGetCurrentTextureStatus_Outdated,
            lib.WGPUSurfaceGetCurrentTextureStatus_Lost,
        ]:
            if texture_id:
                # H: void f(WGPUTexture texture)
                libf.wgpuTextureRelease(texture_id)
                texture_id = 0
            # Try to re-configure, if we can
            self._configure_screen_real()
            # H: void f(WGPUSurface surface, WGPUSurfaceTexture * surfaceTexture)
            libf.wgpuSurfaceGetCurrentTexture(self._surface_id, surface_texture)
            status_int = surface_texture.status
            status_str = status_str_map.get(status_int, "Unknown")
            texture_id = surface_texture.texture

        # If still not optimal, we need to make some decisions ...
        if status_int != lib.WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
            # It's ok if we miss a sporadic frame during resizing, but warn if it becomes too much.
            self._number_of_successive_unsuccesful_textures += 1
            if self._number_of_successive_unsuccesful_textures > 5:
                n = self._number_of_successive_unsuccesful_textures
                self._number_of_successive_unsuccesful_textures = 0
                logger.warning(
                    f"No succesful surface texture obtained for {n} frames: {status_str!r}"
                )
            # Decide what to do
            if status_int == lib.WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
                # Can still use the texture
                pass
            elif status_int in [
                lib.WGPUSurfaceGetCurrentTextureStatus_Timeout,
                lib.WGPUSurfaceGetCurrentTextureStatus_Outdated,
                lib.WGPUSurfaceGetCurrentTextureStatus_Lost,
            ]:
                # Use a dummy texture that we cannot present
                if texture_id:
                    # H: void f(WGPUTexture texture)
                    libf.wgpuTextureRelease(texture_id)
                    texture_id = 0
                self._skip_present_screen = True
                return self._create_texture_bitmap()
            else:
                # WGPUSurfaceGetCurrentTextureStatus_OutOfMemory
                # WGPUSurfaceGetCurrentTextureStatus_DeviceLost
                # WGPUSurfaceGetCurrentTextureStatus_Error
                # This is something we cannot recover from.
                raise RuntimeError(
                    f"Cannot get surface texture: {status_str} ({status_int})."
                )

        # I don't expect this to happen, but let's check just in case.
        if not texture_id:
            raise RuntimeError("Cannot get surface texture (no texture)")

        # Wrap it in a Python texture object

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
        # H: WGPUTextureUsage f(WGPUTexture texture)
        usage = libf.wgpuTextureGetUsage(texture_id)

        label = ""
        # Cannot yet set label, because it's not implemented in wgpu-native
        # label = "surface-texture"
        # H: void f(WGPUTexture texture, WGPUStringView label)
        # libf.wgpuTextureSetLabel(texture_id, to_c_string_view(label))

        tex_info = {
            "size": (width, height, depth),
            "mip_level_count": mip_level_count,
            "sample_count": sample_count,
            "dimension": dimension,
            "format": format,
            "usage": usage,
        }
        device = self._config["device"]
        return GPUTexture(label, texture_id, device, tex_info)

    def _present_screen(self):
        if self._skip_present_screen:
            self._skip_present_screen = False
        else:
            # H: WGPUStatus f(WGPUSurface surface)
            status = libf.wgpuSurfacePresent(self._surface_id)
            if status != lib.WGPUStatus_Success:
                logger.warning("wgpuSurfacePresent failed")

    def _release(self):
        self._drop_texture()
        if self._surface_id is not None and libf is not None:
            self._surface_id, surface_id = None, self._surface_id
            if surface_id:  # is not NULL
                # H: void f(WGPUSurface surface)
                libf.wgpuSurfaceRelease(surface_id)


class GPUObjectBase(classes.GPUObjectBase):
    def _release(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void wgpuDeviceRelease(WGPUDevice device)
            # H: void wgpuBufferRelease(WGPUBuffer buffer)
            # H: void wgpuTextureRelease(WGPUTexture texture)
            # H: void wgpuTextureViewRelease(WGPUTextureView textureView)
            # H: void wgpuSamplerRelease(WGPUSampler sampler)
            # H: void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout bindGroupLayout)
            # H: void wgpuBindGroupRelease(WGPUBindGroup bindGroup)
            # H: void wgpuPipelineLayoutRelease(WGPUPipelineLayout pipelineLayout)
            # H: void wgpuShaderModuleRelease(WGPUShaderModule shaderModule)
            # H: void wgpuComputePipelineRelease(WGPUComputePipeline computePipeline)
            # H: void wgpuRenderPipelineRelease(WGPURenderPipeline renderPipeline)
            # H: void wgpuCommandBufferRelease(WGPUCommandBuffer commandBuffer)
            # H: void wgpuCommandEncoderRelease(WGPUCommandEncoder commandEncoder)
            # H: void wgpuComputePassEncoderRelease(WGPUComputePassEncoder computePassEncoder)
            # H: void wgpuRenderPassEncoderRelease(WGPURenderPassEncoder renderPassEncoder)
            # H: void wgpuRenderBundleEncoderRelease(WGPURenderBundleEncoder renderBundleEncoder)
            # H: void wgpuQueueRelease(WGPUQueue queue)
            # H: void wgpuRenderBundleRelease(WGPURenderBundle renderBundle)
            # H: void wgpuQuerySetRelease(WGPUQuerySet querySet)
            function = type(self)._release_function
            function(internal)


class GPUAdapterInfo(classes.GPUAdapterInfo):
    pass


class GPUAdapter(classes.GPUAdapter):
    def request_device_async(
        self,
        *,
        label: str = "",
        required_features: Sequence[enums.FeatureNameEnum] = (),
        required_limits: dict[str, int | None] | None = None,
        default_queue: structs.QueueDescriptorStruct | None = None,
    ) -> GPUPromise[GPUDevice]:
        required_limits = {} if required_limits is None else required_limits
        if default_queue:
            check_struct("QueueDescriptor", default_queue)
        # Note that although we claim this function is async, the callback always
        # happens inside the call to libf.wgpuAdapterRequestDevice
        return self._request_device_async(
            label, required_features, required_limits, default_queue, ""
        )

    def _request_device_async(
        self,
        label: str,
        required_features: Sequence[enums.FeatureNameEnum],
        required_limits: dict[str, int],
        default_queue: structs.QueueDescriptorStruct,
        trace_path: str,
    ) -> GPUPromise[GPUDevice]:
        # Note that this method is used in extras.py

        # ---- Handle features

        assert isinstance(required_features, (tuple, list, set))

        c_features = set()
        for f in required_features:
            if isinstance(f, str):
                f = f.replace("_", "-")
                f = to_snake_case(f, "-")
                i = enummap.get(f"FeatureName.{f}", None)
                if i is None:
                    i = enum_str2int["NativeFeature"].get(f, None)
                if i is None and f == "subgroups":  # Temporary fix, see #721
                    i = enum_str2int["NativeFeature"].get("subgroup", None)
                if i is None:
                    raise KeyError(f"Unknown feature: '{f}'")
                c_features.add(i)
            else:
                raise TypeError("Features must be given as str.")

        c_features = sorted(c_features)  # makes it a list

        # ----- Set limits

        # H: nextInChain: WGPUChainedStructOut *, maxTextureDimension1D: int, maxTextureDimension2D: int, maxTextureDimension3D: int, maxTextureArrayLayers: int, maxBindGroups: int, maxBindGroupsPlusVertexBuffers: int, maxBindingsPerBindGroup: int, maxDynamicUniformBuffersPerPipelineLayout: int, maxDynamicStorageBuffersPerPipelineLayout: int, maxSampledTexturesPerShaderStage: int, maxSamplersPerShaderStage: int, maxStorageBuffersPerShaderStage: int, maxStorageTexturesPerShaderStage: int, maxUniformBuffersPerShaderStage: int, maxUniformBufferBindingSize: int, maxStorageBufferBindingSize: int, minUniformBufferOffsetAlignment: int, minStorageBufferOffsetAlignment: int, maxVertexBuffers: int, maxBufferSize: int, maxVertexAttributes: int, maxVertexBufferArrayStride: int, maxInterStageShaderVariables: int, maxColorAttachments: int, maxColorAttachmentBytesPerSample: int, maxComputeWorkgroupStorageSize: int, maxComputeInvocationsPerWorkgroup: int, maxComputeWorkgroupSizeX: int, maxComputeWorkgroupSizeY: int, maxComputeWorkgroupSizeZ: int, maxComputeWorkgroupsPerDimension: int
        c_required_limits = new_struct_p(
            "WGPULimits *",
            # not used: nextInChain
            # not used: maxTextureDimension1D
            # not used: maxTextureDimension2D
            # not used: maxTextureDimension3D
            # not used: maxTextureArrayLayers
            # not used: maxBindGroups
            # not used: maxBindGroupsPlusVertexBuffers
            # not used: maxBindingsPerBindGroup
            # not used: maxDynamicUniformBuffersPerPipelineLayout
            # not used: maxDynamicStorageBuffersPerPipelineLayout
            # not used: maxSampledTexturesPerShaderStage
            # not used: maxSamplersPerShaderStage
            # not used: maxStorageBuffersPerShaderStage
            # not used: maxStorageTexturesPerShaderStage
            # not used: maxUniformBuffersPerShaderStage
            # not used: maxUniformBufferBindingSize
            # not used: maxStorageBufferBindingSize
            # not used: minUniformBufferOffsetAlignment
            # not used: minStorageBufferOffsetAlignment
            # not used: maxVertexBuffers
            # not used: maxBufferSize
            # not used: maxVertexAttributes
            # not used: maxVertexBufferArrayStride
            # not used: maxInterStageShaderVariables
            # not used: maxColorAttachments
            # not used: maxColorAttachmentBytesPerSample
            # not used: maxComputeWorkgroupStorageSize
            # not used: maxComputeInvocationsPerWorkgroup
            # not used: maxComputeWorkgroupSizeX
            # not used: maxComputeWorkgroupSizeY
            # not used: maxComputeWorkgroupSizeZ
            # not used: maxComputeWorkgroupsPerDimension
        )

        def canonicalize_limit_name(name):
            if name in self._limits:
                return name
            if "_" in name:
                alt_name = name.replace("_", "-")
                if alt_name in self._limits:
                    return alt_name
            alt_name = to_snake_case(name, "-")
            if alt_name in self._limits:
                return alt_name
            raise KeyError(f"Unknown limit name '{name}'")

        if required_limits:
            assert isinstance(required_limits, dict)
            required_limits = {
                canonicalize_limit_name(key): value
                for key, value in required_limits.items()
            }
        else:
            # If required_limits isn't set, set it to self._limits.  This is the same as
            # setting it to {}, but the loop below goes just a little bit faster.
            required_limits = self._limits

        for key in dir(c_required_limits):
            snake_key = to_snake_case(key, "-")
            # Skip the  pointers
            if snake_key in (
                "next-in-chain",
                "max-push-constant-size",
                "max-non-sampler-bindings",
            ):
                # Skip the chain and the native limits as they are handled in their own
                continue
            # Use the value in required_limits if it exists. Otherwise, the old value
            try:
                value = required_limits[snake_key]
            except KeyError:
                value = self._limits[snake_key]
            setattr(c_required_limits, key, value)

        #  the native only limits are passed in via the next-in-chain struct
        # H: chain: WGPUChainedStructOut, maxPushConstantSize: int, maxNonSamplerBindings: int
        c_required_limits_native = new_struct_p(
            "WGPUNativeLimits *",
            maxPushConstantSize=required_limits.get(
                "max-push-constant-size", self._limits["max-push-constant-size"]
            ),
            maxNonSamplerBindings=required_limits.get(
                "max-non-sampler-bindings", self._limits["max-non-sampler-bindings"]
            ),
            # not used: chain
        )
        c_required_limits_native.chain.next = ffi.NULL
        c_required_limits_native.chain.sType = lib.WGPUSType_NativeLimits

        # here we attached the chain to the struct that's passed further down.
        c_required_limits.nextInChain = ffi.addressof(c_required_limits_native, "chain")
        # ---- Set queue descriptor

        # Note that the default_queue arg is a descriptor (dict for QueueDescriptor), but is currently empty :)
        check_struct("QueueDescriptor", {})

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView
        queue_struct = new_struct(
            "WGPUQueueDescriptor",
            # not used: nextInChain
            label=to_c_string_view("default_queue"),
        )

        # ----- Compose device descriptor extras

        # TODO: is this supported anymore?
        c_trace_path = to_c_string_view(trace_path if trace_path else None)

        # H: chain: WGPUChainedStruct, tracePath: WGPUStringView
        c_device_extras = new_struct_p(
            "WGPUDeviceExtras *",
            tracePath=c_trace_path,
            # not used: chain
        )
        c_device_extras.chain.sType = lib.WGPUSType_DeviceExtras

        # Note that the object returned by ffi.cast() does not own the memory, so we must keep a ref to the uncast object, until wgpu-native has consumed it.
        c_device_next_in_chain = ffi.cast("WGPUChainedStruct * ", c_device_extras)

        # ----- Device lost

        @ffi.callback(
            "void(WGPUDevice const *, WGPUDeviceLostReason, WGPUStringView, void *, void *)"
        )
        def device_lost_callback(c_device, c_reason, c_message, userdata1, userdata2):
            logger.error("DEVICE LOST!")
            reason = enum_int2str["DeviceLostReason"].get(c_reason, "Unknown")
            msg = from_c_string_view(c_message)
            # This is afaik an error that cannot usually be attributed to a specific call,
            # so we cannot raise it as an error. We log it instead.
            # WebGPU provides (promise-based) API for user-code to handle the error.
            # We might want to do something similar, once we have async figured out.
            error_handler.log_error(f"The WGPU device was lost ({reason}):\n{msg}")

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPUDeviceLostCallback, userdata1: void*, userdata2: void*
        device_lost_callback_info = new_struct(
            "WGPUDeviceLostCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowSpontaneous,
            callback=device_lost_callback,
            # not used: userdata1
            # not used: userdata2
        )

        # ----- Uncaptured error

        # TODO: For some errors (seen for errors in wgsl, but not for some others) the error gets logged via the logger as well (duplicate). Probably an issue with wgpu-core.

        @ffi.callback(
            "void(WGPUDevice const *, WGPUErrorType, WGPUStringView, void *, void *)"
        )
        def uncaptured_error_callback(
            c_device, c_type, c_message, userdata1, userdata2
        ):
            # Let our error handler deal with it: the currently running API call will raise an error, or the error will be logged.
            # Note that this call does/should not raise directly; it's a callback from Rust code
            error_type = enum_int2str["ErrorType"].get(c_type, "Unknown")
            msg = from_c_string_view(c_message)
            msg = "\n".join(line.rstrip() for line in msg.splitlines())
            error_handler.handle_error(error_type, msg)

        # H: nextInChain: WGPUChainedStruct *, callback: WGPUUncapturedErrorCallback, userdata1: void*, userdata2: void*
        uncaptured_error_callback_info = new_struct(
            "WGPUUncapturedErrorCallbackInfo",
            # not used: nextInChain
            callback=uncaptured_error_callback,
            # not used: userdata1
            # not used: userdata2
        )

        # ----- Request device

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, requiredFeatureCount: int, requiredFeatures: WGPUFeatureName *, requiredLimits: WGPULimits *, defaultQueue: WGPUQueueDescriptor, deviceLostCallbackInfo: WGPUDeviceLostCallbackInfo, uncapturedErrorCallbackInfo: WGPUUncapturedErrorCallbackInfo
        struct = new_struct_p(
            "WGPUDeviceDescriptor *",
            nextInChain=c_device_next_in_chain,
            label=to_c_string_view(label),
            requiredFeatureCount=len(c_features),
            requiredFeatures=new_array("WGPUFeatureName[]", c_features),
            requiredLimits=c_required_limits,
            defaultQueue=queue_struct,
            deviceLostCallbackInfo=device_lost_callback_info,
            uncapturedErrorCallbackInfo=uncaptured_error_callback_info,
        )

        @ffi.callback(
            "void(WGPURequestDeviceStatus, WGPUDevice, WGPUStringView, void *, void *)"
        )
        def request_device_callback(status, result, c_message, userdata1, userdata2):
            if status != lib.WGPURequestDeviceStatus_Success:
                msg = from_c_string_view(c_message)
                promise._wgpu_set_error(
                    RuntimeError(f"Request device failed ({status}): {msg}")
                )
            else:
                promise._wgpu_set_input(result)

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPURequestDeviceCallback, userdata1: void*, userdata2: void*
        callback_info = new_struct(
            "WGPURequestDeviceCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowProcessEvents,
            callback=request_device_callback,
            # not used: userdata1
            # not used: userdata2
        )

        def handler(device_id):
            limits = _get_limits(device_id, device=True)
            features = _get_features(device_id, device=True)
            # H: WGPUQueue f(WGPUDevice device)
            queue_id = libf.wgpuDeviceGetQueue(device_id)
            queue = GPUQueue("", queue_id, None)

            device = GPUDevice(label, device_id, self, features, limits, queue)

            # Bind some things to the lifetime of the device
            device._uncaptured_error_callback = uncaptured_error_callback
            device._device_lost_callback = device_lost_callback

            return device

        instance = get_wgpu_instance()

        def poller():
            # H: void f(WGPUInstance instance)
            libf.wgpuInstanceProcessEvents(instance)

        promise = GPUPromise(
            "request_device",
            handler,
            loop=self._loop,
            poller=poller,
            keepalive=request_device_callback,
        )

        # H: WGPUFuture f(WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor, WGPURequestDeviceCallbackInfo callbackInfo)
        libf.wgpuAdapterRequestDevice(self._internal, struct, callback_info)

        return promise

    def _release(self):
        if self._internal is not None and libf is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUAdapter adapter)
            libf.wgpuAdapterRelease(internal)


class GPUDevice(classes.GPUDevice, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuDeviceRelease

    # This flag  should be deleted once create_compute_pipeline_async() and
    # create_render_pipeline_async() are actually implemented in the wgpu-native library.
    # they now exist in the header, but are still unimplemented: https://github.com/gfx-rs/wgpu-native/blob/f29ebee88362934f8f9fab530f3ccb7fde2d49a9/src/unimplemented.rs#L66-L82
    _CREATE_PIPELINE_ASYNC_IS_IMPLEMENTED = False

    def _poll(self):
        # Internal function
        if self._internal:
            # H: WGPUBool f(WGPUDevice device, WGPUBool wait, WGPUSubmissionIndex const * submissionIndex)
            libf.wgpuDevicePoll(self._internal, False, ffi.NULL)

    def _poll_wait(self):
        if self._internal:
            # H: WGPUBool f(WGPUDevice device, WGPUBool wait, WGPUSubmissionIndex const * submissionIndex)
            libf.wgpuDevicePoll(self._internal, True, ffi.NULL)

    def create_buffer(
        self,
        *,
        label: str = "",
        size: int,
        usage: flags.BufferUsageFlags,
        mapped_at_creation: bool = False,
    ) -> GPUBuffer:
        return self._create_buffer(label, int(size), usage, bool(mapped_at_creation))

    def _create_buffer(self, label, size, usage, mapped_at_creation):
        # Create a buffer object
        if isinstance(usage, str):
            usage = str_flag_to_int(flags.BufferUsage, usage)
        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, usage: WGPUBufferUsage/int, size: int, mappedAtCreation: WGPUBool/int
        struct = new_struct_p(
            "WGPUBufferDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            size=size,
            usage=int(usage),
            mappedAtCreation=mapped_at_creation,
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
        label: str = "",
        size: tuple[int, int, int] | structs.Extent3DStruct,
        mip_level_count: int = 1,
        sample_count: int = 1,
        dimension: enums.TextureDimensionEnum = "2d",
        format: enums.TextureFormatEnum,
        usage: flags.TextureUsageFlags,
        view_formats: Sequence[enums.TextureFormatEnum] = (),
    ) -> GPUTexture:
        if isinstance(usage, str):
            usage = str_flag_to_int(flags.TextureUsage, usage)
        usage = int(usage)
        size = _tuple_from_extent3d(size)

        # It's easy to accidentally pass 2, when you mean '2d'. Sadly in webgpu.h,
        # the int value for '2d' is actually 1 :/
        if not isinstance(dimension, str):
            raise TypeError(
                f"Texture dimension must be a str, not {dimension.__class__.__name__}"
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

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, usage: WGPUTextureUsage/int, dimension: WGPUTextureDimension, size: WGPUExtent3D, format: WGPUTextureFormat, mipLevelCount: int, sampleCount: int, viewFormatCount: int, viewFormats: WGPUTextureFormat *
        struct = new_struct_p(
            "WGPUTextureDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            size=c_size,
            mipLevelCount=mip_level_count,
            sampleCount=sample_count,
            dimension=dimension,
            format=format,
            usage=usage,
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
        label: str = "",
        address_mode_u: enums.AddressModeEnum = "clamp-to-edge",
        address_mode_v: enums.AddressModeEnum = "clamp-to-edge",
        address_mode_w: enums.AddressModeEnum = "clamp-to-edge",
        mag_filter: enums.FilterModeEnum = "nearest",
        min_filter: enums.FilterModeEnum = "nearest",
        mipmap_filter: enums.MipmapFilterModeEnum = "nearest",
        lod_min_clamp: float = 0,
        lod_max_clamp: float = 32,
        compare: enums.CompareFunctionEnum | None = None,
        max_anisotropy: int = 1,
    ) -> GPUSampler:
        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, addressModeU: WGPUAddressMode, addressModeV: WGPUAddressMode, addressModeW: WGPUAddressMode, magFilter: WGPUFilterMode, minFilter: WGPUFilterMode, mipmapFilter: WGPUMipmapFilterMode, lodMinClamp: float, lodMaxClamp: float, compare: WGPUCompareFunction, maxAnisotropy: int
        struct = new_struct_p(
            "WGPUSamplerDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            addressModeU=address_mode_u,
            addressModeV=address_mode_v,
            addressModeW=address_mode_w,
            magFilter=mag_filter,
            minFilter=min_filter,
            mipmapFilter=mipmap_filter,
            lodMinClamp=lod_min_clamp,
            lodMaxClamp=lod_max_clamp,
            compare=0 if compare is None else compare,  # 0 means undefined
            maxAnisotropy=max_anisotropy,
        )

        # H: WGPUSampler f(WGPUDevice device, WGPUSamplerDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateSampler(self._internal, struct)
        return GPUSampler(label, id, self)

    def create_bind_group_layout(
        self, *, label: str = "", entries: Sequence[structs.BindGroupLayoutEntryStruct]
    ) -> GPUBindGroupLayout:
        c_entries_list = []
        for entry in entries:
            check_struct("BindGroupLayoutEntry", entry)
            buffer = entry.get("buffer")
            sampler = entry.get("sampler")
            texture = entry.get("texture")
            storage_texture = entry.get("storage_texture")
            if buffer is not None:  # Note, it might be an empty dictionary
                info = buffer
                sampler = texture = storage_texture = ()
                check_struct("BufferBindingLayout", info)
                min_binding_size = info.get("min_binding_size", None)
                if min_binding_size is None:
                    min_binding_size = 0  # lib.WGPU_LIMIT_U64_UNDEFINED
                # H: nextInChain: WGPUChainedStruct *, type: WGPUBufferBindingType, hasDynamicOffset: WGPUBool/int, minBindingSize: int
                buffer = new_struct(
                    "WGPUBufferBindingLayout",
                    # not used: nextInChain
                    type=info.get("type", "uniform"),
                    hasDynamicOffset=info.get("has_dynamic_offset", False),
                    minBindingSize=min_binding_size,
                )
            elif sampler is not None:  # It may be an empty dictionary
                info = sampler
                buffer = texture = storage_texture = ()
                check_struct("SamplerBindingLayout", info)
                # H: nextInChain: WGPUChainedStruct *, type: WGPUSamplerBindingType
                sampler = new_struct(
                    "WGPUSamplerBindingLayout",
                    # not used: nextInChain
                    type=info.get("type", "filtering"),
                )
            elif texture is not None:  # It may be an empty dictionary
                info = texture
                buffer = sampler = storage_texture = ()
                check_struct("TextureBindingLayout", info)
                view_dimension = info.get("view_dimension", "2d")
                if not isinstance(view_dimension, str):
                    raise TypeError(
                        f"Texture view dimension must be a str, not {view_dimension.__class__.__name__}"
                    )
                # H: nextInChain: WGPUChainedStruct *, sampleType: WGPUTextureSampleType, viewDimension: WGPUTextureViewDimension, multisampled: WGPUBool/int
                texture = new_struct(
                    "WGPUTextureBindingLayout",
                    # not used: nextInChain
                    sampleType=info.get("sample_type", "float"),
                    viewDimension=view_dimension,
                    multisampled=info.get("multisampled", False),
                )
            elif storage_texture is not None:  # format is required, so not empty
                info = storage_texture
                buffer = sampler = texture = ()
                check_struct("StorageTextureBindingLayout", info)
                view_dimension = info.get("view_dimension", "2d")
                if not isinstance(view_dimension, str):
                    raise TypeError(
                        f"Texture view dimension must be a str, not {view_dimension.__class__.__name__}"
                    )
                # H: nextInChain: WGPUChainedStruct *, access: WGPUStorageTextureAccess, format: WGPUTextureFormat, viewDimension: WGPUTextureViewDimension
                storage_texture = new_struct(
                    "WGPUStorageTextureBindingLayout",
                    # not used: nextInChain
                    access=info.get("access", "write-only"),
                    viewDimension=view_dimension,
                    format=info["format"],
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
            # H: nextInChain: WGPUChainedStruct *, binding: int, visibility: WGPUShaderStage/int, buffer: WGPUBufferBindingLayout, sampler: WGPUSamplerBindingLayout, texture: WGPUTextureBindingLayout, storageTexture: WGPUStorageTextureBindingLayout
            c_entry = new_struct(
                "WGPUBindGroupLayoutEntry",
                # not used: nextInChain
                binding=int(entry["binding"]),
                visibility=int(visibility),
                buffer=buffer,
                sampler=sampler,
                texture=texture,
                storageTexture=storage_texture,
            )
            c_entries_list.append(c_entry)

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, entryCount: int, entries: WGPUBindGroupLayoutEntry *
        struct = new_struct_p(
            "WGPUBindGroupLayoutDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            entries=new_array("WGPUBindGroupLayoutEntry[]", c_entries_list),
            entryCount=len(c_entries_list),
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
        return GPUBindGroupLayout(label, id, self)

    def create_bind_group(
        self,
        *,
        label: str = "",
        layout: GPUBindGroupLayout,
        entries: Sequence[structs.BindGroupEntryStruct],
    ) -> GPUBindGroup:
        c_entries_list = []
        for entry in entries:
            check_struct("BindGroupEntry", entry)
            # The resource can be a sampler, texture view, or buffer descriptor
            resource = entry["resource"]
            if isinstance(resource, GPUSampler):
                # H: nextInChain: WGPUChainedStruct *, binding: int, buffer: WGPUBuffer, offset: int, size: int, sampler: WGPUSampler, textureView: WGPUTextureView
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    # not used: nextInChain
                    binding=int(entry["binding"]),
                    buffer=ffi.NULL,
                    offset=0,
                    size=0,
                    sampler=resource._internal,
                    textureView=ffi.NULL,
                )
            elif isinstance(resource, GPUTextureView):
                # H: nextInChain: WGPUChainedStruct *, binding: int, buffer: WGPUBuffer, offset: int, size: int, sampler: WGPUSampler, textureView: WGPUTextureView
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    # not used: nextInChain
                    binding=int(entry["binding"]),
                    buffer=ffi.NULL,
                    offset=0,
                    size=0,
                    sampler=ffi.NULL,
                    textureView=resource._internal,
                )
            elif isinstance(resource, (structs.BufferBinding, dict)):
                # H: nextInChain: WGPUChainedStruct *, binding: int, buffer: WGPUBuffer, offset: int, size: int, sampler: WGPUSampler, textureView: WGPUTextureView
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    # not used: nextInChain
                    binding=int(entry["binding"]),
                    buffer=resource["buffer"]._internal,
                    offset=resource.get("offset", 0),
                    size=resource.get("size", lib.WGPU_WHOLE_SIZE),
                    sampler=ffi.NULL,
                    textureView=ffi.NULL,
                )
            else:
                raise TypeError(f"Unexpected resource type {type(resource)}")
            c_entries_list.append(c_entry)

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, layout: WGPUBindGroupLayout, entryCount: int, entries: WGPUBindGroupEntry *
        struct = new_struct_p(
            "WGPUBindGroupDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            layout=layout._internal,
            entries=new_array("WGPUBindGroupEntry[]", c_entries_list),
            entryCount=len(c_entries_list),
        )

        # H: WGPUBindGroup f(WGPUDevice device, WGPUBindGroupDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateBindGroup(self._internal, struct)
        return GPUBindGroup(label, id, self)

    def create_pipeline_layout(
        self, *, label: str = "", bind_group_layouts: Sequence[GPUBindGroupLayout]
    ) -> GPUPipelineLayout:
        return self._create_pipeline_layout(label, bind_group_layouts, [])

    def _create_pipeline_layout(
        self,
        label: str,
        bind_group_layouts: Sequence[GPUBindGroupLayout],
        push_constant_layouts,
    ):
        bind_group_layouts_ids = [x._internal for x in bind_group_layouts]
        c_layout_array = new_array("WGPUBindGroupLayout[]", bind_group_layouts_ids)

        c_pipeline_layout_next_in_chain = ffi.NULL
        if push_constant_layouts:
            count = len(push_constant_layouts)
            c_push_constant_ranges = new_array("WGPUPushConstantRange[]", count)
            for layout, c_push_constant_range in zip(
                push_constant_layouts, c_push_constant_ranges, strict=False
            ):
                visibility = layout["visibility"]
                if isinstance(visibility, str):
                    visibility = str_flag_to_int(flags.ShaderStage, visibility)
                c_push_constant_range.stages = visibility
                c_push_constant_range.start = layout["start"]
                c_push_constant_range.end = layout["end"]

            # H: chain: WGPUChainedStruct, pushConstantRangeCount: int, pushConstantRanges: WGPUPushConstantRange *
            c_pipeline_layout_extras = new_struct_p(
                "WGPUPipelineLayoutExtras *",
                pushConstantRangeCount=count,
                pushConstantRanges=c_push_constant_ranges,
                # not used: chain
            )
            c_pipeline_layout_extras.chain.sType = lib.WGPUSType_PipelineLayoutExtras
            # Note that the object returned by ffi.cast() does not own the memory, so we must keep a ref to the uncast object, until wgpu-native has consumed it.
            c_pipeline_layout_next_in_chain = ffi.cast(
                "WGPUChainedStruct *", c_pipeline_layout_extras
            )

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, bindGroupLayoutCount: int, bindGroupLayouts: WGPUBindGroupLayout *
        struct = new_struct_p(
            "WGPUPipelineLayoutDescriptor *",
            nextInChain=c_pipeline_layout_next_in_chain,
            label=to_c_string_view(label),
            bindGroupLayouts=c_layout_array,
            bindGroupLayoutCount=len(bind_group_layouts),
        )

        # H: WGPUPipelineLayout f(WGPUDevice device, WGPUPipelineLayoutDescriptor const * descriptor)
        id = libf.wgpuDeviceCreatePipelineLayout(self._internal, struct)
        return GPUPipelineLayout(label, id, self)

    def create_shader_module(
        self,
        *,
        label: str = "",
        code: str,
        compilation_hints: Sequence[structs.ShaderModuleCompilationHintStruct] = (),
    ) -> GPUShaderModule:
        if False:
            # Trick the check_struct check in the codegen.
            # Compilation_hint are not used, but part of the WebGPU API (for now)
            for compilation_hint in compilation_hints:
                check_struct("ShaderModuleCompilationHint", compilation_hint)
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
                        # H: name: WGPUStringView, value: WGPUStringView
                        new_struct(
                            "WGPUShaderDefine",
                            name=to_c_string_view("gl_VertexID"),
                            value=to_c_string_view("gl_VertexIndex"),
                        )
                    )
                # H: chain: WGPUChainedStruct, stage: WGPUShaderStage/int, code: WGPUStringView, defineCount: int, defines: WGPUShaderDefine *
                source_struct = new_struct_p(
                    "WGPUShaderSourceGLSL *",
                    # not used: chain
                    code=to_c_string_view(code),
                    stage=c_stage,
                    defineCount=len(defines),
                    defines=new_array("WGPUShaderDefine[]", defines),
                )
                source_struct[0].chain.next = ffi.NULL
                source_struct[0].chain.sType = lib.WGPUSType_ShaderSourceGLSL
            else:
                # === WGSL
                # H: chain: WGPUChainedStruct, code: WGPUStringView
                source_struct = new_struct_p(
                    "WGPUShaderSourceWGSL *",
                    # not used: chain
                    code=to_c_string_view(code),
                )
                source_struct[0].chain.next = ffi.NULL
                source_struct[0].chain.sType = lib.WGPUSType_ShaderSourceWGSL
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
                "WGPUShaderSourceSPIRV *",
                # not used: chain
                codeSize=len(data) // 4,
                code=data_u32,
            )
            source_struct[0].chain.next = ffi.NULL
            source_struct[0].chain.sType = lib.WGPUSType_ShaderSourceSPIRV
        else:
            raise TypeError(
                "Shader code must be str for WGSL or GLSL, or bytes for SpirV."
            )

        # Note that the object returned by ffi.cast() does not own the memory, so we must keep a ref to the uncast object, until wgpu-native has consumed it.
        c_shader_module_next_in_chain = ffi.cast("WGPUChainedStruct *", source_struct)

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView
        struct = new_struct_p(
            "WGPUShaderModuleDescriptor *",
            nextInChain=c_shader_module_next_in_chain,
            label=to_c_string_view(label),
        )
        # H: WGPUShaderModule f(WGPUDevice device, WGPUShaderModuleDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateShaderModule(self._internal, struct)
        if id == ffi.NULL:
            raise RuntimeError("Shader module creation failed")
        return GPUShaderModule(label, id, self)

    def create_compute_pipeline(
        self,
        *,
        label: str = "",
        layout: GPUPipelineLayout | enums.AutoLayoutModeEnum,
        compute: structs.ProgrammableStageStruct,
    ) -> GPUComputePipeline:
        descriptor = self._create_compute_pipeline_descriptor(label, layout, compute)
        # H: WGPUComputePipeline f(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateComputePipeline(self._internal, descriptor)
        return GPUComputePipeline(label, id, self)

    def create_compute_pipeline_async(
        self,
        *,
        label: str = "",
        layout: GPUPipelineLayout | enums.AutoLayoutModeEnum,
        compute: structs.ProgrammableStageStruct,
    ) -> GPUPromise[GPUComputePipeline]:
        descriptor = self._create_compute_pipeline_descriptor(label, layout, compute)

        if not self._CREATE_PIPELINE_ASYNC_IS_IMPLEMENTED:
            # H: WGPUComputePipeline f(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor)
            id = libf.wgpuDeviceCreateComputePipeline(self._internal, descriptor)
            result = GPUComputePipeline(label, id, self)
            promise = GPUPromise(
                "create_compute_pipeline_async", None, loop=self._device._loop
            )
            promise._wgpu_set_input(result)
            return promise

        # This code is virtually identical to the code in create_render_pipeline_async.
        # Can they be merged??
        @ffi.callback(
            "void(WGPUCreatePipelineAsyncStatus, WGPUComputePipeline, char *, void *, void *)"
        )
        def callback(status, result, c_message, _userdata1, _userdata2):
            if status != lib.WGPUCreatePipelineAsyncStatus_Success:
                msg = from_c_string_view(c_message)
                promise._wgpu_set_error(
                    RuntimeError(f"create_compute_pipeline failed ({status}): {msg}")
                )
            else:
                promise._wgpu_set_input(result)

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPUCreateComputePipelineAsyncCallback, userdata1: void*, userdata2: void*
        callback_info = new_struct(
            "WGPUCreateComputePipelineAsyncCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowProcessEvents,
            callback=callback,
            # not used: userdata1
            # not used: userdata2
        )

        def handler(id):
            return GPUComputePipeline(label, id, self)

        promise = GPUPromise(
            "create_compute_pipeline",
            handler,
            loop=self._loop,
            poller=self._device._poll,
            keepalive=callback,
        )

        # H: WGPUFuture f(WGPUDevice device, WGPUComputePipelineDescriptor const * descriptor, WGPUCreateComputePipelineAsyncCallbackInfo callbackInfo)
        libf.wgpuDeviceCreateComputePipelineAsync(
            self._internal, descriptor, callback_info
        )

        return promise

    def _create_compute_pipeline_descriptor(
        self,
        label: str,
        layout: GPUPipelineLayout | enums.AutoLayoutModeEnum,
        compute: structs.ProgrammableStage,
    ):
        check_struct("ProgrammableStage", compute)
        c_constants, c_constant_entries = _get_override_constant_entries(compute)
        # H: nextInChain: WGPUChainedStruct *, module: WGPUShaderModule, entryPoint: WGPUStringView, constantCount: int, constants: WGPUConstantEntry *
        c_compute_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            # not used: nextInChain
            module=compute["module"]._internal,
            entryPoint=to_c_string_view(compute.get("entry_point")),
            constantCount=len(c_constant_entries),
            constants=c_constants,
        )

        if isinstance(layout, GPUPipelineLayout):
            layout_id = layout._internal
        elif layout == enums.AutoLayoutMode.auto:
            layout_id = ffi.NULL
        else:
            raise TypeError(
                "create_compute_pipeline() 'layout' arg must be a GPUPipelineLayout or 'auto'"
            )

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, layout: WGPUPipelineLayout, compute: WGPUProgrammableStageDescriptor
        struct = new_struct_p(
            "WGPUComputePipelineDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            layout=layout_id,
            compute=c_compute_stage,
        )
        return struct

    def create_render_pipeline(
        self,
        *,
        label: str = "",
        layout: GPUPipelineLayout | enums.AutoLayoutModeEnum,
        vertex: structs.VertexStateStruct,
        primitive: structs.PrimitiveStateStruct | None = None,
        depth_stencil: structs.DepthStencilStateStruct | None = None,
        multisample: structs.MultisampleStateStruct | None = None,
        fragment: structs.FragmentStateStruct | None = None,
    ) -> GPURenderPipeline:
        primitive = {} if primitive is None else primitive
        multisample = {} if multisample is None else multisample
        descriptor, _keep_alive = self._create_render_pipeline_descriptor(
            label, layout, vertex, primitive, depth_stencil, multisample, fragment
        )
        # H: WGPURenderPipeline f(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateRenderPipeline(self._internal, descriptor)
        return GPURenderPipeline(label, id, self)

    def create_render_pipeline_async(
        self,
        *,
        label: str = "",
        layout: GPUPipelineLayout | enums.AutoLayoutModeEnum,
        vertex: structs.VertexStateStruct,
        primitive: structs.PrimitiveStateStruct | None = None,
        depth_stencil: structs.DepthStencilStateStruct | None = None,
        multisample: structs.MultisampleStateStruct | None = None,
        fragment: structs.FragmentStateStruct | None = None,
    ) -> GPUPromise[GPURenderPipeline]:
        primitive = {} if primitive is None else primitive
        multisample = {} if multisample is None else multisample
        # TODO: wgpuDeviceCreateRenderPipelineAsync is not yet implemented in wgpu-native
        descriptor, _keep_alive = self._create_render_pipeline_descriptor(
            label, layout, vertex, primitive, depth_stencil, multisample, fragment
        )

        if not self._CREATE_PIPELINE_ASYNC_IS_IMPLEMENTED:
            # H: WGPURenderPipeline f(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor)
            id = libf.wgpuDeviceCreateRenderPipeline(self._internal, descriptor)
            result = GPURenderPipeline(label, id, self)
            promise = GPUPromise(
                "create_render_pipeline_async", None, loop=self._device._loop
            )
            promise._wgpu_set_input(result)
            return promise

        @ffi.callback(
            "void(WGPUCreatePipelineAsyncStatus, WGPURenderPipeline, WGPUStringView, void *, void *)"
        )
        def callback(status, result, c_message, _userdata1, _userdata2):
            if status != lib.WGPUCreatePipelineAsyncStatus_Success:
                msg = from_c_string_view(c_message)
                promise._wgpu_set_error(
                    RuntimeError(f"Create renderPipeline failed ({status}): {msg}")
                )
            else:
                promise._wgpu_set_input(result)

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPUCreateRenderPipelineAsyncCallback, userdata1: void*, userdata2: void*
        callback_info = new_struct(
            "WGPUCreateRenderPipelineAsyncCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowProcessEvents,
            callback=callback,
            # not used: userdata1
            # not used: userdata2
        )

        def handler(id):
            return GPURenderPipeline(label, id, self)

        promise = GPUPromise(
            "create_render_pipeline",
            handler,
            loop=self._loop,
            poller=self._device._poll,
            keepalive=callback,
        )

        # H: WGPUFuture f(WGPUDevice device, WGPURenderPipelineDescriptor const * descriptor, WGPUCreateRenderPipelineAsyncCallbackInfo callbackInfo)
        libf.wgpuDeviceCreateRenderPipelineAsync(
            self._internal,
            descriptor,
            callback_info,
        )

        return promise

    def _create_render_pipeline_descriptor(
        self,
        label: str,
        layout: GPUPipelineLayout | enums.AutoLayoutModeEnum,
        vertex: structs.VertexState,
        primitive: structs.PrimitiveState,
        depth_stencil: structs.DepthStencilState,
        multisample: structs.MultisampleState,
        fragment: structs.FragmentState,
    ):
        # We need to keep some objects alive until the struct is consumed by wgpu-native
        keep_alive = []

        depth_stencil = depth_stencil or {}
        multisample = multisample or {}
        primitive = primitive or {}
        # remove the extras so the struct can still be checked
        primitive_extras = {}
        if isinstance(primitive, dict):
            # in case of extras, the struct isn't used but a dict... so we check for it here.
            primitive_extras["polygon_mode"] = primitive.pop("polygon_mode", "Fill")
            primitive_extras["conservative"] = primitive.pop("conservative", False)
        check_struct("VertexState", vertex)
        check_struct("DepthStencilState", depth_stencil)
        check_struct("MultisampleState", multisample)
        check_struct("PrimitiveState", primitive)

        c_vertex_buffer_layout_list = [
            self._create_vertex_buffer_layout(buffer_des)
            for buffer_des in vertex.get("buffers", ())
        ]
        c_vertex_buffer_descriptors_array = new_array(
            "WGPUVertexBufferLayout[]", c_vertex_buffer_layout_list
        )
        c_vertex_constants, c_vertex_entries = _get_override_constant_entries(vertex)
        # H: nextInChain: WGPUChainedStruct *, module: WGPUShaderModule, entryPoint: WGPUStringView, constantCount: int, constants: WGPUConstantEntry *, bufferCount: int, buffers: WGPUVertexBufferLayout *
        c_vertex_state = new_struct(
            "WGPUVertexState",
            # not used: nextInChain
            module=vertex["module"]._internal,
            entryPoint=to_c_string_view(vertex.get("entry_point")),
            constantCount=len(c_vertex_entries),
            constants=c_vertex_constants,
            bufferCount=len(c_vertex_buffer_layout_list),
            buffers=c_vertex_buffer_descriptors_array,
        )

        # explanations for extras: https://docs.rs/wgpu/latest/wgpu/struct.PrimitiveState.html
        polygon_mode = enum_str2int["PolygonMode"].get(
            primitive_extras.get("polygon_mode"), enum_str2int["PolygonMode"]["Fill"]
        )

        # H: chain: WGPUChainedStruct, polygonMode: WGPUPolygonMode, conservative: WGPUBool/int
        c_primitive_state_extras = new_struct_p(
            "WGPUPrimitiveStateExtras *",
            # not used: chain
            polygonMode=polygon_mode,
            conservative=primitive_extras.get("conservative", False),
        )
        c_primitive_state_extras.chain.sType = lib.WGPUSType_PrimitiveStateExtras

        # Note that the object returned by ffi.cast() does not own the memory, so we must keep a ref to the uncast object, until wgpu-native has consumed it.
        c_primitive_state_next_in_chain = ffi.cast(
            "WGPUChainedStruct *", c_primitive_state_extras
        )
        keep_alive.append(c_primitive_state_extras)

        # H: nextInChain: WGPUChainedStruct *, topology: WGPUPrimitiveTopology, stripIndexFormat: WGPUIndexFormat, frontFace: WGPUFrontFace, cullMode: WGPUCullMode, unclippedDepth: WGPUBool/int
        c_primitive_state = new_struct(
            "WGPUPrimitiveState",
            nextInChain=c_primitive_state_next_in_chain,
            topology=primitive.get("topology", "triangle-list"),
            stripIndexFormat=primitive.get("strip_index_format", 0),
            frontFace=primitive.get("front_face", "ccw"),
            cullMode=primitive.get("cull_mode", "none"),
            # not used: unclippedDepth
        )

        c_depth_stencil_state = ffi.NULL
        if depth_stencil:
            c_depth_stencil_state = self._create_depth_stencil_state(depth_stencil)

        # H: nextInChain: WGPUChainedStruct *, count: int, mask: int, alphaToCoverageEnabled: WGPUBool/int
        c_multisample_state = new_struct(
            "WGPUMultisampleState",
            # not used: nextInChain
            count=multisample.get("count", 1),
            mask=multisample.get("mask", 0xFFFFFFFF),
            alphaToCoverageEnabled=multisample.get("alpha_to_coverage_enabled", False),
        )

        c_fragment_state = ffi.NULL
        if fragment is not None:
            c_color_targets_list = [
                self._create_color_target_state(target)
                for target in fragment["targets"]
            ]
            c_color_targets_array = new_array(
                "WGPUColorTargetState[]", c_color_targets_list
            )
            check_struct("FragmentState", fragment)
            c_fragment_constants, c_fragment_entries = _get_override_constant_entries(
                fragment
            )
            # H: nextInChain: WGPUChainedStruct *, module: WGPUShaderModule, entryPoint: WGPUStringView, constantCount: int, constants: WGPUConstantEntry *, targetCount: int, targets: WGPUColorTargetState *
            c_fragment_state = new_struct_p(
                "WGPUFragmentState *",
                # not used: nextInChain
                module=fragment["module"]._internal,
                entryPoint=to_c_string_view(fragment.get("entry_point")),
                constantCount=len(c_fragment_entries),
                constants=c_fragment_constants,
                targetCount=len(c_color_targets_list),
                targets=c_color_targets_array,
            )

        if isinstance(layout, GPUPipelineLayout):
            layout_id = layout._internal
        elif layout == enums.AutoLayoutMode.auto:
            layout_id = ffi.NULL
        else:
            raise TypeError(
                "create_render_pipeline() 'layout' arg must be a GPUPipelineLayout or 'auto'"
            )

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, layout: WGPUPipelineLayout, vertex: WGPUVertexState, primitive: WGPUPrimitiveState, depthStencil: WGPUDepthStencilState *, multisample: WGPUMultisampleState, fragment: WGPUFragmentState *
        struct = new_struct_p(
            "WGPURenderPipelineDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            layout=layout_id,
            vertex=c_vertex_state,
            primitive=c_primitive_state,
            depthStencil=c_depth_stencil_state,
            multisample=c_multisample_state,
            fragment=c_fragment_state,
        )
        return struct, keep_alive

    def _create_color_target_state(self, target):
        if not target.get("blend", None):
            c_blend = ffi.NULL
        else:
            c_alpha_blend, c_color_blend = [
                # H: operation: WGPUBlendOperation, srcFactor: WGPUBlendFactor, dstFactor: WGPUBlendFactor
                new_struct(
                    "WGPUBlendComponent",
                    srcFactor=blend.get("src_factor", "one"),
                    dstFactor=blend.get("dst_factor", "zero"),
                    operation=blend.get("operation", "add"),
                )
                for blend in (
                    target["blend"]["alpha"],
                    target["blend"]["color"],
                )
            ]
            # H: color: WGPUBlendComponent, alpha: WGPUBlendComponent
            c_blend = new_struct_p(
                "WGPUBlendState *",
                color=c_color_blend,
                alpha=c_alpha_blend,
            )
        # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, blend: WGPUBlendState *, writeMask: WGPUColorWriteMask/int
        c_color_state = new_struct(
            "WGPUColorTargetState",
            # not used: nextInChain
            format=target["format"],
            blend=c_blend,
            writeMask=target.get("write_mask", 0xF),
        )
        return c_color_state

    def _create_vertex_buffer_layout(self, buffer_des):
        c_attributes_list = []
        for attribute in buffer_des["attributes"]:
            # H: format: WGPUVertexFormat, offset: int, shaderLocation: int
            c_attribute = new_struct(
                "WGPUVertexAttribute",
                format=attribute["format"],
                offset=attribute["offset"],  # this offset is required
                shaderLocation=attribute["shader_location"],
            )
            c_attributes_list.append(c_attribute)
        c_attributes_array = new_array("WGPUVertexAttribute[]", c_attributes_list)
        # H: stepMode: WGPUVertexStepMode, arrayStride: int, attributeCount: int, attributes: WGPUVertexAttribute *
        c_vertex_buffer_descriptor = new_struct(
            "WGPUVertexBufferLayout",
            arrayStride=buffer_des["array_stride"],
            stepMode=buffer_des.get("step_mode", "vertex"),
            attributes=c_attributes_array,
            attributeCount=len(c_attributes_list),
        )
        return c_vertex_buffer_descriptor

    def _create_depth_stencil_state(self, depth_stencil):
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
        # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, depthWriteEnabled: WGPUOptionalBool, depthCompare: WGPUCompareFunction, stencilFront: WGPUStencilFaceState, stencilBack: WGPUStencilFaceState, stencilReadMask: int, stencilWriteMask: int, depthBias: int, depthBiasSlopeScale: float, depthBiasClamp: float
        c_depth_stencil_state = new_struct_p(
            "WGPUDepthStencilState *",
            # not used: nextInChain
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
        )
        return c_depth_stencil_state

    def create_command_encoder(self, *, label: str = "") -> GPUCommandEncoder:
        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView
        struct = new_struct_p(
            "WGPUCommandEncoderDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
        )

        # H: WGPUCommandEncoder f(WGPUDevice device, WGPUCommandEncoderDescriptor const * descriptor)
        id = libf.wgpuDeviceCreateCommandEncoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

    def create_render_bundle_encoder(
        self,
        *,
        label: str = "",
        color_formats: Sequence[enums.TextureFormatEnum],
        depth_stencil_format: enums.TextureFormatEnum | None = None,
        sample_count: int = 1,
        depth_read_only: bool = False,
        stencil_read_only: bool = False,
    ) -> GPURenderBundleEncoder:
        c_color_formats, color_formats_count = ffi.NULL, 0
        if color_formats:
            color_formats_list = [enummap["TextureFormat." + x] for x in color_formats]
            c_color_formats = new_array("WGPUTextureFormat[]", color_formats_list)
            color_formats_count = len(color_formats_list)

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, colorFormatCount: int, colorFormats: WGPUTextureFormat *, depthStencilFormat: WGPUTextureFormat, sampleCount: int, depthReadOnly: WGPUBool/int, stencilReadOnly: WGPUBool/int
        render_bundle_encoder_descriptor = new_struct_p(
            "WGPURenderBundleEncoderDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            colorFormatCount=color_formats_count,
            colorFormats=c_color_formats,
            depthStencilFormat=depth_stencil_format or 0,
            sampleCount=sample_count,
            depthReadOnly=depth_read_only,
            stencilReadOnly=stencil_read_only,
        )
        # H: WGPURenderBundleEncoder f(WGPUDevice device, WGPURenderBundleEncoderDescriptor const * descriptor)
        render_bundle_encoder_id = libf.wgpuDeviceCreateRenderBundleEncoder(
            self._internal, render_bundle_encoder_descriptor
        )
        result = GPURenderBundleEncoder(label, render_bundle_encoder_id, self)
        result._objects_to_keep_alive = set()
        return result

    def create_query_set(
        self, *, label: str = "", type: enums.QueryTypeEnum, count: int
    ) -> GPUQuerySet:
        return self._create_query_set(label, type, count, None)

    def _create_statistics_query_set(self, label, count, statistics):
        values = []
        for name in statistics:
            key = to_snake_case(name.replace("_", "-"), "-")
            value = enum_str2int["PipelineStatisticName"][key]
            values.append(value)
        values.sort()
        return self._create_query_set(
            label, lib.WGPUNativeQueryType_PipelineStatistics, count, values
        )

    def _create_query_set(self, label, type, count, statistics):
        c_query_set_next_in_chain = ffi.NULL
        if statistics:
            c_statistics = new_array("WGPUPipelineStatisticName[]", statistics)
            # H: chain: WGPUChainedStruct, pipelineStatistics: WGPUPipelineStatisticName *, pipelineStatisticCount: int
            c_query_set_descriptor_extras = new_struct_p(
                "WGPUQuerySetDescriptorExtras *",
                pipelineStatisticCount=len(statistics),
                pipelineStatistics=ffi.cast(
                    "WGPUPipelineStatisticName const *", c_statistics
                ),
                # not used: chain
            )
            c_query_set_descriptor_extras.chain.sType = (
                lib.WGPUSType_QuerySetDescriptorExtras
            )
            # Note that the object returned by ffi.cast() does not own the memory, so we must keep a ref to the uncast object, until wgpu-native has consumed it.
            c_query_set_next_in_chain = ffi.cast(
                "WGPUChainedStruct *", c_query_set_descriptor_extras
            )

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, type: WGPUQueryType, count: int
        query_set_descriptor = new_struct_p(
            "WGPUQuerySetDescriptor *",
            nextInChain=c_query_set_next_in_chain,
            label=to_c_string_view(label),
            type=type,
            count=count,
        )

        # H: WGPUQuerySet f(WGPUDevice device, WGPUQuerySetDescriptor const * descriptor)
        query_id = libf.wgpuDeviceCreateQuerySet(self._internal, query_set_descriptor)
        return GPUQuerySet(label, query_id, self, type, count)

    def _get_lost_async(self) -> GPUPromise[GPUDeviceLostInfo]:
        raise NotImplementedError()

    def destroy(self) -> None:
        # NOTE: destroy means that the wgpu-core object gets into a destroyed state. The wgpu-core object still exists.
        # Therefore we must not set self._internal to None.
        internal = self._internal
        if internal is not None:
            # H: void f(WGPUDevice device)
            libf.wgpuDeviceDestroy(internal)

    def _release(self):
        if self._queue is not None:
            queue, self._queue = self._queue, None
            queue._release()
        super()._release()


class GPUBuffer(classes.GPUBuffer, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuBufferRelease

    def __init__(self, label, internal, device, size, usage, map_state):
        super().__init__(label, internal, device, size, usage, map_state)

        self._mapped_status = 0, 0, 0
        self._mapped_memoryviews = []
        # If mapped at creation, set to write mode (no point in reading zeros)
        if self._map_state == enums.BufferMapState.mapped:
            self._mapped_status = 0, self.size, flags.MapMode.WRITE

    def _get_size(self):
        # H: WGPUBufferUsage f(WGPUBuffer buffer)
        return libf.wgpuBufferGetUsage(self._internal)

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
            raise ValueError("Mapped size must be a multiple of 4.")
        if offset + size > self.size:
            raise ValueError("Mapped range must not extend beyond total buffer size.")
        return offset, size

    def map_async(
        self,
        mode: flags.MapModeFlags | None = None,
        offset: int = 0,
        size: int | None = None,
    ) -> GPUPromise[None]:
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
            promise = GPUPromise("buffer.map", None, loop=self._device._loop)
            promise._wgpu_set_error(
                RuntimeError(
                    f"Can only map a buffer if its currently unmapped, not {self._map_state!r}"
                )
            )
            return promise

        # Sync up when reading, otherwise the memory may be all zeros.
        # See https://github.com/gfx-rs/wgpu-native/issues/305
        if sync_on_read and map_mode & lib.WGPUMapMode_Read:
            if self._mapped_status[2] == 0 and self._usage & flags.BufferUsage.MAP_READ:
                encoder = self._device.create_command_encoder()
                command_buffer = encoder.finish()
                self._device.queue.submit([command_buffer])

        # Setup promise

        @ffi.callback("void(WGPUMapAsyncStatus, WGPUStringView, void *, void *)")
        def buffer_map_callback(status, c_message, _userdata1, _userdata2):
            if status != lib.WGPUMapAsyncStatus_Success:
                msg = from_c_string_view(c_message)
                promise._wgpu_set_error(
                    RuntimeError(f"Could not map buffer ({status} : {msg}).")
                )
            else:
                promise._wgpu_set_input(status)

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPUBufferMapCallback, userdata1: void*, userdata2: void*
        buffer_map_callback_info = new_struct(
            "WGPUBufferMapCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowProcessEvents,
            callback=buffer_map_callback,
            # not used: userdata1
            # not used: userdata2
        )

        def handler(_status):
            self._map_state = enums.BufferMapState.mapped
            self._mapped_status = offset, offset + size, mode
            self._mapped_memoryviews = []

        promise = GPUPromise(
            "buffer.map",
            handler,
            loop=self._device._loop,
            poller=self._device._poll,
            keepalive=buffer_map_callback,
        )

        # Map it
        self._map_state = enums.BufferMapState.pending
        # H: WGPUFuture f(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size, WGPUBufferMapCallbackInfo callbackInfo)
        libf.wgpuBufferMapAsync(
            self._internal,
            map_mode,
            offset,
            size,
            buffer_map_callback_info,
        )

        return promise

    def unmap(self) -> None:
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

    def read_mapped(
        self,
        buffer_offset: int | None = None,
        size: int | None = None,
        *,
        copy: bool = True,
    ) -> ArrayLike:
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
            # is unmapped. bytearray() makes a copy of the data; memoryview() creates a
            # view on the bytearray.
            return memoryview(bytearray(src_m)).cast("B")
        else:
            # Return view on the actual mapped data.
            data = src_m.toreadonly()
            self._mapped_memoryviews.append(data)
            return data

    def write_mapped(self, data: ArrayLike, buffer_offset: int | None = None) -> None:
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
        size = (data.nbytes + 3) & ~3

        offset, size = self._check_range(buffer_offset, size)
        if offset < self._mapped_status[0] or (offset + size) > self._mapped_status[1]:
            raise ValueError(
                "The range for buffer writing is not contained in the currently mapped range."
            )

        # Get mapped memoryview
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        src_ptr = libf.wgpuBufferGetMappedRange(self._internal, offset, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, data.nbytes)

        # Copy data. If not contiguous, this operation may be slower.
        src_m[:] = data

    def _experimental_get_mapped_range(self, buffer_offset=None, size=None):
        """Undocumented and experimental. This API can change or be
        removed without notice. Just here so we can benchmark this
        approach. Returns a mapped memoryview object that can be written
        to. Note that once the buffer unmaps, this memoryview object
        becomes unusable, and accessing it may result in a segfault.
        """
        # Can we even write?
        if self._map_state != enums.BufferMapState.mapped:
            raise RuntimeError("Can only write to a buffer if its mapped.")

        offset, size = self._check_range(buffer_offset, size)
        if offset < self._mapped_status[0] or (offset + size) > self._mapped_status[1]:
            raise ValueError(
                "The range for buffer writing is not contained in the currently mapped range."
            )

        # Get mapped memoryview
        # H: void * f(WGPUBuffer buffer, size_t offset, size_t size)
        src_ptr = libf.wgpuBufferGetMappedRange(self._internal, offset, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, size)

        return src_m

    def destroy(self) -> None:
        # NOTE: destroy means that the wgpu-core object gets into a destroyed state. The wgpu-core object still exists.
        # Therefore we must not set self._internal to None.
        internal = self._internal
        if internal is not None:
            # H: void f(WGPUBuffer buffer)
            libf.wgpuBufferDestroy(internal)

    def _release(self):
        self._release_memoryviews()
        super()._release()


class GPUTexture(classes.GPUTexture, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuTextureRelease

    def create_view(
        self,
        *,
        label: str = "",
        format: enums.TextureFormatEnum | None = None,
        dimension: enums.TextureViewDimensionEnum | None = None,
        usage: flags.TextureUsageFlags = 0,
        aspect: enums.TextureAspectEnum = "all",
        base_mip_level: int = 0,
        mip_level_count: int | None = None,
        base_array_layer: int = 0,
        array_layer_count: int | None = None,
    ) -> GPUTextureView:
        # Resolve defaults
        if not format:
            format = self._tex_info["format"]
        if not dimension:
            dimension = self._tex_info["dimension"]  # from create_texture
        elif not isinstance(dimension, str):
            raise TypeError(
                f"Texture view dimension must be a str, not {dimension.__class__.__name__}"
            )
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

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, format: WGPUTextureFormat, dimension: WGPUTextureViewDimension, baseMipLevel: int, mipLevelCount: int, baseArrayLayer: int, arrayLayerCount: int, aspect: WGPUTextureAspect, usage: WGPUTextureUsage/int
        struct = new_struct_p(
            "WGPUTextureViewDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            format=format,
            dimension=dimension,
            aspect=aspect,
            baseMipLevel=base_mip_level,
            mipLevelCount=mip_level_count,
            baseArrayLayer=base_array_layer,
            arrayLayerCount=array_layer_count,
            usage=usage,
        )

        # H: WGPUTextureView f(WGPUTexture texture, WGPUTextureViewDescriptor const * descriptor)
        id = libf.wgpuTextureCreateView(self._internal, struct)
        return GPUTextureView(label, id, self._device, self, self.size)

    def destroy(self) -> None:
        # NOTE: destroy means that the wgpu-core object gets into a destroyed state. The wgpu-core object still exists.
        # Therefore we must not set self._internal to None.
        internal = self._internal
        if internal is not None:
            # H: void f(WGPUTexture texture)
            libf.wgpuTextureDestroy(internal)


class GPUTextureView(classes.GPUTextureView, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuTextureViewRelease


class GPUSampler(classes.GPUSampler, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuSamplerRelease


class GPUBindGroupLayout(classes.GPUBindGroupLayout, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuBindGroupLayoutRelease


class GPUBindGroup(classes.GPUBindGroup, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuBindGroupRelease


class GPUPipelineLayout(classes.GPUPipelineLayout, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuPipelineLayoutRelease


class GPUShaderModule(classes.GPUShaderModule, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuShaderModuleRelease

    def get_compilation_info_async(self) -> GPUPromise[GPUCompilationInfo]:
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
        # H: WGPUFuture f(WGPUShaderModule shaderModule, WGPUCompilationInfoCallbackInfo callbackInfo)
        # libf.wgpuShaderModuleGetCompilationInfo(self._internal, callback, ffi.NULL)
        #
        # self._device._poll()
        #
        # if info is None:
        #     raise RuntimeError("Could not obtain shader compilation info.")
        #
        #  ... and then turn these WGPUCompilationInfoRequestStatus objects into Python objects ...

        result = []

        # Return a resolved promise
        promise = GPUPromise("get_compilation_info", None, loop=self._device._loop)
        promise._wgpu_set_input(result)
        return promise


class GPUPipelineBase(classes.GPUPipelineBase):
    def get_bind_group_layout(self, index: int | None = None) -> GPUBindGroupLayout:
        # H: WGPUBindGroupLayout wgpuComputePipelineGetBindGroupLayout(WGPUComputePipeline computePipeline, uint32_t groupIndex)
        # H: WGPUBindGroupLayout wgpuRenderPipelineGetBindGroupLayout(WGPURenderPipeline renderPipeline, uint32_t groupIndex)
        function = type(self)._get_bind_group_layout_function
        layout_id = function(self._internal, index)
        return GPUBindGroupLayout("", layout_id, self._device)


class GPUComputePipeline(classes.GPUComputePipeline, GPUPipelineBase, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuComputePipelineRelease

    # GPUPipelineBase
    _get_bind_group_layout_function = libf.wgpuComputePipelineGetBindGroupLayout


class GPURenderPipeline(classes.GPURenderPipeline, GPUPipelineBase, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuRenderPipelineRelease

    # GPUPipelineBase
    _get_bind_group_layout_function = libf.wgpuRenderPipelineGetBindGroupLayout


class GPUCommandBuffer(classes.GPUCommandBuffer, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuCommandBufferRelease


class GPUCommandsMixin(classes.GPUCommandsMixin):
    pass


class GPUBindingCommandsMixin(classes.GPUBindingCommandsMixin):
    def set_bind_group(
        self,
        index: int,
        bind_group: GPUBindGroup,
        dynamic_offsets_data: Sequence[int] = (),
        dynamic_offsets_data_start: int | None = None,
        dynamic_offsets_data_length: int | None = None,
    ) -> None:
        if (
            dynamic_offsets_data_start is not None
            or dynamic_offsets_data_length is not None
        ):
            if (
                dynamic_offsets_data_start is None
                or dynamic_offsets_data_length is None
            ):
                raise ValueError(
                    "Dynamic offsets start and length must be both set or both None."
                )
            if dynamic_offsets_data_start < 0:
                raise ValueError("Dynamic offsets start must be non-negative.")
            if dynamic_offsets_data_length < 0:
                raise ValueError("Dynamic offsets length must be non-negative.")

            dynamic_offsets_data = dynamic_offsets_data[
                dynamic_offsets_data_start : dynamic_offsets_data_start
                + dynamic_offsets_data_length
            ]

        offsets = list(dynamic_offsets_data)
        c_offsets = ffi.new("uint32_t []", offsets)
        self._maybe_keep_alive(bind_group)
        # H: void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder computePassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
        # H: void wgpuRenderPassEncoderSetBindGroup(WGPURenderPassEncoder renderPassEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
        # H: void wgpuRenderBundleEncoderSetBindGroup(WGPURenderBundleEncoder renderBundleEncoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const * dynamicOffsets)
        function = type(self)._set_bind_group_function
        function(self._internal, index, bind_group._internal, len(offsets), c_offsets)

    ##
    # It is unfortunate that there is no common Mixin that includes just
    # GPUComputePassEncoder and GPURenderPassEncodeer, but not GPURenderBundleEncoder.
    # We put set_push_constants, and XX_pipeline_statistics_query here because they
    # don't really fit anywhere else.
    #

    def _set_push_constants(self, visibility, offset, size_in_bytes, data, data_offset):
        # Implementation of set_push_constant. The public API is in extras.py since
        # this is a wgpu extension.

        # We support anything that memoryview supports, i.e. anything
        # that implements the buffer protocol, including, bytes,
        # bytearray, ctypes arrays, numpy arrays, etc.
        m, address = get_memoryview_and_address(data)

        # Deal with offset and size
        offset = int(offset)
        data_offset = int(data_offset)
        size = int(size_in_bytes)
        if isinstance(visibility, str):
            visibility = str_flag_to_int(flags.ShaderStage, visibility)

        if not (0 <= size_in_bytes <= m.nbytes):
            raise ValueError("Invalid size_in_bytes")
        if not (0 <= size_in_bytes <= m.nbytes):
            raise ValueError("Invalid data_offset")
        if size_in_bytes + data_offset > m.nbytes:
            raise ValueError("size_in_bytes + data_offset is too large")

        c_data = ffi.cast("void *", address)  # do we want to add data_offset?
        # H: void wgpuComputePassEncoderSetPushConstants(WGPUComputePassEncoder encoder, uint32_t offset, uint32_t sizeBytes, void const * data)
        # H: void wgpuRenderPassEncoderSetPushConstants(WGPURenderPassEncoder encoder, WGPUShaderStage stages, uint32_t offset, uint32_t sizeBytes, void const * data)
        # H: void wgpuRenderBundleEncoderSetPushConstants(WGPURenderBundleEncoder encoder, WGPUShaderStage stages, uint32_t offset, uint32_t sizeBytes, void const * data)
        function = type(self)._set_push_constants_function
        if function is None:
            self._not_implemented("set_push_constants")
        function(self._internal, int(visibility), offset, size, c_data + data_offset)

    def _begin_pipeline_statistics_query(self, query_set, query_index):
        # H: void wgpuComputePassEncoderBeginPipelineStatisticsQuery(WGPUComputePassEncoder computePassEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
        # H: void wgpuRenderPassEncoderBeginPipelineStatisticsQuery(WGPURenderPassEncoder renderPassEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
        function = type(self)._begin_pipeline_statistics_query_function
        if function is None:
            self._not_implemented("begin_pipeline_statistics")
        function(self._internal, query_set._internal, int(query_index))

    def _end_pipeline_statistics_query(self):
        # H: void wgpuComputePassEncoderEndPipelineStatisticsQuery(WGPUComputePassEncoder computePassEncoder)
        # H: void wgpuRenderPassEncoderEndPipelineStatisticsQuery(WGPURenderPassEncoder renderPassEncoder)
        function = type(self)._end_pipeline_statistics_query_function
        if function is None:
            self._not_implemented("end_pipeline_statistics")
        function(self._internal)

    def _not_implemented(self, name) -> NoReturn:
        raise RuntimeError(f"{type(self).__name__} does not implement {name}")


class GPUDebugCommandsMixin(classes.GPUDebugCommandsMixin):
    # whole class is likely going to be solved better: https://github.com/pygfx/wgpu-py/pull/546
    def push_debug_group(self, group_label: str | None = None) -> None:
        c_group_label = to_c_string_view(group_label)
        # H: void wgpuCommandEncoderPushDebugGroup(WGPUCommandEncoder commandEncoder, WGPUStringView groupLabel)
        # H: void wgpuComputePassEncoderPushDebugGroup(WGPUComputePassEncoder computePassEncoder, WGPUStringView groupLabel)
        # H: void wgpuRenderPassEncoderPushDebugGroup(WGPURenderPassEncoder renderPassEncoder, WGPUStringView groupLabel)
        # H: void wgpuRenderBundleEncoderPushDebugGroup(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView groupLabel)
        function = type(self)._push_debug_group_function
        function(self._internal, c_group_label)

    def pop_debug_group(self) -> None:
        # H: void wgpuCommandEncoderPopDebugGroup(WGPUCommandEncoder commandEncoder)
        # H: void wgpuComputePassEncoderPopDebugGroup(WGPUComputePassEncoder computePassEncoder)
        # H: void wgpuRenderPassEncoderPopDebugGroup(WGPURenderPassEncoder renderPassEncoder)
        # H: void wgpuRenderBundleEncoderPopDebugGroup(WGPURenderBundleEncoder renderBundleEncoder)
        function = type(self)._pop_debug_group_function
        function(self._internal)

    def insert_debug_marker(self, marker_label: str | None = None) -> None:
        c_marker_label = to_c_string_view(marker_label)
        # H: void wgpuCommandEncoderInsertDebugMarker(WGPUCommandEncoder commandEncoder, WGPUStringView markerLabel)
        # H: void wgpuComputePassEncoderInsertDebugMarker(WGPUComputePassEncoder computePassEncoder, WGPUStringView markerLabel)
        # H: void wgpuRenderPassEncoderInsertDebugMarker(WGPURenderPassEncoder renderPassEncoder, WGPUStringView markerLabel)
        # H: void wgpuRenderBundleEncoderInsertDebugMarker(WGPURenderBundleEncoder renderBundleEncoder, WGPUStringView markerLabel)
        function = type(self)._insert_debug_marker_function
        function(self._internal, c_marker_label)

    def _write_timestamp(self, query_set, query_index):
        # H: void wgpuCommandEncoderWriteTimestamp(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
        # H: void wgpuComputePassEncoderWriteTimestamp(WGPUComputePassEncoder computePassEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
        # H: void wgpuRenderPassEncoderWriteTimestamp(WGPURenderPassEncoder renderPassEncoder, WGPUQuerySet querySet, uint32_t queryIndex)
        function = type(self)._write_timestamp_function
        if function is None:
            raise RuntimeError(
                f"{type(self).__name__} does not implement write_timestamp"
            )
        function(self._internal, query_set._internal, int(query_index))


class GPURenderCommandsMixin(classes.GPURenderCommandsMixin):
    def set_pipeline(self, pipeline: GPURenderPipeline | None = None) -> None:
        self._maybe_keep_alive(pipeline)
        pipeline_id = pipeline._internal
        # H: void wgpuRenderPassEncoderSetPipeline(WGPURenderPassEncoder renderPassEncoder, WGPURenderPipeline pipeline)
        # H: void wgpuRenderBundleEncoderSetPipeline(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderPipeline pipeline)
        function = type(self)._set_pipeline_function
        function(self._internal, pipeline_id)

    def set_index_buffer(
        self,
        buffer: GPUBuffer | None = None,
        index_format: enums.IndexFormatEnum | None = None,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        self._maybe_keep_alive(buffer)
        if not size:
            size = lib.WGPU_WHOLE_SIZE
        c_index_format = enummap[f"IndexFormat.{index_format}"]
        # H: void wgpuRenderPassEncoderSetIndexBuffer(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
        # H: void wgpuRenderBundleEncoderSetIndexBuffer(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size)
        function = type(self)._set_index_buffer_function
        function(
            self._internal, buffer._internal, c_index_format, int(offset), int(size)
        )

    def set_vertex_buffer(
        self,
        slot: int | None = None,
        buffer: GPUBuffer | None = None,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        self._maybe_keep_alive(buffer)
        if not size:
            size = lib.WGPU_WHOLE_SIZE
        # H: void wgpuRenderPassEncoderSetVertexBuffer(WGPURenderPassEncoder renderPassEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size)
        # H: void wgpuRenderBundleEncoderSetVertexBuffer(WGPURenderBundleEncoder renderBundleEncoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size)
        function = type(self)._set_vertex_buffer_function
        function(self._internal, int(slot), buffer._internal, int(offset), int(size))

    def draw(
        self,
        vertex_count: int | None = None,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        # H: void wgpuRenderPassEncoderDraw(WGPURenderPassEncoder renderPassEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
        # H: void wgpuRenderBundleEncoderDraw(WGPURenderBundleEncoder renderBundleEncoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
        function = type(self)._draw_function
        function(
            self._internal, vertex_count, instance_count, first_vertex, first_instance
        )

    def draw_indirect(
        self,
        indirect_buffer: GPUBuffer | None = None,
        indirect_offset: int | None = None,
    ) -> None:
        # self._maybe_keep_alive(indirect_buffer)
        buffer_id = indirect_buffer._internal
        # H: void wgpuRenderPassEncoderDrawIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        # H: void wgpuRenderBundleEncoderDrawIndirect(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        function = type(self)._draw_indirect_function
        function(self._internal, buffer_id, int(indirect_offset))

    def draw_indexed(
        self,
        index_count: int | None = None,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        # H: void wgpuRenderPassEncoderDrawIndexed(WGPURenderPassEncoder renderPassEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
        # H: void wgpuRenderBundleEncoderDrawIndexed(WGPURenderBundleEncoder renderBundleEncoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance)
        function = type(self)._draw_indexed_function
        function(
            self._internal,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )

    def draw_indexed_indirect(
        self,
        indirect_buffer: GPUBuffer | None = None,
        indirect_offset: int | None = None,
    ) -> None:
        self._maybe_keep_alive(indirect_buffer)
        buffer_id = indirect_buffer._internal
        # H: void wgpuRenderPassEncoderDrawIndexedIndirect(WGPURenderPassEncoder renderPassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        # H: void wgpuRenderBundleEncoderDrawIndexedIndirect(WGPURenderBundleEncoder renderBundleEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        function = type(self)._draw_indexed_indirect_function
        function(self._internal, buffer_id, int(indirect_offset))


class GPUCommandEncoder(
    classes.GPUCommandEncoder, GPUCommandsMixin, GPUDebugCommandsMixin, GPUObjectBase
):
    # GPUDebugCommandsMixin
    _push_debug_group_function = libf.wgpuCommandEncoderPushDebugGroup
    _pop_debug_group_function = libf.wgpuCommandEncoderPopDebugGroup
    _insert_debug_marker_function = libf.wgpuCommandEncoderInsertDebugMarker
    _write_timestamp_function = libf.wgpuCommandEncoderWriteTimestamp

    # GPUObjectBaseMixin
    _release_function = libf.wgpuCommandEncoderRelease

    def begin_compute_pass(
        self,
        *,
        label: str = "",
        timestamp_writes: structs.ComputePassTimestampWritesStruct | None = None,
    ) -> GPUComputePassEncoder:
        c_timestamp_writes_struct = ffi.NULL
        if timestamp_writes is not None:
            check_struct("ComputePassTimestampWrites", timestamp_writes)
            # H: querySet: WGPUQuerySet, beginningOfPassWriteIndex: int, endOfPassWriteIndex: int
            c_timestamp_writes_struct = new_struct_p(
                "WGPUComputePassTimestampWrites *",
                querySet=timestamp_writes["query_set"]._internal,
                beginningOfPassWriteIndex=timestamp_writes.get(
                    "beginning_of_pass_write_index", lib.WGPU_QUERY_SET_INDEX_UNDEFINED
                ),
                endOfPassWriteIndex=timestamp_writes.get(
                    "end_of_pass_write_index", lib.WGPU_QUERY_SET_INDEX_UNDEFINED
                ),
            )
        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, timestampWrites: WGPUComputePassTimestampWrites *
        struct = new_struct_p(
            "WGPUComputePassDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            timestampWrites=c_timestamp_writes_struct,
        )
        # H: WGPUComputePassEncoder f(WGPUCommandEncoder commandEncoder, WGPUComputePassDescriptor const * descriptor)
        raw_encoder = libf.wgpuCommandEncoderBeginComputePass(self._internal, struct)
        encoder = GPUComputePassEncoder(label, raw_encoder, self._device)
        return encoder

    def begin_render_pass(
        self,
        *,
        label: str = "",
        color_attachments: Sequence[structs.RenderPassColorAttachmentStruct],
        depth_stencil_attachment: structs.RenderPassDepthStencilAttachmentStruct
        | None = None,
        occlusion_query_set: GPUQuerySet | None = None,
        timestamp_writes: structs.RenderPassTimestampWritesStruct | None = None,
        max_draw_count: int = 50000000,
    ) -> GPURenderPassEncoder:
        c_timestamp_writes_struct = ffi.NULL
        if timestamp_writes is not None:
            check_struct("RenderPassTimestampWrites", timestamp_writes)
            # H: querySet: WGPUQuerySet, beginningOfPassWriteIndex: int, endOfPassWriteIndex: int
            c_timestamp_writes_struct = new_struct_p(
                "WGPURenderPassTimestampWrites *",
                querySet=timestamp_writes["query_set"]._internal,
                beginningOfPassWriteIndex=timestamp_writes.get(
                    "beginning_of_pass_write_index", lib.WGPU_QUERY_SET_INDEX_UNDEFINED
                ),
                endOfPassWriteIndex=timestamp_writes.get(
                    "end_of_pass_write_index", lib.WGPU_QUERY_SET_INDEX_UNDEFINED
                ),
            )

        c_color_attachments_list = [
            self._create_render_pass_color_attachment(color_attachment)
            for color_attachment in color_attachments
        ]
        c_color_attachments_array = new_array(
            "WGPURenderPassColorAttachment[]", c_color_attachments_list
        )

        c_depth_stencil_attachment = ffi.NULL
        if depth_stencil_attachment is not None:
            check_struct("RenderPassDepthStencilAttachment", depth_stencil_attachment)
            c_depth_stencil_attachment = self._create_render_pass_stencil_attachment(
                depth_stencil_attachment
            )

        c_occlusion_query_set = ffi.NULL
        if occlusion_query_set is not None:
            c_occlusion_query_set = occlusion_query_set._internal

        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView, colorAttachmentCount: int, colorAttachments: WGPURenderPassColorAttachment *, depthStencilAttachment: WGPURenderPassDepthStencilAttachment *, occlusionQuerySet: WGPUQuerySet, timestampWrites: WGPURenderPassTimestampWrites *
        struct = new_struct_p(
            "WGPURenderPassDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
            colorAttachments=c_color_attachments_array,
            colorAttachmentCount=len(c_color_attachments_list),
            depthStencilAttachment=c_depth_stencil_attachment,
            timestampWrites=c_timestamp_writes_struct,
            occlusionQuerySet=c_occlusion_query_set,
        )

        # H: WGPURenderPassEncoder f(WGPUCommandEncoder commandEncoder, WGPURenderPassDescriptor const * descriptor)
        raw_encoder = libf.wgpuCommandEncoderBeginRenderPass(self._internal, struct)
        encoder = GPURenderPassEncoder(label, raw_encoder, self._device)
        return encoder

    def _create_render_pass_color_attachment(self, color_attachment):
        check_struct("RenderPassColorAttachment", color_attachment)
        texture_view = color_attachment["view"]
        if not isinstance(texture_view, GPUTextureView):
            raise TypeError("Color attachment view must be a GPUTextureView.")
        texture_view_id = texture_view._internal
        c_resolve_target = (
            ffi.NULL
            if color_attachment.get("resolve_target", None) is None
            else color_attachment["resolve_target"]._internal
        )  # this is a TextureViewId or null
        clear_value = color_attachment.get("clear_value", (0, 0, 0, 0))
        if isinstance(clear_value, dict):
            check_struct("Color", clear_value)
            clear_value = _tuple_from_color(clear_value)
        # H: r: float, g: float, b: float, a: float
        c_clear_value = new_struct(
            "WGPUColor",
            r=clear_value[0],
            g=clear_value[1],
            b=clear_value[2],
            a=clear_value[3],
        )
        # H: nextInChain: WGPUChainedStruct *, view: WGPUTextureView, depthSlice: int, resolveTarget: WGPUTextureView, loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearValue: WGPUColor
        c_attachment = new_struct(
            "WGPURenderPassColorAttachment",
            # not used: nextInChain
            view=texture_view_id,
            resolveTarget=c_resolve_target,
            loadOp=color_attachment["load_op"],
            storeOp=color_attachment["store_op"],
            clearValue=c_clear_value,
            depthSlice=lib.WGPU_DEPTH_SLICE_UNDEFINED,  # not implemented yet
            # not used: resolveTarget
        )
        return c_attachment

    # Pulled out from create_render_pass because it was too large.
    def _create_render_pass_stencil_attachment(self, ds_attachment):
        view = ds_attachment["view"]
        depth_read_only = stencil_read_only = False
        depth_load_op = depth_store_op = stencil_load_op = stencil_store_op = 0
        depth_clear_value = stencil_clear_value = 0
        depth_keys_okay = stencil_keys_okay = False
        # All depth texture formats have "depth" in their name.
        if "depth" in view.texture.format:
            depth_read_only = ds_attachment.get("depth_read_only", False)
            if not depth_read_only:
                depth_keys_okay = True
                depth_load_op = ds_attachment["depth_load_op"]
                depth_store_op = ds_attachment["depth_store_op"]
                if depth_load_op == "clear":
                    depth_clear_value = ds_attachment["depth_clear_value"]
        # All stencil texture formats all have "stencil" in their name
        if "stencil" in view.texture.format:
            stencil_read_only = ds_attachment.get("stencil_read_only", False)
            if not stencil_read_only:
                stencil_keys_okay = True
                stencil_load_op = ds_attachment["stencil_load_op"]
                stencil_store_op = ds_attachment["stencil_store_op"]
                # We only need this if load_op == "clear". It has a default value.
                stencil_clear_value = ds_attachment.get("stencil_clear_value", 0)

        # By the spec, we shouldn't allow "depth_load_op" or "depth_store_op"
        # unless we have a non-read-only depth format. Likewise, "stencil_load_op"
        # and "stencil_store_op" aren't allowed unless we have a non-read-only stencil
        # format.
        # But until now, they were required, even if not needed. So let's make it a
        # warning for now, with the possibility of making it an error in the future.
        unexpected_keys = [
            *(("depth_load_op", "depth_store_op") if not depth_keys_okay else ()),
            *(("stencil_load_op", "stencil_store_op") if not stencil_keys_okay else ()),
        ]
        for key in unexpected_keys:
            if ds_attachment.get(key) is not None:
                if not getattr(self._device, f"warned_about_{key}", False):
                    from wgpu import logger

                    logger.warning(f"Unexpected key {key} in depth_stencil_attachment")
                    setattr(self._device, f"warned_about_{key}", True)

        # H: view: WGPUTextureView, depthLoadOp: WGPULoadOp, depthStoreOp: WGPUStoreOp, depthClearValue: float, depthReadOnly: WGPUBool/int, stencilLoadOp: WGPULoadOp, stencilStoreOp: WGPUStoreOp, stencilClearValue: int, stencilReadOnly: WGPUBool/int
        c_depth_stencil_attachment = new_struct_p(
            "WGPURenderPassDepthStencilAttachment *",
            view=view._internal,
            depthLoadOp=depth_load_op,
            depthStoreOp=depth_store_op,
            depthClearValue=float(depth_clear_value),
            depthReadOnly=depth_read_only,
            stencilLoadOp=stencil_load_op,
            stencilStoreOp=stencil_store_op,
            stencilClearValue=int(stencil_clear_value),
            stencilReadOnly=stencil_read_only,
        )
        return c_depth_stencil_attachment

    def clear_buffer(
        self, buffer: GPUBuffer | None = None, offset: int = 0, size: int | None = None
    ) -> None:
        offset = int(offset)
        if offset % 4 != 0:  # pragma: no cover
            raise ValueError("offset must be a multiple of 4")

        if size is not None:
            size = int(size)
            if size <= 0:  # pragma: no cover
                raise ValueError("size must be > 0")
            if size % 4 != 0:  # pragma: no cover
                raise ValueError("size must be a multiple of 4")
            if offset + size > buffer.size:  # pragma: no cover
                raise ValueError("buffer size out of range")
        else:
            size = lib.WGPU_WHOLE_SIZE
            if offset > buffer.size:
                raise ValueError("offset is too large")
        # H: void f(WGPUCommandEncoder commandEncoder, WGPUBuffer buffer, uint64_t offset, uint64_t size)
        libf.wgpuCommandEncoderClearBuffer(
            self._internal, buffer._internal, offset, size
        )

    def copy_buffer_to_buffer(
        self,
        source: GPUBuffer | None = None,
        source_offset: int | None = None,
        destination: GPUBuffer | None = None,
        destination_offset: int | None = None,
        size: int | None = None,
    ) -> None:
        if source_offset % 4 != 0:  # pragma: no cover
            raise ValueError("source_offset must be a multiple of 4")
        if destination_offset % 4 != 0:  # pragma: no cover
            raise ValueError("destination_offset must be a multiple of 4")
        if size is None:
            size = lib.WGPU_WHOLE_SIZE
        elif size % 4 != 0:  # pragma: no cover
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

    def copy_buffer_to_texture(
        self,
        source: structs.TexelCopyBufferInfoStruct | None = None,
        destination: structs.TexelCopyTextureInfoStruct | None = None,
        copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None,
    ) -> None:
        check_struct("TexelCopyBufferInfo", source)
        check_struct("TexelCopyTextureInfo", destination)

        row_alignment = 256
        bytes_per_row = int(source["bytes_per_row"])
        if (bytes_per_row % row_alignment) != 0:
            raise ValueError(
                f"bytes_per_row ({bytes_per_row}) must be a multiple of {row_alignment}"
            )
        if isinstance(destination["texture"], GPUTextureView):
            raise ValueError("copy destination texture must be a texture, not a view")

        size = _tuple_from_extent3d(copy_size)

        # H: layout: WGPUTexelCopyBufferLayout, buffer: WGPUBuffer
        c_source = new_struct_p(
            "WGPUTexelCopyBufferInfo *",
            # H: offset: int, bytesPerRow: int, rowsPerImage: int
            layout=new_struct(
                "WGPUTexelCopyBufferLayout",
                offset=int(source.get("offset", 0)),
                bytesPerRow=bytes_per_row,
                rowsPerImage=int(source.get("rows_per_image", size[1])),
            ),
            buffer=source["buffer"]._internal,
        )

        ori = _tuple_from_origin3d(destination)
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_destination = new_struct_p(
            "WGPUTexelCopyTextureInfo *",
            texture=destination["texture"]._internal,
            mipLevel=int(destination.get("mip_level", 0)),
            origin=c_origin,
            aspect=enums.TextureAspect.all,
        )

        # H: width: int, height: int, depthOrArrayLayers: int
        c_copy_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUTexelCopyBufferInfo const * source, WGPUTexelCopyTextureInfo const * destination, WGPUExtent3D const * copySize)
        libf.wgpuCommandEncoderCopyBufferToTexture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_buffer(
        self,
        source: structs.TexelCopyTextureInfoStruct | None = None,
        destination: structs.TexelCopyBufferInfoStruct | None = None,
        copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None,
    ) -> None:
        check_struct("TexelCopyTextureInfo", source)
        check_struct("TexelCopyBufferInfo", destination)

        row_alignment = 256
        bytes_per_row = int(destination["bytes_per_row"])
        if (bytes_per_row % row_alignment) != 0:
            raise ValueError(
                f"bytes_per_row must ({bytes_per_row}) be a multiple of {row_alignment}"
            )
        if isinstance(source["texture"], GPUTextureView):
            raise ValueError("copy source texture must be a texture, not a view")

        size = _tuple_from_extent3d(copy_size)

        ori = _tuple_from_origin3d(source)
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_source = new_struct_p(
            "WGPUTexelCopyTextureInfo *",
            texture=source["texture"]._internal,
            mipLevel=int(source.get("mip_level", 0)),
            origin=c_origin,
            aspect=0,
        )

        # H: layout: WGPUTexelCopyBufferLayout, buffer: WGPUBuffer
        c_destination = new_struct_p(
            "WGPUTexelCopyBufferInfo *",
            # H: offset: int, bytesPerRow: int, rowsPerImage: int
            layout=new_struct(
                "WGPUTexelCopyBufferLayout",
                offset=int(destination.get("offset", 0)),
                bytesPerRow=bytes_per_row,
                rowsPerImage=int(destination.get("rows_per_image", size[1])),
            ),
            buffer=destination["buffer"]._internal,
        )

        # H: width: int, height: int, depthOrArrayLayers: int
        c_copy_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUTexelCopyTextureInfo const * source, WGPUTexelCopyBufferInfo const * destination, WGPUExtent3D const * copySize)
        libf.wgpuCommandEncoderCopyTextureToBuffer(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_texture(
        self,
        source: structs.TexelCopyTextureInfoStruct | None = None,
        destination: structs.TexelCopyTextureInfoStruct | None = None,
        copy_size: tuple[int, int, int] | structs.Extent3DStruct | None = None,
    ) -> None:
        check_struct("TexelCopyTextureInfo", source)
        check_struct("TexelCopyTextureInfo", destination)

        if isinstance(source["texture"], GPUTextureView):
            raise ValueError("copy source texture must be a texture, not a view")
        if isinstance(destination["texture"], GPUTextureView):
            raise ValueError("copy destination texture must be a texture, not a view")

        ori = _tuple_from_origin3d(source)
        # H: x: int, y: int, z: int
        c_origin1 = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_source = new_struct_p(
            "WGPUTexelCopyTextureInfo *",
            texture=source["texture"]._internal,
            mipLevel=int(source.get("mip_level", 0)),
            origin=c_origin1,
            # not used: aspect
        )

        ori = _tuple_from_origin3d(destination)
        # H: x: int, y: int, z: int
        c_origin2 = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_destination = new_struct_p(
            "WGPUTexelCopyTextureInfo *",
            texture=destination["texture"]._internal,
            mipLevel=int(destination.get("mip_level", 0)),
            origin=c_origin2,
            # not used: aspect
        )

        size = _tuple_from_extent3d(copy_size)
        # H: width: int, height: int, depthOrArrayLayers: int
        c_copy_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUCommandEncoder commandEncoder, WGPUTexelCopyTextureInfo const * source, WGPUTexelCopyTextureInfo const * destination, WGPUExtent3D const * copySize)
        libf.wgpuCommandEncoderCopyTextureToTexture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def finish(self, *, label: str = "") -> GPUCommandBuffer:
        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView
        struct = new_struct_p(
            "WGPUCommandBufferDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
        )
        # H: WGPUCommandBuffer f(WGPUCommandEncoder commandEncoder, WGPUCommandBufferDescriptor const * descriptor)
        id = libf.wgpuCommandEncoderFinish(self._internal, struct)

        return GPUCommandBuffer(label, id, self._device)

    def resolve_query_set(
        self,
        query_set: GPUQuerySet | None = None,
        first_query: int | None = None,
        query_count: int | None = None,
        destination: GPUBuffer | None = None,
        destination_offset: int | None = None,
    ) -> None:
        # H: void f(WGPUCommandEncoder commandEncoder, WGPUQuerySet querySet, uint32_t firstQuery, uint32_t queryCount, WGPUBuffer destination, uint64_t destinationOffset)
        libf.wgpuCommandEncoderResolveQuerySet(
            self._internal,
            query_set._internal,
            int(first_query),
            int(query_count),
            destination._internal,
            int(destination_offset),
        )


class GPUComputePassEncoder(
    classes.GPUComputePassEncoder,
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPUObjectBase,
):
    # GPUDebugCommandsMixin
    _push_debug_group_function = libf.wgpuComputePassEncoderPushDebugGroup
    _pop_debug_group_function = libf.wgpuComputePassEncoderPopDebugGroup
    _insert_debug_marker_function = libf.wgpuComputePassEncoderInsertDebugMarker
    _write_timestamp_function = libf.wgpuComputePassEncoderWriteTimestamp

    # GPUBindingCommandsMixin
    _set_bind_group_function = libf.wgpuComputePassEncoderSetBindGroup
    _begin_pipeline_statistics_query_function = libf.wgpuComputePassEncoderBeginPipelineStatisticsQuery  # fmt: skip
    _end_pipeline_statistics_query_function = libf.wgpuComputePassEncoderEndPipelineStatisticsQuery  # fmt: skip
    _set_push_constants_function = libf.wgpuComputePassEncoderSetPushConstants

    # GPUObjectBaseMixin
    _release_function = libf.wgpuComputePassEncoderRelease

    def set_pipeline(self, pipeline: GPUComputePipeline | None = None) -> None:
        pipeline_id = pipeline._internal
        # H: void f(WGPUComputePassEncoder computePassEncoder, WGPUComputePipeline pipeline)
        libf.wgpuComputePassEncoderSetPipeline(self._internal, pipeline_id)

    def dispatch_workgroups(
        self,
        workgroup_count_x: int | None = None,
        workgroup_count_y: int = 1,
        workgroup_count_z: int = 1,
    ) -> None:
        # H: void f(WGPUComputePassEncoder computePassEncoder, uint32_t workgroupCountX, uint32_t workgroupCountY, uint32_t workgroupCountZ)
        libf.wgpuComputePassEncoderDispatchWorkgroups(
            self._internal, workgroup_count_x, workgroup_count_y, workgroup_count_z
        )

    def dispatch_workgroups_indirect(
        self,
        indirect_buffer: GPUBuffer | None = None,
        indirect_offset: int | None = None,
    ) -> None:
        buffer_id = indirect_buffer._internal
        # H: void f(WGPUComputePassEncoder computePassEncoder, WGPUBuffer indirectBuffer, uint64_t indirectOffset)
        libf.wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def end(self) -> None:
        # H: void f(WGPUComputePassEncoder computePassEncoder)
        libf.wgpuComputePassEncoderEnd(self._internal)

    def _maybe_keep_alive(self, object):
        pass


class GPURenderPassEncoder(
    classes.GPURenderPassEncoder,
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPURenderCommandsMixin,
    GPUObjectBase,
):
    # GPUDebugCommandsMixin
    _push_debug_group_function = libf.wgpuRenderPassEncoderPushDebugGroup
    _pop_debug_group_function = libf.wgpuRenderPassEncoderPopDebugGroup
    _insert_debug_marker_function = libf.wgpuRenderPassEncoderInsertDebugMarker
    _write_timestamp_function = libf.wgpuRenderPassEncoderWriteTimestamp

    # GPUBindingCommandsMixin
    _set_bind_group_function = libf.wgpuRenderPassEncoderSetBindGroup
    _set_push_constants_function = libf.wgpuRenderPassEncoderSetPushConstants
    _begin_pipeline_statistics_query_function = libf.wgpuRenderPassEncoderBeginPipelineStatisticsQuery  # fmt: skip
    _end_pipeline_statistics_query_function = libf.wgpuRenderPassEncoderEndPipelineStatisticsQuery  # fmt: skip

    # GPURenderCommandsMixin
    _set_pipeline_function = libf.wgpuRenderPassEncoderSetPipeline
    _set_index_buffer_function = libf.wgpuRenderPassEncoderSetIndexBuffer
    _set_vertex_buffer_function = libf.wgpuRenderPassEncoderSetVertexBuffer
    _draw_function = libf.wgpuRenderPassEncoderDraw
    _draw_indirect_function = libf.wgpuRenderPassEncoderDrawIndirect
    _draw_indexed_function = libf.wgpuRenderPassEncoderDrawIndexed
    _draw_indexed_indirect_function = libf.wgpuRenderPassEncoderDrawIndexedIndirect

    # GPUObjectBaseMixin
    _release_function = libf.wgpuRenderPassEncoderRelease

    def set_viewport(
        self,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        min_depth: float | None = None,
        max_depth: float | None = None,
    ) -> None:
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

    def set_scissor_rect(
        self,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height)
        libf.wgpuRenderPassEncoderSetScissorRect(
            self._internal, int(x), int(y), int(width), int(height)
        )

    def set_blend_constant(
        self,
        color: tuple[float, float, float, float] | structs.ColorStruct | None = None,
    ) -> None:
        if isinstance(color, dict):
            check_struct("Color", color)
            color = _tuple_from_color(color)
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

    def set_stencil_reference(self, reference: int | None = None) -> None:
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t reference)
        libf.wgpuRenderPassEncoderSetStencilReference(self._internal, int(reference))

    def end(self) -> None:
        # H: void f(WGPURenderPassEncoder renderPassEncoder)
        libf.wgpuRenderPassEncoderEnd(self._internal)

    def execute_bundles(self, bundles: Sequence[GPURenderBundle] | None = None) -> None:
        bundle_ids = [bundle._internal for bundle in bundles]
        c_bundle_info = new_array("WGPURenderBundle[]", bundle_ids)
        # H: void f(WGPURenderPassEncoder renderPassEncoder, size_t bundleCount, WGPURenderBundle const * bundles)
        libf.wgpuRenderPassEncoderExecuteBundles(
            self._internal, len(bundles), c_bundle_info
        )

    def begin_occlusion_query(self, query_index: int | None = None) -> None:
        # H: void f(WGPURenderPassEncoder renderPassEncoder, uint32_t queryIndex)
        libf.wgpuRenderPassEncoderBeginOcclusionQuery(self._internal, int(query_index))

    def end_occlusion_query(self) -> None:
        # H: void f(WGPURenderPassEncoder renderPassEncoder)
        libf.wgpuRenderPassEncoderEndOcclusionQuery(self._internal)

    def _multi_draw_indirect(self, buffer, offset, count):
        # H: void f(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint32_t count)
        libf.wgpuRenderPassEncoderMultiDrawIndirect(
            self._internal, buffer._internal, int(offset), int(count)
        )

    def _multi_draw_indexed_indirect(self, buffer, offset, count):
        # H: void f(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint32_t count)
        libf.wgpuRenderPassEncoderMultiDrawIndexedIndirect(
            self._internal, buffer._internal, int(offset), int(count)
        )

    def _multi_draw_indirect_count(
        self, buffer, offset, count_buffer, count_buffer_offset, max_count
    ):
        # H: void f(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, WGPUBuffer count_buffer, uint64_t count_buffer_offset, uint32_t max_count)
        libf.wgpuRenderPassEncoderMultiDrawIndirectCount(
            self._internal,
            buffer._internal,
            int(offset),
            count_buffer._internal,
            int(count_buffer_offset),
            int(max_count),
        )

    def _multi_draw_indexed_indirect_count(
        self, buffer, offset, count_buffer, count_buffer_offset, max_count
    ):
        # H: void f(WGPURenderPassEncoder encoder, WGPUBuffer buffer, uint64_t offset, WGPUBuffer count_buffer, uint64_t count_buffer_offset, uint32_t max_count)
        libf.wgpuRenderPassEncoderMultiDrawIndexedIndirectCount(
            self._internal,
            buffer._internal,
            int(offset),
            count_buffer._internal,
            int(count_buffer_offset),
            int(max_count),
        )

    def _maybe_keep_alive(self, object):
        pass


class GPURenderBundleEncoder(
    classes.GPURenderBundleEncoder,
    GPUCommandsMixin,
    GPUDebugCommandsMixin,
    GPUBindingCommandsMixin,
    GPURenderCommandsMixin,
    GPUObjectBase,
):
    # GPUDebugCommandsMixin
    _push_debug_group_function = libf.wgpuRenderBundleEncoderPushDebugGroup
    _pop_debug_group_function = libf.wgpuRenderBundleEncoderPopDebugGroup
    _insert_debug_marker_function = libf.wgpuRenderBundleEncoderInsertDebugMarker

    # GPUBindingCommandsMixin
    _set_bind_group_function = libf.wgpuRenderBundleEncoderSetBindGroup
    _set_push_constants_function = libf.wgpuRenderBundleEncoderSetPushConstants
    _begin_pipeline_statistics_query_function = None  # not implemented
    _end_pipeline_statistics_query_function = None  # not implemented
    _write_timestamp_function = None  # not implemented

    # GPURenderCommandsMixin
    _set_pipeline_function = libf.wgpuRenderBundleEncoderSetPipeline
    _set_index_buffer_function = libf.wgpuRenderBundleEncoderSetIndexBuffer
    _set_vertex_buffer_function = libf.wgpuRenderBundleEncoderSetVertexBuffer
    _draw_function = libf.wgpuRenderBundleEncoderDraw
    _draw_indirect_function = libf.wgpuRenderBundleEncoderDrawIndirect
    _draw_indexed_function = libf.wgpuRenderBundleEncoderDrawIndexed
    _draw_indexed_indirect_function = libf.wgpuRenderBundleEncoderDrawIndexedIndirect

    # GPUObjectBaseMixin
    _release_function = libf.wgpuRenderBundleEncoderRelease

    def finish(self, *, label: str = "") -> GPURenderBundle:
        # H: nextInChain: WGPUChainedStruct *, label: WGPUStringView
        struct = new_struct_p(
            "WGPURenderBundleDescriptor *",
            # not used: nextInChain
            label=to_c_string_view(label),
        )
        # H: WGPURenderBundle f(WGPURenderBundleEncoder renderBundleEncoder, WGPURenderBundleDescriptor const * descriptor)
        id = libf.wgpuRenderBundleEncoderFinish(self._internal, struct)
        # The other encoders require that we call self._release() when
        # we're done with it.  But that doesn't seem to be an issue here.
        # We no longer need to keep these objects alive after the call to finish().
        self._objects_to_keep_alive.clear()
        return GPURenderBundle(label, id, self._device)

    def _maybe_keep_alive(self, object):
        self._objects_to_keep_alive.add(object)


class GPUQueue(classes.GPUQueue, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuQueueRelease

    def submit(self, command_buffers: Sequence[GPUCommandBuffer] | None = None) -> None:
        command_buffer_ids = [cb._internal for cb in command_buffers]
        c_command_buffers = new_array("WGPUCommandBuffer[]", command_buffer_ids)
        # H: void f(WGPUQueue queue, size_t commandCount, WGPUCommandBuffer const * commands)
        libf.wgpuQueueSubmit(self._internal, len(command_buffer_ids), c_command_buffers)

    def write_buffer(
        self,
        buffer: GPUBuffer | None = None,
        buffer_offset: int | None = None,
        data: ArrayLike | None = None,
        data_offset: int = 0,
        size: int | None = None,
    ) -> None:
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

    def read_buffer(
        self, buffer: GPUBuffer, buffer_offset: int = 0, size: int | None = None
    ) -> ArrayLike:
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
        tmp_buffer.map_async("READ_NOSYNC").sync_wait()
        data = tmp_buffer.read_mapped()

        # Explicit drop.
        tmp_buffer.destroy()

        return data

    def write_texture(
        self,
        destination: structs.TexelCopyTextureInfoStruct | None = None,
        data: ArrayLike | None = None,
        data_layout: structs.TexelCopyBufferLayoutStruct | None = None,
        size: tuple[int, int, int] | structs.Extent3DStruct | None = None,
    ) -> None:
        # Note that the bytes_per_row restriction does not apply for
        # this function; wgpu-native deals with it.

        check_struct("TexelCopyTextureInfo", destination)
        check_struct("TexelCopyBufferLayout", data_layout)

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

        size = _tuple_from_extent3d(size)

        ori = _tuple_from_origin3d(destination)
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3D",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTexture, mipLevel: int, origin: WGPUOrigin3D, aspect: WGPUTextureAspect
        c_destination = new_struct_p(
            "WGPUTexelCopyTextureInfo *",
            texture=destination["texture"]._internal,
            mipLevel=destination.get("mip_level", 0),
            origin=c_origin,
            aspect=enums.TextureAspect.all,
        )

        # H: offset: int, bytesPerRow: int, rowsPerImage: int
        c_data_layout = new_struct_p(
            "WGPUTexelCopyBufferLayout *",
            offset=data_layout.get("offset", 0),
            bytesPerRow=data_layout["bytes_per_row"],
            rowsPerImage=data_layout.get("rows_per_image", size[1]),
        )

        # H: width: int, height: int, depthOrArrayLayers: int
        c_size = new_struct_p(
            "WGPUExtent3D *",
            width=size[0],
            height=size[1],
            depthOrArrayLayers=size[2],
        )

        # H: void f(WGPUQueue queue, WGPUTexelCopyTextureInfo const * destination, void const * data, size_t dataSize, WGPUTexelCopyBufferLayout const * dataLayout, WGPUExtent3D const * writeSize)
        libf.wgpuQueueWriteTexture(
            self._internal, c_destination, c_data, data_length, c_data_layout, c_size
        )

    _shared_copy_buffer = None, 0

    def read_texture(
        self, source: dict, data_layout: dict, size: tuple[int, int, int]
    ) -> ArrayLike:
        # Note that the bytes_per_row restriction does not apply for
        # this function; we have to deal with it.

        device = source["texture"]._device

        # Get and calculate striding info
        # Note that full_stride (bytes per row) must be a multiple of 256
        ori_offset = data_layout.get("offset", 0)
        ori_stride = data_layout["bytes_per_row"]
        extra_stride = (256 - ori_stride % 256) % 256
        full_stride = ori_stride + extra_stride

        size = _tuple_from_extent3d(size)
        data_length = full_stride * size[1] * size[2]

        # Create temporary buffer
        is_present_texture = source["texture"].label == "present"
        copy_buffer = None
        if is_present_texture:
            copy_buffer, time_since_size_ok = self._shared_copy_buffer
            if copy_buffer is None:
                pass  # No buffer
            elif copy_buffer.size < data_length:
                copy_buffer = None  # Buffer too small
            elif copy_buffer.size < data_length * 4:
                self._shared_copy_buffer = copy_buffer, time.time()  # Bufer size ok
            elif time.time() - time_since_size_ok > 5.0:
                copy_buffer = None  # Too large too long
        if copy_buffer is None:
            buffer_size = data_length
            buffer_size += (4096 - buffer_size % 4096) % 4096
            buf_usage = flags.BufferUsage.COPY_DST | flags.BufferUsage.MAP_READ
            copy_buffer = device._create_buffer(
                "copy-buffer", buffer_size, buf_usage, False
            )
            if is_present_texture:
                self._shared_copy_buffer = copy_buffer, time.time()

        destination = {
            "buffer": copy_buffer,
            "offset": 0,
            "bytes_per_row": full_stride,  # or WGPU_COPY_STRIDE_UNDEFINED ?
            "rows_per_image": data_layout.get("rows_per_image", size[1]),
        }

        # Copy data to temp buffer
        encoder = device.create_command_encoder()
        encoder.copy_texture_to_buffer(source, destination, size)
        command_buffer = encoder.finish()
        self.submit([command_buffer])

        promise = copy_buffer.map_async("READ_NOSYNC", 0, data_length)

        # Download from mappable buffer
        # Because we use `copy=False``, we *must* copy the data.
        if copy_buffer.map_state == "pending":
            promise.sync_wait()
        mapped_data = copy_buffer.read_mapped(copy=False)

        data_length2 = ori_stride * size[1] * size[2] + ori_offset

        # Copy the data
        if extra_stride or ori_offset:
            # Copy per row
            data = memoryview(bytearray(data_length2)).cast(mapped_data.format)
            i_start = ori_offset
            for i in range(size[1] * size[2]):
                row = mapped_data[i * full_stride : i * full_stride + ori_stride]
                data[i_start : i_start + ori_stride] = row
                i_start += ori_stride
        else:
            # Copy as a whole
            data = memoryview(bytearray(mapped_data)).cast(mapped_data.format)

        # Alternative copy solution using Numpy.
        # I expected this to be faster, but does not really seem to be. Seems not worth it
        # since we technically don't depend on Numpy. Leaving here for reference.
        # import numpy as np
        # mapped_data = np.asarray(mapped_data)[:data_length]
        # data = np.empty(data_length2, dtype=mapped_data.dtype)
        # mapped_data.shape = -1, full_stride
        # data.shape = -1, ori_stride
        # data[:] = mapped_data[:, :ori_stride]
        # data.shape = -1
        # data = memoryview(data)

        # Since we use read_mapped(copy=False), we must unmap it *after* we've copied the data.
        copy_buffer.unmap()

        return data

    def on_submitted_work_done_async(self) -> GPUPromise[None]:
        @ffi.callback("void(WGPUQueueWorkDoneStatus, void *, void *)")
        def work_done_callback(status, _userdata1, _userdata2):
            if status == lib.WGPUQueueWorkDoneStatus_Success:
                promise._wgpu_set_input(True)
            else:
                result = {
                    lib.WGPUQueueWorkDoneStatus_InstanceDropped: "InstanceDropped",
                    lib.WGPUQueueWorkDoneStatus_Error: "Error",
                    lib.WGPUQueueWorkDoneStatus_Unknown: "Unknown",
                }.get(status, "Other")
                promise._wgpu_set_error(
                    RuntimeError(f"Queue work done status: {result}")
                )

        # H: nextInChain: WGPUChainedStruct *, mode: WGPUCallbackMode, callback: WGPUQueueWorkDoneCallback, userdata1: void*, userdata2: void*
        work_done_callback_info = new_struct(
            "WGPUQueueWorkDoneCallbackInfo",
            # not used: nextInChain
            mode=lib.WGPUCallbackMode_AllowProcessEvents,
            callback=work_done_callback,
            # not used: userdata1
            # not used: userdata2
        )

        def handler(_value):
            return None

        promise = GPUPromise(
            "on_submitted_work_done",
            handler,
            loop=self._device._loop,
            poller=self._device._poll_wait,
            keepalive=work_done_callback,
        )

        # H: WGPUFuture f(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callbackInfo)
        libf.wgpuQueueOnSubmittedWorkDone(self._internal, work_done_callback_info)

        return promise


class GPURenderBundle(classes.GPURenderBundle, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuRenderBundleRelease


class GPUQuerySet(classes.GPUQuerySet, GPUObjectBase):
    # GPUObjectBaseMixin
    _release_function = libf.wgpuQuerySetRelease

    def destroy(self) -> None:
        # NOTE: wgpuQuerySetDestroy is currently not implemented incorrectly https://github.com/gfx-rs/wgpu-native/pull/509#discussion_r2403822550
        # It calls `drop` instead, i.e. does the same as wgpuQuerySetRelease. This means that we must set self._internal to None.
        # But this means that if we'd call wgpuQuerySetDestroy, and wgpu-native gets fixed, we get a leak :(
        # So instead we make it explicitly do the same as release.
        # TODO: remove this when wgpu-native actually uses destroy.
        wgpu_native_uses_drop = True
        if wgpu_native_uses_drop:
            self._release()
            return

        # Below is the eventual code for this method:

        # NOTE: destroy means that the wgpu-core object gets into a destroyed state. The wgpu-core object still exists.
        # Therefore we must not set self._internal to None.
        internal = self._internal
        if internal is not None:
            # H: void f(WGPUQuerySet querySet)
            libf.wgpuQuerySetDestroy(internal)
            # if we call del objects during our tests on the "destroyed" object, we get a panic
            # by setting this to none, the __del__ call via _release skips it.
            # might mean we retain memory tho?
            self._internal = None


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
