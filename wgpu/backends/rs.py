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
import ctypes
import logging
import ctypes.util
from weakref import WeakKeyDictionary
from typing import List, Dict

from .. import base, flags, enums, structs
from .. import _register_backend
from .._coreutils import ApiDiff

from .rs_ffi import ffi, lib, check_expected_version
from .rs_mappings import cstructfield2enum, enummap, feature_names
from .rs_helpers import (
    get_surface_id_from_canvas,
    get_memoryview_from_address,
    get_memoryview_and_address,
)


logger = logging.getLogger("wgpu")  # noqa
apidiff = ApiDiff()

# The wgpu-native version that we target/expect
__version__ = "0.7.0"
__commit_sha__ = "17cd1fda67572af47718d67ecf700bb0c9548dce"
version_info = tuple(map(int, __version__.split(".")))
check_expected_version(version_info)  # produces a warning on mismatch


# %% Helper functions and objects

swap_chain_status_map = {
    getattr(lib, "WGPUSwapChainStatus_" + x): x
    for x in ("Good", "Suboptimal", "Lost", "Outdated", "Timeout")
}


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
        if canvas is None:
            surface_id = 0
        else:
            surface_id = get_surface_id_from_canvas(canvas)

        # Convert the descriptor
        # H: power_preference: WGPUPowerPreference, compatible_surface: WGPUOption_SurfaceId/int
        struct = new_struct_p(
            "WGPURequestAdapterOptions *",
            power_preference=power_preference,
            compatible_surface=surface_id,
        )

        # Select possible backends. This is not exposed in the WebGPU API
        # 1 => Backend::Empty,
        # 2 => Backend::Vulkan,
        # 4 => Backend::Metal,
        # 8 => Backend::Dx12,  (buggy)
        # 16 => Backend::Dx11,  (not implemented yet)
        # 32 => Backend::Gl,  (not implemented yet)
        backend_mask = 2 | 4  # Vulkan or Metal

        # Do the API call and get the adapter id

        adapter_id = None

        @ffi.callback("void(uint64_t, void *)")
        def _request_adapter_callback(received, userdata):
            nonlocal adapter_id
            adapter_id = received

        # H: void f(const WGPURequestAdapterOptions *desc, WGPUBackendBit mask, WGPURequestAdapterCallback callback, void *userdata)
        lib.wgpu_request_adapter_async(
            struct, backend_mask, _request_adapter_callback, ffi.NULL
        )  # userdata, stub

        # For now, Rust will call the callback immediately
        # todo: when wgpu gets an event loop -> while run wgpu event loop or something

        assert adapter_id is not None

        # H: struct WGPULimits f(WGPUAdapterId adapter_id)
        c_limits = lib.wgpu_adapter_limits(adapter_id)
        limits = {key: getattr(c_limits, key) for key in dir(c_limits)}

        # H: WGPUFeatures f(WGPUAdapterId adapter_id)
        c_features_flag = lib.wgpu_adapter_features(adapter_id)  # noqa
        features = feature_flag_to_feature_names(c_features_flag)

        # Meh, all I got was ints that we'd have to look up. Implement later.
        # H: void f(WGPUAdapterId adapter_id, struct WGPUAdapterInfo *info)
        # lib.wgpu_adapter_get_info(adapter_id, c_info)

        return GPUAdapter("WGPU", adapter_id, features, limits)

    async def request_adapter_async(self, *, canvas, power_preference=None):
        """Async version of ``request_adapter()``.
        This function uses the Rust WGPU library.
        """
        return self.request_adapter(
            canvas=canvas, power_preference=power_preference
        )  # no-cover


class GPUCanvasContext(base.GPUCanvasContext):
    pass


class GPUObjectBase(base.GPUObjectBase):
    pass


class GPUAdapter(base.GPUAdapter):
    def request_device(
        self,
        *,
        label="",
        non_guaranteed_features: "List[enums.FeatureName]" = [],
        non_guaranteed_limits: "Dict[str, int]" = {},
    ):
        return self._request_device(
            label, non_guaranteed_features, non_guaranteed_limits, ""
        )

    @apidiff.add("a sweet bonus feature from wgpu-native")
    def request_device_tracing(
        self,
        trace_path,
        *,
        label="",
        non_guaranteed_features: "list(enums.FeatureName)" = [],
        non_guaranteed_limits: "Dict[str, int]" = {},
    ):
        """Write a trace of all commands to a file so it can be reproduced
        elsewhere. The trace is cross-platform!
        """
        if not os.path.isdir(trace_path):
            os.makedirs(trace_path, exist_ok=True)
        elif os.listdir(trace_path):
            logger.warning(f"Trace directory not empty: {trace_path}")
        return self._request_device(
            label, non_guaranteed_features, non_guaranteed_limits, trace_path
        )

    def _request_device(self, label, features, limits, trace_path):
        c_trace_path = ffi.NULL
        if trace_path:  # no-cover
            c_trace_path = ffi.new("char []", trace_path.encode())

        # Handle features
        # todo: actually enable features
        c_features_flag = 0

        # Handle default limits
        limits2 = base.DEFAULT_ADAPTER_LIMITS.copy()
        limits2.update(limits or {})

        # H: max_bind_groups: int
        c_limits = new_struct(
            "WGPULimits",
            max_bind_groups=limits2["max_bind_groups"],
        )
        # H: label: WGPULabel, features: WGPUFeatures/int, limits: WGPULimits, trace_path: const char
        struct = new_struct_p(
            "WGPUDeviceDescriptor *",
            label=to_c_label(label),
            features=c_features_flag,
            limits=c_limits,
            trace_path=c_trace_path,
        )
        # H: WGPUDeviceId f(WGPUAdapterId adapter_id, const struct WGPUDeviceDescriptor *desc)
        device_id = lib.wgpu_adapter_request_device(self._internal, struct)

        # Get the actual limits reported by the device
        # H: struct WGPULimits f(WGPUDeviceId device_id)
        c_limits = lib.wgpu_device_limits(device_id)
        limits3 = {key: getattr(c_limits, key) for key in dir(c_limits)}

        # Get actual features reported by the device
        # H: WGPUFeatures f(WGPUDeviceId device_id)
        c_features_flag = lib.wgpu_device_features(device_id)
        features = feature_flag_to_feature_names(c_features_flag)

        # Get the queue to which commands can be submitted
        # H: WGPUQueueId f(WGPUDeviceId device_id)
        queue_id = lib.wgpu_device_get_default_queue(device_id)
        queue = GPUQueue("", queue_id, None)

        return GPUDevice(label, device_id, self, features, limits3, queue)

    async def request_device_async(
        self,
        *,
        label="",
        non_guaranteed_features: "List[enums.FeatureName]" = [],
        non_guaranteed_limits: "Dict[str, int]" = {},
    ):
        return self._request_device(
            label, non_guaranteed_features, non_guaranteed_limits, ""
        )  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUAdapterId adapter_id)
            lib.wgpu_adapter_destroy(internal)


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
        # H: uint8_t *f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferSize size)
        dst_ptr = lib.wgpu_buffer_get_mapped_range(buffer._internal, 0, size)
        dst_address = int(ffi.cast("intptr_t", dst_ptr))
        dst_m = get_memoryview_from_address(dst_address, size)
        dst_m[:] = m  # nicer than ctypes.memmove(dst_address, src_address, m.nbytes)

        buffer._unmap()
        return buffer

    def _create_buffer(self, label, size, usage, mapped_at_creation):

        # Create a buffer object
        # H: label: WGPULabel, size: WGPUBufferAddress/int, usage: WGPUBufferUsage/int, mapped_at_creation: bool
        struct = new_struct_p(
            "WGPUBufferDescriptor *",
            label=to_c_label(label),
            size=size,
            usage=usage,
            mapped_at_creation=mapped_at_creation,
        )
        # H: WGPUBufferId f(WGPUDeviceId device_id, const struct WGPUBufferDescriptor *desc)
        id = lib.wgpu_device_create_buffer(self._internal, struct)
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
        # H: width: int, height: int, depth: int
        c_size = new_struct(
            "WGPUExtent3d",
            width=size[0],
            height=size[1],
            depth=size[2],
        )
        # H: label: WGPULabel, size: WGPUExtent3d, mip_level_count: int, sample_count: int, dimension: WGPUTextureDimension, format: WGPUTextureFormat, usage: WGPUTextureUsage/int
        struct = new_struct_p(
            "WGPUTextureDescriptor *",
            label=to_c_label(label),
            size=c_size,
            mip_level_count=mip_level_count,
            sample_count=sample_count,
            dimension=dimension,
            format=format,
            usage=usage,
        )
        # H: WGPUTextureId f(WGPUDeviceId device_id, const struct WGPUTextureDescriptor *desc)
        id = lib.wgpu_device_create_texture(self._internal, struct)

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
        lod_max_clamp: float = 0xFFFFFFFF,
        compare: "enums.CompareFunction" = None,
        max_anisotropy: int = 1,
    ):
        # H: next_in_chain: WGPUChainedStruct *, label: WGPULabel, address_mode_u: WGPUAddressMode, address_mode_v: WGPUAddressMode, address_mode_w: WGPUAddressMode, mag_filter: WGPUFilterMode, min_filter: WGPUFilterMode, mipmap_filter: WGPUFilterMode, lod_min_clamp: float, lod_max_clamp: float, compare: WGPUCompareFunction/int, border_color: WGPUSamplerBorderColor
        struct = new_struct_p(
            "WGPUSamplerDescriptor *",
            label=to_c_label(label),
            address_mode_u=address_mode_u,
            address_mode_v=address_mode_v,
            mag_filter=mag_filter,
            min_filter=min_filter,
            mipmap_filter=mipmap_filter,
            lod_min_clamp=lod_min_clamp,
            lod_max_clamp=lod_max_clamp,
            compare=0 if compare is None else compare,
            # not used: address_mode_w
            # not used: next_in_chain
            # not used: border_color
        )
        # max_anisotropy not yet supported by wgpu-native

        # H: WGPUSamplerId f(WGPUDeviceId device_id, const struct WGPUSamplerDescriptor *desc)
        id = lib.wgpu_device_create_sampler(self._internal, struct)
        return GPUSampler(label, id, self)

    def create_bind_group_layout(
        self, *, label="", entries: "List[structs.BindGroupLayoutEntry]"
    ):
        c_entries_list = []
        for entry in entries:
            check_struct("BindGroupLayoutEntry", entry)
            c_has_dynamic_offset = False
            c_view_dimension = 0
            c_texture_component_type = 0
            c_multisampled = False
            c_storage_texture_format = 0
            if entry.get("buffer"):
                info = entry["buffer"]
                check_struct("BufferBindingLayout", info)
                type = info["type"]
                if type == enums.BufferBindingType.uniform:
                    c_type = lib.WGPUBindingType_UniformBuffer
                elif type == enums.BufferBindingType.storage:
                    c_type = lib.WGPUBindingType_StorageBuffer
                elif type == enums.BufferBindingType.read_only_storage:
                    c_type = lib.WGPUBindingType_ReadonlyStorageBuffer
                else:
                    raise ValueError(f"Unknown buffer binding type {type}")
                c_has_dynamic_offset = info.get("has_dynamic_offset", False)
                min_binding_size = 0  # noqa: not yet supported in wgpy-native
            elif entry.get("sampler"):
                info = entry["sampler"]
                check_struct("SamplerBindingLayout", info)
                type = info["type"]
                if type == enums.SamplerBindingType.filtering:
                    c_type = lib.WGPUBindingType_Sampler
                elif type == enums.SamplerBindingType.comparison:
                    c_type = lib.WGPUBindingType_ComparisonSampler
                elif type == enums.SamplerBindingType.non_filtering:
                    raise NotImplementedError("Not available in wgpu-native")
                else:
                    raise ValueError(f"Unknown sampler binding type {type}")
            elif entry.get("texture"):
                info = entry["texture"]
                check_struct("TextureBindingLayout", info)
                c_type = lib.WGPUBindingType_SampledTexture
                type = info.get("sample_type", "float")
                if type == enums.TextureSampleType.float:
                    c_texture_component_type = lib.WGPUTextureComponentType_Float
                elif type == enums.TextureSampleType.sint:
                    c_texture_component_type = lib.WGPUTextureComponentType_Sint
                elif type == enums.TextureSampleType.uint:
                    c_texture_component_type = lib.WGPUTextureComponentType_Uint
                elif type == enums.TextureSampleType.depth:
                    raise NotImplementedError("Not available in wgpu-native")
                else:
                    raise ValueError(f"Unknown texture sample type {type}")
                field = info.get("view_dimension", "2d")
                c_view_dimension = enummap[f"TextureViewDimension.{field}"]
                c_multisampled = info.get("multisampled", False)
            elif entry.get("storage_texture"):
                info = entry["storage_texture"]
                check_struct("StorageTextureBindingLayout", info)
                access = info["access"]
                if access == enums.StorageTextureAccess.read_only:
                    c_type = lib.WGPUBindingType_ReadonlyStorageTexture
                elif access == enums.StorageTextureAccess.write_only:
                    c_type = lib.WGPUBindingType_WriteonlyStorageTexture
                else:
                    raise ValueError(f"Unknown storage texture binding access {access}")
                field = info.get("view_dimension", "2d")
                c_view_dimension = enummap[f"TextureViewDimension.{field}"]
                field = info["format"]
                c_storage_texture_format = enummap[f"TextureFormat.{field}"]
            else:
                raise ValueError(
                    "Bind group layout entry did not contain field 'buffer', 'sampler', 'texture', nor 'storage_texture'"
                )
            # H: binding: int, visibility: WGPUShaderStage/int, ty: WGPUBindingType/int, has_dynamic_offset: bool, min_buffer_binding_size: int, multisampled: bool, filtering: bool, view_dimension: WGPUTextureViewDimension, texture_component_type: WGPUTextureComponentType/int, storage_texture_format: WGPUTextureFormat, count: int
            c_entry = new_struct(
                "WGPUBindGroupLayoutEntry",
                binding=int(entry["binding"]),
                visibility=int(entry["visibility"]),
                ty=c_type,
                # Used for uniform buffer and storage buffer bindings.
                has_dynamic_offset=c_has_dynamic_offset,
                # Used for sampled texture and storage texture bindings.
                view_dimension=c_view_dimension,
                # Used for sampled texture bindings.
                texture_component_type=c_texture_component_type,
                # Used for sampled texture bindings.
                multisampled=c_multisampled,
                # Used for storage texture bindings.
                storage_texture_format=c_storage_texture_format,
                # not used: min_buffer_binding_size
                # not used: filtering
                # not used: count
            )
            c_entries_list.append(c_entry)

        # H: label: WGPULabel, entries: WGPUBindGroupLayoutEntry *, entries_length: int
        struct = new_struct_p(
            "WGPUBindGroupLayoutDescriptor *",
            label=to_c_label(label),
            entries=ffi.new("WGPUBindGroupLayoutEntry []", c_entries_list),
            entries_length=len(c_entries_list),
        )

        # H: WGPUBindGroupLayoutId f(WGPUDeviceId device_id, const struct WGPUBindGroupLayoutDescriptor *desc)
        id = lib.wgpu_device_create_bind_group_layout(self._internal, struct)

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
                # H: binding: int, buffer: WGPUOption_BufferId/int, offset: WGPUBufferAddress/int, size: WGPUBufferSize/int, sampler: WGPUOption_SamplerId/int, texture_view: WGPUOption_TextureViewId/int
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    binding=int(entry["binding"]),
                    buffer=0,
                    offset=0,
                    size=0,
                    sampler=resource._internal,
                    texture_view=0,
                )
            elif isinstance(resource, GPUTextureView):
                # H: binding: int, buffer: WGPUOption_BufferId/int, offset: WGPUBufferAddress/int, size: WGPUBufferSize/int, sampler: WGPUOption_SamplerId/int, texture_view: WGPUOption_TextureViewId/int
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    binding=int(entry["binding"]),
                    buffer=0,
                    offset=0,
                    size=0,
                    sampler=0,
                    texture_view=resource._internal,
                )
            elif isinstance(resource, dict):  # Buffer binding
                # H: binding: int, buffer: WGPUOption_BufferId/int, offset: WGPUBufferAddress/int, size: WGPUBufferSize/int, sampler: WGPUOption_SamplerId/int, texture_view: WGPUOption_TextureViewId/int
                c_entry = new_struct(
                    "WGPUBindGroupEntry",
                    binding=int(entry["binding"]),
                    buffer=resource["buffer"]._internal,
                    offset=resource["offset"],
                    size=resource["size"],
                    sampler=0,
                    texture_view=0,
                )
            else:
                raise TypeError(f"Unexpected resource type {type(resource)}")
            c_entries_list.append(c_entry)

        c_entries_array = ffi.new("WGPUBindGroupEntry []", c_entries_list)
        # H: label: WGPULabel, layout: WGPUBindGroupLayoutId/int, entries: WGPUBindGroupEntry *, entries_length: int
        struct = new_struct_p(
            "WGPUBindGroupDescriptor *",
            label=to_c_label(label),
            layout=layout._internal,
            entries=c_entries_array,
            entries_length=len(c_entries_list),
        )

        # H: WGPUBindGroupId f(WGPUDeviceId device_id, const struct WGPUBindGroupDescriptor *desc)
        id = lib.wgpu_device_create_bind_group(self._internal, struct)
        return GPUBindGroup(label, id, self, entries)

    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "List[GPUBindGroupLayout]"
    ):

        bind_group_layouts_ids = [x._internal for x in bind_group_layouts]

        c_layout_array = ffi.new("WGPUBindGroupLayoutId []", bind_group_layouts_ids)
        # H: label: WGPULabel, bind_group_layouts: const WGPUBindGroupLayoutId, bind_group_layouts_length: int
        struct = new_struct_p(
            "WGPUPipelineLayoutDescriptor *",
            label=to_c_label(label),
            bind_group_layouts=c_layout_array,
            bind_group_layouts_length=len(bind_group_layouts),
        )

        # H: WGPUPipelineLayoutId f(WGPUDeviceId device_id, const struct WGPUPipelineLayoutDescriptor *desc_base)
        id = lib.wgpu_device_create_pipeline_layout(self._internal, struct)
        return GPUPipelineLayout(label, id, self, bind_group_layouts)

    def create_shader_module(self, *, label="", code: str, source_map: dict = None):

        if isinstance(code, str):
            # WGSL
            # H: chain: WGPUChainedStruct, source: const char
            source_struct = new_struct_p(
                "WGPUShaderModuleWGSLDescriptor *",
                source=ffi.new("char []", code.encode()),
                # not used: chain
            )
            source_struct[0].chain.next = ffi.NULL
            source_struct[0].chain.s_type = lib.WGPUSType_ShaderModuleWGSLDescriptor
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
            # H: chain: WGPUChainedStruct, code_size: int, code: const uint32_t
            source_struct = new_struct_p(
                "WGPUShaderModuleSPIRVDescriptor *",
                code=data_u32,
                code_size=len(data) // 4,
                # not used: chain
            )
            source_struct[0].chain.next = ffi.NULL
            source_struct[0].chain.s_type = lib.WGPUSType_ShaderModuleSPIRVDescriptor

        # H: next_in_chain: WGPUChainedStruct *, label: WGPULabel, flags: WGPUShaderFlags/int
        struct = new_struct_p(
            "WGPUShaderModuleDescriptor *",
            label=to_c_label(label),
            next_in_chain=ffi.cast("WGPUChainedStruct *", source_struct),
            flags=0,  # 1: validate, 2: translate
        )

        # H: WGPUShaderModuleId f(WGPUDeviceId device_id, const struct WGPUShaderModuleDescriptor *desc)
        id = lib.wgpu_device_create_shader_module(self._internal, struct)
        return GPUShaderModule(label, id, self)

    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        compute: "structs.ProgrammableStage",
    ):
        check_struct("ProgrammableStage", compute)
        # H: module: WGPUShaderModuleId/int, entry_point: WGPULabel
        c_compute_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=compute["module"]._internal,
            entry_point=ffi.new("char []", compute["entry_point"].encode()),
        )

        # H: label: WGPULabel, layout: WGPUOption_PipelineLayoutId/int, stage: WGPUProgrammableStageDescriptor
        struct = new_struct_p(
            "WGPUComputePipelineDescriptor *",
            label=to_c_label(label),
            layout=layout._internal,
            stage=c_compute_stage,
        )

        # H: WGPUComputePipelineId f(WGPUDeviceId device_id, const struct WGPUComputePipelineDescriptor *desc)
        id = lib.wgpu_device_create_compute_pipeline(self._internal, struct)
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

        # H: module: WGPUShaderModuleId/int, entry_point: WGPULabel
        c_vertex_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=vertex["module"]._internal,
            entry_point=ffi.new("char []", vertex["entry_point"].encode()),
        )
        c_fragment_stage = ffi.NULL
        if fragment is not None:
            check_struct("FragmentState", fragment)
            # H: module: WGPUShaderModuleId/int, entry_point: WGPULabel
            c_fragment_stage = new_struct_p(
                "WGPUProgrammableStageDescriptor *",
                module=fragment["module"]._internal,
                entry_point=ffi.new("char []", fragment["entry_point"].encode()),
            )
        # H: nextInChain: WGPUChainedStruct *, frontFace: WGPUFrontFace, cullMode: WGPUCullMode/int, depthBias: int, depthBiasSlopeScale: float, depthBiasClamp: float, clampDepth: bool, polygonMode: WGPUPolygonMode
        c_rasterization_state = new_struct(
            "WGPURasterizationStateDescriptor",
            frontFace=primitive.get("front_face", "ccw"),
            cullMode=primitive.get("cull_mode", "none"),
            depthBias=depth_stencil.get("depth_bias", 0),
            depthBiasSlopeScale=depth_stencil.get("depth_bias_slope_scale", 0),
            depthBiasClamp=depth_stencil.get("depth_bias_clamp", 0),
            # not used: nextInChain
            # not used: clampDepth
            # not used: polygonMode
        )
        c_color_states_list = []
        for target in fragment["targets"]:
            alpha_blend = _tuple_from_tuple_or_dict(
                target["blend"]["alpha"],
                ("src_factor", "dst_factor", "operation"),
            )
            # H: operation: WGPUBlendOperation, srcFactor: WGPUBlendFactor, dstFactor: WGPUBlendFactor
            c_alpha_blend = new_struct(
                "WGPUBlendDescriptor",
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
                "WGPUBlendDescriptor",
                srcFactor=color_blend[0],
                dstFactor=color_blend[1],
                operation=color_blend[2],
            )
            # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, alphaBlend: WGPUBlendDescriptor, colorBlend: WGPUBlendDescriptor, writeMask: WGPUColorWrite/int
            c_color_state = new_struct(
                "WGPUColorStateDescriptor",
                format=target["format"],
                alphaBlend=c_alpha_blend,
                colorBlend=c_color_blend,
                writeMask=target.get("write_mask", 0xF),
                # not used: nextInChain
            )
            c_color_states_list.append(c_color_state)
        c_color_states_array = ffi.new(
            "WGPUColorStateDescriptor []", c_color_states_list
        )
        c_depth_stencil_state = ffi.NULL
        if depth_stencil:
            assert (
                depth_stencil.get("format", None) is not None
            ), "depth_stencil needs format"
            stencil_front = depth_stencil.get("stencil_front", {})
            check_struct("StencilFaceState", stencil_front)
            # H: compare: WGPUCompareFunction/int, failOp: WGPUStencilOperation, depthFailOp: WGPUStencilOperation, passOp: WGPUStencilOperation
            c_stencil_front = new_struct(
                "WGPUStencilStateFaceDescriptor",
                compare=stencil_front.get("compare", "always"),
                failOp=stencil_front.get("fail_op", "keep"),
                depthFailOp=stencil_front.get("depth_fail_op", "keep"),
                passOp=stencil_front.get("pass_op", "keep"),
            )
            stencil_back = depth_stencil.get("stencil_back", {})
            check_struct("StencilFaceState", stencil_front)
            # H: compare: WGPUCompareFunction/int, failOp: WGPUStencilOperation, depthFailOp: WGPUStencilOperation, passOp: WGPUStencilOperation
            c_stencil_back = new_struct(
                "WGPUStencilStateFaceDescriptor",
                compare=stencil_back.get("compare", "always"),
                failOp=stencil_back.get("fail_op", "keep"),
                depthFailOp=stencil_back.get("depth_fail_op", "keep"),
                passOp=stencil_back.get("pass_op", "keep"),
            )
            # H: nextInChain: WGPUChainedStruct *, format: WGPUTextureFormat, depthWriteEnabled: bool, depthCompare: WGPUCompareFunction/int, stencilFront: WGPUStencilStateFaceDescriptor, stencilBack: WGPUStencilStateFaceDescriptor, stencilReadMask: int, stencilWriteMask: int
            c_depth_stencil_state = new_struct_p(
                "WGPUDepthStencilStateDescriptor *",
                format=depth_stencil["format"],
                depthWriteEnabled=bool(depth_stencil.get("depth_write_enabled", False)),
                depthCompare=depth_stencil.get("depth_compare", "always"),
                stencilFront=c_stencil_front,
                stencilBack=c_stencil_back,
                stencilReadMask=depth_stencil.get("stencil_read_mask", 0xFFFFFFFF),
                stencilWriteMask=depth_stencil.get("stencil_write_mask", 0xFFFFFFFF),
                # not used: nextInChain
            )
        c_vertex_buffer_descriptors_list = []
        for buffer_des in vertex["buffers"]:
            c_attributes_list = []
            for attribute in buffer_des["attributes"]:
                # H: format: WGPUVertexFormat, offset: int, shaderLocation: int
                c_attribute = new_struct(
                    "WGPUVertexAttributeDescriptor",
                    format=attribute["format"],
                    offset=attribute["offset"],
                    shaderLocation=attribute["shader_location"],
                )
                c_attributes_list.append(c_attribute)
            c_attributes_array = ffi.new(
                "WGPUVertexAttributeDescriptor []", c_attributes_list
            )
            # H: arrayStride: int, stepMode: WGPUInputStepMode, attributeCount: int, attributes: WGPUVertexAttributeDescriptor *
            c_vertex_buffer_descriptor = new_struct(
                "WGPUVertexBufferLayoutDescriptor",
                arrayStride=buffer_des["array_stride"],
                stepMode=buffer_des.get("step_mode", "vertex"),
                attributes=c_attributes_array,
                attributeCount=len(c_attributes_list),
            )
            c_vertex_buffer_descriptors_list.append(c_vertex_buffer_descriptor)
        c_vertex_buffer_descriptors_array = ffi.new(
            "WGPUVertexBufferLayoutDescriptor []", c_vertex_buffer_descriptors_list
        )
        # H: nextInChain: WGPUChainedStruct *, indexFormat: WGPUIndexFormat/int, vertexBufferCount: int, vertexBuffers: WGPUVertexBufferLayoutDescriptor *
        c_vertex_state = new_struct(
            "WGPUVertexStateDescriptor",
            indexFormat=primitive.get("strip_index_format", 0),
            vertexBuffers=c_vertex_buffer_descriptors_array,
            vertexBufferCount=len(c_vertex_buffer_descriptors_list),
            # not used: nextInChain
        )

        # H: nextInChain: WGPUChainedStruct *, label: WGPULabel, layout: WGPUOption_PipelineLayoutId/int, vertexStage: WGPUProgrammableStageDescriptor, fragmentStage: WGPUProgrammableStageDescriptor *, vertexState: WGPUVertexStateDescriptor, primitiveTopology: WGPUPrimitiveTopology, rasterizationState: WGPURasterizationStateDescriptor, sampleCount: int, depthStencilState: WGPUDepthStencilStateDescriptor *, colorStateCount: int, colorStates: WGPUColorStateDescriptor *, sampleMask: int, alphaToCoverageEnabled: bool
        struct = new_struct_p(
            "WGPURenderPipelineDescriptor *",
            label=to_c_label(label),
            layout=layout._internal,
            vertexStage=c_vertex_stage,
            fragmentStage=c_fragment_stage,
            primitiveTopology=primitive["topology"],
            rasterizationState=c_rasterization_state,
            colorStates=c_color_states_array,
            colorStateCount=len(c_color_states_list),
            depthStencilState=c_depth_stencil_state,
            vertexState=c_vertex_state,
            sampleCount=multisample.get("count", 1),
            sampleMask=multisample.get("mask", 0xFFFFFFFF),
            alphaToCoverageEnabled=multisample.get("alpha_to_coverage_enabled", False),
            # not used: nextInChain
        )

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

    def create_command_encoder(self, *, label="", measure_execution_time: bool = False):
        # H: label: WGPULabel
        struct = new_struct_p(
            "WGPUCommandEncoderDescriptor *",
            label=to_c_label(label),
        )

        # H: WGPUCommandEncoderId f(WGPUDeviceId device_id, const struct WGPUCommandEncoderDescriptor *desc)
        id = lib.wgpu_device_create_command_encoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

    # FIXME: new method to implement
    def create_render_bundle_encoder(
        self,
        *,
        label="",
        color_formats: "List[enums.TextureFormat]",
        depth_stencil_format: "enums.TextureFormat" = None,
        sample_count: int = 1,
    ):
        raise NotImplementedError()

    # FIXME: new method to implement
    def create_query_set(
        self,
        *,
        label="",
        type: "enums.QueryType",
        count: int,
        pipeline_statistics: "List[enums.PipelineStatisticName]" = [],
    ):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUDeviceId device_id)
            internal  # todo: crashes (last checked 2021-04)  lib.wgpu_device_destroy(internal)


class GPUBuffer(base.GPUBuffer, GPUObjectBase):
    def map_read(self):
        size = self.size

        # Prepare
        status = 99
        data = memoryview((ctypes.c_uint8 * size)()).cast("B")

        @ffi.callback("void(WGPUBufferMapAsyncStatus, uint8_t*)")
        def _map_read_callback(status_, user_data_p):
            nonlocal status
            status = status_

        # Map it
        # H: void f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferAddress size, WGPUBufferMapCallback callback, uint8_t *user_data)
        lib.wgpu_buffer_map_read_async(
            self._internal, 0, size, _map_read_callback, ffi.NULL
        )

        # Let it do some cycles
        # H: void f(WGPUDeviceId device_id, bool force_wait)
        lib.wgpu_device_poll(self._device._internal, True)

        if status != 0:  # no-cover
            raise RuntimeError(f"Could not read buffer data ({status}).")

        # Copy data
        # H: uint8_t *f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferSize size)
        src_ptr = lib.wgpu_buffer_get_mapped_range(self._internal, 0, size)
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

        @ffi.callback("void(WGPUBufferMapAsyncStatus, uint8_t*)")
        def _map_write_callback(status_, user_data_p):
            nonlocal status
            status = status_

        # Map it
        # H: void f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferAddress size, WGPUBufferMapCallback callback, uint8_t *user_data)
        lib.wgpu_buffer_map_write_async(
            self._internal, 0, size, _map_write_callback, ffi.NULL
        )

        # Let it do some cycles
        # H: void f(WGPUDeviceId device_id, bool force_wait)
        lib.wgpu_device_poll(self._device._internal, True)

        if status != 0:  # no-cover
            raise RuntimeError(f"Could not read buffer data ({status}).")

        # Copy data
        # H: uint8_t *f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferSize size)
        src_ptr = lib.wgpu_buffer_get_mapped_range(self._internal, 0, size)
        src_address = int(ffi.cast("intptr_t", src_ptr))
        src_m = get_memoryview_from_address(src_address, size)
        src_m[:] = data

        self._unmap()

    def _unmap(self):
        # H: void f(WGPUBufferId buffer_id)
        lib.wgpu_buffer_unmap(self._internal)

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBufferId buffer_id, bool now)
            lib.wgpu_buffer_destroy(internal, False)


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
        mip_level_count = base_mip_level or 0
        array_layer_count = array_layer_count or 0

        if format is None or dimension is None:
            if not (
                format is None
                and dimension is None
                and aspect == "all"
                and base_mip_level == 0
                and mip_level_count == 0
                and base_array_layer == 0
                and array_layer_count == 0
            ):
                raise ValueError(
                    "In create_view() if any parameter is given, "
                    + "both format and dimension must be specified."
                )
            # H: WGPUTextureViewId f(WGPUTextureId texture_id, const struct WGPUTextureViewDescriptor *desc)
            id = lib.wgpu_texture_create_view(self._internal, ffi.NULL)

        else:
            # H: label: WGPULabel, format: WGPUTextureFormat, dimension: WGPUTextureViewDimension, aspect: WGPUTextureAspect, base_mip_level: int, level_count: int, base_array_layer: int, array_layer_count: int
            struct = new_struct_p(
                "WGPUTextureViewDescriptor *",
                label=to_c_label(label),
                format=format,
                dimension=dimension,
                aspect=aspect or "all",
                base_mip_level=base_mip_level,
                level_count=mip_level_count,
                base_array_layer=base_array_layer,
                array_layer_count=array_layer_count,
            )
            # H: WGPUTextureViewId f(WGPUTextureId texture_id, const struct WGPUTextureViewDescriptor *desc)
            id = lib.wgpu_texture_create_view(self._internal, struct)

        return GPUTextureView(label, id, self._device, self, self.size)

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTextureId texture_id, bool now)
            lib.wgpu_texture_destroy(internal, False)


class GPUTextureView(base.GPUTextureView, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTextureViewId texture_view_id, bool now)
            internal  # todo: crashes (last checked 2021-04)  todoplib.wgpu_texture_view_destroy(internal, False)


class GPUSampler(base.GPUSampler, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUSamplerId sampler_id)
            lib.wgpu_sampler_destroy(internal)


class GPUBindGroupLayout(base.GPUBindGroupLayout, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroupLayoutId bind_group_layout_id)
            lib.wgpu_bind_group_layout_destroy(internal)


class GPUBindGroup(base.GPUBindGroup, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroupId bind_group_id)
            lib.wgpu_bind_group_destroy(internal)


class GPUPipelineLayout(base.GPUPipelineLayout, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUPipelineLayoutId pipeline_layout_id)
            lib.wgpu_pipeline_layout_destroy(internal)


class GPUShaderModule(base.GPUShaderModule, GPUObjectBase):
    def compilation_info(self):
        return super().compilation_info()

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUShaderModuleId shader_module_id)
            lib.wgpu_shader_module_destroy(internal)


class GPUPipelineBase(base.GPUPipelineBase):
    pass


class GPUComputePipeline(base.GPUComputePipeline, GPUPipelineBase, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUComputePipelineId compute_pipeline_id)
            lib.wgpu_compute_pipeline_destroy(internal)


class GPURenderPipeline(base.GPURenderPipeline, GPUPipelineBase, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderPipelineId render_pipeline_id)
            lib.wgpu_render_pipeline_destroy(internal)


class GPUCommandBuffer(base.GPUCommandBuffer, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUCommandBufferId command_buffer_id)
            internal  # todo: crashes (last checked 2021-04)  lib.wgpu_command_buffer_destroy(internal)


class GPUCommandEncoder(base.GPUCommandEncoder, GPUObjectBase):
    def begin_compute_pass(self, *, label=""):
        # H: label: WGPULabel
        struct = new_struct_p(
            "WGPUComputePassDescriptor *",
            label=to_c_label(label),
        )
        # H: struct WGPUComputePass *f(WGPUCommandEncoderId encoder_id, const struct WGPUComputePassDescriptor *desc)
        raw_pass = lib.wgpu_command_encoder_begin_compute_pass(self._internal, struct)
        return GPUComputePassEncoder(label, raw_pass, self)

    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "List[structs.RenderPassColorAttachment]",
        depth_stencil_attachment: "structs.RenderPassDepthStencilAttachment" = None,
        occlusion_query_set: "GPUQuerySet" = None,
    ):
        # Note that occlusion_query_set is ignored because wgpu-native does not have it.

        c_color_attachments_list = []
        for color_attachment in color_attachments:
            check_struct("RenderPassColorAttachment", color_attachment)
            assert isinstance(color_attachment["view"], GPUTextureView)
            texture_view_id = color_attachment["view"]._internal
            c_resolve_target = (
                0
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
            # H: load_op: WGPULoadOp, store_op: WGPUStoreOp, clear_value: WGPUColor, read_only: bool
            c_channel = new_struct(
                "WGPUPassChannel_Color",
                load_op=c_load_op,
                store_op=color_attachment.get("store_op", "store"),
                clear_value=c_clear_color,
                read_only=True,
            )
            # H: attachment: WGPUTextureViewId/int, resolve_target: WGPUOption_TextureViewId/int, channel: WGPUPassChannel_Color
            c_attachment = new_struct(
                "WGPUColorAttachmentDescriptor",
                attachment=texture_view_id,
                resolve_target=c_resolve_target,
                channel=c_channel,
            )
            c_color_attachments_list.append(c_attachment)
        c_color_attachments_array = ffi.new(
            "WGPUColorAttachmentDescriptor []", c_color_attachments_list
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
            # H: load_op: WGPULoadOp, store_op: WGPUStoreOp, clear_value: float, read_only: bool
            c_depth = new_struct(
                "WGPUPassChannel_f32",
                load_op=c_depth_load_op,
                store_op=depth_stencil_attachment["depth_store_op"],
                clear_value=float(c_depth_clear),
                # not used: read_only
            )
            # H: load_op: WGPULoadOp, store_op: WGPUStoreOp, clear_value: int, read_only: bool
            c_stencil = new_struct(
                "WGPUPassChannel_u32",
                load_op=c_stencil_load_op,
                store_op=depth_stencil_attachment["stencil_store_op"],
                clear_value=int(c_stencil_clear),
                # not used: read_only
            )
            # H: attachment: WGPUTextureViewId/int, depth: WGPUPassChannel_f32, stencil: WGPUPassChannel_u32
            c_depth_stencil_attachment = new_struct_p(
                "WGPUDepthStencilAttachmentDescriptor *",
                attachment=depth_stencil_attachment["view"]._internal,
                depth=c_depth,
                stencil=c_stencil,
            )

        # H: color_attachments: WGPUColorAttachmentDescriptor *, color_attachments_length: int, depth_stencil_attachment: WGPUDepthStencilAttachmentDescriptor *, label: WGPULabel
        struct = new_struct_p(
            "WGPURenderPassDescriptor *",
            label=to_c_label(label),
            color_attachments=c_color_attachments_array,
            color_attachments_length=len(c_color_attachments_list),
            depth_stencil_attachment=c_depth_stencil_attachment,
        )

        # H: struct WGPURenderPass *f(WGPUCommandEncoderId encoder_id, const struct WGPURenderPassDescriptor *desc)
        raw_pass = lib.wgpu_command_encoder_begin_render_pass(self._internal, struct)
        return GPURenderPassEncoder(label, raw_pass, self)

    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        assert source_offset % 4 == 0, "source_offsetmust be a multiple of 4"
        assert destination_offset % 4 == 0, "destination_offset must be a multiple of 4"
        assert size % 4 == 0, "size must be a multiple of 4"

        assert isinstance(source, GPUBuffer)
        assert isinstance(destination, GPUBuffer)
        # H: void f(WGPUCommandEncoderId command_encoder_id, WGPUBufferId source, WGPUBufferAddress source_offset, WGPUBufferId destination, WGPUBufferAddress destination_offset, WGPUBufferAddress size)
        lib.wgpu_command_encoder_copy_buffer_to_buffer(
            self._internal,
            source._internal,
            int(source_offset),
            destination._internal,
            int(destination_offset),
            int(size),
        )

    def copy_buffer_to_texture(self, source, destination, copy_size):
        row_alignment = lib.WGPUCOPY_BYTES_PER_ROW_ALIGNMENT
        bytes_per_row = int(source["bytes_per_row"])
        if (bytes_per_row % row_alignment) != 0:
            raise ValueError(
                f"bytes_per_row must ({bytes_per_row}) be a multiple of {row_alignment}"
            )

        c_source = new_struct_p(
            "WGPUBufferCopyView *",
            buffer=source["buffer"]._internal,
            # H: offset: WGPUBufferAddress/int, bytes_per_row: int, rows_per_image: int
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(source.get("offset", 0)),
                bytes_per_row=bytes_per_row,
                rows_per_image=int(source.get("rows_per_image", 0)),
            ),
        )

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3d",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTextureId/int, mip_level: int, origin: WGPUOrigin3d
        c_destination = new_struct_p(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=int(destination.get("mip_level", 0)),
            origin=c_origin,
        )

        size = _tuple_from_tuple_or_dict(
            copy_size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depth: int
        c_copy_size = new_struct_p(
            "WGPUExtent3d *",
            width=size[0],
            height=size[1],
            depth=size[2],
        )

        # H: void f(WGPUCommandEncoderId command_encoder_id, const struct WGPUBufferCopyView *source, const struct WGPUTextureCopyView *destination, const struct WGPUExtent3d *copy_size)
        lib.wgpu_command_encoder_copy_buffer_to_texture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_buffer(self, source, destination, copy_size):
        row_alignment = lib.WGPUCOPY_BYTES_PER_ROW_ALIGNMENT
        bytes_per_row = int(destination["bytes_per_row"])
        if (bytes_per_row % row_alignment) != 0:
            raise ValueError(
                f"bytes_per_row must ({bytes_per_row}) be a multiple of {row_alignment}"
            )

        ori = _tuple_from_tuple_or_dict(source.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3d",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTextureId/int, mip_level: int, origin: WGPUOrigin3d
        c_source = new_struct_p(
            "WGPUTextureCopyView *",
            texture=source["texture"]._internal,
            mip_level=int(source.get("mip_level", 0)),
            origin=c_origin,
        )

        c_destination = new_struct_p(
            "WGPUBufferCopyView *",
            buffer=destination["buffer"]._internal,
            # H: offset: WGPUBufferAddress/int, bytes_per_row: int, rows_per_image: int
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(destination.get("offset", 0)),
                bytes_per_row=bytes_per_row,
                rows_per_image=int(destination.get("rows_per_image", 0)),
            ),
        )

        size = _tuple_from_tuple_or_dict(
            copy_size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depth: int
        c_copy_size = new_struct_p(
            "WGPUExtent3d *",
            width=size[0],
            height=size[1],
            depth=size[2],
        )

        # H: void f(WGPUCommandEncoderId command_encoder_id, const struct WGPUTextureCopyView *source, const struct WGPUBufferCopyView *destination, const struct WGPUExtent3d *copy_size)
        lib.wgpu_command_encoder_copy_texture_to_buffer(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_texture(self, source, destination, copy_size):

        ori = _tuple_from_tuple_or_dict(source.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin1 = new_struct(
            "WGPUOrigin3d",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTextureId/int, mip_level: int, origin: WGPUOrigin3d
        c_source = new_struct_p(
            "WGPUTextureCopyView *",
            texture=source["texture"]._internal,
            mip_level=int(source.get("mip_level", 0)),
            origin=c_origin1,
        )

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin2 = new_struct(
            "WGPUOrigin3d",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTextureId/int, mip_level: int, origin: WGPUOrigin3d
        c_destination = new_struct_p(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=int(destination.get("mip_level", 0)),
            origin=c_origin2,
        )

        size = _tuple_from_tuple_or_dict(
            copy_size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depth: int
        c_copy_size = new_struct_p(
            "WGPUExtent3d *",
            width=size[0],
            height=size[1],
            depth=size[2],
        )

        # H: void f(WGPUCommandEncoderId command_encoder_id, const struct WGPUTextureCopyView *source, const struct WGPUTextureCopyView *destination, const struct WGPUExtent3d *copy_size)
        lib.wgpu_command_encoder_copy_texture_to_texture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def finish(self, *, label=""):
        # H: label: WGPULabel
        struct = new_struct_p(
            "WGPUCommandBufferDescriptor *",
            label=to_c_label(label),
        )
        # H: WGPUCommandBufferId f(WGPUCommandEncoderId encoder_id, const struct WGPUCommandBufferDescriptor *desc_base)
        id = lib.wgpu_command_encoder_finish(self._internal, struct)
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
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUCommandEncoderId command_encoder_id)
            internal  # todo: crashes (last checked 2021-04)  lib.wgpu_command_encoder_destroy(internal)


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
        c_offsets = ffi.new("WGPUDynamicOffset []", offsets)
        bind_group_id = bind_group._internal
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(struct WGPUComputePass *pass, uint32_t index, WGPUBindGroupId bind_group_id, const WGPUDynamicOffset *offsets, uintptr_t offset_length)
            lib.wgpu_compute_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )
        else:
            # H: void f(struct WGPURenderPass *pass, uint32_t index, WGPUBindGroupId bind_group_id, const WGPUDynamicOffset *offsets, uintptr_t offset_length)
            lib.wgpu_render_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )

    def push_debug_group(self, group_label):
        c_group_label = ffi.new("char []", group_label.encode())
        color = 0
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(struct WGPUComputePass *pass, WGPURawString label, uint32_t color)
            lib.wgpu_compute_pass_push_debug_group(self._internal, c_group_label, color)
        else:
            # H: void f(struct WGPURenderPass *pass, WGPURawString label, uint32_t color)
            lib.wgpu_render_pass_push_debug_group(self._internal, c_group_label, color)

    def pop_debug_group(self):
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(struct WGPUComputePass *pass)
            lib.wgpu_compute_pass_pop_debug_group(self._internal)
        else:
            # H: void f(struct WGPURenderPass *pass)
            lib.wgpu_render_pass_pop_debug_group(self._internal)

    def insert_debug_marker(self, marker_label):
        c_marker_label = ffi.new("char []", marker_label.encode())
        color = 0
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(struct WGPUComputePass *pass, WGPURawString label, uint32_t color)
            lib.wgpu_compute_pass_insert_debug_marker(
                self._internal, c_marker_label, color
            )
        else:
            # H: void f(struct WGPURenderPass *pass, WGPURawString label, uint32_t color)
            lib.wgpu_render_pass_insert_debug_marker(
                self._internal, c_marker_label, color
            )


class GPUComputePassEncoder(
    base.GPUComputePassEncoder, GPUProgrammablePassEncoder, GPUObjectBase
):
    """ """

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(struct WGPUComputePass *pass, WGPUComputePipelineId pipeline_id)
        lib.wgpu_compute_pass_set_pipeline(self._internal, pipeline_id)

    def dispatch(self, x, y=1, z=1):
        # H: void f(struct WGPUComputePass *pass, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
        lib.wgpu_compute_pass_dispatch(self._internal, x, y, z)

    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(struct WGPUComputePass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset)
        lib.wgpu_compute_pass_dispatch_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def end_pass(self):
        # H: void f(struct WGPUComputePass *pass)
        lib.wgpu_compute_pass_end_pass(self._internal)

    # FIXME: new method to implement
    def write_timestamp(self, query_set, query_index):
        raise NotImplementedError()

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(struct WGPUComputePass *pass)
            internal  # todo: crashes (last checked 2021-04) lib.wgpu_compute_pass_destroy(internal)


class GPURenderEncoderBase(base.GPURenderEncoderBase):
    """ """

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(struct WGPURenderPass *pass, WGPURenderPipelineId pipeline_id)
        lib.wgpu_render_pass_set_pipeline(self._internal, pipeline_id)

    def set_index_buffer(self, buffer, index_format, offset=0, size=0):
        if not size:
            size = buffer.size - offset
        c_index_format = enummap[f"IndexFormat.{index_format}"]
        # H: void f(struct WGPURenderPass *pass, WGPUBufferId buffer_id, WGPUIndexFormat index_format, WGPUBufferAddress offset, WGPUOption_BufferSize size)
        lib.wgpu_render_pass_set_index_buffer(
            self._internal, buffer._internal, c_index_format, int(offset), int(size)
        )

    def set_vertex_buffer(self, slot, buffer, offset=0, size=0):
        if not size:
            size = buffer.size - offset
        # H: void f(struct WGPURenderPass *pass, uint32_t slot, WGPUBufferId buffer_id, WGPUBufferAddress offset, WGPUOption_BufferSize size)
        lib.wgpu_render_pass_set_vertex_buffer(
            self._internal, int(slot), buffer._internal, int(offset), int(size)
        )

    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        # H: void f(struct WGPURenderPass *pass, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
        lib.wgpu_render_pass_draw(
            self._internal, vertex_count, instance_count, first_vertex, first_instance
        )

    def draw_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(struct WGPURenderPass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset)
        lib.wgpu_render_pass_draw_indirect(
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
        # H: void f(struct WGPURenderPass *pass, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t base_vertex, uint32_t first_instance)
        lib.wgpu_render_pass_draw_indexed(
            self._internal,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )

    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(struct WGPURenderPass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset)
        lib.wgpu_render_pass_draw_indexed_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(struct WGPURenderPass *pass)
            internal  # todo: crashes (last checked 2021-04-19) lib.wgpu_render_pass_destroy(internal)


class GPURenderPassEncoder(
    base.GPURenderPassEncoder,
    GPUProgrammablePassEncoder,
    GPURenderEncoderBase,
    GPUObjectBase,
):
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        # H: void f(struct WGPURenderPass *pass, float x, float y, float w, float h, float depth_min, float depth_max)
        lib.wgpu_render_pass_set_viewport(
            self._internal,
            float(x),
            float(y),
            float(width),
            float(height),
            float(min_depth),
            float(max_depth),
        )

    def set_scissor_rect(self, x, y, width, height):
        # H: void f(struct WGPURenderPass *pass, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
        lib.wgpu_render_pass_set_scissor_rect(
            self._internal, int(x), int(y), int(width), int(height)
        )

    def set_blend_color(self, color):
        color = _tuple_from_tuple_or_dict(color, "rgba")
        # H: r: float, g: float, b: float, a: float
        c_color = new_struct_p(
            "WGPUColor *",
            r=color[0],
            g=color[1],
            b=color[2],
            a=color[3],
        )
        # H: void f(struct WGPURenderPass *pass, const struct WGPUColor *color)
        lib.wgpu_render_pass_set_blend_color(self._internal, c_color)

    def set_stencil_reference(self, reference):
        # H: void f(struct WGPURenderPass *pass, uint32_t value)
        lib.wgpu_render_pass_set_stencil_reference(self._internal, int(reference))

    def end_pass(self):
        # H: void f(struct WGPURenderPass *pass)
        lib.wgpu_render_pass_end_pass(self._internal)

    # FIXME: new method to implement
    def execute_bundles(self, bundles):
        raise NotImplementedError()

    # FIXME: new method to implement
    def begin_occlusion_query(self, query_index):
        raise NotImplementedError()

    # FIXME: new method to implement
    def end_occlusion_query(self):
        raise NotImplementedError()

    # FIXME: new method to implement
    def write_timestamp(self, query_set, query_index):
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
        c_command_buffers = ffi.new("WGPUCommandBufferId []", command_buffer_ids)
        # H: void f(WGPUQueueId queue_id, const WGPUCommandBufferId *command_buffers, uintptr_t command_buffers_length)
        lib.wgpu_queue_submit(
            self._internal, c_command_buffers, len(command_buffer_ids)
        )

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
        # H: void f(WGPUQueueId queue_id, WGPUBufferId buffer_id, WGPUBufferAddress buffer_offset, const uint8_t *data, uintptr_t data_length)
        lib.wgpu_queue_write_buffer(
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

        m, address = get_memoryview_and_address(data)
        # todo: could we not derive the size from the shape of m?

        c_data = ffi.cast("uint8_t *", address)
        data_length = m.nbytes

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        # H: x: int, y: int, z: int
        c_origin = new_struct(
            "WGPUOrigin3d",
            x=ori[0],
            y=ori[1],
            z=ori[2],
        )
        # H: texture: WGPUTextureId/int, mip_level: int, origin: WGPUOrigin3d
        c_destination = new_struct_p(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=destination.get("mip_level", 0),
            origin=c_origin,
        )

        # H: offset: WGPUBufferAddress/int, bytes_per_row: int, rows_per_image: int
        c_data_layout = new_struct_p(
            "WGPUTextureDataLayout *",
            offset=data_layout.get("offset", 0),
            bytes_per_row=data_layout["bytes_per_row"],
            rows_per_image=data_layout.get("rows_per_image", 0),
        )

        size = _tuple_from_tuple_or_dict(
            size, ("width", "height", "depth_or_array_layers")
        )
        # H: width: int, height: int, depth: int
        c_size = new_struct_p(
            "WGPUExtent3d *",
            width=size[0],
            height=size[1],
            depth=size[2],
        )

        # H: void f(WGPUQueueId queue_id, const struct WGPUTextureCopyView *texture, const uint8_t *data, uintptr_t data_length, const struct WGPUTextureDataLayout *data_layout, const struct WGPUExtent3d *size)
        lib.wgpu_queue_write_texture(
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

    # FIXME: new method to implement -> does not exist in wgpu-native
    def copy_image_bitmap_to_texture(self, source, destination, copy_size):
        raise NotImplementedError()

    # FIXME: new method to implement
    def on_submitted_work_done(self):
        raise NotImplementedError()


class GPUSwapChain(base.GPUSwapChain, GPUObjectBase):
    def __init__(self, label, internal, device, canvas, format, usage):
        super().__init__(label, internal, device, canvas, format, usage)
        assert internal is None  # we set it later
        self._surface_size = (-1, -1)
        self._surface_id = None
        self._create_native_swap_chain_if_needed()

    def _create_native_swap_chain_if_needed(self):
        canvas = self._canvas
        psize = canvas.get_physical_size()
        if psize == self._surface_size:
            return
        self._surface_size = psize

        # logger.info(str((psize, canvas.get_logical_size(), canvas.get_pixel_ratio())))

        # H: usage: WGPUTextureUsage/int, format: WGPUTextureFormat, width: int, height: int, present_mode: WGPUPresentMode
        struct = new_struct_p(
            "WGPUSwapChainDescriptor *",
            usage=self._usage,
            format=self._format,
            width=max(1, psize[0]),
            height=max(1, psize[1]),
            present_mode=1,
        )
        # present_mode -> 0: Immediate, 1: Mailbox, 2: Fifo

        if self._surface_id is None:
            self._surface_id = get_surface_id_from_canvas(canvas)

        # H: WGPUSwapChainId f(WGPUDeviceId device_id, WGPUSurfaceId surface_id, const struct WGPUSwapChainDescriptor *desc)
        self._internal = lib.wgpu_device_create_swap_chain(
            self._device._internal, self._surface_id, struct
        )

    def __enter__(self):
        # Get the current texture view, and make sure it is presented when done
        self._create_native_swap_chain_if_needed()
        # H: WGPUOption_TextureViewId f(WGPUSwapChainId swap_chain_id)
        view_id = lib.wgpu_swap_chain_get_current_texture_view(self._internal)
        size = self._surface_size[0], self._surface_size[1], 1
        return GPUTextureView("swap_chain", view_id, self._device, None, size)

    def __exit__(self, type, value, tb):
        # Present the current texture
        # H: enum WGPUSwapChainStatus f(WGPUSwapChainId swap_chain_id)
        status = lib.wgpu_swap_chain_present(self._internal)
        if status == lib.WGPUSwapChainStatus_Good:
            pass
        elif status == lib.WGPUSwapChainStatus_Suboptimal:  # no-cover
            if not getattr(self, "_warned_swap_chain_suboptimal", False):
                logger.warning(f"Swap chain status of {self} is suboptimal")
                self._warned_swap_chain_suboptimal = True
        else:  # no-cover
            status_str = swap_chain_status_map.get(status, "")
            raise RuntimeError(
                f"Swap chain status is not good: {status_str} ({status})"
            )


class GPURenderBundle(base.GPURenderBundle, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURenderBundleId render_bundle_id)
            lib.wgpu_render_bundle_destroy(internal)


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

    # FIXME: new method to implement
    def destroy(self):
        raise NotImplementedError()


class GPUUncapturedErrorEvent(base.GPUUncapturedErrorEvent):
    pass


# %%


def _copy_docstrings():
    for ob in globals().values():
        if not (isinstance(ob, type) and issubclass(ob, GPUObjectBase)):
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
