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
from typing import List, Union

from .. import base, flags, enums, structs
from .. import _register_backend
from .._coreutils import ApiDiff

from .rs_ffi import ffi, lib, check_expected_version
from .rs_mappings import cstructfield2enum, enummap
from .rs_helpers import (
    get_surface_id_from_canvas,
    get_memoryview_from_address,
    get_memoryview_and_address,
)


logger = logging.getLogger("wgpu")  # noqa
apidiff = ApiDiff()

# The wgpu-native version that we target/expect
__version__ = "0.5.2"
__commit_sha__ = "160be433dbec0fc7a27d25f2aba3423666ccfa10"
version_info = tuple(map(int, __version__.split(".")))
check_expected_version(version_info)  # produces a warning on mismatch


# %% Helper functions and objects

swap_chain_status_map = {
    getattr(lib, "WGPUSwapChainStatus_" + x): x
    for x in ("Good", "Suboptimal", "Lost", "Outdated", "OutOfMemory", "Timeout")
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
            res = []
            for key in fields:
                if "_or_" in key:  # Convenience for e.g. depth_or_array_layers
                    keys = key.split("_or_")
                    keys.append(key)
                    for key in keys:
                        if key in fields:
                            break
                res.append(ob[key])
            return tuple(res)  # tuple(ob[key] for key in fields)
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

        @ffi.callback("void(unsigned long long, void *)")
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
        features = set()
        return GPUAdapter("WGPU", features, adapter_id)

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
        non_guaranteed_limits: "structs.Limits" = {},
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
        extensions = features

        c_trace_path = ffi.NULL
        if trace_path:  # no-cover
            c_trace_path = ffi.new("char []", trace_path.encode())

        # Handle default limits
        limits2 = base.DEFAULT_LIMITS.copy()
        limits2.update(limits or {})

        # H: anisotropic_filtering: bool
        c_extensions = new_struct(
            "WGPUExtensions",
            anisotropic_filtering="anisotropic_filtering" in extensions,
        )
        # H: max_bind_groups: int
        c_limits = new_struct(
            "WGPULimits",
            max_bind_groups=limits2["max_bind_groups"],
        )
        # H: extensions: WGPUExtensions, limits: WGPULimits
        struct = new_struct_p(
            "WGPUDeviceDescriptor *",
            extensions=c_extensions,
            limits=c_limits,
        )
        # H: WGPUDeviceId f(WGPUAdapterId adapter_id, const WGPUDeviceDescriptor *desc, const char *trace_path)
        device_id = lib.wgpu_adapter_request_device(
            self._internal, struct, c_trace_path
        )

        # Get the actual limits reported by the device
        # H: max_bind_groups: int
        c_limits = new_struct_p(
            "WGPULimits *",
            # not used: max_bind_groups
        )
        # H: void f(WGPUDeviceId _device_id, WGPULimits *limits)
        lib.wgpu_device_get_limits(device_id, c_limits)
        limits3 = {key: getattr(c_limits, key) for key in dir(c_limits)}

        # Get the queue to which commands can be submitted
        # H: WGPUQueueId f(WGPUDeviceId device_id)
        queue_id = lib.wgpu_device_get_default_queue(device_id)
        queue = GPUQueue("", queue_id, None)

        return GPUDevice(label, device_id, self, extensions, limits3, queue)

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
            self._internal, id = None, self._internal
            # H: void f(WGPUAdapterId adapter_id)
            lib.wgpu_adapter_destroy(id)


class GPUDevice(base.GPUDevice, GPUObjectBase):

    # FIXME: new method
    def _destroy(self):
        pass

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
        # Create a buffer object
        c_label = ffi.new("char []", label.encode())
        # H: label: WGPULabel, size: WGPUBufferAddress/int, usage: WGPUBufferUsage/int
        struct = new_struct_p(
            "WGPUBufferDescriptor *",
            label=c_label,
            size=size,
            usage=usage,
        )
        # H: WGPUBufferId f(WGPUDeviceId device_id, const WGPUBufferDescriptor *desc)
        id = lib.wgpu_device_create_buffer(self._internal, struct)
        # Return wrapped buffer
        return GPUBuffer(label, id, self, size, usage, "unmapped")

    def create_buffer_with_data(self, *, label="", data, usage: "flags.BufferUsage"):
        # Get a memoryview of the data
        m, src_address = get_memoryview_and_address(data)
        if not m.contiguous:  # no-cover
            raise ValueError("The given texture data is not contiguous")
        m = m.cast("B", shape=(m.nbytes,))
        # Create a buffer object, and get a memory pointer to its mapped memory
        c_label = ffi.new("char []", label.encode())
        # H: label: WGPULabel, size: WGPUBufferAddress/int, usage: WGPUBufferUsage/int
        struct = new_struct_p(
            "WGPUBufferDescriptor *",
            label=c_label,
            size=m.nbytes,
            usage=usage,
        )
        buffer_memory_pointer = ffi.new("uint8_t * *")
        # H: WGPUBufferId f(WGPUDeviceId device_id, const WGPUBufferDescriptor *desc, uint8_t **mapped_ptr_out)
        id = lib.wgpu_device_create_buffer_mapped(
            self._internal, struct, buffer_memory_pointer
        )
        # Copy the data to the mapped memory
        dst_address = int(ffi.cast("intptr_t", buffer_memory_pointer[0]))
        dst_m = get_memoryview_from_address(dst_address, m.nbytes)
        dst_m[:] = m  # nicer than ctypes.memmove(dst_address, src_address, m.nbytes)
        # H: void f(WGPUBufferId buffer_id)
        lib.wgpu_buffer_unmap(id)
        # Return the wrapped buffer
        return GPUBuffer(label, id, self, m.nbytes, usage, "unmapped")

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
        c_label = ffi.new("char []", label.encode())
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
            label=c_label,
            size=c_size,
            mip_level_count=mip_level_count,
            sample_count=sample_count,
            dimension=dimension,
            format=format,
            usage=usage,
        )
        # H: WGPUTextureId f(WGPUDeviceId device_id, const WGPUTextureDescriptor *desc)
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
        c_label = ffi.new("char []", label.encode())
        # H: label: WGPULabel, address_mode_u: WGPUAddressMode, address_mode_v: WGPUAddressMode, address_mode_w: WGPUAddressMode, mag_filter: WGPUFilterMode, min_filter: WGPUFilterMode, mipmap_filter: WGPUFilterMode, lod_min_clamp: float, lod_max_clamp: float, compare: WGPUCompareFunction
        struct = new_struct_p(
            "WGPUSamplerDescriptor *",
            label=c_label,
            address_mode_u=address_mode_u,
            address_mode_v=address_mode_v,
            mag_filter=mag_filter,
            min_filter=min_filter,
            mipmap_filter=mipmap_filter,
            lod_min_clamp=lod_min_clamp,
            lod_max_clamp=lod_max_clamp,
            compare=0 if compare is None else compare,
            # not used: address_mode_w
        )
        # max_anisotropy not yet supported by wgpu-native

        # H: WGPUSamplerId f(WGPUDeviceId device_id, const WGPUSamplerDescriptor *desc)
        id = lib.wgpu_device_create_sampler(self._internal, struct)
        return GPUSampler(label, id, self)

    def create_bind_group_layout(
        self, *, label="", entries: "List[structs.BindGroupLayoutEntry]"
    ):
        c_entries_list = []
        for entry in entries:
            c_has_dynamic_offset = False
            c_view_dimension = 0
            c_texture_component_type = 0
            c_multisampled = False
            c_storage_texture_format = 0
            if entry.get("buffer"):
                info = entry["buffer"]
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
            # H: binding: int, visibility: WGPUShaderStage/int, ty: WGPUBindingType, multisampled: bool, has_dynamic_offset: bool, view_dimension: WGPUTextureViewDimension, texture_component_type: WGPUTextureComponentType, storage_texture_format: WGPUTextureFormat
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
            )
            c_entries_list.append(c_entry)

        c_label = ffi.new("char []", label.encode())
        # H: label: const char, entries: WGPUBindGroupLayoutEntry *, entries_length: int
        struct = new_struct_p(
            "WGPUBindGroupLayoutDescriptor *",
            label=c_label,
            entries=ffi.new("WGPUBindGroupLayoutEntry []", c_entries_list),
            entries_length=len(c_entries_list),
        )

        # H: WGPUBindGroupLayoutId f(WGPUDeviceId device_id, const WGPUBindGroupLayoutDescriptor *desc)
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
            # The resource can be a sampler, texture view, or buffer descriptor
            resource = entry["resource"]
            if isinstance(resource, GPUSampler):
                c_resource_kwargs = {
                    "tag": 1,  # WGPUBindingResource_Tag.WGPUBindingResource_Sampler
                    # H: _0: WGPUSamplerId/int
                    "sampler": new_struct(
                        "WGPUBindingResource_WGPUSampler_Body",
                        _0=resource._internal,
                    ),
                }
            elif isinstance(resource, GPUTextureView):
                c_resource_kwargs = {
                    "tag": 2,  # WGPUBindingResource_Tag.WGPUBindingResource_TextureView
                    # H: _0: WGPUTextureViewId/int
                    "texture_view": new_struct(
                        "WGPUBindingResource_WGPUTextureView_Body",
                        _0=resource._internal,
                    ),
                }
            elif isinstance(resource, dict):  # Buffer binding
                # H: buffer: WGPUBufferId/int, offset: WGPUBufferAddress/int, size: WGPUBufferAddress/int
                c_buffer_entry = new_struct(
                    "WGPUBufferBinding",
                    buffer=resource["buffer"]._internal,
                    offset=resource["offset"],
                    size=resource["size"],
                )
                c_resource_kwargs = {
                    "tag": 0,  # WGPUBindingResource_Tag.WGPUBindingResource_Buffer
                    # H: _0: WGPUBufferBinding
                    "buffer": new_struct(
                        "WGPUBindingResource_WGPUBuffer_Body",
                        _0=c_buffer_entry,
                    ),
                }
            else:
                raise TypeError(f"Unexpected resource type {type(resource)}")
            # Instantiate without write new_struct(), to disable annotation here
            f = new_struct
            c_resource = f("WGPUBindingResource", **c_resource_kwargs)
            # H: binding: int, resource: WGPUBindingResource
            c_entry = new_struct(
                "WGPUBindGroupEntry",
                binding=int(entry["binding"]),
                resource=c_resource,
            )
            c_entries_list.append(c_entry)

        c_label = ffi.new("char []", label.encode())
        c_entries_array = ffi.new("WGPUBindGroupEntry []", c_entries_list)
        # H: label: const char, layout: WGPUBindGroupLayoutId/int, entries: WGPUBindGroupEntry *, entries_length: int
        struct = new_struct_p(
            "WGPUBindGroupDescriptor *",
            label=c_label,
            layout=layout._internal,
            entries=c_entries_array,
            entries_length=len(c_entries_list),
        )

        # H: WGPUBindGroupId f(WGPUDeviceId device_id, const WGPUBindGroupDescriptor *desc)
        id = lib.wgpu_device_create_bind_group(self._internal, struct)
        return GPUBindGroup(label, id, self, entries)

    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "List[GPUBindGroupLayout]"
    ):

        bind_group_layouts_ids = [x._internal for x in bind_group_layouts]

        c_layout_array = ffi.new("WGPUBindGroupLayoutId []", bind_group_layouts_ids)
        # H: bind_group_layouts: const WGPUBindGroupLayoutId, bind_group_layouts_length: int
        struct = new_struct_p(
            "WGPUPipelineLayoutDescriptor *",
            bind_group_layouts=c_layout_array,
            bind_group_layouts_length=len(bind_group_layouts),
        )

        # H: WGPUPipelineLayoutId f(WGPUDeviceId device_id, const WGPUPipelineLayoutDescriptor *desc)
        id = lib.wgpu_device_create_pipeline_layout(self._internal, struct)
        return GPUPipelineLayout(label, id, self, bind_group_layouts)

    def create_shader_module(self, *, label="", code: str, source_map: dict = None):

        if isinstance(code, bytes):
            data = code  # Assume it's Spirv
        elif hasattr(code, "to_bytes"):
            data = code.to_bytes()
        elif hasattr(code, "to_spirv"):
            data = code.to_spirv()
        else:
            raise TypeError("Need bytes or ob with ob.to_spirv() for shader.")

        magic_nr = b"\x03\x02#\x07"  # 0x7230203
        if data[:4] != magic_nr:
            raise ValueError("Given shader data does not look like a SpirV module")

        # From bytes to WGPUU32Array
        data_u8 = ffi.new("uint8_t[]", data)
        data_u32 = ffi.cast("uint32_t *", data_u8)
        c_code = ffi.new(
            "WGPUU32Array *", {"bytes": data_u32, "length": len(data) // 4}
        )[0]

        # H: code: WGPUU32Array
        struct = new_struct_p(
            "WGPUShaderModuleDescriptor *",
            code=c_code,
        )

        # H: WGPUShaderModuleId f(WGPUDeviceId device_id, const WGPUShaderModuleDescriptor *desc)
        id = lib.wgpu_device_create_shader_module(self._internal, struct)
        return GPUShaderModule(label, id, self)

    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        compute: "structs.ProgrammableStage",
    ):

        # H: module: WGPUShaderModuleId/int, entry_point: WGPURawString
        c_compute_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=compute["module"]._internal,
            entry_point=ffi.new("char []", compute["entry_point"].encode()),
        )

        # H: layout: WGPUPipelineLayoutId/int, compute_stage: WGPUProgrammableStageDescriptor
        struct = new_struct_p(
            "WGPUComputePipelineDescriptor *",
            layout=layout._internal,
            compute_stage=c_compute_stage,
        )

        # H: WGPUComputePipelineId f(WGPUDeviceId device_id, const WGPUComputePipelineDescriptor *desc)
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

    # FIXME: was create_render_pipeline(
    #     self, *, label="",
    #     layout: "GPUPipelineLayout" = None,
    #     vertex_stage: "structs.ProgrammableStageDescriptor",
    #     fragment_stage: "structs.ProgrammableStageDescriptor" = None,
    #     primitive_topology: "enums.PrimitiveTopology",
    #     rasterization_state: "structs.RasterizationStateDescriptor" = {},
    #     color_states: "List[structs.ColorStateDescriptor]",
    #     depth_stencil_state: "structs.DepthStencilStateDescriptor" = None,
    #     vertex_state: "structs.VertexStateDescriptor" = {},
    #     sample_count: int = 1,
    #     sample_mask: int = 0xFFFFFFFF,
    #     alpha_to_coverage_enabled: bool = False
    # ):
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

        # Little helper, remove after june 2021 or so
        assert "stencil_front" not in depth_stencil, "stencil_front -> front"

        # H: module: WGPUShaderModuleId/int, entry_point: WGPURawString
        c_vertex_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=vertex["module"]._internal,
            entry_point=ffi.new("char []", vertex["entry_point"].encode()),
        )
        c_fragment_stage = ffi.NULL
        if fragment is not None:
            # H: module: WGPUShaderModuleId/int, entry_point: WGPURawString
            c_fragment_stage = new_struct_p(
                "WGPUProgrammableStageDescriptor *",
                module=fragment["module"]._internal,
                entry_point=ffi.new("char []", fragment["entry_point"].encode()),
            )
        # H: front_face: WGPUFrontFace, cull_mode: WGPUCullMode, depth_bias: int, depth_bias_slope_scale: float, depth_bias_clamp: float
        c_rasterization_state = new_struct_p(
            "WGPURasterizationStateDescriptor *",
            front_face=primitive.get("front_face", "ccw"),
            cull_mode=primitive.get("cull_mode", "none"),
            depth_bias=depth_stencil.get("depth_bias", 0),
            depth_bias_slope_scale=depth_stencil.get("depth_bias_slope_scale", 0),
            depth_bias_clamp=depth_stencil.get("depth_bias_clamp", 0),
        )
        c_color_states_list = []
        for target in fragment["targets"]:
            alpha_blend = _tuple_from_tuple_or_dict(
                target["blend"]["alpha"],
                ("src_factor", "dst_factor", "operation"),
            )
            # H: src_factor: WGPUBlendFactor, dst_factor: WGPUBlendFactor, operation: WGPUBlendOperation
            c_alpha_blend = new_struct(
                "WGPUBlendDescriptor",
                src_factor=alpha_blend[0],
                dst_factor=alpha_blend[1],
                operation=alpha_blend[2],
            )
            color_blend = _tuple_from_tuple_or_dict(
                target["blend"]["color"],
                ("src_factor", "dst_factor", "operation"),
            )
            # H: src_factor: WGPUBlendFactor, dst_factor: WGPUBlendFactor, operation: WGPUBlendOperation
            c_color_blend = new_struct(
                "WGPUBlendDescriptor",
                src_factor=color_blend[0],
                dst_factor=color_blend[1],
                operation=color_blend[2],
            )
            # H: format: WGPUTextureFormat, alpha_blend: WGPUBlendDescriptor, color_blend: WGPUBlendDescriptor, write_mask: WGPUColorWrite/int
            c_color_state = new_struct(
                "WGPUColorStateDescriptor",
                format=target["format"],
                alpha_blend=c_alpha_blend,
                color_blend=c_color_blend,
                write_mask=target.get("write_mask", 0xF),
            )
            c_color_states_list.append(c_color_state)
        c_color_states_array = ffi.new(
            "WGPUColorStateDescriptor []", c_color_states_list
        )
        if depth_stencil.get("front", None) is None:
            c_depth_stencil_state = ffi.NULL
        else:
            stencil_front = depth_stencil.get("front", {})
            # H: compare: WGPUCompareFunction, fail_op: WGPUStencilOperation, depth_fail_op: WGPUStencilOperation, pass_op: WGPUStencilOperation
            c_stencil_front = new_struct(
                "WGPUStencilStateFaceDescriptor",
                compare=stencil_front.get("compare", "always"),
                fail_op=stencil_front.get("fail_op", "keep"),
                depth_fail_op=stencil_front.get("depth_fail_op", "keep"),
                pass_op=stencil_front.get("pass_op", "keep"),
            )
            stencil_back = depth_stencil.get("back", {})
            # H: compare: WGPUCompareFunction, fail_op: WGPUStencilOperation, depth_fail_op: WGPUStencilOperation, pass_op: WGPUStencilOperation
            c_stencil_back = new_struct(
                "WGPUStencilStateFaceDescriptor",
                compare=stencil_back.get("compare", "always"),
                fail_op=stencil_back.get("fail_op", "keep"),
                depth_fail_op=stencil_back.get("depth_fail_op", "keep"),
                pass_op=stencil_back.get("pass_op", "keep"),
            )
            # H: format: WGPUTextureFormat, depth_write_enabled: bool, depth_compare: WGPUCompareFunction, stencil_front: WGPUStencilStateFaceDescriptor, stencil_back: WGPUStencilStateFaceDescriptor, stencil_read_mask: int, stencil_write_mask: int
            c_depth_stencil_state = new_struct_p(
                "WGPUDepthStencilStateDescriptor *",
                format=depth_stencil["format"],
                depth_write_enabled=bool(
                    depth_stencil.get("depth_write_enabled", False)
                ),
                depth_compare=depth_stencil.get("depth_compare", "always"),
                stencil_front=c_stencil_front,
                stencil_back=c_stencil_back,
                stencil_read_mask=depth_stencil.get("stencil_read_mask", 0xFFFFFFFF),
                stencil_write_mask=depth_stencil.get("stencil_write_mask", 0xFFFFFFFF),
            )
        c_vertex_buffer_descriptors_list = []
        for buffer_des in vertex["buffers"]:
            c_attributes_list = []
            for attribute in buffer_des["attributes"]:
                # H: offset: WGPUBufferAddress/int, format: WGPUVertexFormat, shader_location: WGPUShaderLocation/int
                c_attribute = new_struct(
                    "WGPUVertexAttributeDescriptor",
                    format=attribute["format"],
                    offset=attribute["offset"],
                    shader_location=attribute["shader_location"],
                )
                c_attributes_list.append(c_attribute)
            c_attributes_array = ffi.new(
                "WGPUVertexAttributeDescriptor []", c_attributes_list
            )
            # H: array_stride: WGPUBufferAddress/int, step_mode: WGPUInputStepMode, attributes: WGPUVertexAttributeDescriptor *, attributes_length: int
            c_vertex_buffer_descriptor = new_struct(
                "WGPUVertexBufferLayoutDescriptor",
                array_stride=buffer_des["array_stride"],
                step_mode=buffer_des.get("step_mode", "vertex"),
                attributes=c_attributes_array,
                attributes_length=len(c_attributes_list),
            )
            c_vertex_buffer_descriptors_list.append(c_vertex_buffer_descriptor)
        c_vertex_buffer_descriptors_array = ffi.new(
            "WGPUVertexBufferLayoutDescriptor []", c_vertex_buffer_descriptors_list
        )
        # H: index_format: WGPUIndexFormat, vertex_buffers: WGPUVertexBufferLayoutDescriptor *, vertex_buffers_length: int
        c_vertex_state = new_struct(
            "WGPUVertexStateDescriptor",
            index_format=primitive.get("strip_index_format", "uint32"),
            vertex_buffers=c_vertex_buffer_descriptors_array,
            vertex_buffers_length=len(c_vertex_buffer_descriptors_list),
        )

        # H: layout: WGPUPipelineLayoutId/int, vertex_stage: WGPUProgrammableStageDescriptor, fragment_stage: WGPUProgrammableStageDescriptor *, primitive_topology: WGPUPrimitiveTopology, rasterization_state: WGPURasterizationStateDescriptor *, color_states: WGPUColorStateDescriptor *, color_states_length: int, depth_stencil_state: WGPUDepthStencilStateDescriptor *, vertex_state: WGPUVertexStateDescriptor, sample_count: int, sample_mask: int, alpha_to_coverage_enabled: bool
        struct = new_struct_p(
            "WGPURenderPipelineDescriptor *",
            layout=layout._internal,
            vertex_stage=c_vertex_stage,
            fragment_stage=c_fragment_stage,
            primitive_topology=primitive["topology"],
            rasterization_state=c_rasterization_state,
            color_states=c_color_states_array,
            color_states_length=len(c_color_states_list),
            depth_stencil_state=c_depth_stencil_state,
            vertex_state=c_vertex_state,
            sample_count=multisample["count"],
            sample_mask=multisample["mask"],
            alpha_to_coverage_enabled=multisample["alpha_to_coverage_enabled"],
        )

        # H: WGPURenderPipelineId f(WGPUDeviceId device_id, const WGPURenderPipelineDescriptor *desc)
        id = lib.wgpu_device_create_render_pipeline(self._internal, struct)
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
        c_label = ffi.new("char []", label.encode())
        # H: label: const char
        struct = new_struct_p(
            "WGPUCommandEncoderDescriptor *",
            label=c_label,
        )

        # H: WGPUCommandEncoderId f(WGPUDeviceId device_id, const WGPUCommandEncoderDescriptor *desc)
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


class GPUBuffer(base.GPUBuffer, GPUObjectBase):

    # def map(self, mode, offset=0, size=0):
    #     if not size:
    #         size = self.size - offset
    #     if not (offset == 0 and size == self.size):  # no-cover
    #         raise ValueError(
    #             "Cannot (yet) map buffers with nonzero offset and non-full size."
    #         )
    #
    #     if mode == flags.MapMode.READ:
    #         return self._map_read()
    #     elif mode == flags.MapMode.WRITE:
    #         return self._map_write()
    #     else:  # no-cover
    #         raise ValueError(f"Invalid MapMode flag: {mode}")

    def read_data(self, offset=0, size=0):
        if not size:
            size = self.size - offset
        assert 0 <= offset < self.size
        assert 0 <= size <= (self.size - offset)

        mapped_mem = self._map_read(offset, size)
        new_mem = memoryview((ctypes.c_uint8 * mapped_mem.nbytes)()).cast("B")
        new_mem[:] = mapped_mem
        self._unmap()
        return new_mem

    async def read_data_async(self, offset=0, size=0):
        return self.read_data(offset, size)

    def write_data(self, data, offset=0):
        m = memoryview(data).cast("B")
        if not m.contiguous:  # no-cover
            raise ValueError("The given buffer data is not contiguous")
        size = m.nbytes
        assert 0 <= offset < self.size
        assert 0 <= size <= (self.size - offset)

        mapped_mem = self._map_write(offset, size)
        mapped_mem[:] = m
        self._unmap()

    def _map_read(self, start, size):
        data = None

        @ffi.callback("void(WGPUBufferMapAsyncStatus, uint8_t*, uint8_t*)")
        def _map_read_callback(status, buffer_data_p, user_data_p):
            nonlocal data
            if status == 0:
                address = int(ffi.cast("intptr_t", buffer_data_p))
                data = get_memoryview_from_address(address, size)

        # H: void f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferAddress size, WGPUBufferMapReadCallback callback, uint8_t *userdata)
        lib.wgpu_buffer_map_read_async(
            self._internal, start, size, _map_read_callback, ffi.NULL
        )

        # Let it do some cycles
        self._state = "mapping pending"
        self._map_mode = flags.MapMode.READ
        # H: void f(WGPUDeviceId device_id, bool force_wait)
        lib.wgpu_device_poll(self._device._internal, True)

        if data is None:  # no-cover
            raise RuntimeError("Could not read buffer data.")

        self._state = "mapped"
        return data

    def _map_write(self, start, size):
        data = None

        @ffi.callback("void(WGPUBufferMapAsyncStatus, uint8_t*, uint8_t*)")
        def _map_write_callback(status, buffer_data_p, user_data_p):
            nonlocal data
            if status == 0:
                address = int(ffi.cast("intptr_t", buffer_data_p))
                data = get_memoryview_from_address(address, size)

        # H: void f(WGPUBufferId buffer_id, WGPUBufferAddress start, WGPUBufferAddress size, WGPUBufferMapWriteCallback callback, uint8_t *userdata)
        lib.wgpu_buffer_map_write_async(
            self._internal, start, size, _map_write_callback, ffi.NULL
        )

        # Let it do some cycles
        self._state = "mapping pending"
        # H: void f(WGPUDeviceId device_id, bool force_wait)
        lib.wgpu_device_poll(self._device._internal, True)

        if data is None:  # no-cover
            raise RuntimeError("Could not read buffer data.")

        self._state = "mapped"
        self._map_mode = flags.MapMode.WRITE
        return memoryview(data)

    def _unmap(self):
        # H: void f(WGPUBufferId buffer_id)
        lib.wgpu_buffer_unmap(self._internal)
        self._state = "unmapped"
        self._map_mode = 0

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            self._state = "destroyed"
            self._map_mode = 0
            # H: void f(WGPUBufferId buffer_id)
            lib.wgpu_buffer_destroy(internal)


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
            # H: WGPUTextureViewId f(WGPUTextureId texture_id, const WGPUTextureViewDescriptor *desc)
            id = lib.wgpu_texture_create_view(self._internal, ffi.NULL)

        else:
            c_label = ffi.new("char []", label.encode())
            # H: label: WGPULabel, format: WGPUTextureFormat, dimension: WGPUTextureViewDimension, aspect: WGPUTextureAspect, base_mip_level: int, level_count: int, base_array_layer: int, array_layer_count: int
            struct = new_struct_p(
                "WGPUTextureViewDescriptor *",
                label=c_label,
                format=format,
                dimension=dimension,
                aspect=aspect or "all",
                base_mip_level=base_mip_level,
                level_count=mip_level_count,
                base_array_layer=base_array_layer,
                array_layer_count=array_layer_count,
            )
            # H: WGPUTextureViewId f(WGPUTextureId texture_id, const WGPUTextureViewDescriptor *desc)
            id = lib.wgpu_texture_create_view(self._internal, struct)

        return GPUTextureView(label, id, self._device, self, self.size)

    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUTextureId texture_id)
            lib.wgpu_texture_destroy(internal)


class GPUTextureView(base.GPUTextureView, GPUObjectBase):
    pass


class GPUSampler(base.GPUSampler, GPUObjectBase):
    pass


class GPUBindGroupLayout(base.GPUBindGroupLayout, GPUObjectBase):
    pass


class GPUBindGroup(base.GPUBindGroup, GPUObjectBase):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPUBindGroupLayoutId bind_group_layout_id)
            lib.wgpu_bind_group_layout_destroy(internal)


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
    pass


class GPUCommandEncoder(base.GPUCommandEncoder, GPUObjectBase):
    def begin_compute_pass(self, *, label=""):
        # H: todo: int
        struct = new_struct_p(
            "WGPUComputePassDescriptor *",
            todo=0,
        )
        # H: WGPURawPass *f(WGPUCommandEncoderId encoder_id, const WGPUComputePassDescriptor *_desc)
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
            # H: attachment: int, resolve_target: int, load_op: WGPULoadOp, store_op: WGPUStoreOp, clear_color: WGPUColor
            c_attachment = new_struct(
                "WGPURenderPassColorAttachmentDescriptor",
                attachment=texture_view_id,
                resolve_target=c_resolve_target,
                load_op=c_load_op,
                store_op=color_attachment.get("store_op", "store"),
                clear_color=c_clear_color,
            )
            c_color_attachments_list.append(c_attachment)
        c_color_attachments_array = ffi.new(
            "WGPURenderPassColorAttachmentDescriptor []", c_color_attachments_list
        )

        c_depth_stencil_attachment = ffi.NULL
        if depth_stencil_attachment is not None:
            c_depth_load_op, c_depth_clear = _loadop_and_clear_from_value(
                depth_stencil_attachment["depth_load_value"]
            )
            c_stencil_load_op, c_stencil_clear = _loadop_and_clear_from_value(
                depth_stencil_attachment["stencil_load_value"]
            )
            # H: attachment: int, depth_load_op: WGPULoadOp, depth_store_op: WGPUStoreOp, clear_depth: float, stencil_load_op: WGPULoadOp, stencil_store_op: WGPUStoreOp, clear_stencil: int
            c_depth_stencil_attachment = new_struct_p(
                "WGPURenderPassDepthStencilAttachmentDescriptor *",
                attachment=depth_stencil_attachment["view"]._internal,
                depth_load_op=c_depth_load_op,
                depth_store_op=depth_stencil_attachment["depth_store_op"],
                clear_depth=float(c_depth_clear),
                stencil_load_op=c_stencil_load_op,
                stencil_store_op=depth_stencil_attachment["stencil_store_op"],
                clear_stencil=int(c_stencil_clear),
            )

        # H: color_attachments: WGPURenderPassColorAttachmentDescriptorBase_TextureViewId *, color_attachments_length: int, depth_stencil_attachment: WGPURenderPassDepthStencilAttachmentDescriptorBase_TextureViewId *
        struct = new_struct_p(
            "WGPURenderPassDescriptor *",
            color_attachments=c_color_attachments_array,
            color_attachments_length=len(c_color_attachments_list),
            depth_stencil_attachment=c_depth_stencil_attachment,
        )

        # H: WGPURawPass *f(WGPUCommandEncoderId encoder_id, const WGPURenderPassDescriptor *desc)
        raw_pass = lib.wgpu_command_encoder_begin_render_pass(self._internal, struct)
        return GPURenderPassEncoder(label, raw_pass, self)

    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
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

        c_source = new_struct_p(
            "WGPUBufferCopyView *",
            buffer=source["buffer"]._internal,
            # H: offset: WGPUBufferAddress/int, bytes_per_row: int, rows_per_image: int
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(source.get("offset", 0)),
                bytes_per_row=int(source["bytes_per_row"]),
                rows_per_image=int(source.get("rows_per_image", 0)),
            ),
        )

        ori = _tuple_from_tuple_or_dict(destination["origin"], "xyz")
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

        # H: void f(WGPUCommandEncoderId command_encoder_id, const WGPUBufferCopyView *source, const WGPUTextureCopyView *destination, const WGPUExtent3d *copy_size)
        lib.wgpu_command_encoder_copy_buffer_to_texture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_buffer(self, source, destination, copy_size):

        ori = _tuple_from_tuple_or_dict(source["origin"], "xyz")
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
                bytes_per_row=int(destination["bytes_per_row"]),
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

        # H: void f(WGPUCommandEncoderId command_encoder_id, const WGPUTextureCopyView *source, const WGPUBufferCopyView *destination, const WGPUExtent3d *copy_size)
        lib.wgpu_command_encoder_copy_texture_to_buffer(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def copy_texture_to_texture(self, source, destination, copy_size):

        ori = _tuple_from_tuple_or_dict(source["origin"], "xyz")
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

        ori = _tuple_from_tuple_or_dict(destination["origin"], "xyz")
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

        # H: void f(WGPUCommandEncoderId command_encoder_id, const WGPUTextureCopyView *source, const WGPUTextureCopyView *destination, const WGPUExtent3d *copy_size)
        lib.wgpu_command_encoder_copy_texture_to_texture(
            self._internal,
            c_source,
            c_destination,
            c_copy_size,
        )

    def finish(self, *, label=""):
        # H: todo: int
        struct = new_struct_p(
            "WGPUCommandBufferDescriptor *",
            todo=0,
        )
        # H: WGPUCommandBufferId f(WGPUCommandEncoderId encoder_id, const WGPUCommandBufferDescriptor *desc)
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
            # H: void f(WGPURawPass *pass, uint32_t index, WGPUBindGroupId bind_group_id, const WGPUDynamicOffset *offsets, uintptr_t offset_length)
            lib.wgpu_compute_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )
        else:
            # H: void f(WGPURawPass *pass, uint32_t index, WGPUBindGroupId bind_group_id, const WGPUDynamicOffset *offsets, uintptr_t offset_length)
            lib.wgpu_render_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )

    def push_debug_group(self, group_label):
        c_group_label = ffi.new("char []", group_label.encode())
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPURawPass *_pass, WGPURawString _label)
            lib.wgpu_compute_pass_push_debug_group(self._internal, c_group_label)
        else:
            # H: void f(WGPURawPass *_pass, WGPURawString _label)
            lib.wgpu_render_pass_push_debug_group(self._internal, c_group_label)

    def pop_debug_group(self):
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPURawPass *_pass)
            lib.wgpu_compute_pass_pop_debug_group(self._internal)
        else:
            # H: void f(WGPURawPass *_pass)
            lib.wgpu_render_pass_pop_debug_group(self._internal)

    def insert_debug_marker(self, marker_label):
        c_marker_label = ffi.new("char []", marker_label.encode())
        if isinstance(self, GPUComputePassEncoder):
            # H: void f(WGPURawPass *_pass, WGPURawString _label)
            lib.wgpu_compute_pass_insert_debug_marker(self._internal, c_marker_label)
        else:
            # H: void f(WGPURawPass *_pass, WGPURawString _label)
            lib.wgpu_render_pass_insert_debug_marker(self._internal, c_marker_label)


class GPUComputePassEncoder(
    base.GPUComputePassEncoder, GPUProgrammablePassEncoder, GPUObjectBase
):
    """"""

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(WGPURawPass *pass, WGPUComputePipelineId pipeline_id)
        lib.wgpu_compute_pass_set_pipeline(self._internal, pipeline_id)

    def dispatch(self, x, y=1, z=1):
        # H: void f(WGPURawPass *pass, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
        lib.wgpu_compute_pass_dispatch(self._internal, x, y, z)

    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPURawPass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset)
        lib.wgpu_compute_pass_dispatch_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def end_pass(self):
        # H: void f(WGPUComputePassId pass_id)
        lib.wgpu_compute_pass_end_pass(self._internal)

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURawPass *pass)
            internal  # todo: crashes lib.wgpu_compute_pass_destroy(internal)

    # FIXME: new method to implement
    def write_timestamp(self, query_set, query_index):
        raise NotImplementedError()


class GPURenderEncoderBase(base.GPURenderEncoderBase):
    """"""

    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        # H: void f(WGPURawPass *pass, WGPURenderPipelineId pipeline_id)
        lib.wgpu_render_pass_set_pipeline(self._internal, pipeline_id)

    # FIXME: was set_index_buffer(self, buffer, offset=0, size=0):
    def set_index_buffer(self, buffer, index_format, offset=0, size=0):
        if not size:
            size = buffer.size - offset
        # H: void f(WGPURawPass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset, WGPUBufferAddress size)
        lib.wgpu_render_pass_set_index_buffer(
            self._internal, buffer._internal, int(offset), int(size)
        )

    def set_vertex_buffer(self, slot, buffer, offset=0, size=0):
        if not size:
            size = buffer.size - offset
        # H: void f(WGPURawPass *pass, uint32_t slot, WGPUBufferId buffer_id, WGPUBufferAddress offset, WGPUBufferAddress size)
        lib.wgpu_render_pass_set_vertex_buffer(
            self._internal, int(slot), buffer._internal, int(offset), int(size)
        )

    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        # H: void f(WGPURawPass *pass, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
        lib.wgpu_render_pass_draw(
            self._internal, vertex_count, instance_count, first_vertex, first_instance
        )

    def draw_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        # H: void f(WGPURawPass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset)
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
        # H: void f(WGPURawPass *pass, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t base_vertex, uint32_t first_instance)
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
        # H: void f(WGPURawPass *pass, WGPUBufferId buffer_id, WGPUBufferAddress offset)
        lib.wgpu_render_pass_draw_indexed_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            # H: void f(WGPURawPass *pass)
            internal  # todo: crashes lib.wgpu_render_pass_destroy(internal)


class GPURenderPassEncoder(
    base.GPURenderPassEncoder,
    GPUProgrammablePassEncoder,
    GPURenderEncoderBase,
    GPUObjectBase,
):
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        # H: void f(WGPURawPass *pass, float x, float y, float w, float h, float depth_min, float depth_max)
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
        # H: void f(WGPURawPass *pass, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
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
        # H: void f(WGPURawPass *pass, const WGPUColor *color)
        lib.wgpu_render_pass_set_blend_color(self._internal, c_color)

    def set_stencil_reference(self, reference):
        # H: void f(WGPURawPass *pass, uint32_t value)
        lib.wgpu_render_pass_set_stencil_reference(self._internal, int(reference))

    # Not sure what this function exists in the Rust API, because there is no
    # way to create bundles yet?
    # def execute_bundles(self, bundles):
    #     bundles2 = []
    #     for bundle in bundles:
    #         if isinstance(bundle, GPURenderBundle):
    #             bundles2.append(bundle._internal)
    #         else:
    #             bundles2.append(int(bundle))
    #
    #     c_bundles_array = ffi.new("WGPURenderBundleId []", bundles2)
    # H: void f(WGPURawPass *_pass, const WGPURenderBundleId *_bundles, uintptr_t _bundles_length)
    #     lib.wgpu_render_pass_execute_bundles(
    #         self._internal, c_bundles_array, len(bundles2),
    #     )

    def end_pass(self):
        # H: void f(WGPURenderPassId pass_id)
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
    pass

    # Not yet implemented in wgpu-native
    # def finish(self, *, label=""):
    #     ...

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

        # Checks
        if not m.contiguous:  # no-cover
            raise ValueError("The given buffer data is not contiguous")

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

    def write_texture(self, destination, data, data_layout, size):

        m, address = get_memoryview_and_address(data)
        # todo: could we not derive the size from the shape of m?

        # Checks
        if not m.contiguous:  # no-cover
            raise ValueError("The given texture data is not contiguous")

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

        # H: void f(WGPUQueueId queue_id, const WGPUTextureCopyView *texture, const uint8_t *data, uintptr_t data_length, const WGPUTextureDataLayout *data_layout, const WGPUExtent3d *size)
        lib.wgpu_queue_write_texture(
            self._internal, c_destination, c_data, data_length, c_data_layout, c_size
        )

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

        # H: WGPUSwapChainId f(WGPUDeviceId device_id, WGPUSurfaceId surface_id, const WGPUSwapChainDescriptor *desc)
        self._internal = lib.wgpu_device_create_swap_chain(
            self._device._internal, self._surface_id, struct
        )

    def __enter__(self):
        # Get the current texture view, and make sure it is presented when done
        self._create_native_swap_chain_if_needed()
        # H: WGPUSwapChainOutput f(WGPUSwapChainId swap_chain_id)
        sc_output = lib.wgpu_swap_chain_get_next_texture(self._internal)
        status, view_id = sc_output.status, sc_output.view_id
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
        size = self._surface_size[0], self._surface_size[1], 1
        return GPUTextureView("swap_chain", view_id, self._device, None, size)

    def __exit__(self, type, value, tb):
        # Present the current texture
        # H: void f(WGPUSwapChainId swap_chain_id)
        lib.wgpu_swap_chain_present(self._internal)


class GPURenderBundle(base.GPURenderBundle, GPUObjectBase):
    pass


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
