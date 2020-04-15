"""
WGPU backend implementation based on wgpu-native

The wgpu-native project (https://github.com/gfx-rs/wgpu) is a Rust library
based on gfx-hal, which wraps Metal, Vulkan, DX12 and more in the
future. It can compile into a dynamic library exposing a C-API,
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
* Run the wgpu.help() call that is listed above each API method. This
  will usually give you all info that you need from webgpu.idl and
  wgpu.h.
* Though sometimes you may also need webgpu.idl and wgpu.h from the
  resource dir as a reference.
* Use new_struct() to create a C structure with minimal boilerplate.
  It also converts string enum values to their corresponding integers.
* When we update the upstream webgpu.idl or wgpu.h, the
  codegen-script.py should be run. This will update base.py and this
  module. Then run git diff to see what changed in webgpu.idl, wgpu.h
  and in this file, and make adjustments as needed.
"""


import os
import sys
import ctypes
import logging
import ctypes.util
from weakref import WeakKeyDictionary

from cffi import FFI, __version_info__ as cffi_version_info

from .. import base, flags
from .. import _register_backend
from .._coreutils import get_resource_filename
from .._mappings import cstructfield2enum, enummap


logger = logging.getLogger("wgpu")  # noqa


if cffi_version_info < (1, 10):  # no-cover
    raise ImportError(f"{__name__} needs cffi 1.10 or later.")


def _get_wgpu_h():
    """ Read header file and strip some stuff that cffi would stumble on.
    """
    lines = []
    with open(get_resource_filename("wgpu.h")) as f:
        for line in f.readlines():
            if not line.startswith(
                (
                    "#include ",
                    "#define WGPU_LOCAL",
                    "#define WGPUColor",
                    "#define WGPUOrigin3d_ZERO",
                    "#if defined",
                    "#endif",
                )
            ):
                lines.append(line)
    return "".join(lines)


def _get_wgpu_lib_path():
    """ Get the path to the wgpu library, taking into account the
    WGPU_LIB_PATH environment variable.
    """

    # If path is given, use that or fail trying
    override_path = os.getenv("WGPU_LIB_PATH", "").strip()
    if override_path:
        return override_path

    # Get lib filename for supported platforms
    if sys.platform.startswith("win"):  # no-cover
        lib_filename = "wgpu_native.dll"
    elif sys.platform.startswith("darwin"):  # no-cover
        lib_filename = "libwgpu_native.dylib"
    elif sys.platform.startswith("linux"):  # no-cover
        lib_filename = "libwgpu_native.so"
    else:  # no-cover
        raise RuntimeError(
            f"No WGPU library shipped for platform {sys.platform}. Set WGPU_LIB_PATH instead."
        )

    # Note that this can be a false positive, e.g. ARM linux.
    embedded_path = get_resource_filename(lib_filename)
    if not os.path.isfile(embedded_path):  # no-cover
        raise RuntimeError(f"Could not find WGPU library in {embedded_path}")
    else:
        return embedded_path


# Configure cffi and load the dynamic library
# NOTE: `import wgpu.backends.rs` is used in pyinstaller tests to verify
# that we can load the DLL after freezing
ffi = FFI()
ffi.cdef(_get_wgpu_h())
ffi.set_source("wgpu.h", None)
_lib = ffi.dlopen(_get_wgpu_lib_path())


# Object to be able to bind the lifetime of objects to other objects
_refs_per_struct = WeakKeyDictionary()


# Some enum keys need a shortcut
_cstructfield2enum_alt = {
    "load_op": "LoadOp",
    "store_op": "StoreOp",
    "depth_store_op": "StoreOp",
    "stencil_store_op": "StoreOp",
}


def new_struct(ctype, **kwargs):
    """ Create an ffi struct. Provides a flatter syntax and converts our
    string enums to int enums needed in C.
    """
    struct = ffi.new(ctype)
    for key, val in kwargs.items():
        if isinstance(val, str) and isinstance(getattr(struct, key), int):
            if key in _cstructfield2enum_alt:
                structname = _cstructfield2enum_alt[key]
            else:
                structname = cstructfield2enum[ctype.strip(" *")[4:] + "." + key]
            ival = enummap[structname + "." + val]
            setattr(struct, key, ival)
        else:
            setattr(struct, key, val)
    # Some kwargs may be other ffi objects, and some may represent
    # pointers. These need special care because them "being in" the
    # current struct does not prevent them from being cleaned up by
    # Python's garbage collector. Keeping hold of these objects in the
    # calling code is painful and prone to missing cases, so we solve
    # the issue here. We cannot attach an attribute to the struct directly,
    # so we use a global WeakKeyDictionary. Also see issue #52.
    _refs_per_struct[struct] = kwargs
    return struct


def get_surface_id_from_canvas(canvas):
    """ Get an id representing the surface to render to. The way to
    obtain this id differs per platform and GUI toolkit.
    """
    win_id = canvas.get_window_id()

    if sys.platform.startswith("win"):  # no-cover
        # wgpu_create_surface_from_windows_hwnd(void *_hinstance, void *hwnd)
        hwnd = ffi.cast("void *", int(win_id))
        hinstance = ffi.NULL
        return _lib.wgpu_create_surface_from_windows_hwnd(hinstance, hwnd)

    elif sys.platform.startswith("darwin"):  # no-cover
        # wgpu_create_surface_from_metal_layer(void *layer)
        # This is what the triangle example from wgpu-native does:
        # #if WGPU_TARGET == WGPU_TARGET_MACOS
        #     {
        #         id metal_layer = NULL;
        #         NSWindow *ns_window = glfwGetCocoaWindow(window);
        #         [ns_window.contentView setWantsLayer:YES];
        #         metal_layer = [CAMetalLayer layer];
        #         [ns_window.contentView setLayer:metal_layer];
        #         surface = wgpu_create_surface_from_metal_layer(metal_layer);
        #     }
        window = ctypes.c_void_p(win_id)

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        content_view_sel = objc.sel_registerName(b"contentView")
        set_wants_layer_sel = objc.sel_registerName(b"setWantsLayer:")
        responds_to_sel_sel = objc.sel_registerName(b"respondsToSelector:")
        layer_sel = objc.sel_registerName(b"layer")
        set_layer_sel = objc.sel_registerName(b"setLayer:")

        # Try some duck typing to see what kind of object the window pointer points to
        # Qt doesn't return a NSWindow, but a QNSView instead, which is subclass of NSView.
        if objc.objc_msgSend(
            window, responds_to_sel_sel, ctypes.c_void_p(content_view_sel)
        ):
            # NSWindow instances respond to contentView selector
            content_view = objc.objc_msgSend(window, content_view_sel)
        elif objc.objc_msgSend(window, responds_to_sel_sel, ctypes.c_void_p(layer_sel)):
            # NSView instances respond to layer selector
            # Let's assume that the given window pointer is actually the content view
            content_view = window
        else:
            # If the code reaches this part, we know that `window` is an
            # objective-c object but the type is neither NSView or NSWindow.
            raise RuntimeError("Received unidentified objective-c object.")

        # [ns_window.contentView setWantsLayer:YES]
        objc.objc_msgSend(content_view, set_wants_layer_sel, True)

        # metal_layer = [CAMetalLayer layer];
        ca_metal_layer_class = objc.objc_getClass(b"CAMetalLayer")
        metal_layer = objc.objc_msgSend(ca_metal_layer_class, layer_sel)

        # [ns_window.content_view setLayer:metal_layer];
        objc.objc_msgSend(content_view, set_layer_sel, ctypes.c_void_p(metal_layer))

        metal_layer_ffi_pointer = ffi.cast("void *", metal_layer)
        return _lib.wgpu_create_surface_from_metal_layer(metal_layer_ffi_pointer)

    elif sys.platform.startswith("linux"):  # no-cover
        # wgpu_create_surface_from_wayland(void *surface, void *display)
        # wgpu_create_surface_from_xlib(const void **display, uint64_t window)
        display_id = canvas.get_display_id()
        is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
        if is_wayland:
            # todo: works, but have not yet been able to test drawing to the window
            surface = ffi.cast("void *", win_id)
            display = ffi.cast("void *", display_id)
            return _lib.wgpu_create_surface_from_wayland(surface, display)
        else:
            display = ffi.cast("void **", display_id)
            return _lib.wgpu_create_surface_from_xlib(display, win_id)

    else:  # no-cover
        raise RuntimeError("Cannot get surface id: unsupported platform.")


def _tuple_from_tuple_or_dict(ob, fields):
    """ Given a tuple/list/dict, return a tuple. Also checks tuple size.

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
    """ In WebGPU, the load op, can be given either as "load" or a value.
    The latter translates to "clear" plus that value in wgpu-native.
    The value can be float/int/color, but we don't deal with that here.
    """
    if isinstance(value, str):
        assert value == "load"
        return 1, 0  # WGPULoadOp_Load and a stub value
    else:
        return 0, value  # WGPULoadOp_Clear and the value


# %% The API


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
def request_adapter(*, canvas, power_preference: "GPUPowerPreference"):
    """ Get a :class:`GPUAdapter`, the object that represents an abstract wgpu
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
    struct = new_struct(
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

    _lib.wgpu_request_adapter_async(
        struct, backend_mask, _request_adapter_callback, ffi.NULL
    )  # userdata, stub

    # For now, Rust will call the callback immediately
    # todo: when wgpu gets an event loop -> while run wgpu event loop or something

    assert adapter_id is not None
    extensions = []
    return GPUAdapter("WGPU", extensions, adapter_id)


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
async def request_adapter_async(*, canvas, power_preference: "GPUPowerPreference"):
    """ Async version of ``request_adapter()``.
    This function uses the Rust WGPU library.
    """
    return request_adapter(canvas=canvas, power_preference=power_preference)  # no-cover


# Mark as the backend at import time
_register_backend(request_adapter, request_adapter_async)


class GPUAdapter(base.GPUAdapter):
    def __init__(self, name, extensions, id):
        super().__init__(name, extensions)
        self._id = id

    # wgpu.help('DeviceDescriptor', 'adapterrequestdevice', dev=True)
    def request_device(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):

        # Handle default limits
        limits2 = base.default_limits.copy()
        limits2.update(limits or {})

        c_extensions = new_struct(
            "WGPUExtensions *",
            anisotropic_filtering="anisotropic_filtering" in extensions,
        )
        c_limits = new_struct(
            "WGPULimits *", max_bind_groups=limits2["max_bind_groups"]
        )
        struct = new_struct(
            "WGPUDeviceDescriptor *", extensions=c_extensions[0], limits=c_limits[0]
        )

        device_id = _lib.wgpu_adapter_request_device(self._id, struct)

        # Get the actual limits reported by the device
        c_limits = new_struct("WGPULimits *")
        _lib.wgpu_device_get_limits(device_id, c_limits)
        limits3 = {key: getattr(c_limits, key) for key in dir(c_limits)}

        # Get the queue to which commands can be submitted
        queue_id = _lib.wgpu_device_get_default_queue(device_id)
        queue = GPUQueue("", queue_id, self)

        return GPUDevice(label, device_id, self, extensions, limits3, queue)

    # wgpu.help('DeviceDescriptor', 'adapterrequestdevice', dev=True)
    async def request_device_async(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        return self.request_device(extensions=extensions, limits=limits)  # no-cover

    def _destroy(self):
        if self._id is not None:
            self._id, id = None, self._id
            _lib.wgpu_adapter_destroy(id)


class GPUDevice(base.GPUDevice):
    # wgpu.help('BufferDescriptor', 'devicecreatebuffer', dev=True)
    def create_buffer(self, *, label="", size: int, usage: "GPUBufferUsageFlags"):
        c_label = ffi.new("char []", label.encode())
        size = int(size)
        struct = new_struct(
            "WGPUBufferDescriptor *", label=c_label, size=size, usage=usage
        )

        id = _lib.wgpu_device_create_buffer(self._internal, struct)
        return GPUBuffer(label, id, self, size, usage, "unmapped", None)

    # wgpu.help('BufferDescriptor', 'devicecreatebuffermapped', dev=True)
    def create_buffer_mapped(
        self, *, label="", size: int, usage: "GPUBufferUsageFlags"
    ):

        c_label = ffi.new("char []", label.encode())
        size = int(size)
        struct = new_struct(
            "WGPUBufferDescriptor *", label=c_label, size=size, usage=usage
        )

        # Pointer that device_create_buffer_mapped sets, so that we can write stuff
        # there
        buffer_memory_pointer = ffi.new("uint8_t * *")

        id = _lib.wgpu_device_create_buffer_mapped(
            self._internal, struct, buffer_memory_pointer
        )

        # Map a ctypes array onto the data
        pointer_as_int = int(ffi.cast("intptr_t", buffer_memory_pointer[0]))
        mem_as_ctypes = (ctypes.c_uint8 * size).from_address(pointer_as_int)

        return GPUBuffer(label, id, self, size, usage, "mapped", mem_as_ctypes)

    # wgpu.help('TextureDescriptor', 'devicecreatetexture', dev=True)
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
        c_label = ffi.new("char []", label.encode())
        size = _tuple_from_tuple_or_dict(size, ("width", "height", "depth"))
        c_size = new_struct(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )
        struct = new_struct(
            "WGPUTextureDescriptor *",
            label=c_label,
            size=c_size[0],
            array_layer_count=1,
            mip_level_count=mip_level_count,
            sample_count=sample_count,
            dimension=dimension,
            format=format,
            usage=usage,
        )
        id = _lib.wgpu_device_create_texture(self._internal, struct)

        return GPUTexture(label, id, self)

    # wgpu.help('SamplerDescriptor', 'devicecreatesampler', dev=True)
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
        struct = new_struct(
            "WGPUSamplerDescriptor *",
            address_mode_u=address_mode_u,
            address_mode_v=address_mode_v,
            mag_filter=mag_filter,
            min_filter=min_filter,
            mipmap_filter=mipmap_filter,
            lod_min_clamp=lod_min_clamp,
            lod_max_clamp=lod_max_clamp,
            compare=0 if compare is None else compare,
        )

        id = _lib.wgpu_device_create_sampler(self._internal, struct)
        return base.GPUSampler(label, id, self)

    # wgpu.help('BindGroupLayoutDescriptor', 'devicecreatebindgrouplayout', dev=True)
    def create_bind_group_layout(
        self, *, label="", entries: "GPUBindGroupLayoutEntry-list"
    ):
        c_entries_list = []
        for entry in entries:
            type = entry["type"]
            if "texture" in type:
                need = {"view_dimension", "texture_component_type"}
                if "storage" in type:
                    need.add("storage_texture_format")
                assert all(
                    x in entry for x in need
                ), f"{type} binding should specify {need}"
            c_entry = new_struct(
                "WGPUBindGroupLayoutEntry *",
                binding=int(entry["binding"]),
                visibility=int(entry["visibility"]),
                ty=type,
                multisampled=bool(entry.get("multisampled", False)),
                has_dynamic_offset=bool(entry.get("has_dynamic_offset", False)),
                view_dimension=entry.get("view_dimension", "2d"),
                texture_component_type=entry.get("texture_component_type", "float"),
                storage_texture_format=entry.get("storage_texture_format", 0),
            )
            c_entries_list.append(c_entry[0])

        c_label = ffi.new("char []", label.encode())
        c_entries_array = ffi.new("WGPUBindGroupLayoutEntry []", c_entries_list)
        struct = new_struct(
            "WGPUBindGroupLayoutDescriptor *",
            label=c_label,
            entries=c_entries_array,
            entries_length=len(c_entries_list),
        )

        id = _lib.wgpu_device_create_bind_group_layout(self._internal, struct)

        return base.GPUBindGroupLayout(label, id, self, entries)

    # wgpu.help('BindGroupDescriptor', 'devicecreatebindgroup', dev=True)
    def create_bind_group(
        self,
        *,
        label="",
        layout: "GPUBindGroupLayout",
        entries: "GPUBindGroupEntry-list",
    ):

        c_entries_list = []
        for entry in entries:
            # The resource can be a sampler, texture view, or buffer descriptor
            resource = entry["resource"]
            if isinstance(resource, base.GPUSampler):
                c_resource_kwargs = {
                    "tag": 1,  # WGPUBindingResource_Tag.WGPUBindingResource_Sampler
                    "sampler": new_struct(
                        "WGPUBindingResource_WGPUSampler_Body *", _0=resource._internal
                    )[0],
                }
            elif isinstance(resource, base.GPUTextureView):
                c_resource_kwargs = {
                    "tag": 2,  # WGPUBindingResource_Tag.WGPUBindingResource_TextureView
                    "texture_view": new_struct(
                        "WGPUBindingResource_WGPUTextureView_Body *",
                        _0=resource._internal,
                    )[0],
                }
            elif isinstance(resource, dict):  # Buffer binding
                c_buffer_entry = new_struct(
                    "WGPUBufferBinding *",
                    buffer=resource["buffer"]._internal,
                    offset=resource["offset"],
                    size=resource["size"],
                )
                c_resource_kwargs = {
                    "tag": 0,  # WGPUBindingResource_Tag.WGPUBindingResource_Buffer
                    "buffer": new_struct(
                        "WGPUBindingResource_WGPUBuffer_Body *", _0=c_buffer_entry[0]
                    )[0],
                }
            else:
                raise TypeError(f"Unexpected resource type {type(resource)}")
            c_resource = new_struct("WGPUBindingResource *", **c_resource_kwargs)
            c_entry = new_struct(
                "WGPUBindGroupEntry *",
                binding=int(entry["binding"]),
                resource=c_resource[0],
            )
            c_entries_list.append(c_entry[0])

        c_label = ffi.new("char []", label.encode())
        c_entries_array = ffi.new("WGPUBindGroupEntry []", c_entries_list)
        struct = new_struct(
            "WGPUBindGroupDescriptor *",
            label=c_label,
            layout=layout._internal,
            entries=c_entries_array,
            entries_length=len(c_entries_list),
        )

        id = _lib.wgpu_device_create_bind_group(self._internal, struct)
        return base.GPUBindGroup(label, id, self, entries)

    # wgpu.help('PipelineLayoutDescriptor', 'devicecreatepipelinelayout', dev=True)
    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "GPUBindGroupLayout-list"
    ):

        bind_group_layouts_ids = [x._internal for x in bind_group_layouts]

        c_layout_array = ffi.new("WGPUBindGroupLayoutId []", bind_group_layouts_ids)
        struct = new_struct(
            "WGPUPipelineLayoutDescriptor *",
            bind_group_layouts=c_layout_array,
            bind_group_layouts_length=len(bind_group_layouts),
        )

        id = _lib.wgpu_device_create_pipeline_layout(self._internal, struct)
        return base.GPUPipelineLayout(label, id, self, bind_group_layouts)

    # wgpu.help('ShaderModuleDescriptor', 'devicecreateshadermodule', dev=True)
    def create_shader_module(self, *, label="", code: str):

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
        )

        struct = new_struct("WGPUShaderModuleDescriptor *", code=c_code[0])

        id = _lib.wgpu_device_create_shader_module(self._internal, struct)
        return base.GPUShaderModule(label, id, self)

    # wgpu.help('ComputePipelineDescriptor', 'devicecreatecomputepipeline', dev=True)
    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        compute_stage: "GPUProgrammableStageDescriptor",
    ):

        c_compute_stage = new_struct(
            "WGPUProgrammableStageDescriptor *",
            module=compute_stage["module"]._internal,
            entry_point=ffi.new("char []", compute_stage["entry_point"].encode()),
        )

        struct = new_struct(
            "WGPUComputePipelineDescriptor *",
            layout=layout._internal,
            compute_stage=c_compute_stage[0],
        )

        id = _lib.wgpu_device_create_compute_pipeline(self._internal, struct)
        return base.GPUComputePipeline(label, id, self)

    # wgpu.help('RenderPipelineDescriptor', 'devicecreaterenderpipeline', dev=True)
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
        c_vertex_stage = new_struct(
            "WGPUProgrammableStageDescriptor *",
            module=vertex_stage["module"]._internal,
            entry_point=ffi.new("char []", vertex_stage["entry_point"].encode()),
        )
        c_fragment_stage = ffi.NULL
        if fragment_stage is not None:
            c_fragment_stage = new_struct(
                "WGPUProgrammableStageDescriptor *",
                module=fragment_stage["module"]._internal,
                entry_point=ffi.new("char []", fragment_stage["entry_point"].encode()),
            )
        c_rasterization_state = new_struct(
            "WGPURasterizationStateDescriptor *",
            front_face=rasterization_state.get("front_face", "ccw"),
            cull_mode=rasterization_state.get("cull_mode", "none"),
            depth_bias=rasterization_state.get("depth_bias", 0),
            depth_bias_slope_scale=rasterization_state.get("depth_bias_slope_scale", 0),
            depth_bias_clamp=rasterization_state.get("depth_bias_clamp", 0),
        )
        c_color_states_list = []
        for color_state in color_states:
            alpha_blend = _tuple_from_tuple_or_dict(
                color_state["alpha_blend"], ("src_factor", "dst_factor", "operation"),
            )
            c_alpha_blend = new_struct(
                "WGPUBlendDescriptor *",
                src_factor=alpha_blend[0],
                dst_factor=alpha_blend[1],
                operation=alpha_blend[2],
            )
            color_blend = _tuple_from_tuple_or_dict(
                color_state["color_blend"], ("src_factor", "dst_factor", "operation"),
            )
            c_color_blend = new_struct(
                "WGPUBlendDescriptor *",
                src_factor=color_blend[0],
                dst_factor=color_blend[1],
                operation=color_blend[2],
            )
            c_color_state = new_struct(
                "WGPUColorStateDescriptor *",
                format=color_state["format"],
                alpha_blend=c_alpha_blend[0],
                color_blend=c_color_blend[0],
                write_mask=color_state.get("write_mask", 0xF),
            )
            c_color_states_list.append(c_color_state[0])
        c_color_states_array = ffi.new(
            "WGPUColorStateDescriptor []", c_color_states_list
        )
        if depth_stencil_state is None:
            c_depth_stencil_state = ffi.NULL
        else:
            stencil_front = depth_stencil_state.get("stencil_front", {})
            c_stencil_front = new_struct(
                "WGPUStencilStateFaceDescriptor *",
                compare=stencil_front.get("compare", "always"),
                fail_op=stencil_front.get("fail_op", "keep"),
                depth_fail_op=stencil_front.get("depth_fail_op", "keep"),
                pass_op=stencil_front.get("pass_op", "keep"),
            )
            stencil_back = depth_stencil_state.get("stencil_back", {})
            c_stencil_back = new_struct(
                "WGPUStencilStateFaceDescriptor *",
                compare=stencil_back.get("compare", "always"),
                fail_op=stencil_back.get("fail_op", "keep"),
                depth_fail_op=stencil_back.get("depth_fail_op", "keep"),
                pass_op=stencil_back.get("pass_op", "keep"),
            )
            c_depth_stencil_state = new_struct(
                "WGPUDepthStencilStateDescriptor *",
                format=depth_stencil_state["format"],
                depth_write_enabled=bool(
                    depth_stencil_state.get("depth_write_enabled", False)
                ),
                depth_compare=depth_stencil_state.get("depth_compare", "always"),
                stencil_front=c_stencil_front[0],
                stencil_back=c_stencil_back[0],
                stencil_read_mask=depth_stencil_state.get(
                    "stencil_read_mask", 0xFFFFFFFF
                ),
                stencil_write_mask=depth_stencil_state.get(
                    "stencil_write_mask", 0xFFFFFFFF
                ),
            )
        c_vertex_buffer_descriptors_list = []
        for buffer_des in vertex_state["vertex_buffers"]:
            c_attributes_list = []
            for attribute in buffer_des["attributes"]:
                c_attribute = new_struct(
                    "WGPUVertexAttributeDescriptor *",
                    format=attribute["format"],
                    offset=attribute["offset"],
                    shader_location=attribute["shader_location"],
                )
                c_attributes_list.append(c_attribute[0])
            c_attributes_array = ffi.new(
                "WGPUVertexAttributeDescriptor []", c_attributes_list
            )
            c_vertex_buffer_descriptor = new_struct(
                "WGPUVertexBufferLayoutDescriptor *",
                array_stride=buffer_des["array_stride"],
                step_mode=buffer_des.get("step_mode", "vertex"),
                attributes=c_attributes_array,
                attributes_length=len(c_attributes_list),
            )
            c_vertex_buffer_descriptors_list.append(c_vertex_buffer_descriptor[0])
        c_vertex_buffer_descriptors_array = ffi.new(
            "WGPUVertexBufferLayoutDescriptor []", c_vertex_buffer_descriptors_list
        )
        c_vertex_state = new_struct(
            "WGPUVertexStateDescriptor *",
            index_format=vertex_state.get("index_format", "uint32"),
            vertex_buffers=c_vertex_buffer_descriptors_array,
            vertex_buffers_length=len(c_vertex_buffer_descriptors_list),
        )

        struct = new_struct(
            "WGPURenderPipelineDescriptor *",
            layout=layout._internal,
            vertex_stage=c_vertex_stage[0],
            fragment_stage=c_fragment_stage,
            primitive_topology=primitive_topology,
            rasterization_state=c_rasterization_state,
            color_states=c_color_states_array,
            color_states_length=len(c_color_states_list),
            depth_stencil_state=c_depth_stencil_state,
            vertex_state=c_vertex_state[0],
            sample_count=sample_count,
            sample_mask=sample_mask,
            alpha_to_coverage_enabled=alpha_to_coverage_enabled,
        )

        id = _lib.wgpu_device_create_render_pipeline(self._internal, struct)
        return base.GPURenderPipeline(label, id, self)

    # wgpu.help('CommandEncoderDescriptor', 'devicecreatecommandencoder', dev=True)
    def create_command_encoder(self, *, label=""):
        c_label = ffi.new("char []", label.encode())
        struct = new_struct("WGPUCommandEncoderDescriptor *", label=c_label)

        id = _lib.wgpu_device_create_command_encoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

    # Not yet implemented in wgpu-native
    # def create_render_bundle_encoder(
    #     self,
    #     *,
    #     label="",
    #     color_formats: "GPUTextureFormat-list",
    #     depth_stencil_format: "GPUTextureFormat" = None,
    #     sample_count: "GPUSize32" = 1,
    # ):
    #     pass

    def configure_swap_chain(self, canvas, format, usage=None):
        usage = flags.TextureUsage.OUTPUT_ATTACHMENT if usage is None else usage
        return GPUSwapChain(self, canvas, format, usage)


class GPUBuffer(base.GPUBuffer):
    # wgpu.help('buffermapreadasync', dev=True)
    def map_read(self):
        data = None

        @ffi.callback("void(WGPUBufferMapAsyncStatus, uint8_t*, uint8_t*)")
        def _map_read_callback(status, buffer_data_p, user_data_p):
            nonlocal data
            if status == 0:
                pointer_as_int = int(ffi.cast("intptr_t", buffer_data_p))
                mem_as_ctypes = (ctypes.c_uint8 * size).from_address(pointer_as_int)
                data = mem_as_ctypes

        start, size = 0, self.size
        _lib.wgpu_buffer_map_read_async(
            self._internal, start, size, _map_read_callback, ffi.NULL
        )

        # Let it do some cycles
        _lib.wgpu_device_poll(self._device._internal, True)

        if data is None:  # no-cover
            raise RuntimeError("Could not read buffer data.")

        self._state = "mapped"
        self._mapping = data
        return data

    # wgpu.help('buffermapwriteasync', dev=True)
    def map_write(self):
        data = None

        @ffi.callback("void(WGPUBufferMapAsyncStatus, uint8_t*, uint8_t*)")
        def _map_write_callback(status, buffer_data_p, user_data_p):
            nonlocal data
            if status == 0:
                pointer_as_int = int(ffi.cast("intptr_t", buffer_data_p))
                mem_as_ctypes = (ctypes.c_uint8 * size).from_address(pointer_as_int)
                data = mem_as_ctypes

        start, size = 0, self.size
        _lib.wgpu_buffer_map_write_async(
            self._internal, start, size, _map_write_callback, ffi.NULL
        )

        # Let it do some cycles
        _lib.wgpu_device_poll(self._device._internal, True)

        if data is None:  # no-cover
            raise RuntimeError("Could not read buffer data.")

        self._state = "mapped"
        self._mapping = data
        return data

    # wgpu.help('buffermapreadasync', dev=True)
    async def map_read_async(self):
        # todo: actually make this async
        return self.map_read()  # no-cover

    # wgpu.help('buffermapwriteasync', dev=True)
    async def map_write_async(self):
        # todo: actually make this async
        return self.map_write()  # no-cover

    # wgpu.help('bufferunmap', dev=True)
    def unmap(self):
        if self._state == "mapped":
            _lib.wgpu_buffer_unmap(self._internal)
            self._state = "unmapped"
            self._mapping = None

    # wgpu.help('bufferdestroy', dev=True)
    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            self._state = "destroyed"
            self._mapping = None
            _lib.wgpu_buffer_destroy(internal)


class GPUTexture(base.GPUTexture):

    # wgpu.help('TextureViewDescriptor', 'texturecreateview', dev=True)
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
                    "In create_view() if any paramter is given, "
                    + "both format and dimension must be specified."
                )
            id = _lib.wgpu_texture_create_view(self._internal, ffi.NULL)

        else:
            struct = new_struct(
                "WGPUTextureViewDescriptor *",
                format=format,
                dimension=dimension,
                aspect=aspect,
                base_mip_level=base_mip_level,
                level_count=mip_level_count,
                base_array_layer=base_array_layer,
                array_layer_count=array_layer_count,
            )
            id = _lib.wgpu_texture_create_view(self._internal, struct)

        return base.GPUTextureView(label, id, self)

    # wgpu.help('texturedestroy', dev=True)
    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_texture_destroy(internal)


class GPUCommandEncoder(base.GPUCommandEncoder):
    # wgpu.help('ComputePassDescriptor', 'commandencoderbegincomputepass', dev=True)
    def begin_compute_pass(self, *, label=""):
        struct = new_struct("WGPUComputePassDescriptor *", todo=0)
        raw_pass = _lib.wgpu_command_encoder_begin_compute_pass(self._internal, struct)
        return GPUComputePassEncoder(label, raw_pass, self)

    # wgpu.help('RenderPassDescriptor', 'commandencoderbeginrenderpass', dev=True)
    def begin_render_pass(
        self,
        *,
        label="",
        color_attachments: "GPURenderPassColorAttachmentDescriptor-list",
        depth_stencil_attachment: "GPURenderPassDepthStencilAttachmentDescriptor" = None,
        occlusion_query_set: "GPUQuerySet" = None,
    ):
        # Note that occlusion_query_set is ignored because wgpu-native does not have it.

        c_color_attachments_list = []
        for color_attachment in color_attachments:
            assert isinstance(color_attachment["attachment"], base.GPUTextureView)
            texture_view_id = color_attachment["attachment"]._internal
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
            c_clear_color = new_struct(
                "WGPUColor *", r=clr[0], g=clr[1], b=clr[2], a=clr[3]
            )
            c_attachment = new_struct(
                "WGPURenderPassColorAttachmentDescriptor *",
                attachment=texture_view_id,
                resolve_target=c_resolve_target,
                load_op=c_load_op,
                store_op=color_attachment.get("store_op", "store"),
                clear_color=c_clear_color[0],
            )
            c_color_attachments_list.append(c_attachment[0])
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
            c_depth_stencil_attachment = new_struct(
                "WGPURenderPassDepthStencilAttachmentDescriptor *",
                attachment=depth_stencil_attachment["attachment"]._internal,
                depth_load_op=c_depth_load_op,
                depth_store_op=depth_stencil_attachment["depth_store_op"],
                clear_depth=float(c_depth_clear),
                stencil_load_op=c_stencil_load_op,
                stencil_store_op=depth_stencil_attachment["stencil_store_op"],
                clear_stencil=int(c_stencil_clear),
            )

        struct = new_struct(
            "WGPURenderPassDescriptor *",
            color_attachments=c_color_attachments_array,
            color_attachments_length=len(c_color_attachments_list),
            depth_stencil_attachment=c_depth_stencil_attachment,
        )

        raw_pass = _lib.wgpu_command_encoder_begin_render_pass(self._internal, struct)
        return GPURenderPassEncoder(label, raw_pass, self)

    # wgpu.help('Buffer', 'Size64', 'commandencodercopybuffertobuffer', dev=True)
    def copy_buffer_to_buffer(
        self, source, source_offset, destination, destination_offset, size
    ):
        assert isinstance(source, GPUBuffer)
        assert isinstance(destination, GPUBuffer)
        _lib.wgpu_command_encoder_copy_buffer_to_buffer(
            self._internal,
            source._internal,
            int(source_offset),
            destination._internal,
            int(destination_offset),
            int(size),
        )

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopybuffertotexture', dev=True)
    def copy_buffer_to_texture(self, source, destination, copy_size):

        c_source = new_struct(
            "WGPUBufferCopyView *",
            buffer=source["buffer"]._internal,
            offset=int(source.get("offset", 0)),
            bytes_per_row=int(source["bytes_per_row"]),
            rows_per_image=int(source.get("rows_per_image", 0)),
        )

        ori = _tuple_from_tuple_or_dict(destination["origin"], "xyz")
        c_origin = new_struct("WGPUOrigin3d *", x=ori[0], y=ori[1], z=ori[2])
        c_destination = new_struct(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=int(destination.get("mip_level", 0)),
            array_layer=int(destination.get("array_layer", 0)),
            origin=c_origin[0],
        )

        size = _tuple_from_tuple_or_dict(copy_size, ("width", "height", "depth"))
        c_copy_size = new_struct(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_command_encoder_copy_buffer_to_texture(
            self._internal, c_source, c_destination, c_copy_size[0],
        )

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopytexturetobuffer', dev=True)
    def copy_texture_to_buffer(self, source, destination, copy_size):

        ori = _tuple_from_tuple_or_dict(source["origin"], "xyz")
        c_origin = new_struct("WGPUOrigin3d *", x=ori[0], y=ori[1], z=ori[2])
        c_source = new_struct(
            "WGPUTextureCopyView *",
            texture=source["texture"]._internal,
            mip_level=int(source.get("mip_level", 0)),
            array_layer=int(source.get("array_layer", 0)),
            origin=c_origin[0],
        )

        c_destination = new_struct(
            "WGPUBufferCopyView *",
            buffer=destination["buffer"]._internal,
            offset=int(destination.get("offset", 0)),
            bytes_per_row=int(destination["bytes_per_row"]),
            rows_per_image=int(destination.get("rows_per_image", 0)),
        )

        size = _tuple_from_tuple_or_dict(copy_size, ("width", "height", "depth"))
        c_copy_size = new_struct(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_command_encoder_copy_texture_to_buffer(
            self._internal, c_source, c_destination, c_copy_size[0],
        )

    # wgpu.help('Extent3D', 'TextureCopyView', 'commandencodercopytexturetotexture', dev=True)
    def copy_texture_to_texture(self, source, destination, copy_size):

        ori = _tuple_from_tuple_or_dict(source["origin"], "xyz")
        c_origin1 = new_struct("WGPUOrigin3d *", x=ori[0], y=ori[1], z=ori[2])
        c_source = new_struct(
            "WGPUTextureCopyView *",
            texture=source["texture"]._internal,
            mip_level=int(source.get("mip_level", 0)),
            array_layer=int(source.get("array_layer", 0)),
            origin=c_origin1[0],
        )

        ori = _tuple_from_tuple_or_dict(destination["origin"], "xyz")
        c_origin2 = new_struct("WGPUOrigin3d *", x=ori[0], y=ori[1], z=ori[2])
        c_destination = new_struct(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=int(destination.get("mip_level", 0)),
            array_layer=int(destination.get("array_layer", 0)),
            origin=c_origin2[0],
        )

        size = _tuple_from_tuple_or_dict(copy_size, ("width", "height", "depth"))
        c_copy_size = new_struct(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_command_encoder_copy_texture_to_texture(
            self._internal, c_source, c_destination, c_copy_size[0],
        )

    # wgpu.help('CommandBufferDescriptor', 'commandencoderfinish', dev=True)
    def finish(self, *, label=""):
        struct = new_struct("WGPUCommandBufferDescriptor *", todo=0)
        id = _lib.wgpu_command_encoder_finish(self._internal, struct)
        return base.GPUCommandBuffer(label, id, self)

    # todo: these do not exist yet for command_encoder in wgpu-native
    # def push_debug_group(self, group_label):
    # def pop_debug_group(self):
    # def insert_debug_marker(self, marker_label):


class GPUProgrammablePassEncoder(base.GPUProgrammablePassEncoder):
    # wgpu.help('BindGroup', 'Index32', 'Size32', 'Size64', 'programmablepassencodersetbindgroup', dev=True)
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
            _lib.wgpu_compute_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )
        else:
            _lib.wgpu_render_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )

    # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    def push_debug_group(self, group_label):
        c_group_label = ffi.new("char []", group_label.encode())
        if isinstance(self, GPUComputePassEncoder):
            _lib.wgpu_compute_pass_push_debug_group(self._internal, c_group_label)
        else:
            _lib.wgpu_render_pass_push_debug_group(self._internal, c_group_label)

    # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    def pop_debug_group(self):
        if isinstance(self, GPUComputePassEncoder):
            _lib.wgpu_compute_pass_pop_debug_group(self._internal)
        else:
            _lib.wgpu_render_pass_pop_debug_group(self._internal)

    # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    def insert_debug_marker(self, marker_label):
        c_marker_label = ffi.new("char []", marker_label.encode())
        if isinstance(self, GPUComputePassEncoder):
            _lib.wgpu_compute_pass_insert_debug_marker(self._internal, c_marker_label)
        else:
            _lib.wgpu_render_pass_insert_debug_marker(self._internal, c_marker_label)


class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('ComputePipeline', 'computepassencodersetpipeline', dev=True)
    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        _lib.wgpu_compute_pass_set_pipeline(self._internal, pipeline_id)

    # wgpu.help('Size32', 'computepassencoderdispatch', dev=True)
    def dispatch(self, x, y=1, z=1):
        _lib.wgpu_compute_pass_dispatch(self._internal, x, y, z)

    # wgpu.help('Buffer', 'Size64', 'computepassencoderdispatchindirect', dev=True)
    def dispatch_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        _lib.wgpu_compute_pass_dispatch_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    # wgpu.help('computepassencoderendpass', dev=True)
    def end_pass(self):
        _lib.wgpu_compute_pass_end_pass(self._internal)

    # todo: const uint8_t *wgpu_compute_pass_finish(WGPURawPass *pass, uintptr_t *length);

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            internal  # todo: crashes _lib.wgpu_compute_pass_destroy(internal)


class GPURenderEncoderBase(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('RenderPipeline', 'renderencoderbasesetpipeline', dev=True)
    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        _lib.wgpu_render_pass_set_pipeline(self._internal, pipeline_id)

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasesetindexbuffer', dev=True)
    def set_index_buffer(self, buffer, offset=0, size=0):
        _lib.wgpu_render_pass_set_index_buffer(
            self._internal, buffer._internal, int(offset), int(size)
        )

    # wgpu.help('Buffer', 'Index32', 'Size64', 'renderencoderbasesetvertexbuffer', dev=True)
    def set_vertex_buffer(self, slot, buffer, offset=0, size=0):
        _lib.wgpu_render_pass_set_vertex_buffer(
            self._internal, int(slot), buffer._internal, int(offset), int(size)
        )

    # wgpu.help('Size32', 'renderencoderbasedraw', dev=True)
    def draw(self, vertex_count, instance_count=1, first_vertex=0, first_instance=0):
        _lib.wgpu_render_pass_draw(
            self._internal, vertex_count, instance_count, first_vertex, first_instance
        )

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasedrawindirect', dev=True)
    def draw_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        _lib.wgpu_render_pass_draw_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    # wgpu.help('SignedOffset32', 'Size32', 'renderencoderbasedrawindexed', dev=True)
    def draw_indexed(
        self,
        index_count,
        instance_count=1,
        first_index=0,
        base_vertex=0,
        first_instance=0,
    ):
        _lib.wgpu_render_pass_draw_indexed(
            self._internal,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasedrawindexedindirect', dev=True)
    def draw_indexed_indirect(self, indirect_buffer, indirect_offset):
        buffer_id = indirect_buffer._internal
        _lib.wgpu_render_pass_draw_indexed_indirect(
            self._internal, buffer_id, int(indirect_offset)
        )

    # todo: uint8_t *wgpu_render_pass_finish(WGPURawPass *pass, uintptr_t *length);

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            internal  # todo: crashes _lib.wgpu_render_pass_destroy(internal)


class GPURenderPassEncoder(GPURenderEncoderBase):

    # Note: this does not inherit from base.GPURenderPassEncoder!

    # wgpu.help('renderpassencodersetviewport', dev=True)
    def set_viewport(self, x, y, width, height, min_depth, max_depth):
        _lib.wgpu_render_pass_set_viewport(
            self._internal,
            float(x),
            float(y),
            float(width),
            float(height),
            float(min_depth),
            float(max_depth),
        )

    # wgpu.help('IntegerCoordinate', 'renderpassencodersetscissorrect', dev=True)
    def set_scissor_rect(self, x, y, width, height):
        _lib.wgpu_render_pass_set_scissor_rect(
            self._internal, int(x), int(y), int(width), int(height)
        )

    # wgpu.help('Color', 'renderpassencodersetblendcolor', dev=True)
    def set_blend_color(self, color):
        color = _tuple_from_tuple_or_dict(color, "rgba")
        c_color = new_struct(
            "WGPUColor *", r=color[0], g=color[1], b=color[2], a=color[3]
        )
        _lib.wgpu_render_pass_set_blend_color(self._internal, c_color)

    # wgpu.help('StencilValue', 'renderpassencodersetstencilreference', dev=True)
    def set_stencil_reference(self, reference):
        _lib.wgpu_render_pass_set_stencil_reference(self._internal, int(reference))

    # Not sure what this function exists in the Rust API, because there is no
    # way to create bundles yet?
    # def execute_bundles(self, bundles):
    #     bundles2 = []
    #     for bundle in bundles:
    #         if isinstance(bundle, base.GPURenderBundle):
    #             bundles2.append(bundle._internal)
    #         else:
    #             bundles2.append(int(bundle))
    #
    #     c_bundles_array = ffi.new("WGPURenderBundleId []", bundles2)
    #     _lib.wgpu_render_pass_execute_bundles(
    #         self._internal, c_bundles_array, len(bundles2),
    #     )

    # wgpu.help('renderpassencoderendpass', dev=True)
    def end_pass(self):
        _lib.wgpu_render_pass_end_pass(self._internal)


class GPURenderBundleEncoder(base.GPURenderBundleEncoder):
    pass

    # Not yet implemented in wgpu-native
    # def finish(self, *, label=""):
    #     ...


class GPUQueue(base.GPUQueue):
    # wgpu.help('queuesubmit', dev=True)
    def submit(self, command_buffers):
        command_buffer_ids = [cb._internal for cb in command_buffers]
        c_command_buffers = ffi.new("WGPUCommandBufferId []", command_buffer_ids)
        _lib.wgpu_queue_submit(
            self._internal, c_command_buffers, len(command_buffer_ids)
        )

    # Seems not yet implemented in wgpu-native
    # def copy_image_bitmap_to_texture(self, source, destination, copy_size):
    #     ...


class GPUSwapChain(base.GPUSwapChain):
    def __init__(self, device, canvas, format, usage):
        super().__init__("", None, device)
        self._canvas = canvas
        self._format = format
        self._usage = usage
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

        struct = new_struct(
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

        self._internal = _lib.wgpu_device_create_swap_chain(
            self._device._internal, self._surface_id, struct
        )

    def __enter__(self):
        # Get the current texture view, and make sure it is presented when done
        self._create_native_swap_chain_if_needed()
        swap_chain_output = _lib.wgpu_swap_chain_get_next_texture(self._internal)
        return base.GPUTextureView("swap_chain", swap_chain_output.view_id, self)

    def __exit__(self, type, value, tb):
        # Present the current texture
        _lib.wgpu_swap_chain_present(self._internal)


# %%


def _copy_docstrings():
    for ob in globals().values():
        if not (isinstance(ob, type) and issubclass(ob, base.GPUObject)):
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
