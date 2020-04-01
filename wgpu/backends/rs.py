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

from .. import base
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
    paths = []

    override_path = os.getenv("WGPU_LIB_PATH", "").strip()
    if override_path:
        paths.append(override_path)

    lib_filename = None
    if sys.platform.startswith("win"):  # no-cover
        lib_filename = "wgpu_native.dll"
    elif sys.platform.startswith("darwin"):  # no-cover
        lib_filename = "libwgpu_native.dylib"
    elif sys.platform.startswith("linux"):  # no-cover
        lib_filename = "libwgpu_native.so"
    if lib_filename:
        # Note that this can be a false positive, e.g. ARM linux.
        embedded_path = get_resource_filename(lib_filename)
        paths.append(embedded_path)

    for path in paths:
        if os.path.isfile(path):
            return path
    else:  # no-cover
        raise RuntimeError(f"Could not find WGPU library, checked: {paths}")


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
def request_adapter(*, power_preference: "GPUPowerPreference"):
    """ Request an GPUAdapter, the object that represents the implementation of WGPU.
    This function uses the Rust WGPU library.

    Params:
        power_preference(enum): "high-performance" or "low-power"
    """

    # Convert the descriptor
    struct = new_struct(
        "WGPURequestAdapterOptions *", power_preference=power_preference
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

    _lib.wgpu_request_adapter_async(
        struct, backend_mask, _request_adapter_callback, ffi.NULL
    )  # userdata, stub

    # For now, Rust will call the callback immediately
    # todo: when wgpu gets an event loop -> while run wgpu event loop or something

    assert adapter_id is not None
    extensions = []
    return GPUAdapter("WGPU", extensions, adapter_id)


# wgpu.help('RequestAdapterOptions', 'requestadapter', dev=True)
async def request_adapter_async(*, power_preference: "GPUPowerPreference"):
    """ Async version of ``request_adapter()``.
    This function uses the Rust WGPU library.
    """
    return request_adapter(power_preference=power_preference)  # no-cover


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

        # Fill in defaults of limits
        limits = limits or {}
        limits2 = {}
        for key in ["max_bind_groups"]:
            limits2[key] = limits.get(key, base.default_limits[key])

        extensions = tuple(extensions)

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

        id = _lib.wgpu_adapter_request_device(self._id, struct)

        queue_id = _lib.wgpu_device_get_queue(id)
        queue = GPUQueue("", queue_id, self)

        return GPUDevice(label, id, self, extensions, limits2, queue)

    # wgpu.help('DeviceDescriptor', 'adapterrequestdevice', dev=True)
    async def request_device_async(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        return self.request_device(
            label=label, extensions=extensions, limits=limits
        )  # no-cover


class GPUDevice(base.GPUDevice):
    # wgpu.help('BufferDescriptor', 'devicecreatebuffer', dev=True)
    def create_buffer(
        self, *, label="", size: "GPUSize64", usage: "GPUBufferUsageFlags"
    ):
        size = int(size)

        struct = new_struct("WGPUBufferDescriptor *", size=size, usage=usage)

        id = _lib.wgpu_device_create_buffer(self._internal, struct)
        return GPUBuffer(label, id, self, size, usage, "unmapped", None)

    # wgpu.help('BufferDescriptor', 'devicecreatebuffermapped', dev=True)
    def create_buffer_mapped(
        self, *, label="", size: "GPUSize64", usage: "GPUBufferUsageFlags"
    ):

        size = int(size)

        struct = new_struct("WGPUBufferDescriptor *", size=size, usage=usage)

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
        array_layer_count: "GPUIntegerCoordinate" = 1,
        mip_level_count: "GPUIntegerCoordinate" = 1,
        sample_count: "GPUSize32" = 1,
        dimension: "GPUTextureDimension" = "2d",
        format: "GPUTextureFormat",
        usage: "GPUTextureUsageFlags",
    ):
        size = _tuple_from_tuple_or_dict(size, ("width", "height", "depth"))
        c_size = new_struct(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )
        struct = new_struct(
            "WGPUTextureDescriptor *",
            size=c_size[0],
            array_layer_count=array_layer_count,
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
        compare: "GPUCompareFunction" = "never",
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
            compare_function=compare,
        )

        id = _lib.wgpu_device_create_sampler(self._internal, struct)
        return base.GPUSampler(label, id, self)

    # wgpu.help('BindGroupLayoutDescriptor', 'devicecreatebindgrouplayout', dev=True)
    def create_bind_group_layout(
        self, *, label="", bindings: "GPUBindGroupLayoutBinding-list"
    ):

        c_bindings_list = []
        for binding in bindings:
            c_binding = new_struct(
                "WGPUBindGroupLayoutBinding *",
                binding=int(binding["binding"]),
                visibility=int(binding["visibility"]),
                ty=binding["type"],
                texture_dimension=binding.get("texture_dimension", "2d"),
                # ???=binding.get("textureComponentType", "float"),
                multisampled=bool(binding.get("multisampled", False)),
                dynamic=bool(binding.get("has_dynamic_offset", False)),
            )
            c_bindings_list.append(c_binding[0])

        c_bindings_array = ffi.new("WGPUBindGroupLayoutBinding []", c_bindings_list)
        struct = new_struct(
            "WGPUBindGroupLayoutDescriptor *",
            bindings=c_bindings_array,
            bindings_length=len(c_bindings_list),
        )

        id = _lib.wgpu_device_create_bind_group_layout(self._internal, struct)

        return base.GPUBindGroupLayout(label, id, self, bindings)

    # wgpu.help('BindGroupDescriptor', 'devicecreatebindgroup', dev=True)
    def create_bind_group(
        self,
        *,
        label="",
        layout: "GPUBindGroupLayout",
        bindings: "GPUBindGroupBinding-list",
    ):

        c_bindings_list = []
        for binding in bindings:
            # The resource can be a sampler, texture view, or buffer descriptor
            resource = binding["resource"]
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
                c_buffer_binding = new_struct(
                    "WGPUBufferBinding *",
                    buffer=resource["buffer"]._internal,
                    offset=resource["offset"],
                    size=resource["size"],
                )
                c_resource_kwargs = {
                    "tag": 0,  # WGPUBindingResource_Tag.WGPUBindingResource_Buffer
                    "buffer": new_struct(
                        "WGPUBindingResource_WGPUBuffer_Body *", _0=c_buffer_binding[0]
                    )[0],
                }
            else:
                raise TypeError("Unexpected resource type {type(resource)}")  # no-cover
            c_resource = new_struct("WGPUBindingResource *", **c_resource_kwargs)
            c_binding = new_struct(
                "WGPUBindGroupBinding *",
                binding=int(binding["binding"]),
                resource=c_resource[0],
            )
            c_bindings_list.append(c_binding[0])

        c_bindings_array = ffi.new("WGPUBindGroupBinding []", c_bindings_list)
        struct = new_struct(
            "WGPUBindGroupDescriptor *",
            layout=layout._internal,
            bindings=c_bindings_array,
            bindings_length=len(c_bindings_list),
        )

        id = _lib.wgpu_device_create_bind_group(self._internal, struct)
        return base.GPUBindGroup(label, id, self, bindings)

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
    def create_shader_module(self, *, label="", code: "GPUShaderCode"):

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
        c_vertex_stage = new_struct(
            "WGPUProgrammableStageDescriptor *",
            module=vertex_stage["module"]._internal,
            entry_point=ffi.new("char []", vertex_stage["entry_point"].encode()),
        )
        c_fragment_stage = new_struct(
            "WGPUProgrammableStageDescriptor *",
            module=fragment_stage["module"]._internal,
            entry_point=ffi.new("char []", fragment_stage["entry_point"].encode()),
        )
        c_rasterization_state = new_struct(
            "WGPURasterizationStateDescriptor *",
            front_face=rasterization_state["front_face"],
            cull_mode=rasterization_state["cull_mode"],
            depth_bias=rasterization_state["depth_bias"],
            depth_bias_slope_scale=rasterization_state["depth_bias_slope_scale"],
            depth_bias_clamp=rasterization_state["depth_bias_clamp"],
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
                write_mask=color_state["write_mask"],
            )  # enum
            c_color_states_list.append(c_color_state[0])
        c_color_states_array = ffi.new(
            "WGPUColorStateDescriptor []", c_color_states_list
        )
        if depth_stencil_state is None:
            c_depth_stencil_state = ffi.NULL
        else:
            stencil_front = depth_stencil_state["stencil_front"]
            c_stencil_front = new_struct(
                "WGPUStencilStateFaceDescriptor *",
                compare=stencil_front["compare"],
                fail_op=stencil_front["fail_op"],
                depth_fail_op=stencil_front["depth_fail_op"],
                pass_op=stencil_front["pass_op"],
            )
            stencil_back = depth_stencil_state["stencil_back"]
            c_stencil_back = new_struct(
                "WGPUStencilStateFaceDescriptor *",
                compare=stencil_back["compare"],
                fail_op=stencil_back["fail_op"],
                depth_fail_op=stencil_back["depth_fail_op"],
                pass_op=stencil_back["pass_op"],
            )
            c_depth_stencil_state = new_struct(
                "WGPUDepthStencilStateDescriptor *",
                format=depth_stencil_state["format"],
                depth_write_enabled=bool(depth_stencil_state["depth_write_enabled"]),
                depth_compare=depth_stencil_state["depth_compare"],
                stencil_front=c_stencil_front[0],
                stencil_back=c_stencil_back[0],
                stencil_read_mask=depth_stencil_state["stencil_read_mask"],
                stencil_write_mask=depth_stencil_state["stencil_write_mask"],
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
                "WGPUVertexBufferDescriptor *",
                stride=buffer_des["array_stride"],
                step_mode=buffer_des["stepmode"],
                attributes=c_attributes_array,
                attributes_length=len(c_attributes_list),
            )
            c_vertex_buffer_descriptors_list.append(c_vertex_buffer_descriptor[0])
        c_vertex_buffer_descriptors_array = ffi.new(
            "WGPUVertexBufferDescriptor []", c_vertex_buffer_descriptors_list
        )
        c_vertex_input = new_struct(
            "WGPUVertexInputDescriptor *",
            index_format=vertex_state["index_format"],
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
            vertex_input=c_vertex_input[0],
            sample_count=sample_count,
            sample_mask=sample_mask,
            alpha_to_coverage_enabled=alpha_to_coverage_enabled,
        )

        id = _lib.wgpu_device_create_render_pipeline(self._internal, struct)
        return base.GPURenderPipeline(label, id, self)

    # wgpu.help('CommandEncoderDescriptor', 'devicecreatecommandencoder', dev=True)
    def create_command_encoder(self, *, label=""):

        struct = new_struct("WGPUCommandEncoderDescriptor *", todo=0)

        id = _lib.wgpu_device_create_command_encoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

    # Not yet implemented in wgpu-native
    # def create_render_bundle_encoder(
    #     self,
    #     *,
    #     label="",
    #     color_formats: "GPUTextureFormat-list",
    #     depth_stencil_format: "GPUTextureFormat",
    #     sample_count: "GPUSize32" = 1,
    # ):
    #     pass

    def _gui_configure_swap_chain(self, canvas, format, usage):
        """ Get a swapchain object from a canvas object. Called by BaseCanvas.
        """
        # Note: canvas should implement the BaseCanvas interface.
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

    # wgpu.help('bufferdestroy', dev=True)
    def destroy(self):
        if self._state != "destroyed":
            self._state = "destroyed"
            _lib.wgpu_buffer_destroy(self._internal)


class GPUTexture(base.GPUTexture):

    _destroyed = False

    # wgpu.help('TextureViewDescriptor', 'texturecreateview', dev=True)
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

    def create_default_view(self, *, label=""):
        # This method is available in wgpu-rs, and it's kinda nice :)
        id = _lib.wgpu_texture_create_view(self._internal, ffi.NULL)
        return base.GPUTextureView(label, id, self)

    # wgpu.help('texturedestroy', dev=True)
    def destroy(self):
        if not self._destroyed:
            self._destroyed = True
            _lib.wgpu_texture_destroy(self._internal)


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
        depth_stencil_attachment: "GPURenderPassDepthStencilAttachmentDescriptor",
    ):

        c_color_attachments_list = []
        for color_attachment in color_attachments:
            assert isinstance(color_attachment["attachment"], base.GPUTextureView)
            texture_view_id = color_attachment["attachment"]._internal
            c_resolve_target = (
                ffi.NULL
                if color_attachment["resolve_target"] is None
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
                store_op=color_attachment["store_op"],
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
            row_pitch=int(source["row_pitch"]),
            image_height=int(source["image_height"]),
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
            row_pitch=int(destination["row_pitch"]),
            image_height=int(destination["image_height"]),
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


class GPUProgrammablePassEncoder(base.GPUProgrammablePassEncoder):
    # wgpu.help('BindGroup', 'Index32', 'Size64', 'programmablepassencodersetbindgroup', dev=True)
    def set_bind_group(
        self,
        index,
        bind_group,
        dynamic_offsets_data,
        dynamic_offsets_data_start,
        dynamic_offsets_data_length,
    ):
        offsets = list(dynamic_offsets_data)
        c_offsets = ffi.new("WGPUBufferAddress []", offsets)
        bind_group_id = bind_group._internal
        if isinstance(self, GPUComputePassEncoder):
            _lib.wgpu_compute_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )
        else:
            _lib.wgpu_render_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )

    # # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    # def push_debug_group(self, group_label):
    #     ...
    #
    # # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    # def pop_debug_group(self):
    #     ...
    #
    # # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    # def insert_debug_marker(self, marker_label):
    #     ...


class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('ComputePipeline', 'computepassencodersetpipeline', dev=True)
    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        _lib.wgpu_compute_pass_set_pipeline(self._internal, pipeline_id)

    # wgpu.help('Size32', 'computepassencoderdispatch', dev=True)
    def dispatch(self, x, y, z):
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


class GPURenderEncoderBase(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('RenderPipeline', 'renderencoderbasesetpipeline', dev=True)
    def set_pipeline(self, pipeline):
        pipeline_id = pipeline._internal
        _lib.wgpu_render_pass_set_pipeline(self._internal, pipeline_id)

    # wgpu.help('Buffer', 'Size64', 'renderencoderbasesetindexbuffer', dev=True)
    def set_index_buffer(self, buffer, offset):
        _lib.wgpu_render_pass_set_index_buffer(self._internal, buffer._internal, offset)

    # wgpu.help('Buffer', 'Index32', 'Size64', 'renderencoderbasesetvertexbuffer', dev=True)
    def set_vertex_buffer(self, slot, buffer, offset):
        buffers, offsets = [buffer], [offset]
        c_buffer_ids = ffi.new("WGPUBufferId []", [b._internal for b in buffers])
        c_offsets = ffi.new("WGPUBufferAddress []", [int(i) for i in offsets])
        _lib.wgpu_render_pass_set_vertex_buffers(
            self._internal, slot, c_buffer_ids, c_offsets, len(buffers)
        )

    # wgpu.help('Size32', 'renderencoderbasedraw', dev=True)
    def draw(self, vertex_count, instance_count, first_vertex, first_instance):
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
        self, index_count, instance_count, first_index, base_vertex, first_instance
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

        if self._surface_id is None:
            self._surface_id = get_surface_id_from_canvas(canvas)

        self._internal = _lib.wgpu_device_create_swap_chain(
            self._device._internal, self._surface_id, struct
        )

    def get_current_texture_view(self):
        # todo: should we cache instances (on their id)?
        # otherwise we have multiple instances mapping to same internal texture
        self._create_native_swap_chain_if_needed()
        swap_chain_output = _lib.wgpu_swap_chain_get_next_texture(self._internal)
        return base.GPUTextureView("swap_chain", swap_chain_output.view_id, self)

    def _gui_present(self):
        """ Present the current texture. This is not part of the public API,
        instead, GUI backends should call this at the right moment.
        """
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
