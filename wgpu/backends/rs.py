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

from .. import base, flags, _structs
from .. import _register_backend
from .._coreutils import get_resource_filename, logger_set_level_callbacks
from .._mappings import cstructfield2enum, enummap


logger = logging.getLogger("wgpu")  # noqa

# wgpu-native version that we target/expect
__version__ = "0.5.2"
__commit_sha__ = "160be433dbec0fc7a27d25f2aba3423666ccfa10"
version_info = tuple(map(int, __version__.split(".")))


if cffi_version_info < (1, 10):  # no-cover
    raise ImportError(f"{__name__} needs cffi 1.10 or later.")


# %% Load the lib and integrate logging system


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


# Get the actual wgpu-native version
_version_int = _lib.wgpu_get_version()
version_info_lib = tuple((_version_int >> bits) & 0xFF for bits in (16, 8, 0))
if version_info_lib != version_info:  # no-cover
    logger.warning(
        f"Expected wgpu-native version {version_info} but got {version_info_lib}"
    )


@ffi.callback("void(int level, const char *)")
def _logger_callback(level, c_msg):
    """ Called when Rust emits a log message.
    """
    msg = ffi.string(c_msg).decode(errors="ignore")  # make a copy
    # todo: We currently skip some false negatives to avoid spam.
    false_negatives = (
        "Unknown decoration",
        "Failed to parse shader",
        "Shader module will not be validated",
    )
    if msg.startswith(false_negatives):
        return
    m = {
        _lib.WGPULogLevel_Error: logger.error,
        _lib.WGPULogLevel_Warn: logger.warning,
        _lib.WGPULogLevel_Info: logger.info,
        _lib.WGPULogLevel_Debug: logger.debug,
        _lib.WGPULogLevel_Trace: logger.debug,
    }
    func = m.get(level, logger.warning)
    func(msg)


def _logger_set_level_callback(level):
    """ Called when the log level is set from Python.
    """
    if level >= 40:
        _lib.wgpu_set_log_level(_lib.WGPULogLevel_Error)
    elif level >= 30:
        _lib.wgpu_set_log_level(_lib.WGPULogLevel_Warn)
    elif level >= 20:
        _lib.wgpu_set_log_level(_lib.WGPULogLevel_Info)
    elif level >= 10:
        _lib.wgpu_set_log_level(_lib.WGPULogLevel_Debug)
    elif level >= 5:
        _lib.wgpu_set_log_level(_lib.WGPULogLevel_Trace)  # extra level
    else:
        _lib.wgpu_set_log_level(_lib.WGPULogLevel_Off)


# Connect Rust logging with Python logging
_lib.wgpu_set_log_callback(_logger_callback)
logger_set_level_callbacks.append(_logger_set_level_callback)
_logger_set_level_callback(logger.level)


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
    """ Create a pointer to an ffi struct. Provides a flatter syntax
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
    """ Create an ffi value struct. The passed kwargs are also bound
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
            if key in _cstructfield2enum_alt:
                structname = _cstructfield2enum_alt[key]
            else:
                structname = cstructfield2enum[ctype.strip(" *")[4:] + "." + key]
            ival = enummap[structname + "." + val]
            setattr(struct_p, key, ival)
        else:
            setattr(struct_p, key, val)
    return struct_p


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


def _get_memoryview_and_address(data):
    """ Get a memoryview for the given data and its memory address.
    The data object must support the buffer protocol.
    """

    # To get the address from a memoryview, there are multiple options.
    # The most obvious is using ctypes:
    #
    #   c_array = (ctypes.c_uint8 * nbytes).from_buffer(m)
    #   address = ctypes.addressof(c_array)
    #
    # Unfortunately, this call fails if the memoryview is readonly, e.g. if
    # the data is a bytes object or readonly numpy array. One could then
    # use from_buffer_copy(), but that introduces an extra data copy, which
    # can hurt performance when the data is large.
    #
    # Another alternative that can be used for objects implementing the array
    # interface (like numpy arrays) is to directly read the address:
    #
    #   address = data.__array_interface__["data"][0]
    #
    # But what seems to work best (at the moment) is using cffi.

    # Convert data to a memoryview. That way we have something consistent
    # to work with, which supports all objects implementing the buffer protocol.
    m = memoryview(data)

    # Get the address via ffi. In contrast to ctypes, this also
    # works for readonly data (e.g. bytes)
    c_data = ffi.from_buffer("uint8_t []", m)
    address = int(ffi.cast("uintptr_t", c_data))

    return m, address


def _get_memoryview_from_address(address, nbytes, format="B"):
    """ Get a memoryview from an int memory address and a byte count,
    """
    # The default format is "<B", which seems to confuse some memoryview
    # operations, so we always cast it.
    c_array = (ctypes.c_uint8 * nbytes).from_address(address)
    return memoryview(c_array).cast(format, shape=(nbytes,))


def _check_struct(what, d):
    """ Check that the given dict does not have any unexpected keys
    (which may be there because of typos or api changes).
    """
    fields = set(d.keys())
    ref_fields = getattr(_structs, what).keys()
    unexpected = fields.difference(ref_fields)
    if unexpected:
        s1 = ", ".join(unexpected)
        s2 = ", ".join(ref_fields)
        raise ValueError(f"Unexpected keys: {s1}.\n  -> for {what}: {s2}.")


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
        return self._request_device(label, extensions, limits, "")

    def request_device_tracing(
        self,
        trace_path,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        """ Write a trace of all commands to a file so it can be reproduced
        elsewhere. The trace is cross-platform!
        """
        if not os.path.isdir(trace_path):
            os.makedirs(trace_path, exist_ok=True)
        elif os.listdir(trace_path):
            logger.warning(f"Trace directory not empty: {trace_path}")
        return self._request_device(label, extensions, limits, trace_path)

    def _request_device(self, label, extensions, limits, trace_path):

        c_trace_path = ffi.NULL
        if trace_path:  # no-cover
            c_trace_path = ffi.new("char []", trace_path.encode())

        # Handle default limits
        _check_struct("Limits", limits)
        limits2 = base.default_limits.copy()
        limits2.update(limits or {})

        c_extensions = new_struct(
            "WGPUExtensions",
            anisotropic_filtering="anisotropic_filtering" in extensions,
        )
        c_limits = new_struct("WGPULimits", max_bind_groups=limits2["max_bind_groups"])
        struct = new_struct_p(
            "WGPUDeviceDescriptor *", extensions=c_extensions, limits=c_limits
        )
        device_id = _lib.wgpu_adapter_request_device(self._id, struct, c_trace_path)

        # Get the actual limits reported by the device
        c_limits = new_struct_p("WGPULimits *")
        _lib.wgpu_device_get_limits(device_id, c_limits)
        limits3 = {key: getattr(c_limits, key) for key in dir(c_limits)}

        # Get the queue to which commands can be submitted
        queue_id = _lib.wgpu_device_get_default_queue(device_id)
        queue = GPUQueue("", queue_id, None)

        return GPUDevice(label, device_id, self, extensions, limits3, queue)

    # wgpu.help('DeviceDescriptor', 'adapterrequestdevice', dev=True)
    async def request_device_async(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        return self._request_device(label, extensions, limits, "")  # no-cover

    def _destroy(self):
        if self._id is not None:
            self._id, id = None, self._id
            _lib.wgpu_adapter_destroy(id)


class GPUDevice(base.GPUDevice):

    # wgpu.help('BufferDescriptor', 'devicecreatebuffer', dev=True)
    def create_buffer(
        self,
        *,
        label="",
        size: int,
        usage: "GPUBufferUsageFlags",
        mapped_at_creation: bool = False,
    ):
        size = int(size)
        if mapped_at_creation:
            raise ValueError(
                "In wgpu-py, mapped_at_creation must be False. Use create_buffer_with_data() instead."
            )
        # Create a buffer object
        c_label = ffi.new("char []", label.encode())
        struct = new_struct_p(
            "WGPUBufferDescriptor *", label=c_label, size=size, usage=usage
        )
        id = _lib.wgpu_device_create_buffer(self._internal, struct)
        # Return wrapped buffer
        return GPUBuffer(label, id, self, size, usage, "unmapped")

    def create_buffer_with_data(self, *, label="", data, usage: "GPUBufferUsageFlags"):
        # Get a memoryview of the data
        m, src_address = _get_memoryview_and_address(data)
        if not m.contiguous:  # no-cover
            raise ValueError("The given texture data is not contiguous")
        m = m.cast("B", shape=(m.nbytes,))
        # Create a buffer object, and get a memory pointer to its mapped memory
        c_label = ffi.new("char []", label.encode())
        struct = new_struct_p(
            "WGPUBufferDescriptor *", label=c_label, size=m.nbytes, usage=usage
        )
        buffer_memory_pointer = ffi.new("uint8_t * *")
        id = _lib.wgpu_device_create_buffer_mapped(
            self._internal, struct, buffer_memory_pointer
        )
        # Copy the data to the mapped memory
        dst_address = int(ffi.cast("intptr_t", buffer_memory_pointer[0]))
        dst_m = _get_memoryview_from_address(dst_address, m.nbytes)
        dst_m[:] = m  # nicer than ctypes.memmove(dst_address, src_address, m.nbytes)
        _lib.wgpu_buffer_unmap(id)
        # Return the wrapped buffer
        return GPUBuffer(label, id, self, m.nbytes, usage, "unmapped")

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
            "WGPUExtent3d", width=size[0], height=size[1], depth=size[2],
        )
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
        id = _lib.wgpu_device_create_texture(self._internal, struct)

        tex_info = {
            "size": size,
            "mip_level_count": mip_level_count,
            "sample_count": sample_count,
            "dimension": dimension,
            "format": format,
            "usage": usage,
        }
        return GPUTexture(label, id, self, tex_info)

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
        c_label = ffi.new("char []", label.encode())
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
        )

        id = _lib.wgpu_device_create_sampler(self._internal, struct)
        return base.GPUSampler(label, id, self)

    # wgpu.help('BindGroupLayoutDescriptor', 'devicecreatebindgrouplayout', dev=True)
    def create_bind_group_layout(
        self, *, label="", entries: "GPUBindGroupLayoutEntry-list"
    ):
        c_entries_list = []
        for entry in entries:
            _check_struct("BindGroupLayoutEntry", entry)
            type = entry["type"]
            if "texture" in type:
                need = {"view_dimension"}
                if "storage" in type:
                    need.add("storage_texture_format")
                else:
                    need.add("texture_component_type")
                assert all(
                    x in entry for x in need
                ), f"{type} binding should specify {need}"
            c_entry = new_struct(
                "WGPUBindGroupLayoutEntry",
                binding=int(entry["binding"]),
                visibility=int(entry["visibility"]),
                ty=type,
                # Used for uniform buffer and storage buffer bindings.
                has_dynamic_offset=bool(entry.get("has_dynamic_offset", False)),
                # Used for sampled texture and storage texture bindings.
                view_dimension=entry.get("view_dimension", "2d"),
                # Used for sampled texture bindings.
                texture_component_type=entry.get("texture_component_type", "float"),
                # Used for sampled texture bindings.
                multisampled=bool(entry.get("multisampled", False)),
                # Used for storage texture bindings.
                storage_texture_format=entry.get("storage_texture_format", 0),
            )
            c_entries_list.append(c_entry)

        c_label = ffi.new("char []", label.encode())
        struct = new_struct_p(
            "WGPUBindGroupLayoutDescriptor *",
            label=c_label,
            entries=ffi.new("WGPUBindGroupLayoutEntry []", c_entries_list),
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
            _check_struct("BindGroupEntry", entry)
            # The resource can be a sampler, texture view, or buffer descriptor
            resource = entry["resource"]
            if isinstance(resource, base.GPUSampler):
                c_resource_kwargs = {
                    "tag": 1,  # WGPUBindingResource_Tag.WGPUBindingResource_Sampler
                    "sampler": new_struct(
                        "WGPUBindingResource_WGPUSampler_Body", _0=resource._internal
                    ),
                }
            elif isinstance(resource, base.GPUTextureView):
                c_resource_kwargs = {
                    "tag": 2,  # WGPUBindingResource_Tag.WGPUBindingResource_TextureView
                    "texture_view": new_struct(
                        "WGPUBindingResource_WGPUTextureView_Body",
                        _0=resource._internal,
                    ),
                }
            elif isinstance(resource, dict):  # Buffer binding
                _check_struct("BufferBinding", resource)
                c_buffer_entry = new_struct(
                    "WGPUBufferBinding",
                    buffer=resource["buffer"]._internal,
                    offset=resource["offset"],
                    size=resource["size"],
                )
                c_resource_kwargs = {
                    "tag": 0,  # WGPUBindingResource_Tag.WGPUBindingResource_Buffer
                    "buffer": new_struct(
                        "WGPUBindingResource_WGPUBuffer_Body", _0=c_buffer_entry
                    ),
                }
            else:
                raise TypeError(f"Unexpected resource type {type(resource)}")
            c_resource = new_struct("WGPUBindingResource", **c_resource_kwargs)
            c_entry = new_struct(
                "WGPUBindGroupEntry",
                binding=int(entry["binding"]),
                resource=c_resource,
            )
            c_entries_list.append(c_entry)

        c_label = ffi.new("char []", label.encode())
        c_entries_array = ffi.new("WGPUBindGroupEntry []", c_entries_list)
        struct = new_struct_p(
            "WGPUBindGroupDescriptor *",
            label=c_label,
            layout=layout._internal,
            entries=c_entries_array,
            entries_length=len(c_entries_list),
        )

        id = _lib.wgpu_device_create_bind_group(self._internal, struct)
        return GPUBindGroup(label, id, self, entries)

    # wgpu.help('PipelineLayoutDescriptor', 'devicecreatepipelinelayout', dev=True)
    def create_pipeline_layout(
        self, *, label="", bind_group_layouts: "GPUBindGroupLayout-list"
    ):

        bind_group_layouts_ids = [x._internal for x in bind_group_layouts]

        c_layout_array = ffi.new("WGPUBindGroupLayoutId []", bind_group_layouts_ids)
        struct = new_struct_p(
            "WGPUPipelineLayoutDescriptor *",
            bind_group_layouts=c_layout_array,
            bind_group_layouts_length=len(bind_group_layouts),
        )

        id = _lib.wgpu_device_create_pipeline_layout(self._internal, struct)
        return GPUPipelineLayout(label, id, self, bind_group_layouts)

    # wgpu.help('ShaderModuleDescriptor', 'devicecreateshadermodule', dev=True)
    def create_shader_module(self, *, label="", code: str, source_map: "dict" = None):

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

        struct = new_struct_p("WGPUShaderModuleDescriptor *", code=c_code)

        id = _lib.wgpu_device_create_shader_module(self._internal, struct)
        return GPUShaderModule(label, id, self)

    # wgpu.help('ComputePipelineDescriptor', 'devicecreatecomputepipeline', dev=True)
    def create_compute_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
        compute_stage: "GPUProgrammableStageDescriptor",
    ):
        _check_struct("ProgrammableStageDescriptor", compute_stage)

        c_compute_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=compute_stage["module"]._internal,
            entry_point=ffi.new("char []", compute_stage["entry_point"].encode()),
        )

        struct = new_struct_p(
            "WGPUComputePipelineDescriptor *",
            layout=layout._internal,
            compute_stage=c_compute_stage,
        )

        id = _lib.wgpu_device_create_compute_pipeline(self._internal, struct)
        return GPUComputePipeline(label, id, self, layout)

    # wgpu.help('RenderPipelineDescriptor', 'devicecreaterenderpipeline', dev=True)
    def create_render_pipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout" = None,
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
        _check_struct("ProgrammableStageDescriptor", vertex_stage)
        _check_struct("RasterizationStateDescriptor", rasterization_state)
        _check_struct("VertexStateDescriptor", vertex_state)

        c_vertex_stage = new_struct(
            "WGPUProgrammableStageDescriptor",
            module=vertex_stage["module"]._internal,
            entry_point=ffi.new("char []", vertex_stage["entry_point"].encode()),
        )
        c_fragment_stage = ffi.NULL
        if fragment_stage is not None:
            _check_struct("ProgrammableStageDescriptor", fragment_stage)
            c_fragment_stage = new_struct_p(
                "WGPUProgrammableStageDescriptor *",
                module=fragment_stage["module"]._internal,
                entry_point=ffi.new("char []", fragment_stage["entry_point"].encode()),
            )
        c_rasterization_state = new_struct_p(
            "WGPURasterizationStateDescriptor *",
            front_face=rasterization_state.get("front_face", "ccw"),
            cull_mode=rasterization_state.get("cull_mode", "none"),
            depth_bias=rasterization_state.get("depth_bias", 0),
            depth_bias_slope_scale=rasterization_state.get("depth_bias_slope_scale", 0),
            depth_bias_clamp=rasterization_state.get("depth_bias_clamp", 0),
        )
        c_color_states_list = []
        for color_state in color_states:
            _check_struct("ColorStateDescriptor", color_state)
            alpha_blend = _tuple_from_tuple_or_dict(
                color_state["alpha_blend"], ("src_factor", "dst_factor", "operation"),
            )
            c_alpha_blend = new_struct(
                "WGPUBlendDescriptor",
                src_factor=alpha_blend[0],
                dst_factor=alpha_blend[1],
                operation=alpha_blend[2],
            )
            color_blend = _tuple_from_tuple_or_dict(
                color_state["color_blend"], ("src_factor", "dst_factor", "operation"),
            )
            c_color_blend = new_struct(
                "WGPUBlendDescriptor",
                src_factor=color_blend[0],
                dst_factor=color_blend[1],
                operation=color_blend[2],
            )
            c_color_state = new_struct(
                "WGPUColorStateDescriptor",
                format=color_state["format"],
                alpha_blend=c_alpha_blend,
                color_blend=c_color_blend,
                write_mask=color_state.get("write_mask", 0xF),
            )
            c_color_states_list.append(c_color_state)
        c_color_states_array = ffi.new(
            "WGPUColorStateDescriptor []", c_color_states_list
        )
        if depth_stencil_state is None:
            c_depth_stencil_state = ffi.NULL
        else:
            _check_struct("DepthStencilStateDescriptor", depth_stencil_state)
            stencil_front = depth_stencil_state.get("stencil_front", {})
            _check_struct("StencilStateFaceDescriptor", stencil_front)
            c_stencil_front = new_struct(
                "WGPUStencilStateFaceDescriptor",
                compare=stencil_front.get("compare", "always"),
                fail_op=stencil_front.get("fail_op", "keep"),
                depth_fail_op=stencil_front.get("depth_fail_op", "keep"),
                pass_op=stencil_front.get("pass_op", "keep"),
            )
            stencil_back = depth_stencil_state.get("stencil_back", {})
            c_stencil_back = new_struct(
                "WGPUStencilStateFaceDescriptor",
                compare=stencil_back.get("compare", "always"),
                fail_op=stencil_back.get("fail_op", "keep"),
                depth_fail_op=stencil_back.get("depth_fail_op", "keep"),
                pass_op=stencil_back.get("pass_op", "keep"),
            )
            c_depth_stencil_state = new_struct_p(
                "WGPUDepthStencilStateDescriptor *",
                format=depth_stencil_state["format"],
                depth_write_enabled=bool(
                    depth_stencil_state.get("depth_write_enabled", False)
                ),
                depth_compare=depth_stencil_state.get("depth_compare", "always"),
                stencil_front=c_stencil_front,
                stencil_back=c_stencil_back,
                stencil_read_mask=depth_stencil_state.get(
                    "stencil_read_mask", 0xFFFFFFFF
                ),
                stencil_write_mask=depth_stencil_state.get(
                    "stencil_write_mask", 0xFFFFFFFF
                ),
            )
        c_vertex_buffer_descriptors_list = []
        for buffer_des in vertex_state["vertex_buffers"]:
            _check_struct("VertexBufferLayoutDescriptor", buffer_des)
            c_attributes_list = []
            for attribute in buffer_des["attributes"]:
                _check_struct("VertexAttributeDescriptor", attribute)
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
        c_vertex_state = new_struct(
            "WGPUVertexStateDescriptor",
            index_format=vertex_state.get("index_format", "uint32"),
            vertex_buffers=c_vertex_buffer_descriptors_array,
            vertex_buffers_length=len(c_vertex_buffer_descriptors_list),
        )

        struct = new_struct_p(
            "WGPURenderPipelineDescriptor *",
            layout=layout._internal,
            vertex_stage=c_vertex_stage,
            fragment_stage=c_fragment_stage,
            primitive_topology=primitive_topology,
            rasterization_state=c_rasterization_state,
            color_states=c_color_states_array,
            color_states_length=len(c_color_states_list),
            depth_stencil_state=c_depth_stencil_state,
            vertex_state=c_vertex_state,
            sample_count=sample_count,
            sample_mask=sample_mask,
            alpha_to_coverage_enabled=alpha_to_coverage_enabled,
        )

        id = _lib.wgpu_device_create_render_pipeline(self._internal, struct)
        return GPURenderPipeline(label, id, self, layout)

    # wgpu.help('CommandEncoderDescriptor', 'devicecreatecommandencoder', dev=True)
    def create_command_encoder(self, *, label=""):
        c_label = ffi.new("char []", label.encode())
        struct = new_struct_p("WGPUCommandEncoderDescriptor *", label=c_label)

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
                data = _get_memoryview_from_address(address, size)

        _lib.wgpu_buffer_map_read_async(
            self._internal, start, size, _map_read_callback, ffi.NULL
        )

        # Let it do some cycles
        self._state = "mapping pending"
        self._map_mode = flags.MapMode.READ
        _lib.wgpu_device_poll(self._device._internal, True)

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
                data = _get_memoryview_from_address(address, size)

        _lib.wgpu_buffer_map_write_async(
            self._internal, start, size, _map_write_callback, ffi.NULL
        )

        # Let it do some cycles
        self._state = "mapping pending"
        _lib.wgpu_device_poll(self._device._internal, True)

        if data is None:  # no-cover
            raise RuntimeError("Could not read buffer data.")

        self._state = "mapped"
        self._map_mode = flags.MapMode.WRITE
        return memoryview(data)

    def _unmap(self):
        _lib.wgpu_buffer_unmap(self._internal)
        self._state = "unmapped"
        self._map_mode = 0

    # wgpu.help('bufferdestroy', dev=True)
    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            self._state = "destroyed"
            self._map_mode = 0
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
            c_label = ffi.new("char []", label.encode())
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
            id = _lib.wgpu_texture_create_view(self._internal, struct)

        return base.GPUTextureView(label, id, self._device, self, self.texture_size)

    # wgpu.help('texturedestroy', dev=True)
    def destroy(self):
        self._destroy()  # no-cover

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_texture_destroy(internal)


class GPUBindGroup(base.GPUBindGroup):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_bind_group_layout_destroy(internal)


class GPUPipelineLayout(base.GPUPipelineLayout):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_pipeline_layout_destroy(internal)


class GPUShaderModule(base.GPUShaderModule):
    # wgpu.help('shadermodulecompilationinfo', dev=True)
    def compilation_info(self):
        return super().compilation_info()

    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_shader_module_destroy(internal)


class GPUComputePipeline(base.GPUComputePipeline):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_compute_pipeline_destroy(internal)


class GPURenderPipeline(base.GPURenderPipeline):
    def _destroy(self):
        if self._internal is not None:
            self._internal, internal = None, self._internal
            _lib.wgpu_render_pipeline_destroy(internal)


class GPUCommandEncoder(base.GPUCommandEncoder):
    # wgpu.help('ComputePassDescriptor', 'commandencoderbegincomputepass', dev=True)
    def begin_compute_pass(self, *, label=""):
        struct = new_struct_p("WGPUComputePassDescriptor *", todo=0)
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
            _check_struct("RenderPassColorAttachmentDescriptor", color_attachment)
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
                "WGPUColor", r=clr[0], g=clr[1], b=clr[2], a=clr[3]
            )
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
            _check_struct(
                "RenderPassDepthStencilAttachmentDescriptor", depth_stencil_attachment
            )
            c_depth_load_op, c_depth_clear = _loadop_and_clear_from_value(
                depth_stencil_attachment["depth_load_value"]
            )
            c_stencil_load_op, c_stencil_clear = _loadop_and_clear_from_value(
                depth_stencil_attachment["stencil_load_value"]
            )
            c_depth_stencil_attachment = new_struct_p(
                "WGPURenderPassDepthStencilAttachmentDescriptor *",
                attachment=depth_stencil_attachment["attachment"]._internal,
                depth_load_op=c_depth_load_op,
                depth_store_op=depth_stencil_attachment["depth_store_op"],
                clear_depth=float(c_depth_clear),
                stencil_load_op=c_stencil_load_op,
                stencil_store_op=depth_stencil_attachment["stencil_store_op"],
                clear_stencil=int(c_stencil_clear),
            )

        struct = new_struct_p(
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

        _check_struct("BufferCopyView", source)
        _check_struct("TextureCopyView", destination)

        c_source = new_struct_p(
            "WGPUBufferCopyView *",
            buffer=source["buffer"]._internal,
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(source.get("offset", 0)),
                bytes_per_row=int(source["bytes_per_row"]),
                rows_per_image=int(source.get("rows_per_image", 0)),
            ),
        )

        ori = _tuple_from_tuple_or_dict(destination["origin"], "xyz")
        c_origin = new_struct("WGPUOrigin3d", x=ori[0], y=ori[1], z=ori[2])
        c_destination = new_struct_p(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=int(destination.get("mip_level", 0)),
            origin=c_origin,
        )

        size = _tuple_from_tuple_or_dict(copy_size, ("width", "height", "depth"))
        c_copy_size = new_struct_p(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_command_encoder_copy_buffer_to_texture(
            self._internal, c_source, c_destination, c_copy_size,
        )

    # wgpu.help('BufferCopyView', 'Extent3D', 'TextureCopyView', 'commandencodercopytexturetobuffer', dev=True)
    def copy_texture_to_buffer(self, source, destination, copy_size):

        _check_struct("TextureCopyView", source)
        _check_struct("BufferCopyView", destination)

        ori = _tuple_from_tuple_or_dict(source["origin"], "xyz")
        c_origin = new_struct("WGPUOrigin3d", x=ori[0], y=ori[1], z=ori[2])
        c_source = new_struct_p(
            "WGPUTextureCopyView *",
            texture=source["texture"]._internal,
            mip_level=int(source.get("mip_level", 0)),
            origin=c_origin,
        )

        c_destination = new_struct_p(
            "WGPUBufferCopyView *",
            buffer=destination["buffer"]._internal,
            layout=new_struct(
                "WGPUTextureDataLayout",
                offset=int(destination.get("offset", 0)),
                bytes_per_row=int(destination["bytes_per_row"]),
                rows_per_image=int(destination.get("rows_per_image", 0)),
            ),
        )

        size = _tuple_from_tuple_or_dict(copy_size, ("width", "height", "depth"))
        c_copy_size = new_struct_p(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_command_encoder_copy_texture_to_buffer(
            self._internal, c_source, c_destination, c_copy_size,
        )

    # wgpu.help('Extent3D', 'TextureCopyView', 'commandencodercopytexturetotexture', dev=True)
    def copy_texture_to_texture(self, source, destination, copy_size):

        _check_struct("TextureCopyView", source)
        _check_struct("TextureCopyView", destination)

        ori = _tuple_from_tuple_or_dict(source["origin"], "xyz")
        c_origin1 = new_struct("WGPUOrigin3d", x=ori[0], y=ori[1], z=ori[2])
        c_source = new_struct_p(
            "WGPUTextureCopyView *",
            texture=source["texture"]._internal,
            mip_level=int(source.get("mip_level", 0)),
            origin=c_origin1,
        )

        ori = _tuple_from_tuple_or_dict(destination["origin"], "xyz")
        c_origin2 = new_struct("WGPUOrigin3d", x=ori[0], y=ori[1], z=ori[2])
        c_destination = new_struct_p(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=int(destination.get("mip_level", 0)),
            origin=c_origin2,
        )

        size = _tuple_from_tuple_or_dict(copy_size, ("width", "height", "depth"))
        c_copy_size = new_struct_p(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_command_encoder_copy_texture_to_texture(
            self._internal, c_source, c_destination, c_copy_size,
        )

    # wgpu.help('CommandBufferDescriptor', 'commandencoderfinish', dev=True)
    def finish(self, *, label=""):
        struct = new_struct_p("WGPUCommandBufferDescriptor *", todo=0)
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
        if not size:
            size = buffer.size - offset
        _lib.wgpu_render_pass_set_index_buffer(
            self._internal, buffer._internal, int(offset), int(size)
        )

    # wgpu.help('Buffer', 'Index32', 'Size64', 'renderencoderbasesetvertexbuffer', dev=True)
    def set_vertex_buffer(self, slot, buffer, offset=0, size=0):
        if not size:
            size = buffer.size - offset
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
        c_color = new_struct_p(
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

    # wgpu.help('Buffer', 'Size64', 'queuewritebuffer', dev=True)
    def write_buffer(self, buffer, buffer_offset, data, data_offset=0, size=None):

        # We support anything that memoryview supports, i.e. anything
        # that implements the buffer protocol, including, bytes,
        # bytearray, ctypes arrays, numpy arrays, etc.
        m, address = _get_memoryview_and_address(data)
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
        _lib.wgpu_queue_write_buffer(
            self._internal, buffer._internal, buffer_offset, c_data, data_length
        )

    # wgpu.help('Extent3D', 'TextureCopyView', 'TextureDataLayout', 'queuewritetexture', dev=True)
    def write_texture(self, destination, data, data_layout, size):

        m, address = _get_memoryview_and_address(data)
        # todo: could we not derive the size from the shape of m?

        # Checks
        if not m.contiguous:  # no-cover
            raise ValueError("The given texture data is not contiguous")

        c_data = ffi.cast("uint8_t *", address)
        data_length = m.nbytes

        ori = _tuple_from_tuple_or_dict(destination.get("origin", (0, 0, 0)), "xyz")
        c_origin = new_struct("WGPUOrigin3d", x=ori[0], y=ori[1], z=ori[2])
        c_destination = new_struct_p(
            "WGPUTextureCopyView *",
            texture=destination["texture"]._internal,
            mip_level=destination.get("mip_level", 0),
            origin=c_origin,
        )

        c_data_layout = new_struct_p(
            "WGPUTextureDataLayout *",
            offset=data_layout.get("offset", 0),
            bytes_per_row=data_layout["bytes_per_row"],
            rows_per_image=data_layout.get("rows_per_image", 0),
        )

        size = _tuple_from_tuple_or_dict(size, ("width", "height", "depth"))
        c_size = new_struct_p(
            "WGPUExtent3d *", width=size[0], height=size[1], depth=size[2],
        )

        _lib.wgpu_queue_write_texture(
            self._internal, c_destination, c_data, data_length, c_data_layout, c_size
        )


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

        self._internal = _lib.wgpu_device_create_swap_chain(
            self._device._internal, self._surface_id, struct
        )

    def __enter__(self):
        # Get the current texture view, and make sure it is presented when done
        self._create_native_swap_chain_if_needed()
        sc_output = _lib.wgpu_swap_chain_get_next_texture(self._internal)
        status, view_id = sc_output.status, sc_output.view_id
        if status == _lib.WGPUSwapChainStatus_Good:
            pass
        elif status == _lib.WGPUSwapChainStatus_Suboptimal:  # no-cover
            if not getattr(self, "_warned_swap_chain_suboptimal", False):
                logger.warning(f"Swap chain status of {self} is suboptimal")
                self._warned_swap_chain_suboptimal = True
        else:  # no-cover
            status_str = swap_chain_status_map.get(status, "")
            raise RuntimeError(
                f"Swap chain status is not good: {status_str} ({status})"
            )
        size = self._surface_size[0], self._surface_size[1], 1
        return base.GPUTextureView("swap_chain", view_id, self._device, None, size)

    def __exit__(self, type, value, tb):
        # Present the current texture
        _lib.wgpu_swap_chain_present(self._internal)


swap_chain_status_map = {
    getattr(_lib, "WGPUSwapChainStatus_" + x): x
    for x in ("Good", "Suboptimal", "Lost", "Outdated", "OutOfMemory", "Timeout")
}

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
