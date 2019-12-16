"""
WGPU backend implementation based on the wgpu library.

The Rust wgpu project (https://github.com/gfx-rs/wgpu) is a Rust library
based on gfx-hal, which wraps Metal, Vulkan, DX12 and more in the
future. It can compile into a dynamic library exposing a C-API,
accomanied by a C header file. We wrap this using cffi, which uses the
header file to do most type conversions for us.
"""

import os
import ctypes

from cffi import FFI

from . import _api
from . import _register_backend
from .utils import get_resource_dir
from ._constants import cstructfield2enum, enummap


os.environ["RUST_BACKTRACE"] = "0"  # Set to 1 for more trace info

# Read header file and strip some stuff that cffi would stumble on
lines = []
with open(os.path.join(get_resource_dir(), "wgpu.h")) as f:
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


# Configure cffi
ffi = FFI()
ffi.cdef("".join(lines))
ffi.set_source("wgpu.h", None)

# Load the dynamic library
_lib = ffi.dlopen(os.path.join(get_resource_dir(), "wgpu_native-debug.dll"))


def new_struct(ctype, **kwargs):
    """ Create an ffi struct. Provides a flatter syntax and converts our
    string enums to int enums needed in C.
    """
    struct = ffi.new(ctype)
    for key, val in kwargs.items():
        if isinstance(val, str) and isinstance(getattr(struct, key), int):
            structname = cstructfield2enum[ctype.strip(" *")[4:] + "." + key]
            ival = enummap[structname + "." + val]
            setattr(struct, key, ival)
        else:
            setattr(struct, key, val)
    return struct


# wgpu.help('requestadapter', 'RequestAdapterOptions', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
async def requestAdapter(options: dict = None):
    """ Request an GPUAdapter, the object that represents the implementation of WGPU.
    Use options (RequestAdapterOptions) to specify e.g. power preference.

    This function uses the Rust WGPU library.
    """
    if options is None:
        options = {"powerPreference": "high-performance"}

    # Convert the descriptor
    struct = new_struct(
        "WGPURequestAdapterOptions *", power_preference=options["powerPreference"]
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
        struct, backend_mask, _request_adapter_callback, ffi.NULL  # userdata, stub
    )

    # For now, Rust will call the callback immediately
    # todo: when wgpu gets an event loop -> while run wgpu event loop or something

    assert adapter_id is not None
    extensions = {}
    return GPUAdapter("WGPU", extensions, adapter_id)


# Mark as the backend on import time
_register_backend(requestAdapter)


class GPUAdapter(_api.GPUAdapter):
    def __init__(self, name, extensions, id):
        super().__init__(name, extensions)
        self._id = id

    # wgpu.help('adapterrequestdevice', 'DeviceDescriptor', dev=True)
    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    async def requestDevice(self, des: dict = None):
        return self.requestDeviceSync(des)

    def requestDeviceSync(self, des: dict = None):

        extensions = des["extensions"]
        limits = des["limits"]

        c_extensions = new_struct(
            "WGPUExtensions *", anisotropic_filtering=extensions["anisotropicFiltering"]
        )
        c_limits = new_struct("WGPULimits *", max_bind_groups=limits["maxBindGroups"])
        struct = new_struct(
            "WGPUDeviceDescriptor *", extensions=c_extensions[0], limits=c_limits[0]
        )

        id = _lib.wgpu_adapter_request_device(self._id, struct)
        label = des.get("label", "")
        queue = None
        return GPUDevice(label, id, self, extensions, limits, queue)


class GPUDevice(_api.GPUDevice):

    # wgpu.help('devicecreatebuffer', 'BufferDescriptor', dev=True)
    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
    def createBuffer(self, des: dict):
        struct = new_struct(
            "WGPUBufferDescriptor *", size=des["size"], usage=des["usage"]
        )

        id = _lib.wgpu_device_create_buffer(self._internal, struct, mem)
        label = des.get("label", "")
        return GPUBuffer(label, id, self, des["size"], des["usage"], "unmapped", None)

    # wgpu.help('devicecreatebuffermapped', 'BufferDescriptor', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    def createBufferMapped(self, des: dict):

        size = int(des["size"])

        struct = new_struct("WGPUBufferDescriptor *", size=size, usage=des["usage"])

        # Pointer that device_create_buffer_mapped sets, so that we can write stuff there
        buffer_memory_pointer = ffi.new("uint8_t * *")

        id = _lib.wgpu_device_create_buffer_mapped(
            self._internal, struct, buffer_memory_pointer
        )

        # Map a numpy array onto the data
        pointer_as_int = int(ffi.cast("intptr_t", buffer_memory_pointer[0]))
        mem_as_ctypes = (ctypes.c_uint8 * size).from_address(pointer_as_int)
        # mem_as_numpy = np.frombuffer(mem_as_ctypes, np.uint8)

        label = des.get("label", "")
        return GPUBuffer(label, id, self, size, des["usage"], "mapped", mem_as_ctypes)

    # wgpu.help('devicecreatebindgrouplayout', 'BindGroupLayoutDescriptor', dev=True)
    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def createBindGroupLayout(self, des: dict):

        c_bindings_list = []
        for binding_des in des["bindings"]:
            c_binding = new_struct(
                "WGPUBindGroupLayoutBinding *",
                binding=int(binding_des["binding"]),
                visibility=int(binding_des["visibility"]),  # WGPUShaderStage
                ty=binding_des["BindingType"],
                texture_dimension=binding_des["textureDimension"],
                multisampled=bool(binding_des["multisampled"]),
                dynamic=bool(binding_des["hasDynamicOffset"]),
            )
            c_bindings_list.append(c_binding)

        c_bindings_array = ffi.new("WGPUBindGroupLayoutBinding []", c_bindings_list)
        struct = new_struct(
            "WGPUBindGroupLayoutDescriptor *",
            bindings=c_bindings_array,
            bindings_length=len(c_bindings_list),
        )

        id = _lib.wgpu_device_create_bind_group_layout(self._internal, struct)
        label = des.get("label", "")

        return GPUBindGroupLayout(label, id, self, des["bindings"])

    # wgpu.help('devicecreatebindgroup', 'BindGroupDescriptor', dev=True)
    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
    def createBindGroup(self, des: dict):
        pass


class GPUBuffer(_api.GPUBuffer):

    # wgpu.help('bufferunmap', dev=True)
    # IDL: void unmap();
    def unmap(self):
        if self._state == "mapped":
            _lib.wgpu_buffer_unmap(self._internal)
            self._state = "unmapped"

    # wgpu.help('bufferdestroy', dev=True)
    # IDL: void destroy();
    def destroy(self):
        if self._state != "destroyed":
            self._state = "destroyed"
            _lib.wgpu_buffer_destroy(self._internal)


class GPUBindGroupLayout(_api.GPUBindGroupLayout):
    pass


def _copy_docstrings():
    for ob in globals().values():
        if not (isinstance(ob, type) and issubclass(ob, _api.GPUObject)):
            continue
        elif ob.__module__ != __name__:
            continue
        base = ob.mro()[1]
        ob.__doc__ = base.__doc__
        for name, attr in ob.__dict__.items():
            if name.startswith("_") or not hasattr(attr, "__doc__"):
                continue
            base_attr = getattr(base, name, None)
            if base_attr is not None:
                attr.__doc__ = base_attr.__doc__


_copy_docstrings()
