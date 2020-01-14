"""
WGPU backend implementation based on the wgpu library.

The Rust wgpu project (https://github.com/gfx-rs/wgpu) is a Rust library
based on gfx-hal, which wraps Metal, Vulkan, DX12 and more in the
future. It can compile into a dynamic library exposing a C-API,
accomanied by a C header file. We wrap this using cffi, which uses the
header file to do most type conversions for us.
"""

import os
import sys
import ctypes

from cffi import FFI, __version_info__ as cffi_version_info

from .. import classes
from .. import _register_backend
from ..utils import get_resource_filename
from .._mappings import cstructfield2enum, enummap


if cffi_version_info < (1, 10):
    raise ImportError(f"{__name__} needs cffi 1.10 or later.")


def _get_wgpu_h():
    # Read header file and strip some stuff that cffi would stumble on
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
    paths = []

    override_path = os.getenv("WGPU_LIB_PATH", "").strip()
    if override_path:
        paths.append(override_path)

    lib_filename = None
    if sys.platform.startswith("win"):
        lib_filename = "wgpu_native.dll"
    elif sys.platform.startswith("darwin"):
        lib_filename = "libwgpu_native.dylib"
    elif sys.platform.startswith("linux"):
        lib_filename = "libwgpu_native.so"
    if lib_filename:
        # Note that this can be a false positive, e.g. ARM linux.
        embedded_path = get_resource_filename(lib_filename)
        paths.append(embedded_path)

    for path in paths:
        if os.path.isfile(path):
            return path
    else:
        raise RuntimeError(f"Could not find WGPU library, checked: {paths}")


# Configure cffi and load the dynamic library
ffi = FFI()
ffi.cdef(_get_wgpu_h())
ffi.set_source("wgpu.h", None)
_lib = ffi.dlopen(_get_wgpu_lib_path())


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


def get_surface_id_from_canvas(canvas):
    win_id = canvas.getWindowId()
    if sys.platform.startswith("win"):
        # wgpu_create_surface_from_windows_hwnd(void *_hinstance, void *hwnd)
        hwnd = ffi.cast("void *", int(win_id))
        hinstance = ffi.NULL
        return _lib.wgpu_create_surface_from_windows_hwnd(hinstance, hwnd)
    elif sys.platform.startswith("darwin"):
        # wgpu_create_surface_from_metal_layer(void *layer)
        # todo: MacOS support
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
        layer = ffi.cast("void *", win_id)
        return _lib.wgpu_create_surface_from_metal_layer(layer)
    elif sys.platform.startswith("linux"):
        # wgpu_create_surface_from_wayland(void *surface, void *display)
        # wgpu_create_surface_from_xlib(const void **display, uint64_t window)
        display_id = canvas.getDisplayId()
        is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
        if is_wayland:
            # todo: works, but have not yet been able to test drawing to the window
            surface = ffi.cast("void *", win_id)
            display = ffi.cast("void *", display_id)
            return _lib.wgpu_create_surface_from_wayland(surface, display)
        else:
            display = ffi.cast("void **", display_id)
            return _lib.wgpu_create_surface_from_xlib(display, win_id)
    # Else ...
    raise RuntimeError("Cannot get surface id: unsupported platform.")


# %% The API


# wgpu.help('requestadapter', 'RequestAdapterOptions', dev=True)
# IDL: Promise<GPUAdapter> requestAdapter(optional GPURequestAdapterOptions options = {});
async def requestAdapter(*, powerPreference: "GPUPowerPreference"):
    """ Request an GPUAdapter, the object that represents the implementation of WGPU.
    This function uses the Rust WGPU library.

    Params:
        powerPreference(enum): "high-performance" or "low-power"
    """

    # Convert the descriptor
    struct = new_struct("WGPURequestAdapterOptions *", power_preference=powerPreference)

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


# Mark as the backend on import time
_register_backend(requestAdapter)


class GPUAdapter(classes.GPUAdapter):
    def __init__(self, name, extensions, id):
        super().__init__(name, extensions)
        self._id = id

    # wgpu.help('adapterrequestdevice', 'DeviceDescriptor', dev=True)
    # IDL: Promise<GPUDevice> requestDevice(optional GPUDeviceDescriptor descriptor = {});
    async def requestDevice(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):
        return self.requestDeviceSync(label=label, extensions=extensions, limits=limits)

    def requestDeviceSync(
        self,
        *,
        label="",
        extensions: "GPUExtensionName-list" = [],
        limits: "GPULimits" = {},
    ):

        extensions = tuple(extensions)

        c_extensions = new_struct(
            "WGPUExtensions *",
            anisotropic_filtering="anisotropicFiltering" in extensions,
        )
        c_limits = new_struct("WGPULimits *", max_bind_groups=limits["maxBindGroups"])
        struct = new_struct(
            "WGPUDeviceDescriptor *", extensions=c_extensions[0], limits=c_limits[0]
        )

        id = _lib.wgpu_adapter_request_device(self._id, struct)

        queue_id = _lib.wgpu_device_get_queue(id)
        queue = GPUQueue("", queue_id, self)

        return GPUDevice(label, id, self, extensions, limits, queue)


class GPUDevice(classes.GPUDevice):

    # wgpu.help('devicecreatebuffer', 'BufferDescriptor', dev=True)
    # IDL: GPUBuffer createBuffer(GPUBufferDescriptor descriptor);
    def createBuffer(
        self, *, label="", size: "GPUBufferSize", usage: "GPUBufferUsageFlags"
    ):
        size = int(size)

        struct = new_struct("WGPUBufferDescriptor *", size=size, usage=usage)

        id = _lib.wgpu_device_create_buffer(
            self._internal, struct, mem
        )  # TODO: mem is undefined
        return GPUBuffer(label, id, self, size, usage, "unmapped", None)

    # wgpu.help('devicecreatebuffermapped', 'BufferDescriptor', dev=True)
    # IDL: GPUMappedBuffer createBufferMapped(GPUBufferDescriptor descriptor);
    def createBufferMapped(
        self, *, label="", size: "GPUBufferSize", usage: "GPUBufferUsageFlags"
    ):

        size = int(size)

        struct = new_struct("WGPUBufferDescriptor *", size=size, usage=usage)

        # Pointer that device_create_buffer_mapped sets, so that we can write stuff
        # there
        buffer_memory_pointer = ffi.new("uint8_t * *")

        id = _lib.wgpu_device_create_buffer_mapped(
            self._internal, struct, buffer_memory_pointer
        )

        # Map a numpy array onto the data
        pointer_as_int = int(ffi.cast("intptr_t", buffer_memory_pointer[0]))
        mem_as_ctypes = (ctypes.c_uint8 * size).from_address(pointer_as_int)
        # mem_as_numpy = np.frombuffer(mem_as_ctypes, np.uint8)

        return GPUBuffer(label, id, self, size, usage, "mapped", mem_as_ctypes)

    # wgpu.help('devicecreatebindgrouplayout', 'BindGroupLayoutDescriptor', dev=True)
    # IDL: GPUBindGroupLayout createBindGroupLayout(GPUBindGroupLayoutDescriptor descriptor);
    def createBindGroupLayout(
        self, *, label="", bindings: "GPUBindGroupLayoutBinding-list"
    ):

        c_bindings_list = []
        for binding in bindings:
            c_binding = new_struct(
                "WGPUBindGroupLayoutBinding *",
                binding=int(binding.binding),
                visibility=int(binding.visibility),
                ty=binding.BindingType,
                texture_dimension=binding.textureDimension,
                multisampled=bool(binding.multisampled),
                dynamic=bool(binding.hasDynamicOffset),
            )  # WGPUShaderStage
            c_bindings_list.append(c_binding)

        c_bindings_array = ffi.new("WGPUBindGroupLayoutBinding []", c_bindings_list)
        struct = new_struct(
            "WGPUBindGroupLayoutDescriptor *",
            bindings=c_bindings_array,
            bindings_length=len(c_bindings_list),
        )

        id = _lib.wgpu_device_create_bind_group_layout(self._internal, struct)

        return classes.GPUBindGroupLayout(label, id, self, bindings)

    # wgpu.help('devicecreatebindgroup', 'BindGroupDescriptor', dev=True)
    # IDL: GPUBindGroup createBindGroup(GPUBindGroupDescriptor descriptor);
    def createBindGroup(
        self,
        *,
        label="",
        layout: "GPUBindGroupLayout",
        bindings: "GPUBindGroupBinding-list",
    ):

        c_bindings_list = []
        for binding in bindings:
            c_binding = new_struct(
                "WGPUBindGroupBinding *",
                binding=int(binding.binding),
                resource=binding.resource,
            )  # todo: xxxx WGPUBindingResource
            c_bindings_list.append(c_binding)

        c_bindings_array = ffi.new("WGPUBindGroupBinding []", c_bindings_list)
        struct = new_struct(
            "WGPUBindGroupDescriptor *",
            layout=layout._internal,
            bindings=c_bindings_array,
            bindings_length=len(c_bindings_list),
        )

        id = _lib.wgpu_device_create_bind_group(self._internal, struct)
        return classes.GPUBindGroup(label, id, self, bindings)

    # wgpu.help('devicecreatepipelinelayout', 'PipelineLayoutDescriptor', dev=True)
    # IDL: GPUPipelineLayout createPipelineLayout(GPUPipelineLayoutDescriptor descriptor);
    def createPipelineLayout(
        self, *, label="", bindGroupLayouts: "GPUBindGroupLayout-list"
    ):

        bindGroupLayouts_ids = [x._internal for x in bindGroupLayouts]

        c_layout_array = ffi.new("WGPUBindGroupLayoutId []", bindGroupLayouts_ids)
        struct = new_struct(
            "WGPUPipelineLayoutDescriptor *",
            bind_group_layouts=c_layout_array,
            bind_group_layouts_length=len(bindGroupLayouts),
        )

        id = _lib.wgpu_device_create_pipeline_layout(self._internal, struct)
        return classes.GPUPipelineLayout(label, id, self, bindGroupLayouts)

    # wgpu.help('devicecreateshadermodule', 'ShaderModuleDescriptor', dev=True)
    # IDL: GPUShaderModule createShaderModule(GPUShaderModuleDescriptor descriptor);
    def createShaderModule(self, *, label="", code: "GPUShaderCode"):

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
        return classes.GPUShaderModule(label, id, self)

    # wgpu.help('devicecreaterenderpipeline', 'RenderPipelineDescriptor', dev=True)
    # IDL: GPURenderPipeline createRenderPipeline(GPURenderPipelineDescriptor descriptor);
    def createRenderPipeline(
        self,
        *,
        label="",
        layout: "GPUPipelineLayout",
        vertexStage: "GPUProgrammableStageDescriptor",
        fragmentStage: "GPUProgrammableStageDescriptor",
        primitiveTopology: "GPUPrimitiveTopology",
        rasterizationState: "GPURasterizationStateDescriptor" = {},
        colorStates: "GPUColorStateDescriptor-list",
        depthStencilState: "GPUDepthStencilStateDescriptor",
        vertexState: "GPUVertexStateDescriptor" = {},
        sampleCount: int = 1,
        sampleMask: int = 0xFFFFFFFF,
        alphaToCoverageEnabled: bool = False,
    ):

        refs = []  # to avoid premature gc collection
        c_vertex_stage = new_struct(
            "WGPUProgrammableStageDescriptor *",
            module=vertexStage["module"]._internal,
            entry_point=ffi.new("char []", vertexStage["entryPoint"].encode()),
        )
        c_fragment_stage = new_struct(
            "WGPUProgrammableStageDescriptor *",
            module=fragmentStage["module"]._internal,
            entry_point=ffi.new("char []", fragmentStage["entryPoint"].encode()),
        )
        c_rasterization_state = new_struct(
            "WGPURasterizationStateDescriptor *",
            front_face=rasterizationState["frontFace"],
            cull_mode=rasterizationState["cullMode"],
            depth_bias=rasterizationState["depthBias"],
            depth_bias_slope_scale=rasterizationState["depthBiasSlopeScale"],
            depth_bias_clamp=rasterizationState["depthBiasClamp"],
        )
        c_color_states_list = []
        for colorState in colorStates:
            alphaBlend = colorState["alphaBlend"]
            if not isinstance(alphaBlend, (list, tuple)):  # support dict and tuple
                alphaBlend = (
                    alphaBlend["srcFactor"],
                    alphaBlend["dstFactor"],
                    alphaBlend["operation"],
                )
            c_alpha_blend = new_struct(
                "WGPUBlendDescriptor *",
                src_factor=alphaBlend[0],
                dst_factor=alphaBlend[1],
                operation=alphaBlend[2],
            )
            colorBlend = colorState["colorBlend"]
            if not isinstance(colorBlend, (list, tuple)):  # support dict and tuple
                colorBlend = (
                    colorBlend["srcFactor"],
                    colorBlend["dstFactor"],
                    colorBlend["operation"],
                )
            c_color_blend = new_struct(
                "WGPUBlendDescriptor *",
                src_factor=colorBlend[0],
                dst_factor=colorBlend[1],
                operation=colorBlend[2],
            )
            c_color_state = new_struct(
                "WGPUColorStateDescriptor *",
                format=colorState["format"],
                alpha_blend=c_alpha_blend[0],
                color_blend=c_color_blend[0],
                write_mask=colorState["writeMask"],
            )  # enum
            refs.extend([c_alpha_blend, c_color_blend])
            c_color_states_list.append(c_color_state[0])
        c_color_states_array = ffi.new(
            "WGPUColorStateDescriptor []", c_color_states_list
        )
        if depthStencilState is None:
            c_depth_stencil_state = ffi.NULL
        else:
            raise NotImplementedError()
            # c_depth_stencil_state = new_struct(
            #     "WGPUDepthStencilStateDescriptor *",
            #     format=
            #     depth_write_enabled=
            #     depth_compare
            #     stencil_front
            #     stencil_back
            #     stencil_read_mask
            #     stencil_write_mask
            # )
        c_vertex_buffer_descriptors_list = []
        for buffer_des in vertexState["vertexBuffers"]:
            c_attributes_list = []
            for attribute in buffer_des["attributes"]:
                c_attribute = new_struct(
                    "WGPUVertexAttributeDescriptor *",
                    format=attribute["format"],
                    offset=attribute["offset"],
                    shader_location=attribute["shaderLocation"],
                )
                c_attributes_list.append(c_attribute)
            c_attributes_array = ffi.new(
                "WGPUVertexAttributeDescriptor []", c_attributes_list
            )
            c_vertex_buffer_descriptor = new_struct(
                "WGPUVertexBufferDescriptor *",
                stride=buffer_des["arrayStride"],
                step_mode=buffer_des["stepmode"],
                attributes=c_attributes_array,
                attributes_length=len(c_attributes_list),
            )
            refs.append(c_attributes_list)
            c_vertex_buffer_descriptors_list.append(c_vertex_buffer_descriptor)
        c_vertex_buffer_descriptors_array = ffi.new(
            "WGPUVertexBufferDescriptor []", c_vertex_buffer_descriptors_list
        )
        c_vertex_input = new_struct(
            "WGPUVertexInputDescriptor *",
            index_format=vertexState["indexFormat"],
            vertex_buffers=c_vertex_buffer_descriptors_array,
            vertex_buffers_length=len(c_vertex_buffer_descriptors_list),
        )

        struct = new_struct(
            "WGPURenderPipelineDescriptor *",
            layout=layout._internal,
            vertex_stage=c_vertex_stage[0],
            fragment_stage=c_fragment_stage,
            primitive_topology=primitiveTopology,
            rasterization_state=c_rasterization_state,
            color_states=c_color_states_array,
            color_states_length=len(c_color_states_list),
            depth_stencil_state=c_depth_stencil_state,
            vertex_input=c_vertex_input[0],
            sample_count=sampleCount,
            sample_mask=sampleMask,
            alpha_to_coverage_enabled=alphaToCoverageEnabled,
        )  # c-pointer  # enum

        id = _lib.wgpu_device_create_render_pipeline(self._internal, struct)
        return classes.GPURenderPipeline(label, id, self)

    # wgpu.help('devicecreatecommandencoder', 'CommandEncoderDescriptor', dev=True)
    # IDL: GPUCommandEncoder createCommandEncoder(optional GPUCommandEncoderDescriptor descriptor = {});
    def createCommandEncoder(self, *, label=""):

        struct = new_struct("WGPUCommandEncoderDescriptor *", todo=0)

        id = _lib.wgpu_device_create_command_encoder(self._internal, struct)
        return GPUCommandEncoder(label, id, self)

    def _gui_configureSwapChain(self, canvas, format, usage):
        """ Get a swapchain object from a canvas object. Called by BaseCanvas.
        """
        # Note: canvas should implement the BaseCanvas interface.
        return GPUSwapChain(self, canvas, format, usage)


class GPUBuffer(classes.GPUBuffer):

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


class GPUTexture(classes.GPUTexture):

    # wgpu.help('texturecreateview', 'TextureViewDescriptor', dev=True)
    # IDL: GPUTextureView createView(optional GPUTextureViewDescriptor descriptor = {});
    def createView(
        self,
        *,
        label="",
        format: "GPUTextureFormat",
        dimension: "GPUTextureViewDimension",
        aspect: "GPUTextureAspect" = "all",
        baseMipLevel: int = 0,
        mipLevelCount: int = 0,
        baseArrayLayer: int = 0,
        arrayLayerCount: int = 0,
    ):

        struct = new_struct(
            "WGPUTextureViewDescriptor *",
            dimension=dimension,
            aspect=aspect,
            base_mip_level=baseMipLevel,
            level_count=mipLevelCount,
            base_array_layer=baseArrayLayer,
            array_layer_count=arrayLayerCount,
        )

        id = _lib.wgpu_texture_create_view(self._internal, struct)
        return classes.GPUTextureView(label, id, self)

    # wgpu.help('texturedestroy', dev=True)
    # IDL: void destroy();
    def destroy(self):
        _lib.wgpu_texture_destroy(self._internal)


class GPUCommandEncoder(classes.GPUCommandEncoder):

    # wgpu.help('commandencoderbeginrenderpass', 'RenderPassDescriptor', dev=True)
    # IDL: GPURenderPassEncoder beginRenderPass(GPURenderPassDescriptor descriptor);
    def beginRenderPass(
        self,
        *,
        label="",
        colorAttachments: "GPURenderPassColorAttachmentDescriptor-list",
        depthStencilAttachment: "GPURenderPassDepthStencilAttachmentDescriptor",
    ):

        refs = []

        c_color_attachments_list = []
        for colorAttachment in colorAttachments:
            assert isinstance(colorAttachment["attachment"], classes.GPUTextureView)
            texture_view_id = colorAttachment["attachment"]._internal
            if colorAttachment["resolveTarget"] is None:
                c_resolve_target = ffi.NULL
            else:
                raise NotImplementedError()
            if isinstance(colorAttachment["loadValue"], str):
                assert colorAttachment["loadValue"] == "load"
                c_load_op = 1  # WGPULoadOp_Load
                c_clear_color = ffi.new("WGPUColor *", dict(r=0, g=0, b=0, a=0))
            else:
                c_load_op = 0  # WGPULoadOp_Clear
                clr = colorAttachment["loadValue"]
                if isinstance(clr, dict):
                    c_clear_color = ffi.new("WGPUColor *", *clr)
                else:
                    c_clear_color = ffi.new(
                        "WGPUColor *", dict(r=clr[0], g=clr[1], b=clr[2], a=clr[3])
                    )
            c_attachment = new_struct(
                "WGPURenderPassColorAttachmentDescriptor *",
                attachment=texture_view_id,
                resolve_target=c_resolve_target,
                load_op=c_load_op,
                store_op=colorAttachment["storeOp"],
                clear_color=c_clear_color[0],
            )
            refs.append(c_clear_color)
            c_color_attachments_list.append(c_attachment[0])
        c_color_attachments_array = ffi.new(
            "WGPURenderPassColorAttachmentDescriptor []", c_color_attachments_list
        )

        c_depth_stencil_attachment = ffi.NULL
        if depthStencilAttachment is not None:
            raise NotImplementedError()

        struct = new_struct(
            "WGPURenderPassDescriptor *",
            color_attachments=c_color_attachments_array,
            color_attachments_length=len(c_color_attachments_list),
            depth_stencil_attachment=c_depth_stencil_attachment,
        )

        id = _lib.wgpu_command_encoder_begin_render_pass(self._internal, struct)
        return GPURenderPassEncoder(label, id, self)

    # wgpu.help('commandencoderfinish', 'CommandBufferDescriptor', dev=True)
    # IDL: GPUCommandBuffer finish(optional GPUCommandBufferDescriptor descriptor = {});
    def finish(self, *, label=""):
        struct = new_struct("WGPUCommandBufferDescriptor *", todo=0)
        id = _lib.wgpu_command_encoder_finish(self._internal, struct)
        return classes.GPUCommandBuffer(label, id, self)


class GPUProgrammablePassEncoder(classes.GPUProgrammablePassEncoder):

    # wgpu.help('programmablepassencodersetbindgroup', 'BindGroup', dev=True)
    # IDL: void setBindGroup(unsigned long index, GPUBindGroup bindGroup,  Uint32Array dynamicOffsetsData,  unsigned long long dynamicOffsetsDataStart,  unsigned long long dynamicOffsetsDataLength);
    def setBindGroup(
        self,
        index,
        bindGroup,
        dynamicOffsetsData,
        dynamicOffsetsDataStart,
        dynamicOffsetsDataLength,
    ):
        offsets = list(dynamicOffsetsData)
        c_offsets = ffi.new("WGPUBufferAddress []", offsets)
        bind_group_id = bindGroup._internal
        if isinstance(self, GPUComputePassEncoder):
            _lib.wgpu_compute_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )
        else:
            _lib.wgpu_render_pass_set_bind_group(
                self._internal, index, bind_group_id, c_offsets, len(offsets)
            )

    # wgpu.help('programmablepassencoderpushdebuggroup', dev=True)
    # IDL: void pushDebugGroup(DOMString groupLabel);
    def pushDebugGroup(self, groupLabel):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderpopdebuggroup', dev=True)
    # IDL: void popDebugGroup();
    def popDebugGroup(self):
        raise NotImplementedError()

    # wgpu.help('programmablepassencoderinsertdebugmarker', dev=True)
    # IDL: void insertDebugMarker(DOMString markerLabel);
    def insertDebugMarker(self, markerLabel):
        raise NotImplementedError()


class GPUComputePassEncoder(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('computepassencodersetpipeline', 'ComputePipeline', dev=True)
    # IDL: void setPipeline(GPUComputePipeline pipeline);
    def setPipeline(self, pipeline):
        raise NotImplementedError()

    # wgpu.help('computepassencoderdispatch', dev=True)
    # IDL: void dispatch(unsigned long x, optional unsigned long y = 1, optional unsigned long z = 1);
    def dispatch(self, x, y, z):
        raise NotImplementedError()

    # wgpu.help('computepassencoderdispatchindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void dispatchIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def dispatchIndirect(self, indirectBuffer, indirectOffset):
        raise NotImplementedError()

    # wgpu.help('computepassencoderendpass', dev=True)
    # IDL: void endPass();
    def endPass(self):
        raise NotImplementedError()


class GPURenderEncoderBase(GPUProgrammablePassEncoder):
    """
    """

    # wgpu.help('renderencoderbasesetpipeline', 'RenderPipeline', dev=True)
    # IDL: void setPipeline(GPURenderPipeline pipeline);
    def setPipeline(self, pipeline):
        pipeline_id = pipeline._internal
        _lib.wgpu_render_pass_set_pipeline(self._internal, pipeline_id)

    # wgpu.help('renderencoderbasesetindexbuffer', 'Buffer', 'BufferSize', dev=True)
    # IDL: void setIndexBuffer(GPUBuffer buffer, optional GPUBufferSize offset = 0);
    def setIndexBuffer(self, buffer, offset):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasesetvertexbuffer', 'Buffer', 'BufferSize', dev=True)
    # IDL: void setVertexBuffer(unsigned long slot, GPUBuffer buffer, optional GPUBufferSize offset = 0);
    def setVertexBuffer(self, slot, buffer, offset):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedraw', dev=True)
    # IDL: void draw(unsigned long vertexCount, unsigned long instanceCount,  unsigned long firstVertex, unsigned long firstInstance);
    def draw(self, vertexCount, instanceCount, firstVertex, firstInstance):
        _lib.wgpu_render_pass_draw(
            self._internal, vertexCount, instanceCount, firstVertex, firstInstance
        )

    # wgpu.help('renderencoderbasedrawindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void drawIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def drawIndirect(self, indirectBuffer, indirectOffset):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindexed', dev=True)
    # IDL: void drawIndexed(unsigned long indexCount, unsigned long instanceCount,  unsigned long firstIndex, long baseVertex, unsigned long firstInstance);
    def drawIndexed(
        self, indexCount, instanceCount, firstIndex, baseVertex, firstInstance
    ):
        raise NotImplementedError()

    # wgpu.help('renderencoderbasedrawindexedindirect', 'Buffer', 'BufferSize', dev=True)
    # IDL: void drawIndexedIndirect(GPUBuffer indirectBuffer, GPUBufferSize indirectOffset);
    def drawIndexedIndirect(self, indirectBuffer, indirectOffset):
        raise NotImplementedError()


# todo: this does not inherit from classes.GPURenderPassEncoder. Use multiple
# inheritance or leave it?


class GPURenderPassEncoder(GPURenderEncoderBase):
    """
    """

    # wgpu.help('renderpassencodersetviewport', dev=True)
    # IDL: void setViewport(float x, float y,  float width, float height,  float minDepth, float maxDepth);
    def setViewport(self, x, y, width, height, minDepth, maxDepth):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetscissorrect', dev=True)
    # IDL: void setScissorRect(unsigned long x, unsigned long y, unsigned long width, unsigned long height);
    def setScissorRect(self, x, y, width, height):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetblendcolor', 'Color', dev=True)
    # IDL: void setBlendColor(GPUColor color);
    def setBlendColor(self, color):
        raise NotImplementedError()

    # wgpu.help('renderpassencodersetstencilreference', dev=True)
    # IDL: void setStencilReference(unsigned long reference);
    def setStencilReference(self, reference):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderexecutebundles', dev=True)
    # IDL: void executeBundles(sequence<GPURenderBundle> bundles);
    def executeBundles(self, bundles):
        raise NotImplementedError()

    # wgpu.help('renderpassencoderendpass', dev=True)
    # IDL: void endPass();
    def endPass(self):
        _lib.wgpu_render_pass_end_pass(self._internal)


class GPUQueue(classes.GPUQueue):

    # wgpu.help('queuesubmit', dev=True)
    # IDL: void submit(sequence<GPUCommandBuffer> commandBuffers);
    def submit(self, commandBuffers):
        command_buffer_ids = [cb._internal for cb in commandBuffers]
        c_command_buffers = ffi.new("WGPUCommandBufferId []", command_buffer_ids)
        _lib.wgpu_queue_submit(
            self._internal, c_command_buffers, len(command_buffer_ids)
        )


class GPUSwapChain(classes.GPUSwapChain):
    def __init__(self, device, canvas, format, usage):
        super().__init__("", None, device)
        self._canvas = canvas
        self._format = format
        self._usage = usage
        self._surface_size = (-1, -1)
        self._surface_id = None
        self._create_native_swapchain_if_needed()

    def _create_native_swapchain_if_needed(self):
        cur_size = self._canvas.getSizeAndPixelRatio()  # width, height, ratio
        if cur_size == self._surface_size:
            return

        self._surface_size = cur_size

        struct = new_struct(
            "WGPUSwapChainDescriptor *",
            usage=self._usage,
            format=self._format,
            width=cur_size[0],
            height=cur_size[1],
            present_mode=1,
        )  # vsync or not vsync

        if self._surface_id is None:
            self._surface_id = get_surface_id_from_canvas(self._canvas)

        self._internal = _lib.wgpu_device_create_swap_chain(
            self._device._internal, self._surface_id, struct
        )  # device-id

    def getCurrentTextureView(self):
        # todo: should we cache instances (on their id)?
        # otherwise we have multiple instances mapping to same internal texture
        self._create_native_swapchain_if_needed()
        swapChainOutput = _lib.wgpu_swap_chain_get_next_texture(self._internal)
        return classes.GPUTextureView("swapchain", swapChainOutput.view_id, self)

    def _gui_present(self):
        """ Present the current texture. This is not part of the public API,
        instead, GUI backends should call this at the right moment.
        """
        _lib.wgpu_swap_chain_present(self._internal)


# %%


def _copy_docstrings():
    for ob in globals().values():
        if not (isinstance(ob, type) and issubclass(ob, classes.GPUObject)):
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
