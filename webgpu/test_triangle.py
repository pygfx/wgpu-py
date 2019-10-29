"""
Example use of webgpu API to draw a triangle.

Ported from:
https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/main.rs
https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c

For reference, the same kind of example using the Vulkan API is 700 lines:
https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

"""

import time
import ctypes
import asyncio

import webgpu
import webgpu.wgpu_gl
import webgpu.wgpu_ctypes
import webgpu.wgpu_ffi


# Select backend. When using GL, I need to turn NVidia driver on, because of advanced shaders.
BACKEND = "FFI"
assert BACKEND in ("GL", "CTYPES", "FFI")


## Create window


def create_window(*args):
    if BACKEND == "GL":
        return create_window_glfw(*args)
    else:
        return create_window_qt(*args)


def create_window_glfw(width, height, name, instance_handle):
    import glfw
    glfw.init()

    # todo: depends on what backend is used, I guess?
    if BACKEND in ("CTYPES", "FFI"):
        # Create Window
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)
        window = glfw.create_window(width, height, name, None, None)

        # todo: how to get the window handle for glfw?
        # This would have been nice ... but glfw.get_win32_window does not exist
        # https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c#L167
        # hwnd = glfw.glfwGetWin32Window(window)
        # HINSTANCE hinstance = GetModuleHandle(NULL);
        # surface = wgpu_create_surface_from_windows_hwnd(hinstance, hwnd);
    else:
        glfw.window_hint(glfw.RESIZABLE, False)
        window = glfw.create_window(width, height, name, None, None)
        glfw.make_context_current(window)
        surface = window  # todo: surface_id

    async def _keep_glfw_alive():
        while True:
            await asyncio.sleep(0.1)
            if glfw.window_should_close(window):
                glfw.terminate()
                break
            else:
                glfw.poll_events()

    asyncio.get_event_loop().create_task(_keep_glfw_alive())
    return surface


def create_window_qt(width, height, name, instance_handle):
    from PyQt5 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication([])

    if BACKEND in ("CTYPES", "FFI"):
        window = QtGui.QWindow(None)
        # Use winId() or effectiveWinId() to get the Windows window handle
        hwnd = webgpu.wgpu_ffi.ffi.cast("void *", int(window.winId()))
        hinstance = webgpu.wgpu_ffi.ffi.NULL
        surface = wgpu.create_surface_from_windows_hwnd(hinstance, hwnd)
    else:
        # class MyQt5OpenGLWidget(QtWidgets.QOpenGLWidget):
        #     def paintEvent(self, event):
        #         self.makeCurrent()
        # todo: GL does currently not work on Qt - I'm fighting contexts / swap buffers
        window = QtGui.QWindow(None)
        window.setSurfaceType(QtGui.QWindow.OpenGLSurface)
        window.ctx = QtGui.QOpenGLContext(window)
        window.ctx.setFormat(window.requestedFormat())
        window.ctx.create()
        window.ctx.makeCurrent(window)
        surface = window

    window.resize(width, height)
    window.setTitle(name + " | " + BACKEND)
    window.show()

    async def _keep_qt_alive():
        while window.isVisible():
            await asyncio.sleep(0.1)
            app.flush()
            app.processEvents()

    asyncio.get_event_loop().create_task(_keep_qt_alive())
    return surface

##

# Instantiate gl-based wgpu context
if BACKEND == "GL":
    wgpu = webgpu.wgpu_gl.GlWGPU()
elif BACKEND == "CTYPES":
    wgpu = webgpu.wgpu_ctypes.RsWGPU()
elif BACKEND == "FFI":
    wgpu = webgpu.wgpu_ffi.RsWGPU()
else:
    raise RuntimeError(f"Invalid backend {BACKEND}")


adapter_id = wgpu.request_adapter(
    wgpu.create_RequestAdapterOptions(
        power_preference=wgpu.PowerPreference_Default,
        backends=1 | 2 | 4 | 8  # backend bits - no idea what this means
        )
)

device_des = wgpu.create_DeviceDescriptor(
    extensions=wgpu.create_Extensions(anisotropic_filtering=False),
    limits=wgpu.create_Limits(max_bind_groups=8)
)

device_id = wgpu.adapter_request_device(adapter_id, device_des)

surface_id = create_window(640, 480, "Triangle WGPU", device_id)

# gl_VertexIndex vs gl_InstanceID


def make_code(vert_or_frag):
    # Get filename and load file
    assert vert_or_frag in ("vert", "frag")
    filename = "shaders/triangle." + vert_or_frag
    if BACKEND != "GL":
        filename += ".spv"
    data = open(filename, 'rb').read()
    # Process the data
    if BACKEND == "GL":
        return data.decode()
    else:
        # todo: if using the wgpu lib, need SpirV code as uint32 array
        import cffi
        ffi = cffi.FFI()
        x = ffi.new("uint8_t[]", data)
        y = ffi.cast("uint32_t *", x)
        return dict(bytes=y, length=len(data)//4)


vs_module = wgpu.device_create_shader_module(
    device_id,
    wgpu.create_ShaderModuleDescriptor(code=make_code("vert"))
)

fs_module = wgpu.device_create_shader_module(
    device_id,
    wgpu.create_ShaderModuleDescriptor(code=make_code("frag"))
)

# todo: I think this is where uniforms go
bind_group_layout = wgpu.device_create_bind_group_layout(
    device_id,
    wgpu.create_BindGroupLayoutDescriptor(bindings=(), bindings_length=0)
)

bind_group = wgpu.device_create_bind_group(
    device_id,
    wgpu.create_BindGroupDescriptor(layout=bind_group_layout, bindings=(), bindings_length=0)
)

pipeline_layout = wgpu.device_create_pipeline_layout(
    device_id,
    wgpu.create_PipelineLayoutDescriptor(bind_group_layouts=(bind_group, ), bind_group_layouts_length=1)
)


# todo: a lot of these functions have device_id as first arg - this smells like a class, perhaps
# todo: several descriptor args have a list, and another arg to provide the length of that list, because C

render_pipeline = wgpu.device_create_render_pipeline(
    device_id,
    wgpu.create_RenderPipelineDescriptor(
        layout=pipeline_layout,
        vertex_stage=wgpu.create_ProgrammableStageDescriptor(
            module=vs_module,
            entry_point="main",
        ),
        # fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
        fragment_stage=wgpu.create_ProgrammableStageDescriptor(
            module=fs_module,
            entry_point="main",
        ),
        primitive_topology=wgpu.PrimitiveTopology_TriangleList,
        # rasterization_state: Some(wgpu::RasterizationStateDescriptor {
        rasterization_state=wgpu.create_RasterizationStateDescriptor(
            front_face=wgpu.FrontFace_Ccw,
            cull_mode=wgpu.CullMode_None,
            depth_bias=0,
            depth_bias_slope_scale=0.0,
            depth_bias_clamp=0.0,
        ),
        # color_states: &[wgpu::ColorStateDescriptor {
        color_states=wgpu.create_ColorStateDescriptor(
            format=wgpu.TextureFormat_Bgra8UnormSrgb,
            alpha_blend=wgpu.create_BlendDescriptor(
                    src_factor=wgpu.BlendFactor_One,
                    dst_factor=wgpu.BlendFactor_Zero,
                    operation=wgpu.BlendOperation_Add),
            color_blend=wgpu.create_BlendDescriptor(
                    src_factor=wgpu.BlendFactor_One,
                    dst_factor=wgpu.BlendFactor_Zero,
                    operation=wgpu.BlendOperation_Add),
            write_mask=wgpu.ColorWrite_ALL,  # write_mask: wgpu::ColorWrite::ALL,
        ),
        color_states_length=1,
        depth_stencil_state=None,
        vertex_input=wgpu.create_VertexInputDescriptor(
            index_format=wgpu.IndexFormat_Uint16,
            vertex_buffers=(),
            vertex_buffers_length=0,
        ),
        sample_count=1,
        sample_mask=1, # todo: or FFFFFFFFFF-ish?
        alpha_to_coverage_enabled=False,
    )
)


swap_chain = wgpu.device_create_swap_chain(
    device_id=device_id,
    surface_id=surface_id,
    desc=wgpu.create_SwapChainDescriptor(
        usage=wgpu.TextureUsage_OUTPUT_ATTACHMENT,  # usage
        format=wgpu.TextureFormat_Bgra8UnormSrgb, # format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width=640, # width: size.width.round() as u32,
        height=480, # height: size.height.round() as u32,
        present_mode=wgpu.PresentMode_Vsync,  # present_mode: wgpu::PresentMode::Vsync,
    )
)


def drawFrame():
    next_texture = wgpu.swap_chain_get_next_texture(swap_chain)
    command_encoder = wgpu.device_create_command_encoder(
        device_id,
        wgpu.create_CommandEncoderDescriptor(todo=0),
    )

    rpass = wgpu.command_encoder_begin_render_pass(
        command_encoder,
        wgpu.create_RenderPassDescriptor(
            color_attachments=(
                wgpu.create_RenderPassColorAttachmentDescriptor(
                    # attachment=next_texture["view_id"],
                    # todo: arg! need struct2dict function in ffi implementation
                    attachment=next_texture["view_id"] if isinstance(next_texture, dict) else next_texture.view_id,
                    resolve_target=None, # resolve_target: None,
                    load_op=wgpu.LoadOp_Clear,  # load_op: wgpu::LoadOp::Clear,
                    store_op=wgpu.StoreOp_Store,  # store_op: wgpu::StoreOp::Store,
                    clear_color=dict(r=1, g=1, b=0, a=1), # clear_color: wgpu::Color::GREEN,
                ),
            ),
            color_attachments_length=1,
            depth_stencil_attachment=None,  # depth_stencil_attachement
        )
    )

    wgpu.render_pass_set_pipeline(rpass, render_pipeline)
    wgpu.render_pass_set_bind_group(rpass, 0, bind_group, [], 0)
    wgpu.render_pass_draw(rpass, 3, 1, 0, 0)

    queue = wgpu.device_get_queue(device_id)
    wgpu.render_pass_end_pass(rpass)
    cmd_buf = wgpu.command_encoder_finish(command_encoder, None)
    wgpu.queue_submit(queue, [cmd_buf], 1)
    wgpu.swap_chain_present(swap_chain)


# todo: stop when window is closed ...
async def drawer():
    while True:
        await asyncio.sleep(0.1)
        drawFrame()

asyncio.get_event_loop().create_task(drawer())

