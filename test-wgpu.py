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

import wgpu
# import wgpu.wgpu_gl
import wgpu.wgpu_ctypes
import wgpu.wgpu_ffi


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
        hwnd = wgpu.wgpu_ffi.ffi.cast("void *", int(window.winId()))
        hinstance = wgpu.wgpu_ffi.ffi.NULL
        surface = ctx.create_surface_from_windows_hwnd(hinstance, hwnd)
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

def vertex_shader(input, output):

    pass
    output.define("pos", vec4, "gl_Position")

    # out gl_PerVertex {
    #     vec4 gl_Position;
    # };

    #   const vec2 positions[3] = vec2[3](
    #     vec2(0.0, -0.5),
    #     vec2(0.5, 0.5),
    #     vec2(-0.5, 0.5)
    # );

    #   void main() {
    #     //gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0); // original
    #     gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0); // changed
    # }

##

# Instantiate gl-based wgpu context
if BACKEND == "GL":
    ctx = wgpu.wgpu_gl.GlWGPU()
elif BACKEND == "CTYPES":
    ctx = wgpu.wgpu_ctypes.RsWGPU()
elif BACKEND == "FFI":
    ctx = wgpu.wgpu_ffi.RsWGPU()
else:
    raise RuntimeError(f"Invalid backend {BACKEND}")


adapter_id = ctx.request_adapter(
    ctx.create_RequestAdapterOptions(
        power_preference=ctx.PowerPreference_Default,
        # backends=1 | 2 | 4 | 8  # backend bits - no idea what this means
        backends=8  # 2 and 8 are available, but 2 does not work on HP laptop
        # oh, and although 8 works, it wants zero bind groups :/
        )
)

device_des = ctx.create_DeviceDescriptor(
    extensions=ctx.create_Extensions(anisotropic_filtering=False),
    limits=ctx.create_Limits(max_bind_groups=0)
)

device_id = ctx.adapter_request_device(adapter_id, device_des)

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


vs_module = ctx.device_create_shader_module(
    device_id,
    ctx.create_ShaderModuleDescriptor(code=make_code("vert"))
)

fs_module = ctx.device_create_shader_module(
    device_id,
    ctx.create_ShaderModuleDescriptor(code=make_code("frag"))
)

# todo: I think this is where uniforms go
bind_group_layout = ctx.device_create_bind_group_layout(
    device_id,
    ctx.create_BindGroupLayoutDescriptor(bindings=(), bindings_length=0)
)

bind_group = ctx.device_create_bind_group(
    device_id,
    ctx.create_BindGroupDescriptor(layout=bind_group_layout, bindings=(), bindings_length=0)
)

pipeline_layout = ctx.device_create_pipeline_layout(
    device_id,
    # ctx.create_PipelineLayoutDescriptor(bind_group_layouts=(bind_group, ), bind_group_layouts_length=1)
    ctx.create_PipelineLayoutDescriptor(bind_group_layouts=(), bind_group_layouts_length=0)
)


# todo: a lot of these functions have device_id as first arg - this smells like a class, perhaps
# todo: several descriptor args have a list, and another arg to provide the length of that list, because C

render_pipeline = ctx.device_create_render_pipeline(
    device_id,
    ctx.create_RenderPipelineDescriptor(
        layout=pipeline_layout,
        vertex_stage=ctx.create_ProgrammableStageDescriptor(
            module=vs_module,
            entry_point="main",
        ),
        # fragment_stage: Some(ctx::ProgrammableStageDescriptor {
        fragment_stage=ctx.create_ProgrammableStageDescriptor(
            module=fs_module,
            entry_point="main",
        ),
        primitive_topology=ctx.PrimitiveTopology_TriangleList,
        # rasterization_state: Some(ctx::RasterizationStateDescriptor {
        rasterization_state=ctx.create_RasterizationStateDescriptor(
            front_face=ctx.FrontFace_Ccw,
            cull_mode=ctx.CullMode_None,
            depth_bias=0,
            depth_bias_slope_scale=0.0,
            depth_bias_clamp=0.0,
        ),
        # color_states: &[ctx::ColorStateDescriptor {
        color_states=ctx.create_ColorStateDescriptor(
            format=ctx.TextureFormat_Bgra8UnormSrgb,
            alpha_blend=ctx.create_BlendDescriptor(
                    src_factor=ctx.BlendFactor_One,
                    dst_factor=ctx.BlendFactor_Zero,
                    operation=ctx.BlendOperation_Add),
            color_blend=ctx.create_BlendDescriptor(
                    src_factor=ctx.BlendFactor_One,
                    dst_factor=ctx.BlendFactor_Zero,
                    operation=ctx.BlendOperation_Add),
            write_mask=ctx.ColorWrite_ALL,  # write_mask: ctx::ColorWrite::ALL,
        ),
        color_states_length=1,
        depth_stencil_state=None,
        vertex_input=ctx.create_VertexInputDescriptor(
            index_format=ctx.IndexFormat_Uint16,
            vertex_buffers=(),
            vertex_buffers_length=0,
        ),
        sample_count=1,
        sample_mask=1, # todo: or FFFFFFFFFF-ish?
        alpha_to_coverage_enabled=False,
    )
)


swap_chain = ctx.device_create_swap_chain(
    device_id=device_id,
    surface_id=surface_id,
    desc=ctx.create_SwapChainDescriptor(
        usage=ctx.TextureUsage_OUTPUT_ATTACHMENT,  # usage
        format=ctx.TextureFormat_Bgra8UnormSrgb, # format: ctx::TextureFormat::Bgra8UnormSrgb,
        width=640, # width: size.width.round() as u32,
        height=480, # height: size.height.round() as u32,
        present_mode=ctx.PresentMode_Vsync,  # present_mode: ctx::PresentMode::Vsync,
    )
)


def drawFrame():
    next_texture = ctx.swap_chain_get_next_texture(swap_chain)
    command_encoder = ctx.device_create_command_encoder(
        device_id,
        ctx.create_CommandEncoderDescriptor(todo=0),
    )

    rpass = ctx.command_encoder_begin_render_pass(
        command_encoder,
        ctx.create_RenderPassDescriptor(
            color_attachments=(
                ctx.create_RenderPassColorAttachmentDescriptor(
                    # attachment=next_texture["view_id"],
                    # todo: arg! need struct2dict function in ffi implementation
                    attachment=next_texture["view_id"] if isinstance(next_texture, dict) else next_texture.view_id,
                    resolve_target=None, # resolve_target: None,
                    load_op=ctx.LoadOp_Clear,  # load_op: ctx::LoadOp::Clear,
                    store_op=ctx.StoreOp_Store,  # store_op: ctx::StoreOp::Store,
                    clear_color=dict(r=0.5, g=255, b=0, a=255), # clear_color: ctx::Color::GREEN,
                ),
            ),
            color_attachments_length=1,
            depth_stencil_attachment=None,  # depth_stencil_attachement
        )
    )

    ctx.render_pass_set_pipeline(rpass, render_pipeline)
    # ctx.render_pass_set_bind_group(rpass, 0, bind_group, [], 0)
    ctx.render_pass_draw(rpass, 3, 1, 0, 0)

    queue = ctx.device_get_queue(device_id)
    ctx.render_pass_end_pass(rpass)
    cmd_buf = ctx.command_encoder_finish(command_encoder, None)
    ctx.queue_submit(queue, [cmd_buf], 1)
    ctx.swap_chain_present(swap_chain)


# todo: stop when window is closed ...
async def drawer():
    while True:
        await asyncio.sleep(0.1)
        # print("draw")
        drawFrame()

asyncio.get_event_loop().create_task(drawer())

