"""
Example use of webgpu API to draw a triangle.

Ported from:
https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/main.rs
https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c

For reference, the same kind of example using the Vulkan API is 700 lines:
https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

"""

import ctypes
import asyncio

import glfw

import webgpu
import webgpu.wgpu_gl


## Create window


glfw.init()


def create_window(width, height, name):

    # Create a window
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(width, height, name, None, None)
    return window

    # # Create the Vulkan surface object
    # ffi = vk.ffi
    # surface = ctypes.c_void_p(0)
    # # instance = ctypes.cast(int(ffi.cast('intptr_t', instance_handle)), ctypes.c_void_p)
    # instance = ctypes.cast(
    #     int(ffi.cast("uintptr_t", instance_handle)), ctypes.c_void_p
    # )
    # glfw.create_window_surface(instance, window, None, ctypes.byref(surface))
    # surface = ffi.cast("VkSurfaceKHR", surface.value)
    # if surface is None:
    #     raise Exception("failed to create window surface!")
    # return surface


def integrate_asyncio(loop):
    loop.create_task(_keep_glfw_alive())


async def _keep_glfw_alive():
    while True:
        await asyncio.sleep(0.1)
        if glfw.window_should_close(window):
            glfw.terminate()
            break
        else:
            glfw.poll_events()


window = create_window(640, 480, "Triangle WGPU")
integrate_asyncio(asyncio.get_event_loop())


##

wgpu = webgpu.wgpu_gl.GlWGPU()

adapter_id = wgpu.request_adapter()
    wgpu.create_RequestAdapterOptions(
        wgpu.PowerPreference_Default,
        2 | 4 | 8  # backend bits - no idea what this means
        )
)

device_des = wgpu.create_DeviceDescriptor(
    wgpu.create_Extensions(False),
    wgpu.create_Limits(99))
)

device_id = wgpu.adapter_request_device(adapter_id, device_des)


vertex_code = """
"""

fragment_code = """
"""

vs_module = wgpu.device_create_shader_module(
    device_id,
    wgpu.create_ShaderModuleDescriptor(vertex_code)
)

fs_module = wgpu.device_create_shader_module(
    device_id,
    wgpu.create_ShaderModuleDescriptor(fragment_code)
)

bind_group_layout = wgpu.device_create_bind_group_layout(
    wgpu.create_BindGroupLayoutDescriptor([], 0)  # what must this list contain?
)

bind_group = wgpu.device_create_bind_group(
    wgpu.create_BindGroupDescriptor(bind_group_layout, [], 0)
)

pipeline_layout = wgpu.device_create_pipeline_layout(
    wgpu.create_PipelineLayoutDescriptor([bind_group], 1)
)

# todo: this gets hard to read -> change how we do structs
# at the least it should force keyword only args

render_pipeline = wgpu.device_create_render_pipeline(
    wgpu.create_RenderPipelineDescriptor
        pipeline_layout,
        #vertex_stage: wgpu::ProgrammableStageDescriptor {
        wgpu.create_ProgrammableStageDescriptor(
            vs_module,  # module
            "main", # entry_point
        ),
        # fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
        wgpu.create_ProgrammableStageDescriptor(
            fs_module,  # module
            "main", # entry_point
        ),
        wgpu.PrimitiveTopology_TriangleList, # primitive_topology
        # rasterization_state: Some(wgpu::RasterizationStateDescriptor {
        wgpu.create_RasterizationStateDescriptor(
            wgpu.FrontFace_Ccw, # front_face
            wgpu.CullMode_None, # cull_mode
            0,  # depth_bias: 0,
            0.0, # depth_bias_slope_scale: 0.0,
            0.0,  # depth_bias_clamp: 0.0,
        ),
        # color_states: &[wgpu::ColorStateDescriptor {
        wgpu.create_ColorStateDescriptor(
            wgpu.TextureFormat_Bgra8UnormSrgb, # format: wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu.create_BlendDescriptor(wgpu.BlendFactor_One, wgpu.BlendFactor_Zero, wgpu.BlendOperation_Add), # alpha_blend: wgpu::BlendDescriptor::REPLACE,
            wgpu.create_BlendDescriptor(wgpu.BlendFactor_One, wgpu.BlendFactor_Zero, wgpu.BlendOperation_Add), # color_blend: wgpu::BlendDescriptor::REPLACE,
            wgpu.ColorWrite_ALL,  # write_mask: wgpu::ColorWrite::ALL,
        ),
        1,  # color_states_length
        None, # depth_stencil_state: None,
        wgpu.create_VertexInputDescriptor(
            wgpu.IndexFormat_Uint16, # index_format
            [],  # vertex_buffers
            0,  # vertex_buffers_length
        ),
        wgpu.IndexFormat_Uint16, # index_format: wgpu::IndexFormat::Uint16,
        1, # sample_count: 1,
        1, # todo: or FFFFFFFFFF-ish?   sample_mask: 1,
        False,  # alpha_to_coverage_enabled: false,
    )
)


swap_chain = wgpu.device_create_swap_chain(
    device_id,
    surface_id,
    wgpu.create_SwapChainDescriptor(
        wgpu.TextureUsage_OUTPUT_ATTACHMENT,  # usage
        wgpu.TextureFormat_Bgra8UnormSrgb, # format: wgpu::TextureFormat::Bgra8UnormSrgb,
        640, # width: size.width.round() as u32,
        480, # height: size.height.round() as u32,
        wgpu.PresentMode_Vsync,  # present_mode: wgpu::PresentMode::Vsync,
    )
)


def drawFrame():
    next_texture = wgpu.swap_chain_get_next_texture(swap_chain)
    command_encoder = wgpu.device_create_command_encoder(
        device_id,
        wgpu.create_CommandEncoderDescriptor(),
    )

    rpass = wgpu.command_encoder_begin_render_pass(
        command_encoder,
        wgpu.create_RenderPassDescriptor(
            [  # color attachements
                wgpu.create_RenderPassColorAttachmentDescriptor(
                    next_texture["view_id"],
                    None, # resolve_target: None,
                    wgpu.LoadOp_Clear,  # load_op: wgpu::LoadOp::Clear,
                    wgpu.StoreOp_Store,  # store_op: wgpu::StoreOp::Store,
                    (1, 1, 0, 1), # clear_color: wgpu::Color::GREEN,
                ),
            ],
            1,
            None,  # depth_stencil_attachement
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

