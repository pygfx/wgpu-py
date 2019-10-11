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
    # glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # this disabled opengl?
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

glfw.make_context_current(window)

# todo: surface_id
surface_id = window

##

# Instantiate gl-based wgpu context
wgpu = webgpu.wgpu_gl.GlWGPU()

adapter_id = wgpu.request_adapter(
    wgpu.create_RequestAdapterOptions(
        power_preference=wgpu.PowerPreference_Default,
        backends=2 | 4 | 8  # backend bits - no idea what this means
        )
)

device_des = wgpu.create_DeviceDescriptor(
    extensions=wgpu.create_Extensions(anisotropic_filtering=False),
    limits=wgpu.create_Limits(max_bind_groups=99)
)

device_id = wgpu.adapter_request_device(adapter_id, device_des)


# gl_VertexIndex vs gl_InstanceID

vertex_code = """
#version 450
#extension GL_ARB_separate_shader_objects : enable
//#extension GL_KHR_vulkan_glsl : enable

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
    fragColor = colors[gl_VertexID];
}
"""

fragment_code = """
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    float x = 0.5;
    outColor = vec4(fragColor, x + 0.5);
}
"""

vs_module = wgpu.device_create_shader_module(
    device_id,
    wgpu.create_ShaderModuleDescriptor(code=vertex_code)
)

fs_module = wgpu.device_create_shader_module(
    device_id,
    wgpu.create_ShaderModuleDescriptor(code=fragment_code)
)

# todo: I think this is where uniforms go
bind_group_layout = wgpu.device_create_bind_group_layout(
    device_id,
    wgpu.create_BindGroupLayoutDescriptor(bindings=[], bindings_length=0)
)

bind_group = wgpu.device_create_bind_group(
    device_id,
    wgpu.create_BindGroupDescriptor(layout=bind_group_layout, bindings=[], bindings_length=0)
)

pipeline_layout = wgpu.device_create_pipeline_layout(
    device_id,
    wgpu.create_PipelineLayoutDescriptor(bind_group_layouts=[bind_group], bind_group_layouts_length=1)
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
            vertex_buffers=[],
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
            color_attachments=[
                wgpu.create_RenderPassColorAttachmentDescriptor(
                    attachment=next_texture["view_id"],
                    resolve_target=None, # resolve_target: None,
                    load_op=wgpu.LoadOp_Clear,  # load_op: wgpu::LoadOp::Clear,
                    store_op=wgpu.StoreOp_Store,  # store_op: wgpu::StoreOp::Store,
                    clear_color=(1, 1, 0, 1), # clear_color: wgpu::Color::GREEN,
                ),
            ],
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

