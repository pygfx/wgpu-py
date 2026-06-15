"""
The triangle example, using GLSL shaders.
"""

import wgpu
from rendercanvas.auto import RenderCanvas, loop


vertex_shader = """
#version 450 core
layout(location = 0) out vec4 color;
void main()
{
    vec2 positions[3] = vec2[3](
        vec2(0.0, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.75)
    );
    vec3 colors[3] = vec3[3](  // srgb colors
        vec3(1.0, 1.0, 0.0),
        vec3(1.0, 0.0, 1.0),
        vec3(0.0, 1.0, 1.0)
    );
    int index = int(gl_VertexID);
    gl_Position = vec4(positions[index], 0.0, 1.0);
    color = vec4(colors[index], 1.0);
}
"""

fragment_shader = """
#version 450 core
out vec4 FragColor;
layout(location = 0) in vec4 color;
void main()
{
    vec3 physical_color = pow(color.rgb, vec3(2.2));  // gamma correct
    FragColor = vec4(physical_color, color.a);
}
"""

# adapter is required to create the device object, which is the general entry point to create most wgpu objects.
# for convenience and interoperability `wgpu.utils.get_default_device()` and associated configuration are provided.
adapter = wgpu.gpu.request_adapter_sync()
device = adapter.request_device_sync()

# setting up a canvas, so we can see what we draw
canvas = RenderCanvas(size=(640, 480), title="wgpu triangle example")
context = canvas.get_wgpu_context()
render_texture_format = context.get_preferred_format(device.adapter)
context.configure(device=device, format=render_texture_format)

# creating the shader module compiles the shader code for your GPU.
vert_shader = device.create_shader_module(label="triangle_vert", code=vertex_shader)
frag_shader = device.create_shader_module(label="triangle_frag", code=fragment_shader)

# in wgpu-py, methods that take descriptors will take the keyword arguments instead.
# descriptors and other structs can still be accessed via wgpu.structs or top level wgpu.
render_pipeline = device.create_render_pipeline(
    **wgpu.RenderPipelineDescriptor(
        layout=wgpu.AutoLayoutMode.auto,
        vertex=wgpu.VertexState(module=vert_shader),
        depth_stencil=None,
        multisample=None,
        fragment=wgpu.FragmentState(
            module=frag_shader,
            targets=[wgpu.ColorTargetState(format=render_texture_format)],
        ),
    )
)


# this function gets called for every frame. It ends with submitting a buffer of work onto the GPU queue.
def drawing_function():
    command_encoder = device.create_command_encoder()
    current_texture_view = context.get_current_texture().create_view()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=current_texture_view,
                clear_value=(0, 0.2, 0, 1),  # a green background
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
            )
        ],
    )
    render_pass.set_pipeline(render_pipeline)
    render_pass.draw(3)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])


if __name__ == "__main__":
    canvas.request_draw(drawing_function)
    loop.run()
