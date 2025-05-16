"""
The triangle example, using GLSL shaders.

"""

import wgpu


# %% Shaders


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


# %% The wgpu calls


def setup_drawing_sync(canvas, power_preference="high-performance", limits=None):
    """Regular function to set up a viz on the given canvas."""

    adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
    device = adapter.request_device_sync(required_limits=limits)

    render_pipeline = get_render_pipeline(canvas, device)
    return get_draw_function(canvas, device, render_pipeline)


def get_render_pipeline(canvas, device):
    vert_shader = device.create_shader_module(label="triangle_vert", code=vertex_shader)
    frag_shader = device.create_shader_module(
        label="triangle_frag", code=fragment_shader
    )

    # No bind group and layout, we should not create empty ones.
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    present_context = canvas.get_context("wgpu")
    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)

    return device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": vert_shader,
            "entry_point": "main",
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=None,
        multisample=None,
        fragment={
            "module": frag_shader,
            "entry_point": "main",
            "targets": [
                {
                    "format": render_texture_format,
                    "blend": {
                        "color": {},
                        "alpha": {},
                    },
                },
            ],
        },
    )


def get_draw_function(canvas, device, render_pipeline):
    def draw_frame():
        current_texture = canvas.get_context("wgpu").get_current_texture()
        command_encoder = device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(render_pipeline)
        # render_pass.set_bind_group(0, no_bind_group)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    return draw_frame


if __name__ == "__main__":
    from rendercanvas.auto import RenderCanvas, loop

    canvas = RenderCanvas(size=(640, 480), title="wgpu triangle glsl example")
    draw_frame = setup_drawing_sync(canvas)
    canvas.request_draw(draw_frame)
    loop.run()
