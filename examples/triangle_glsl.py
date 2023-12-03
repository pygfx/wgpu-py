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


def main(canvas, power_preference="high-performance", limits=None):
    """Regular function to setup a viz on the given canvas."""
    adapter = wgpu.gpu.request_adapter(power_preference=power_preference)
    device = adapter.request_device(required_limits=limits)
    return _main(canvas, device)


async def main_async(canvas):
    """Async function to setup a viz on the given canvas."""
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async(required_limits={})
    return _main(canvas, device)


def _main(canvas, device):
    vert_shader = device.create_shader_module(label="triangle_vert", code=vertex_shader)
    frag_shader = device.create_shader_module(
        label="triangle_frag", code=fragment_shader
    )

    # No bind group and layout, we should not create empty ones.
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    present_context = canvas.get_context()
    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": vert_shader,
            "entry_point": "main",
            "buffers": [],
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
                        "color": (
                            wgpu.BlendFactor.one,
                            wgpu.BlendFactor.zero,
                            wgpu.BlendOperation.add,
                        ),
                        "alpha": (
                            wgpu.BlendFactor.one,
                            wgpu.BlendFactor.zero,
                            wgpu.BlendOperation.add,
                        ),
                    },
                },
            ],
        },
    )

    def draw_frame():
        current_texture = present_context.get_current_texture()
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
        # render_pass.set_bind_group(0, no_bind_group, [], 0, 1)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    canvas.request_draw(draw_frame)
    return device


if __name__ == "__main__":
    from wgpu.gui.auto import WgpuCanvas, run

    canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle")
    main(canvas)
    run()
