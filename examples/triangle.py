"""
Example use of the wgpu API to draw a triangle. This example is set up
so it can be run on canvases provided by any backend. Running this file
as a script will use the auto-backend (using either glfw or jupyter).


Similar example in other languages / API's:

* Rust wgpu:
  https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/main.rs
* C wgpu:
  https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c
* Python Vulkan:
  https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

"""

import wgpu


# %% Shaders


shader_source = """
struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};
struct VertexOutput {
    @location(0) color : vec4<f32>,
    @builtin(position) pos: vec4<f32>,
};

@stage(vertex)
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
    let index = i32(in.vertex_index);
    let p: vec2<f32> = positions[index];

    var out: VertexOutput;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.color = vec4<f32>(p, 0.5, 1.0);
    return out;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"""


# %% The wgpu calls


def main(canvas, power_preference="high-performance", limits=None):
    """Regular function to setup a viz on the given canvas."""
    # Note: passing the canvas here can (oddly enough) prevent the
    # adapter from being found. Seen with wx/Linux.
    adapter = wgpu.request_adapter(canvas=None, power_preference=power_preference)
    device = adapter.request_device(required_limits=limits)
    return _main(canvas, device)


async def main_async(canvas):
    """Async function to setup a viz on the given canvas."""
    adapter = await wgpu.request_adapter_async(
        canvas=canvas, power_preference="high-performance"
    )
    device = await adapter.request_device_async(required_limits={})
    return _main(canvas, device)


def _main(canvas, device):

    shader = device.create_shader_module(code=shader_source)

    # No bind group and layout, we should not create empty ones.
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    present_context = canvas.get_context()
    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
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
            "module": shader,
            "entry_point": "fs_main",
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
        current_texture_view = present_context.get_current_texture()
        command_encoder = device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
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

    import wgpu.backends.rs  # noqa: F401, Select Rust backend
    from wgpu.gui.auto import WgpuCanvas, run

    canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle")
    main(canvas)
    run()
