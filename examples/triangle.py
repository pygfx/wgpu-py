"""
Example use of webgpu API to draw a triangle. See the triangle_glfw.py
script (and related scripts) for actually running this.

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
    [[builtin(vertex_index)]] vertex_index : u32;
};
struct VertexOutput {
    [[location(0)]] color : vec4<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
    let p: vec2<f32> = positions[in.vertex_index];

    var out: VertexOutput;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.color = vec4<f32>(p, 0.5, 1.0);
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return in.color;
}
"""


# %% The wgpu calls


def main(canvas):
    """Regular function to setup a viz on the given canvas."""
    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    return _main(canvas, device)


async def main_async(canvas):
    """Async function to setup a viz on the given canvas."""
    adapter = await wgpu.request_adapter_async(
        canvas=canvas, power_preference="high-performance"
    )
    device = await adapter.request_device_async(extensions=[], limits={})
    return _main(canvas, device)


def _main(canvas, device):

    shader = device.create_shader_module(code=shader_source)

    bind_group_layout = device.create_bind_group_layout(entries=[])  # zero bindings
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[])

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            # "strip_index_format": wgpu.IndexFormat.uint32,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        },
        depth_stencil=None,
        multisample={
            "count": 1,
            "mask": 0xFFFFFFFF,
            "alpha_to_coverage_enabled": False,
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": wgpu.TextureFormat.bgra8unorm_srgb,
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

    swap_chain = canvas.configure_swap_chain(device=device)

    def draw_frame():
        with swap_chain as current_texture_view:
            command_encoder = device.create_command_encoder()

            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": current_texture_view,
                        "resolve_target": None,
                        "load_value": (0, 0, 0, 1),  # LoadOp.load or color
                        "store_op": wgpu.StoreOp.store,
                    }
                ],
            )

            render_pass.set_pipeline(render_pipeline)
            render_pass.set_bind_group(
                0, bind_group, [], 0, 999999
            )  # last 2 elements not used
            render_pass.draw(3, 1, 0, 0)
            render_pass.end_pass()
            device.queue.submit([command_encoder.finish()])

    canvas.request_draw(draw_frame)
