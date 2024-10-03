"""
Example use of the wgpu API to draw a triangle.

The triangle is a classic example representing the simplest possible
visualisation because it does not need buffers or textures. The same
example in other languages / API's:

* Rust wgpu:
  https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/main.rs
* C wgpu:
  https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c
* Python Vulkan:
  https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

This example is set up so it can be run with any canvas. Running this file
as a script will use the auto-backend.

"""

import wgpu


# %% Entrypoints (sync and async)


def setup_drawing_sync(canvas, power_preference="high-performance", limits=None):
    """Setup to draw a triangle on the given canvas.

    The given canvas must implement WgpuCanvasInterface, but nothing more.
    Returns the draw function.
    """

    adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
    device = adapter.request_device_sync(required_limits=limits)

    pipeline_kwargs = get_render_pipeline_kwargs(canvas, device)

    render_pipeline = device.create_render_pipeline(**pipeline_kwargs)

    return get_draw_function(canvas, device, render_pipeline, asynchronous=False)


async def setup_drawing_async(canvas, limits=None):
    """Setup to async-draw a triangle on the given canvas.

    The given canvas must implement WgpuCanvasInterface, but nothing more.
    Returns the draw function.
    """

    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async(required_limits=limits)

    pipeline_kwargs = get_render_pipeline_kwargs(canvas, device)

    render_pipeline = await device.create_render_pipeline_async(**pipeline_kwargs)

    return get_draw_function(canvas, device, render_pipeline, asynchronous=True)


# %% Functions to create wgpu objects


def get_render_pipeline_kwargs(canvas, device):
    context = canvas.get_context()
    render_texture_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=render_texture_format)

    shader = device.create_shader_module(code=shader_source)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

    return dict(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
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
                        "color": {},
                        "alpha": {},
                    },
                },
            ],
        },
    )


def get_draw_function(canvas, device, render_pipeline, *, asynchronous):
    def draw_frame_sync():
        current_texture = canvas.get_context().get_current_texture()
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

    async def draw_frame_async():
        draw_frame_sync()  # nothing async here

    if asynchronous:
        return draw_frame_async
    else:
        return draw_frame_sync


# %% WGSL


shader_source = """
struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};
struct VertexOutput {
    @location(0) color : vec4<f32>,
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, 0.75),
    );
    var colors = array<vec3<f32>, 3>(  // srgb colors
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 1.0),
    );
    let index = i32(in.vertex_index);
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[index], 0.0, 1.0);
    out.color = vec4<f32>(colors[index], 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let physical_color = pow(in.color.rgb, vec3<f32>(2.2));  // gamma correct
    return vec4<f32>(physical_color, in.color.a);
}
"""


if __name__ == "__main__":
    from wgpu.gui.auto import WgpuCanvas, run

    canvas = WgpuCanvas(size=(640, 480), title="wgpu triangle example")
    draw_frame = setup_drawing_sync(canvas)
    canvas.request_draw(draw_frame)
    run()
