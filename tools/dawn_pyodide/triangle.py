# Render a triangle to an offscreen texture and check its pixels, and (in the
# browser) to an HTML <canvas> via Emdawnwebgpu's canvas surface.
import sys

import wgpu

SHADER = """
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(0.0, 0.6), vec2(-0.6, -0.6), vec2(0.6, -0.6));
    var c = array<vec3<f32>, 3>(vec3(1.0, 0.2, 0.2), vec3(0.2, 1.0, 0.2), vec3(0.2, 0.4, 1.0));
    var out: VOut;
    out.pos = vec4(p[i], 0.0, 1.0);
    out.color = vec4(c[i], 1.0);
    return out;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"""


def make_pipeline(device, fmt):
    module = device.create_shader_module(code=SHADER)
    return device.create_render_pipeline(
        layout="auto",
        vertex={"module": module, "entry_point": "vs_main"},
        fragment={
            "module": module,
            "entry_point": "fs_main",
            "targets": [{"format": fmt}],
        },
        primitive={"topology": "triangle-list"},
    )


def draw(device, pipeline, view):
    encoder = device.create_command_encoder()
    rpass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": view,
                "clear_value": (0.1, 0.1, 0.12, 1),
                "load_op": "clear",
                "store_op": "store",
            }
        ]
    )
    rpass.set_pipeline(pipeline)
    rpass.draw(3)
    rpass.end()
    device.queue.submit([encoder.finish()])


async def render_offscreen_and_check(device, size=64):
    fmt = "rgba8unorm"
    tex = device.create_texture(
        size=(size, size, 1),
        format=fmt,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    draw(device, make_pipeline(device, fmt), tex.create_view())
    # read_texture() is a sync call; in Pyodide it suspends via JSPI
    data = device.queue.read_texture(
        {"texture": tex}, {"bytes_per_row": size * 4}, (size, size, 1)
    )
    data = bytes(data)
    center = data[
        (size // 2 * size + size // 2) * 4 : (size // 2 * size + size // 2) * 4 + 4
    ]
    corner = data[0:4]
    print("offscreen center pixel", tuple(center), "corner pixel", tuple(corner))
    assert center[3] == 255 and sum(center[:3]) > 200, center  # inside the triangle
    assert tuple(corner) == (26, 26, 31, 255), corner  # clear color
    print("TRIANGLE OFFSCREEN OK")


async def main(canvas=None):
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async()
    await render_offscreen_and_check(device)
    if canvas is not None:
        context = wgpu.gpu.get_canvas_context({"window": canvas, "platform": "browser"})
        context.set_physical_size(canvas.width, canvas.height)
        fmt = context.get_preferred_format(adapter)
        context.configure(device=device, format=fmt)
        pipeline = make_pipeline(device, fmt)
        draw(device, pipeline, context.get_current_texture().create_view())
        context.present()
        print(f"TRIANGLE CANVAS OK (format {fmt})")
    return device


if __name__ == "__main__" and sys.platform != "emscripten":
    import asyncio

    asyncio.run(main())
