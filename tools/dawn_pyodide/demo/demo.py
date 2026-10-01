# wgpu-py in the browser: the Dawn backend (a Cython extension) on top of
# Emdawnwebgpu, i.e. the browser's own WebGPU implementation (hardware
# accelerated). Deployed as a demo page by .github/workflows/dawn.yml; the same
# demo as for the cffi variant in #840.
import array
import struct
import time

import js
from pyodide.ffi import create_proxy

import wgpu

COMPUTE_SHADER = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  b[id.x] = a[id.x] * a[id.x];
}
"""

RENDER_SHADER = """
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> };
@group(0) @binding(0) var<uniform> angle: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(0.0, 0.7), vec2(-0.6, -0.5), vec2(0.6, -0.5));
    var c = array<vec3<f32>, 3>(vec3(1.0, 0.25, 0.2), vec3(0.2, 0.9, 0.3), vec3(0.2, 0.4, 1.0));
    let a = angle.x;
    let q = vec2(p[i].x * cos(a) - p[i].y * sin(a), p[i].x * sin(a) + p[i].y * cos(a));
    var out: VOut;
    out.pos = vec4(q, 0.0, 1.0);
    out.color = vec4(c[i], 1.0);
    return out;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> { return in.color; }
"""


async def run_compute(device, log):
    n = 1 << 20
    data = array.array("f", range(n))
    a = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
    b = device.create_buffer(
        size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    readback = device.create_buffer(
        size=n * 4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )
    pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": device.create_shader_module(code=COMPUTE_SHADER)},
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": a}},
            {"binding": 1, "resource": {"buffer": b}},
        ],
    )
    t0 = time.perf_counter()
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind_group)
    cpass.dispatch_workgroups(n // 64)
    cpass.end()
    encoder.copy_buffer_to_buffer(b, 0, readback, 0, n * 4)
    device.queue.submit([encoder.finish()])
    await readback.map_async(wgpu.MapMode.READ)
    out = readback.read_mapped().cast("f")
    readback.unmap()
    dt = time.perf_counter() - t0
    ok = all(out[i] == float(i * i) for i in (0, 1, 2, 3, 1000, 4095))
    log(
        f"Compute: squared {n:,} floats on the GPU in {dt * 1000:.1f} ms "
        f"(incl. readback): out[:4] = {list(out[:4])}, out[4095] = {out[4095]}"
    )
    log("Compute result: " + ("CORRECT" if ok else "WRONG"))
    return ok


async def main(canvas, log):
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    info = adapter.info
    log(f"wgpu backend: {type(wgpu.gpu).__module__}")
    log(
        f"Adapter: vendor={info['vendor']!r} architecture={info['architecture']!r} "
        f"device={info['device']!r} description={info['description']!r} "
        f"backend={info['backend_type']}"
    )
    if "swiftshader" in (info["architecture"] + info["vendor"]).lower():
        log(
            "This is a software (SwiftShader) adapter: the browser has no usable GPU here."
        )
    else:
        log("This is a hardware adapter: the browser's WebGPU on your GPU.")
    device = await adapter.request_device_async()
    ok = await run_compute(device, log)

    # Render a rotating triangle to the <canvas>
    context = wgpu.gpu.get_canvas_context({"window": canvas, "platform": "browser"})
    context.set_physical_size(canvas.width, canvas.height)
    fmt = context.get_preferred_format(adapter)
    context.configure(device=device, format=fmt)
    module = device.create_shader_module(code=RENDER_SHADER)
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": module},
        fragment={"module": module, "targets": [{"format": fmt}]},
    )
    ubuf = device.create_buffer(
        size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": ubuf}}],
    )
    t_start = time.perf_counter()
    frames = [0]

    def draw_frame(_timestamp=None):
        angle = (time.perf_counter() - t_start) * 0.8
        device.queue.write_buffer(ubuf, 0, struct.pack("4f", angle, 0, 0, 0))
        view = context.get_current_texture().create_view()
        encoder = device.create_command_encoder()
        rpass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "clear_value": (0.08, 0.08, 0.1, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ]
        )
        rpass.set_pipeline(pipeline)
        rpass.set_bind_group(0, bind_group)
        rpass.draw(3)
        rpass.end()
        device.queue.submit([encoder.finish()])
        context.present()
        frames[0] += 1
        js.requestAnimationFrame(draw_proxy)

    draw_proxy = create_proxy(draw_frame)
    draw_frame()
    log(f"Rendering a triangle to the canvas ({fmt}).")
    return ok
