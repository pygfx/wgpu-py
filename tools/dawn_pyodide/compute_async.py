# Compute shader round trip through the public wgpu API, async.
# Runs natively (WGPUPY_BACKEND=dawn python compute_async.py) and in Pyodide.
import sys
import struct
import asyncio
import wgpu

SHADER = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  b[id.x] = a[id.x] * 2.0 + 1.0;
}
"""


async def main():
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    print("wgpu backend:", type(wgpu.gpu).__module__)
    print("adapter info:", dict(adapter.info))
    device = await adapter.request_device_async()
    n = 1024
    data = struct.pack(f"{n}f", *range(n))
    buf_a = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
    buf_b = device.create_buffer(
        size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    readback = device.create_buffer(
        size=n * 4, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
    )
    module = device.create_shader_module(code=SHADER)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"}
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": buf_a}},
            {"binding": 1, "resource": {"buffer": buf_b}},
        ],
    )
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind_group)
    cpass.dispatch_workgroups(n // 64)
    cpass.end()
    encoder.copy_buffer_to_buffer(buf_b, 0, readback, 0, n * 4)
    device.queue.submit([encoder.finish()])
    await readback.map_async(wgpu.MapMode.READ)
    out = readback.read_mapped().cast("f")
    readback.unmap()
    expected = [i * 2.0 + 1 for i in range(n)]
    assert list(out) == expected, list(out[:8])
    print("result[:4] =", list(out[:4]), "result[-1] =", out[-1])
    await device.queue.on_submitted_work_done_async()
    print("COMPUTE ASYNC OK")


if sys.platform != "emscripten":
    asyncio.run(main())
