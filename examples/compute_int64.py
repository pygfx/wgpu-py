"""
simple example of using the int64 shader feature
"""

import wgpu

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync(required_features=["shader-int64"])

add_shader = """
@group(0) @binding(0)
var<storage, read_write> data: array<i64>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x;
    let a_idx = i * 3u;
    let b_idx = a_idx + 1u;
    let c_idx = a_idx + 2u;
    data[c_idx] = data[a_idx] + data[b_idx];
}
"""
a = 0x1234567890
b = 0x9876543210
data = memoryview(bytearray(6*8)).cast("q") # signed long long >= 64 bits
data[0] = a
data[1] = b
data[3] = -a
data[4] = -b

buffer = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={
        "module": device.create_shader_module(code=add_shader),
        "entry_point": "main",
    }
)

bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {
            "binding": 0,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        }
    ],
)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(2) # we do two calculations
compute_pass.end()
device.queue.submit([command_encoder.finish()])
result = device.queue.read_buffer(buffer)
res = result.cast("q").tolist()
c = a + b
assert res[2] == c, f"expected {c}, got {res[2]}"
print(f"{a:#x} + {b:#x} = {c:#x}")
d = -a + -b
assert res[5] == d, f"expected {d}, got {res[5]}"
print(f"{-a:#x} + {-b:#x} = {d:#x}")

