"""
A simple compute example demonstrating GPU workgroups and invocation IDs.

Each thread writes its global, local, and workgroup IDs into a storage buffer
so the relationship between them can be inspected.
"""

import numpy as np
import wgpu

# define workgroup configuration

workgroup_size = 4
workgroups = 3
total_threads = workgroup_size * workgroups

# Each thread writes 3 uint32 values
output_elements = total_threads * 3
output_bytes = output_elements * 4

# compute shader

shader_source = f"""
@group(0) @binding(0)
var<storage, read_write> out: array<u32>;

@compute
@workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id)  local_id  : vec3<u32>,
    @builtin(workgroup_id)         wg_id     : vec3<u32>,
) {{
    let base: u32 = global_id.x * 3u;

    out[base]     = global_id.x;
    out[base + 1] = local_id.x;
    out[base + 2] = wg_id.x;
}}
"""

# adapter and device

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# storage buffer

output_buffer = device.create_buffer(
    size=output_bytes,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
)

# Shader module and compute pipeline

shader_module = device.create_shader_module(code=shader_source)

compute_pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={"module": shader_module, "entry_point": "main"},
)

# bind group

bind_group = device.create_bind_group(
    layout=compute_pipeline.get_bind_group_layout(0),
    entries=[
        {
            "binding": 0,
            "resource": {
                "buffer": output_buffer,
                "offset": 0,
                "size": output_buffer.size,
            },
        }
    ],
)

# encode, dispatch and submit

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()

compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(workgroups, 1, 1)

compute_pass.end()

device.queue.submit([command_encoder.finish()])

# results

raw = device.queue.read_buffer(output_buffer)
values = np.frombuffer(raw, dtype=np.uint32)

print(
    f"Dispatched {workgroups} workgroup(s) of {workgroup_size} thread(s) each "
    f"({total_threads} threads total).\n"
)

print(f"{'Thread':>6}  {'global_id':>9}  {'local_id':>8}  {'workgroup_id':>12} ")

for i in range(total_threads):
    global_id = values[i * 3]
    local_id = values[i * 3 + 1]
    workgroup_id = values[i * 3 + 2]

    print(f"{i:>6}  {global_id:>9}  {local_id:>8}  {workgroup_id:>12}")

    # verify invocation ID relationships
    assert global_id == i
    assert local_id == i % workgroup_size
    assert workgroup_id == i // workgroup_size

print("Invocation ID mapping verified")
