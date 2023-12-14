"""
A simple example to profile a compute pass using ComputePassTimestampWrites.
"""

import wgpu

"""
Define the number of elements, global and local sizes.
Change these and see how it affects performance.
"""
n = 512 * 512
local_size = [32, 1, 1]
global_size = [n // local_size[0], 1, 1]

shader_source = f"""
@group(0) @binding(0)
var<storage,read> data1: array<i32>;

@group(0) @binding(1)
var<storage,read> data2: array<i32>;

@group(0) @binding(2)
var<storage,read_write> data3: array<i32>;

@compute
@workgroup_size({','.join(map(str, local_size))})
fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
    let i: u32 = index.x;
    data3[i] = data1[i] + data2[i];
}}
"""

# Define two arrays
data1 = memoryview(bytearray(n * 4)).cast("i")
data2 = memoryview(bytearray(n * 4)).cast("i")

# Initialize the arrays
for i in range(n):
    data1[i] = i

for i in range(n):
    data2[i] = i * 2

adapter = wgpu.gpu.request_adapter(power_preference="high-performance")

# Request a device with the timestamp_query feature, so we can profile our computation
device = adapter.request_device(required_features=[wgpu.FeatureName.timestamp_query])
cshader = device.create_shader_module(code=shader_source)

# Create buffer objects, input buffer is mapped.
buffer1 = device.create_buffer_with_data(data=data1, usage=wgpu.BufferUsage.STORAGE)
buffer2 = device.create_buffer_with_data(data=data2, usage=wgpu.BufferUsage.STORAGE)
buffer3 = device.create_buffer(
    size=data1.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)

# Setup layout and bindings
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": buffer3, "offset": 0, "size": buffer3.size},
    },
]

# Put everything together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

"""
Create a QuerySet to store the 'beginning_of_pass' and 'end_of_pass' timestamps.
Set the 'count' parameter to 2, as this set will contain 2 timestamps.
"""
query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=2)
command_encoder = device.create_command_encoder()

# Pass our QuerySet and the indices into it, where the timestamps will be written.
compute_pass = command_encoder.begin_compute_pass(
    timestamp_writes={
        "query_set": query_set,
        "beginning_of_pass_write_index": 0,
        "end_of_pass_write_index": 1,
    }
)

"""
Create the buffer to store our query results.
Each timestamp is 8 bytes. We mark the buffer usage to be QUERY_RESOLVE,
as we will use this buffer in a resolve_query_set call later.
"""
query_buf = device.create_buffer(
    size=8 * query_set.count,
    usage=wgpu.BufferUsage.QUERY_RESOLVE
    | wgpu.BufferUsage.STORAGE
    | wgpu.BufferUsage.COPY_SRC
    | wgpu.BufferUsage.COPY_DST,
)
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
compute_pass.dispatch_workgroups(*global_size)  # x y z
compute_pass.end()

# Resolve our queries, and store the results in the destination buffer we created above.
command_encoder.resolve_query_set(
    query_set=query_set,
    first_query=0,
    query_count=2,
    destination=query_buf,
    destination_offset=0,
)
device.queue.submit([command_encoder.finish()])

"""
Read the query buffer to get the timestamps.
Index 0: beginning timestamp
Index 1: end timestamp
"""
timestamps = device.queue.read_buffer(query_buf).cast("Q").tolist()
print(f"Adding two {n} sized arrays took {(timestamps[1]-timestamps[0])/1000} us")

# Read result
out = device.queue.read_buffer(buffer3).cast("i")
result = out.tolist()

# Calculate the result on the CPU for comparison
result_cpu = [a + b for a, b in zip(data1, data2)]

# Ensure results are the same
assert result == result_cpu
