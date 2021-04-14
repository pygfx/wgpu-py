"""
Example compute shader that does ... nothing but copy a value from one
buffer into another.
"""

import wgpu
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
from pyshader import python2shader, ivec3, i32, Array


# %% Shader and data


@python2shader
def compute_shader(
    index: ("input", "GlobalInvocationId", ivec3),
    data1: ("buffer", 0, Array(i32)),
    data2: ("buffer", 1, Array(i32)),
):
    i = index.x
    data2[i] = data1[i]


# Create input data as a memoryview
n = 20
data = memoryview(bytearray(n * 4)).cast("i")
for i in range(n):
    data[i] = i


# %% The short version, using memoryview

# The first arg is the input data, per binding
# The second arg are the ouput types, per binding
out = compute_with_buffers({0: data}, {1: (n, "i")}, compute_shader, n=n)

# The result is a dict matching the output types
# Select data from buffer at binding 1
result = out[1]
print(result.tolist())


# %% The short version, using numpy

# import numpy as np
#
# numpy_data = np.frombuffer(data, np.int32)
# out = compute_with_buffers({0: numpy_data}, {1: numpy_data.nbytes}, compute_shader, n=n)
# result = np.frombuffer(out[1], dtype=np.int32)
# print(result)


# %% The long version using the wgpu API

# Create device and shader object
device = wgpu.utils.get_default_device()
cshader = device.create_shader_module(code=compute_shader)

# Create buffer objects, input buffer is mapped.
buffer1 = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
buffer2 = device.create_buffer(
    size=data.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ
)

# Setup layout and bindings
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
    {
        "binding": 1,
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
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
compute_pass.dispatch(n, 1, 1)  # x y z
compute_pass.end_pass()
device.queue.submit([command_encoder.finish()])

# Read result
result = buffer2.read_data().cast("i")
print(result.tolist())
