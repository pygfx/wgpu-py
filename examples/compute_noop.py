"""
Example compute shader that does ... nothing but copy a value from one
buffer into another.
"""

import ctypes

import wgpu
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
from pyshader import python2shader, i32, Array


# %% Shader and data


@python2shader
def compute_shader(
    index: ("input", "GlobalInvocationId", i32),
    data1: ("buffer", 0, Array(i32)),
    data2: ("buffer", 1, Array(i32)),
):
    data2[index] = data1[index]


# Create input data as a ctypes array
n = 20
IntArrayType = ctypes.c_int32 * n
data = IntArrayType(*range(n))


# %% The short version, using ctypes arrays

# The first arg is the input data, per binding
# The second arg are the ouput types, per binding
out = compute_with_buffers({0: data}, {1: IntArrayType}, compute_shader, n=n)

# The result is a dict matching the output types
# Select data from buffer at binding 1
result = out[1]
print(result[:])


# %% The short version, using numpy

# import numpy as np
#
# numpy_data = np.arange(n, dtype=np.int32)
#
# data = IntArrayType.from_address(numpy_data.ctypes.data)
# out = compute_with_buffers({0: data}, {1: IntArrayType}, compute_shader, n=n)
# print(np.frombuffer(out[1], dtype=np.int32))


# %% The long version using the wgpu API

# Create device and shader object
device = wgpu.utils.get_default_device()
cshader = device.create_shader_module(code=compute_shader)

# Create buffer objects, input buffer is mapped.
buffer1 = device.create_buffer_mapped(
    size=ctypes.sizeof(data), usage=wgpu.BufferUsage.STORAGE
)
buffer2 = device.create_buffer(
    size=ctypes.sizeof(data), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ
)

# Cast buffer array
array1 = IntArrayType.from_buffer(buffer1.mapping)
# With Numpy this would be:
# array1 = np.frombuffer(buffer1.mapping, np.int32)

# Copy data and then unmap
array1[:] = data
buffer1.unmap()

# Setup layout and bindings
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "type": wgpu.BindingType.storage_buffer,
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "type": wgpu.BindingType.storage_buffer,
    },
]
bindings = [
    {"binding": 0, "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},},
    {"binding": 1, "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},},
]

# Put everything together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout, compute_stage={"module": cshader, "entry_point": "main"},
)
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
compute_pass.dispatch(n, 1, 1)  # x y z
compute_pass.end_pass()
device.default_queue.submit([command_encoder.finish()])

# Read result
result = buffer2.map_read()
result = IntArrayType.from_buffer(result)  # cast
print(result[:])

# With Numpy this would be:
# print(np.frombuffer(buffer2.mapRead(), np.int32))
