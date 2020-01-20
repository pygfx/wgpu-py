"""
Example compute shader that does ... nothing but copy a value from one
buffer into another
"""
import asyncio

import numpy as np
import wgpu
from python_shader import python2shader, i32, Array

# Select backend
import wgpu.backend.rs


# %% Shader


@python2shader
def compute_shader(input, buffer):
    input.define("index", "GlobalInvocationId", i32)
    buffer.define("data1", 0, Array(i32))
    buffer.define("data2", 1, Array(i32))

    buffer.data2[input.index] = buffer.data1[input.index]


# %% The wgpu calls


async def main():

    adapter = await wgpu.requestAdapter(powerPreference="high-performance")
    device = await adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())

    cshader = device.createShaderModule(code=compute_shader)

    n = 10
    buffer1 = device.createBufferMapped(
        size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ
    )
    # buffer2 = device.createBufferMapped(size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ)
    buffer2 = device.createBuffer(
        size=n * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ
    )
    array1 = np.frombuffer(buffer1.mapping, np.int32)
    array1[:] = np.arange(0, n)

    buffer1.unmap()
    buffer2.unmap()

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
        {
            "binding": 0,
            "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
        },
    ]

    bind_group_layout = device.createBindGroupLayout(bindings=binding_layouts)
    pipeline_layout = device.createPipelineLayout(bindGroupLayouts=[bind_group_layout])
    bind_group = device.createBindGroup(layout=bind_group_layout, bindings=bindings)

    compute_pipeline = device.createComputePipeline(
        layout=pipeline_layout, computeStage={"module": cshader, "entryPoint": "main"},
    )

    command_encoder = device.createCommandEncoder()
    compute_pass = command_encoder.beginComputePass()
    compute_pass.setPipeline(compute_pipeline)
    compute_pass.setBindGroup(0, bind_group, [], 0, 999999)  # last 2 elements not used
    compute_pass.dispatch(n, 1, 1)  # x y z
    compute_pass.endPass()
    device.defaultQueue.submit([command_encoder.finish()])

    print(np.frombuffer(await buffer2.mapReadAsync(), np.int32))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
