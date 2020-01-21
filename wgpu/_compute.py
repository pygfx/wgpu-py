"""
Simple high-level utilities for doing compute on the GPU.

The purpose of wgpu-py is to provide a Pythonic wrapper around wgpu-native.
In principal, a higher-level API is not within the scope of the project.
However, by providing a few un-opinionated utility functions, other projects
(like python-shader) can use wgpu (e.g. in their tests) without having
to keep track of changes in wgpu itself.
"""

import sys

import wgpu

# Note: these funcs need numpy, but we don't import it because its not a strict dep of wgpu-py


def compute_with_buffers(input_arrays, output_arrays, shader, n=None):
    """ Apply the given compute shader to the given input_arrays and return
    output arrays. Both input and output arrays are represented on the GPU
    using storage buffer objects.

    Params:
        input_arrays (dict): A dict mapping an integer binding to a numpy array.
        output_arrays (dict): A dict mapping an integer binding to an outputs spec.
            The spec must be a tuple ``(*shape, dtype)``. A buffer is created
            for each binding and its data is read and returned. The binding can be
            one that is also used as an input array (in which case the GPU buffer
            is the same).
        shader (bytes, shader-object): Any compatible shader object representing
            a SpirV compute shader.
        n (int, tuple, optional): The number provided to dispatch. Can be an int
            or a 3-tuple of ints to specify (x, y, z). If not given or None,
            the first value of the first spec in output_arrays is used.
    """

    # How can we get numpy arrays if numpy is not imported?
    if "numpy" not in sys.modules:
        raise RuntimeError("need numpy arrays.")
    np = sys.modules["numpy"]

    # Check input arrays
    if not isinstance(input_arrays, dict):  # empty is ok
        raise ValueError("input_arrays must be a dict.")
    for key, array in input_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of dict input_arrays must be int.")
        if not isinstance(array, np.ndarray):
            raise TypeError("input arrays must be numpy arrays.")

    # Check output arrays
    output_arrays2 = {}
    if not isinstance(output_arrays, dict) or not output_arrays:
        raise ValueError("output_arrays must be a nonempty dict.")
    for key, spec in output_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of dict output_arrays must be int.")
        if not (isinstance(spec, tuple) and len(spec) >= 2):
            raise TypeError("output arrays must be (*shape, dtype) tuples.")
        try:
            shape = tuple(int(i) for i in spec[:-1])
            output_arrays2[key] = shape + (np.dtype(spec[-1]),)
        except Exception as err:
            raise TypeError(f"output arrays must be (*shape, dtype) tuples: {str(err)}")

    # Get x, y, z from n
    if n is None:
        spec = list(output_arrays2.values())[0]
        x, y, z = spec[0], 1, 1
    elif isinstance(n, int):
        x, y, z = int(n), 1, 1
    elif isinstance(n, tuple) and len(n) == 3:
        x, y, z = int(n[0]), int(n[1]), int(n[2])
    else:
        raise TypeError("compute_with_buffers: n must be None, an int, or 3-int tuple.")
    if not (x >= 1 and y >= 1 and z >= 1):
        raise ValueError("compute_with_buffers: n value(s) must be >= 1.")

    # Create a device and compile the shader
    adapter = wgpu.requestAdapter(powerPreference="high-performance")
    device = adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())
    cshader = device.createShaderModule(code=shader)

    # Create buffers for input and output arrays
    buffers = {}
    for binding_index, array in input_arrays.items():
        # Create the buffer object
        usage = usage = wgpu.BufferUsage.STORAGE
        if binding_index in output_arrays2:
            usage |= wgpu.BufferUsage.MAP_READ
        buffer = device.createBufferMapped(size=array.nbytes, usage=usage)
        # Copy data from array to buffer
        mapped_array = np.frombuffer(buffer.mapping, array.dtype)
        mapped_array.shape = array.shape
        mapped_array[:] = array
        buffer.unmap()
        del mapped_array
        # Store
        buffers[binding_index] = buffer
    for binding_index, spec in output_arrays2.items():
        if binding_index in input_arrays:
            continue  # We already have this buffer
        usage = usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ
        nbytes = np.prod(spec[:-1]) * spec[-1].itemsize
        buffers[binding_index] = device.createBuffer(size=nbytes, usage=usage)

    # Create bindings and binding layouts
    bindings = []
    binding_layouts = []
    for binding_index, buffer in buffers.items():
        bindings.append(
            {
                "binding": binding_index,
                "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
            }
        )
        binding_layouts.append(
            {
                "binding": binding_index,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "type": wgpu.BindingType.storage_buffer,
            }
        )

    # Put buffers together
    bind_group_layout = device.createBindGroupLayout(bindings=binding_layouts)
    pipeline_layout = device.createPipelineLayout(bindGroupLayouts=[bind_group_layout])
    bind_group = device.createBindGroup(layout=bind_group_layout, bindings=bindings)

    # Create a pipeline and "run it"
    compute_pipeline = device.createComputePipeline(
        layout=pipeline_layout, computeStage={"module": cshader, "entryPoint": "main"},
    )
    command_encoder = device.createCommandEncoder()
    compute_pass = command_encoder.beginComputePass()
    compute_pass.setPipeline(compute_pipeline)
    compute_pass.setBindGroup(0, bind_group, [], 0, 999999)  # last 2 elements not used
    compute_pass.dispatch(x, y, z)
    compute_pass.endPass()
    device.defaultQueue.submit([command_encoder.finish()])

    # Read the current data of the output buffers
    output_arrays = {}
    for binding_index, spec in output_arrays2.items():
        shape, dtype = spec[:-1], spec[-1]
        buffer = buffers[binding_index]
        data = buffer.mapRead()  # slow, can also be done async
        output_array = np.frombuffer(data, dtype)
        output_array.shape = shape
        output_arrays[binding_index] = output_array

    return output_arrays
