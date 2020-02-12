"""
Simple high-level utilities for doing compute on the GPU.
"""

import ctypes

import wgpu


def compute_with_buffers(input_arrays, output_arrays, shader, n=None):
    """ Apply the given compute shader to the given input_arrays and return
    output arrays. Both input and output arrays are represented on the GPU
    using storage buffer objects.

    Params:
        input_arrays (dict): A dict mapping int bindings to ctypes arrays.
            The type of the array does not need to match the type with which
            the shader will interpret the buffer data (though it probably
            makes the code easier to follow).
        output_arrays (dict): A dict mapping int bindings to ctypes
            array types. This function uses the given type to determine
            the buffer size (in bytes), and returns arrays of matching
            type. Example: ``ctypes.c_float * 20``. If you don't care
            about the type (e.g. because you re-cast it later), you can
            just specify the buffer size using ``ctypes.c_ubyte * nbytes``.
        shader (bytes, shader-object): The SpirV representing the shader,
            as raw bytes or an object implementing ``to_spirv()``
            (e.g. a python_shader SpirV module).
        n (int, tuple, optional): The dispatch counts. Can be an int
            or a 3-tuple of ints to specify (x, y, z). If not given or None,
            the length of the first output array type is used.

    Returns:
        output (dict): A dict mapping int bindings to ctypes arrays. The
            keys match those of ``output_arrays``, and the arrays are instances
            of the corresponding array types.
    """

    # Check input arrays
    if not isinstance(input_arrays, dict):  # empty is ok
        raise ValueError("input_arrays must be a dict.")
    for key, array in input_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of input_arrays must be int.")
        if not isinstance(array, ctypes.Array):
            raise TypeError("values of input_arrays must be ctypes arrays.")

    # Check output arrays
    if not isinstance(output_arrays, dict) or not output_arrays:
        raise ValueError("output_arrays must be a nonempty dict.")
    for key, array_type in output_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of output_arrays must be int.")
        if not (isinstance(array_type, type) and issubclass(array_type, ctypes.Array)):
            raise TypeError("values of output_arrays must be ctypes array subclasses.")

    # Get x, y, z from n
    if n is None:
        array_type = list(output_arrays.values())[0]
        x, y, z = array_type._length_, 1, 1
    elif isinstance(n, int):
        x, y, z = int(n), 1, 1
    elif isinstance(n, tuple) and len(n) == 3:
        x, y, z = int(n[0]), int(n[1]), int(n[2])
    else:
        raise TypeError("compute_with_buffers: n must be None, an int, or 3-int tuple.")
    if not (x >= 1 and y >= 1 and z >= 1):
        raise ValueError("compute_with_buffers: n value(s) must be >= 1.")

    # Create a device and compile the shader
    adapter = wgpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device(extensions=[], limits={})
    cshader = device.create_shader_module(code=shader)

    # Create buffers for input and output arrays
    buffers = {}
    for binding_index, array in input_arrays.items():
        # Create the buffer object
        nbytes = ctypes.sizeof(array)
        usage = wgpu.BufferUsage.STORAGE
        if binding_index in output_arrays:
            usage |= wgpu.BufferUsage.MAP_READ
        buffer = device.create_buffer_mapped(size=nbytes, usage=usage)
        # Copy data from array to buffer
        ctypes.memmove(buffer.mapping, array, nbytes)
        buffer.unmap()
        # Store
        buffers[binding_index] = buffer
    for binding_index, array_type in output_arrays.items():
        if binding_index in input_arrays:
            continue  # We already have this buffer
        nbytes = ctypes.sizeof(array_type)
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.MAP_READ
        buffers[binding_index] = device.create_buffer(size=nbytes, usage=usage)

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
    bind_group_layout = device.create_bind_group_layout(bindings=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, bindings=bindings)

    # Create a pipeline and "run it"
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute_stage={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(
        0, bind_group, [], 0, 999999
    )  # last 2 elements not used
    compute_pass.dispatch(x, y, z)
    compute_pass.end_pass()
    device.default_queue.submit([command_encoder.finish()])

    # Read the current data of the output buffers
    output = {}
    for binding_index, array_type in output_arrays.items():
        buffer = buffers[binding_index]
        array_uint8 = buffer.map_read()  # slow, can also be done async
        output[binding_index] = array_type.from_buffer(array_uint8)

    return output
