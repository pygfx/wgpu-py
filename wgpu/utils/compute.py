"""
Simple high-level utilities for doing compute on the GPU.
"""

import ctypes

import wgpu.utils


def compute_with_buffers(input_arrays, output_arrays, shader, n=None):
    """Apply the given compute shader to the given input_arrays and return
    output arrays. Both input and output arrays are represented on the GPU
    using storage buffer objects.

    Parameters:
        input_arrays (dict): A dict mapping int bindings to arrays. The array
            can be anything that supports the buffer protocol, including
            bytes, memoryviews, ctypes arrays and numpy arrays. The
            type and shape of the array does not need to match the type
            with which the shader will interpret the buffer data (though
            it probably makes your code easier to follow).
        output_arrays (dict): A dict mapping int bindings to output shapes.
            If the value is int, it represents the size (in bytes) of
            the buffer. If the value is a tuple, its last element
            specifies the format (see below), and the preceding elements
            specify the shape. These are used to ``cast()`` the
            memoryview object before it is returned. If the value is a
            ctypes array type, the result will be cast to that instead
            of a memoryview. Note that any buffer that is NOT in the
            output arrays dict will be considered readonly in the shader.
        shader (str or bytes): The shader as a string of WGSL code or SpirV bytes.
        n (int, tuple, optional): The dispatch counts. Can be an int
            or a 3-tuple of ints to specify (x, y, z). If not given or None,
            the length of the first output array type is used.

    Returns:
        output (dict): A dict mapping int bindings to memoryviews.

    The format characters to cast a ``memoryview`` are hard to remember, so
    here's a refresher:

    * "b" and "B" are signed and unsiged 8-bit ints.
    * "h" and "H" are signed and unsiged 16-bit ints.
    * "i" and "I" are signed and unsiged 32-bit ints.
    * "e" and "f" are 16-bit and 32-bit floats.
    """

    # Check input arrays
    if not isinstance(input_arrays, dict):  # empty is ok
        raise TypeError("input_arrays must be a dict.")
    for key, array in input_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of input_arrays must be int.")
        # Simply wrapping in a memoryview ensures that it supports the buffer protocol
        memoryview(array)

    # Check output arrays
    output_infos = {}
    if not isinstance(output_arrays, dict) or not output_arrays:
        raise TypeError("output_arrays must be a nonempty dict.")
    for key, array_descr in output_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of output_arrays must be int.")
        if isinstance(array_descr, str) and "x" in array_descr:
            array_descr = tuple(array_descr.split("x"))
        if isinstance(array_descr, int):
            output_infos[key] = {
                "length": array_descr,
                "nbytes": array_descr,
                "format": "B",
                "shape": (array_descr,),
            }
        elif isinstance(array_descr, tuple):
            format = array_descr[-1]
            try:
                format_size = FORMAT_SIZES[format]
            except KeyError:
                raise ValueError(f"Invalid format for output array {key}: {format}")
            shape = tuple(int(i) for i in array_descr[:-1])
            if not (shape and all(i > 0 for i in shape)):
                raise ValueError(f"Invalid shape for output array {key}: {shape}")
            nbytes = format_size
            for i in shape:
                nbytes *= i
            output_infos[key] = {
                "length": shape[0],
                "nbytes": nbytes,
                "format": format,
                "shape": shape,
            }
        elif isinstance(array_descr, type) and issubclass(array_descr, ctypes.Array):
            output_infos[key] = {
                "length": array_descr._length_,
                "nbytes": ctypes.sizeof(array_descr),
                "ctypes_array_type": array_descr,
            }
        else:
            raise TypeError(
                f"Invalid value for output array description: {array_descr}"
            )

    # Get nx, ny, nz from n
    if n is None:
        output_info = list(output_infos.values())[0]
        nx, ny, nz = output_info["length"], 1, 1
    elif isinstance(n, int):
        nx, ny, nz = int(n), 1, 1
    elif isinstance(n, tuple) and len(n) == 3:
        nx, ny, nz = int(n[0]), int(n[1]), int(n[2])
    else:
        raise TypeError("compute_with_buffers: n must be None, an int, or 3-int tuple.")
    if not (nx >= 1 and ny >= 1 and nz >= 1):
        raise ValueError("compute_with_buffers: n value(s) must be >= 1.")

    # Create a device and compile the shader
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader)

    # Create buffers for input and output arrays
    buffers = {}
    for index, array in input_arrays.items():
        usage = wgpu.BufferUsage.STORAGE
        if index in output_arrays:
            usage |= wgpu.BufferUsage.COPY_SRC
        buffer = device.create_buffer_with_data(data=array, usage=usage)
        buffers[index] = buffer
    for index, info in output_infos.items():
        if index in input_arrays:
            continue  # We already have this buffer
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        buffers[index] = device.create_buffer(size=info["nbytes"], usage=usage)

    # Create bindings and binding layouts
    bindings = []
    binding_layouts = []
    for index, buffer in buffers.items():
        bindings.append(
            {
                "binding": index,
                "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
            }
        )
        storage_types = (
            wgpu.BufferBindingType.read_only_storage,
            wgpu.BufferBindingType.storage,
        )
        binding_layouts.append(
            {
                "binding": index,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": storage_types[index in output_infos],
                    "has_dynamic_offset": False,
                },
            }
        )

    # Put buffers together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create a pipeline and "run it"
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 args not used
    compute_pass.dispatch_workgroups(nx, ny, nz)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    # Read the current data of the output buffers
    output = {}
    for index, info in output_infos.items():
        buffer = buffers[index]
        # m = buffer.read_data()  # old API
        m = device.queue.read_buffer(buffer)  # slow, can also be done async
        if "ctypes_array_type" in info:
            output[index] = info["ctypes_array_type"].from_buffer(m)
        else:
            output[index] = m.cast(info["format"], shape=info["shape"])

    return output


FORMAT_SIZES = {"b": 1, "B": 1, "h": 2, "H": 2, "i": 4, "I": 4, "e": 2, "f": 4}

# It's tempting to allow for other formats, like "int32" and "f4", but
# users who like numpy will simply specify the number of bytes and
# convert the result. Users who will work with the memoryview directly
# should not be confused with other formats than memoryview.cast()
# normally supports.
