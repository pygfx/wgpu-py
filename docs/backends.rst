The wgpu backends
=================

What do backends do?
--------------------

The heavy lifting (i.e communication with the hardware) in wgpu is performed by
one of its backends.

Backends can be selected explicitly by importing them:

.. code-block:: py

    import wgpu.backends.wgpu_natve

There is also an `auto` backend to help keep code portable:

.. code-block:: py

    import wgpu.backends.auto

In most cases, however, you don't need any of the above imports, because
a backend is automatically selected in the first call to :func:`wgpu.GPU.request_adapter`.

Each backend can also provide additional (backend-specific)
functionality. To keep the main API clean and portable, this extra
functionality is provided as a functional API that has to be imported
from the specific backend.


The wgpu_native backend
-----------------------

.. code-block:: py

    import wgpu.backends.wgpu_natve


This backend wraps `wgpu-native <https://github.com/gfx-rs/wgpu-native>`__,
which is a C-api for `wgpu <https://github.com/gfx-rs/wgpu>`__, a Rust library
that wraps Vulkan, Metal, DirectX12 and more.
This is the main backend for wgpu-core. The only working backend, right now, to be precise.
It also works out of the box, because the wgpu-native DLL is shipped with wgpu-py.

The wgpu_native backend provides a few extra functionalities:

.. py:function:: wgpu.backends.wgpu_native.request_device_sync(adapter, trace_path, *, label="", required_features, required_limits, default_queue)

    An alternative to :func:`wgpu.GPUAdapter.request_adapter`, that streams a trace
    of all low level calls to disk, so the visualization can be replayed (also on other systems),
    investigated, and debugged.

    The trace_path argument is ignored on drivers that do not support tracing.

    :param adapter: The adapter to create a device for.
    :param trace_path: The path to an (empty) directory. Is created if it does not exist.
    :param label: A human readable label. Optional.
    :param required_features: The features (extensions) that you need. Default [].
    :param required_limits: the various limits that you need. Default {}.
    :param default_queue: Descriptor for the default queue. Optional.
    :return: Device
    :rtype: wgpu.GPUDevice

The wgpu_native backend provides support for immediates.
Immediates are not yet part of the WebGPU spec, but the headers for native webgpu have converged officially.

Immediates offer a way to set send a small amount of data to the GPU in the command encoder directly, no need for uniform buffer uploads.
They are restricted to rather small sizes, usually 128 or 265 bytes.

Given an adapter, first determine if it supports immediates::

    >> "immediates" in adapter.features
    True

If immediates are supported, determine the maximum number of bytes that can
be allocated for immediates::

    >> adapter.limits["max-immediate-size"]
    256

You must tell the adapter to create a device that supports immediates,
and you must tell it the number of bytes of immediates that you are using.
Overestimating is okay::

    device = adapter.request_device_sync(
        required_features=["immediates"],
        required_limits={"max-immediate-size": 256},
    )

Creating a immediate data struct in your shader code is similar to the way you would create
a uniform buffer.
The same data can be accessed across all shader stages: vertex, fragment and compute::

    struct Immediates {
        vertex_transform: vec4x4f,
        fragment_color: vec4f,
        pick_position: vec2f,
        frame_counter: u32,
    }
    var<immediate> immediate_data: Immediates;

To the pipeline layout for this shader, use
``wgpu.backends.wpgu_native.create_pipeline_layout`` instead of
``device.create_pipeline_layout``.  It takes an additional argument,
``immediate_size`` simply the number of bytes of immediate data you are using.

Finally, you set the value of the immediates by using
``wgpu.backends.wpgu_native.set_immediates``::

    set_immediates(pass_encoder, offset=0, size_in_bytes=64, data=<64 bytes>, data_offset=0)

.. py:function:: wgpu.backends.wpgu_native.create_pipeline_layout(device, *, label="", bind_group_layouts, immediate_size=0)

   This method provides the same functionality as :func:`wgpu.GPUDevice.create_pipeline_layout`,
   but provides an extra `immediate_size` argument.
   When using immediates, this argument is the number of bytes of immediate data you are using.

    :param device: The device on which we are creating the pipeline layout
    :param label: An optional label
    :param bind_group_layouts: 
    :param immediate_size: number of bytes for immediates data.

.. py:function:: wgpu.backends.wgpu_native.set_immediates(render_pass_encoder,offset, size_in_bytes, data, data_offset=0)

    This function requires that the underlying GPU implement `immediates`.
    These immediates are a buffer of bytes available to all shader stages.

    :param render_pass_encoder: The render pass encoder to which we are providing immediates.
    :param offset: The offset into the immediate data at which the bytes are to be written
    :param size_in_bytes: The number of bytes to copy from the data
    :param data: The data to copy to the buffer
    :param data_offset: The starting offset in the data at which to begin copying.


There are four functions that allow you to perform multiple draw calls at once.
Two take the number of draws to perform as an argument; two have this value in a buffer.

Typically, these calls do not reduce work or increase parallelism on the GPU. Rather
they reduce driver overhead on the CPU.

The first two require that you enable the feature ``"multi-draw-indirect"``.

.. py:function:: wgpu.backends.wgpu_native.multi_draw_indirect(render_pass_encoder, buffer, *, offset=0, count)

    Equivalent to::
        for i in range(count):
            render_pass_encoder.draw_indirect(buffer, offset + i * 16)

    :param render_pass_encoder: The current render pass encoder.
    :param buffer: The indirect buffer containing the arguments. Must have length
                   at least offset + 16 * count.
    :param offset: The byte offset in the indirect buffer containing the first argument.
                   Must be a multiple of 4.
    :param count: The number of draw operations to perform.

.. py:function:: wgpu.backends.wgpu_native.multi_draw_indexed_indirect(render_pass_encoder, buffer, *, offset=0, count)

    Equivalent to::

        for i in range(count):
            render_pass_encoder.draw_indexed_indirect(buffer, offset + i * 2-)


    :param render_pass_encoder: The current render pass encoder.
    :param buffer: The indirect buffer containing the arguments. Must have length
                   at least offset + 20 * count.
    :param offset: The byte offset in the indirect buffer containing the first argument.
                   Must be a multiple of 4.
    :param count: The number of draw operations to perform.

The second two require that you enable the feature ``"multi-draw-indirect-count"``.
They are identical to the previous two, except that the ``count`` argument is replaced by
three arguments. The value at ``count_buffer_offset`` in ``count_buffer`` is treated as
an unsigned 32-bit integer. The ``count`` is the minimum of this value and ``max_count``.

.. py:function:: wgpu.backends.wgpu_native.multi_draw_indirect_count(render_pass_encoder, buffer, *, offset=0, count_buffer, count_offset=0, max_count)

    Equivalent to::

        count = min(<u32 at count_buffer_offset in count_buffer>, max_count)
        for i in range(count):
            render_pass_encoder.draw_indirect(buffer, offset + i * 16)

    :param render_pass_encoder: The current render pass encoder.
    :param buffer: The indirect buffer containing the arguments. Must have length
                   at least offset + 16 * max_count.
    :param offset: The byte offset in the indirect buffer containing the first argument.
                   Must be a multiple of 4.
    :param count_buffer: The indirect buffer containing the count.
    :param count_buffer_offset: The offset into count_buffer.
                   Must be a multiple of 4.
    :param max_count: The maximum number of draw operations to perform.

.. py:function:: wgpu.backends.wgpu_native.multi_draw_indexed_indirect_count(render_pass_encoder, buffer, *, offset=0, count_buffer, count_offset=0, max_count)

    Equivalent to::

        count = min(<u32 at count_buffer_offset in count_buffer>, max_count)
        for i in range(count):
            render_pass_encoder.draw_indexed_indirect(buffer, offset + i * 2-)

    :param render_pass_encoder: The current render pass encoder.
    :param buffer: The indirect buffer containing the arguments. Must have length
                   at least offset + 20 * max_count.
    :param offset: The byte offset in the indirect buffer containing the first argument.
                   Must be a multiple of 4.
    :param count_buffer: The indirect buffer containing the count.
    :param count_buffer_offset: The offset into count_buffer.
                   Must be a multiple of 4.
    :param max_count: The maximum number of draw operations to perform.

Some GPUS allow you to collect timestamps other than via the ``timestamp_writes=`` argument
to ``command_encoder.begin_compute_pass`` and ``command_encoder.begin_render_pass``.

When ``write_timestamp`` is called with a command encoder as its first argument, a
timestamp is written to the indicated query set at the indicated index when all previous
command recorded into the same command encoder have been executed. This usage requires
that the features ``"timestamp-query"`` and ``"timestamp-query-inside-encoders"`` are
both enabled.

When ``write_timestamp`` is called with a render pass or compute pass as its first
argument, a timestamp is written to the indicated query set at the indicated index at
that point in this queue. This usage requires
that the features ``"timestamp-query"`` and ``"timestamp-query-inside-passes"`` are
both enabled.

.. py:function:: wgpu.backends.wgpu_native.write_timestamp(encoder, query_set, query_index)

    Writes a timestamp to the timestamp query set and the indicated index.

    :param encoder: The ComputePassEncoder, RenderPassEncoder, or CommandEncoder.
    :param query_set: The timestamp query set into which to save the result.
    :param index: The index of the query set into which to write the result.


Some GPUs allow you collect statistics on their pipelines. Those GPUs that support this
have the feature "pipeline-statistics-query", and you must enable this feature when
getting the device.
You create a query set using the function
``wgpu.backends.wgpu_native.create_statistics_query_set``.

The possible statistics are:

*    ``PipelineStatisticName.VertexShaderInvocations`` = "vertex-shader-invocations"
      * The number of times the vertex shader is called.
*    ``PipelineStatisticName.ClipperInvocations`` = "clipper-invocations"
      * The number of triangles generated by the vertex shader.
*    ``PipelineStatisticName.ClipperPrimitivesOut`` = "clipper-primitives-out"
      * The number of primitives output by the clipper.
*    ``PipelineStatisticName.FragmentShaderInvocations`` = "fragment-shader-invocations"
      * The number of times the fragment shader is called.
*    ``PipelineStatisticName.ComputeShaderInvocations`` = "compute-shader-invocations"
      * The number of times the compute shader is called.

The statistics argument is a list or a tuple of statistics names.  Each element of the
sequence must either be:

*    The enumeration, e.g. ``PipelineStatisticName.FragmentShaderInvocations``
*    A camel case string, e.g. ``"VertexShaderInvocations"``
*    A snake-case string, e.g. ``"vertex-shader-invocations"``
*    An underscored string, e.g.  ``"vertex_shader_invocations"``

You may use any number of these statistics in a query set. Each result is an 8-byte
unsigned integer, and the total size of each entry in the query set is 8 times
the number of statistics chosen.

The statistics are always output to the query set in the order above, even if they are
given in a different order in the list.

.. py:function:: wgpu.backends.wgpu_native.create_statistics_query_set(device, count, statistics)

    Create a query set that could hold count entries for the specified statistics.
    The statistics are specified as a list of strings.

    :param device: The device.
    :param count: Number of entries that go into the query set.
    :param statistics: A sequence of strings giving the desired statistics.

.. py:function:: wgpu.backends.wgpu_native.begin_pipeline_statistics_query(encoder, query_set, index)

    Start collecting statistics.

    :param encoder: The ComputePassEncoder or RenderPassEncoder.
    :param query_set: The query set into which to save the result.
    :param index: The index of the query set into which to write the result.

.. py:function:: wgpu.backends.wgpu_native.end_pipeline_statistics_query(encoder, query_set, index)

    Stop collecting statistics and write them into the query set.

    :param encoder: The ComputePassEncoder or RenderPassEncoder.

.. py:function:: wgpu.backends.wgpu_native.set_instance_extras(backends, flags, dx12_compiler, gles3_minor_version, fence_behavior, dxc_path, dxc_max_shader_model, budget_for_device_creation, budget_for_device_loss)

    Sets the global instance with extras. Needs to be called before instance is created (in enumerate_adapters or request_adapter).
    Most of these options are for specific backends, and might not create an instance or crash when used in the wrong combinations. 

    :param backends: bitflags as list[str], which backends to enable on the instance level. Defaults to ``["All"]``. Can be any combination of ``["Vulkan", "GL", "Metal", "DX12", "BrowserWebGPU"]`` or the premade combinations ``["All", "Primary", "secondary"]``. Note that your device needs to support these backends, for detailed information see https://docs.rs/wgpu/latest/wgpu/struct.Backends.html
    :param flags: bitflags as list[str], debug flags for the compiler. Defaults to ``["Default"]``, can be any combination of ``["Debug", "Validation", "DiscardHalLabels"]``.
    :param dx12_compiler: enum/str, either "Fxc", "Dxc" or "Undefined". Defaults to "Fxc" same as "Undefined". Dxc requires additional library files.
    :param gles3_minor_version: enum/int 0, 1 or 2. Defaults to "Atomic" (handled by driver).
    :param fence_behavior: enum/int, "Normal" or "AutoFinish", Default to "Normal".
    :param dxc_path: str, path to dxcompiler.dll, defaults to ``None``. None looks in the resource directory.
    :param dxc_max_shader_model: float between 6.0 and 6.7, Maximum shader model the given dll supports. Defaults to 6.5.
    :param budget_for_device_creation: Optional[int], between 0 and 100, to specify memory budget threshold for when creating resources (buffer, textures...) will fail. Defaults to None.
    :param budget_for_device_loss: Optional[int], between 0 and 100, to specify memory budget threshold when the device will be lost. Defaults to None.

Use like the following before the instance is created, which happens during request_adapter or enumerate_adapters.

.. code-block:: py

    import wgpu
    from wgpu.backends.wgpu_native.extras import set_instance_extras
    set_instance_extras(
        backends=["Vulkan"],
        flags=["Debug"],
    )

    # ...

    for a in wgpu.gpu.enumerate_adapters_sync():
        print(a.summary)

For additional usage examples look at `extras_dxc.py` and `extras_debug.py` in the examples directory.
Limited documentation on instance extras can be found in `wgpu.h`.

The js_webgpu backend
---------------------

.. code-block:: py

    import wgpu.backends.js_webgpu


This backend calls into the JavaScript WebGPU API. For this, the Python code would need
access to JavaScript - this backend is intended for use-cases like `PScript <https://github.com/flexxui/pscript>`__ `PyScript <https://github.com/pyscript/pyscript>`__, and `RustPython <https://github.com/RustPython/RustPython>`__.

This backend is still a stub, see `issue #407 <https://github.com/pygfx/wgpu-py/issues/407>`__ for details.
