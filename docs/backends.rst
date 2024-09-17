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

.. py:function:: wgpu.backends.wgpu_native.request_device_tracing(adapter, trace_path, *, label="", required_features, required_limits, default_queue)

    An alternative to :func:`wgpu.GPUAdapter.request_adapter`, that streams a trace
    of all low level calls to disk, so the visualization can be replayed (also on other systems),
    investigated, and debugged.

    :param adapter: The adapter to create a device for.
    :param trace_path: The path to an (empty) directory. Is created if it does not exist.
    :param label: A human readable label. Optional.
    :param required_features: The features (extensions) that you need. Default [].
    :param required_limits: the various limits that you need. Default {}.
    :param default_queue: Descriptor for the default queue. Optional.
    :return: Device
    :rtype: wgpu.GPUDevice

The wgpu_native backend provides support for push constants.
Since WebGPU does not support this feature, documentation on its use is hard to find.
A full explanation of push constants and its use in Vulkan can be found
`here <https://vkguide.dev/docs/chapter-3/push_constants/>`_.
Using push constants in WGPU closely follows the Vulkan model.

The advantage of push constants is that they are typically faster to update than uniform buffers.
Modifications to push constants are included in the command encoder; updating a uniform
buffer involves sending a separate command to the GPU.
The disadvantage of push constants is that their size limit is much smaller. The limit
is guaranteed to be at least 128 bytes, and 256 bytes is typical.

Given an adapter, first determine if it supports push constants::

    >> "push-constants" in adapter.features
    True

If push constants are supported, determine the maximum number of bytes that can
be allocated for push constants::

    >> adapter.limits["max-push-constant-size"]
    256

You must tell the adapter to create a device that supports push constants,
and you must tell it the number of bytes of push constants that you are using.
Overestimating is okay::

    device = adapter.request_device(
        required_features=["push-constants"],
        required_limits={"max-push-constant-size": 256},
    )

Creating a push constant in your shader code is similar to the way you would create
a uniform buffer.
The fields that are only used in the ``@vertex`` shader should be separated from the fields
that are only used in the ``@fragment`` shader which should be separated from the fields
used in both shaders::

    struct PushConstants {
        // vertex shader
        vertex_transform: vec4x4f,
        // fragment shader
        fragment_transform: vec4x4f,
        // used in both
        generic_transform: vec4x4f,
    }
    var<push_constant> push_constants: PushConstants;

To the pipeline layout for this shader, use
``wgpu.backends.wpgu_native.create_pipeline_layout`` instead of
``device.create_pipelinelayout``.  It takes an additional argument,
``push_constant_layouts``, describing
the layout of the push constants.  For example, in the above example::

    push_constant_layouts = [
        {"visibility": ShaderState.VERTEX, "start": 0, "end": 64},
        {"visibility": ShaderStage.FRAGMENT, "start": 64, "end": 128},
        {"visibility": ShaderState.VERTEX + ShaderStage.FRAGMENT , "start": 128, "end": 192},
    ],

Finally, you set the value of the push constant by using
``wgpu.backends.wpgu_native.set_push_constants``::

    set_push_constants(this_pass, ShaderStage.VERTEX, 0, 64, <64 bytes>)
    set_push_constants(this_pass, ShaderStage.FRAGMENT, 64, 128, <64 bytes>)
    set_push_constants(this_pass, ShaderStage.VERTEX + ShaderStage.FRAGMENT, 128, 192, <64 bytes>)

Bytes must be set separately for each of the three shader stages.  If the push constant has
already been set, on the next use you only need to call ``set_push_constants`` on those
bytes you wish to change.

.. py:function:: wgpu.backends.wpgu_native.create_pipeline_layout(device, *, label="", bind_group_layouts, push_constant_layouts=[])

   This method provides the same functionality as :func:`wgpu.GPUDevice.create_pipeline_layout`,
   but provides an extra `push_constant_layouts` argument.
   When using push constants, this argument is a list of dictionaries, where each item
   in the dictionary has three fields: `visibility`, `start`, and `end`.

    :param device: The device on which we are creating the pipeline layout
    :param label: An optional label
    :param bind_group_layouts:
    :param push_constant_layouts: Described above.

.. py:function:: wgpu.backends.wgpu_native.set_push_constants(render_pass_encoder, visibility, offset, size_in_bytes, data, data_offset=0)

    This function requires that the underlying GPU implement `push_constants`.
    These push constants are a buffer of bytes available to the `fragment` and `vertex`
    shaders. They are similar to a bound buffer, but the buffer is set using this
    function call.

    :param render_pass_encoder: The render pass encoder to which we are pushing constants.
    :param visibility: The stages (vertex, fragment, or both) to which these constants are visible
    :param offset: The offset into the push constants at which the bytes are to be written
    :param size_in_bytes: The number of bytes to copy from the ata
    :param data: The data to copy to the buffer
    :param data_offset: The starting offset in the data at which to begin copying.


There are two functions that allow you to perform multiple draw calls at once.
Both require that you enable the feature "multi-draw-indirect".

Typically, these calls do not reduce work or increase parallelism on the GPU. Rather
they reduce driver overhead on the CPU.

.. py:function:: wgpu.backends.wgpu_native.multi_draw_indirect(render_pass_encoder, buffer, *, offset=0, count):

     Equivalent to::
        for i in range(count):
            render_pass_encoder.draw_indirect(buffer, offset + i * 16)

    :param render_pass_encoder: The current render pass encoder.
    :param buffer: The indirect buffer containing the arguments.
    :param offset: The byte offset in the indirect buffer containing the first argument.
    :param count: The number of draw operations to perform.

.. py:function:: wgpu.backends.wgpu_native.multi_draw_indexed_indirect(render_pass_encoder, buffer, *, offset=0, count):

     Equivalent to::
        for i in range(count):
            render_pass_encoder.draw_indexed_indirect(buffer, offset + i * 2-)


    :param render_pass_encoder: The current render pass encoder.
    :param buffer: The indirect buffer containing the arguments.
    :param offset: The byte offset in the indirect buffer containing the first argument.
    :param count: The number of draw operations to perform.


The js_webgpu backend
---------------------

.. code-block:: py

    import wgpu.backends.js_webgpu


This backend calls into the JavaScript WebGPU API. For this, the Python code would need
access to JavaScript - this backend is intended for use-cases like `PScript <https://github.com/flexxui/pscript>`__ `PyScript <https://github.com/pyscript/pyscript>`__, and `RustPython <https://github.com/RustPython/RustPython>`__.

This backend is still a stub, see `issue #407 <https://github.com/pygfx/wgpu-py/issues/407>`__ for details.
