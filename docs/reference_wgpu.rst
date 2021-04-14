WGPU API
========

This document describes the wgpu API. It is basically a Pythonic version of the
`WebGPU API <https://gpuweb.github.io/gpuweb/>`_. It exposes an API
for performing operations, such as rendering and computation, on a
Graphics Processing Unit.

.. warning::
    The WebGPU API is still being developed and occasionally there are backwards
    incompatible changes. Since we mostly follow the WebGPU API, there may be
    backwards incompatible changes to wgpu-py too. This will be so until
    the WebGPU API settles as a standard.


How to read this API
--------------------

The classes in this API all have a name staring with "GPU", this helps
discern them from flags and enums. These classes are never instantiated
directly; new objects are returned by certain methods.

Most methods in this API have no positional arguments; each argument
must be referenced by name. Some argument values must be a dict, these
can be thought of as "nested" arguments.

Many arguments (and dict fields) must be a
:doc:`flags <reference_flags>` or :doc:`enums <reference_enums>`.
Flags are integer bitmasks that can be *orred* together. Enum values are
strings in this API.

Some arguments have a default value. Most do not.


Selecting the backend
---------------------

Before you can use this API, you have to select a backend. Eventually
there may be multiple backends, but at the moment
there is only one backend, which is based on the Rust libary
`wgpu-native <https://github.com/gfx-rs/wgpu>`_. You select
the backend by importing it:


.. code-block:: py

    import wgpu.backends.rs


The ``wgpu-py`` package comes with the ``wgpu-native`` library. If you want
to use your own version of that library instead, set the ``WGPU_LIB_PATH``
environment variable.


Differences from WebGPU
-----------------------

This API is derived from the WebGPU spec, but differs in a few ways.
For example, methods that in WebGPU accept a descriptor/struct/dict,
here accept the fields in that struct as keyword arguments.


.. autodata:: wgpu.base.apidiff
    :annotation: Differences of base API:


.. autodata:: wgpu.backends.rs.apidiff
    :annotation: Differences of rs backend:


Adapter
-------

To start using the GPU for computations or rendering, a device object
is required. One first requests an adapter, which represens a GPU
implementation on the current system. The device can then be requested
from the adapter.


.. autoclass:: wgpu.GPU

.. autofunction:: wgpu.request_adapter

.. autofunction:: wgpu.request_adapter_async

.. autoclass:: wgpu.GPUAdapter
    :members:


Device
------

The device is the central object; most other GPU objects are created from it.
It is recommended to request a device object once, or perhaps twice.
But not for every operation (e.g. in unit tests).
Also see :func:`wgpu.utils.get_default_device`.


.. autoclass:: wgpu.GPUObjectBase
    :members:


.. autoclass:: wgpu.GPUDevice
    :members:


Buffers and textures
--------------------

Buffers and textures are used to provide your shaders with data.

.. autoclass:: wgpu.GPUBuffer
    :members:

.. autoclass:: wgpu.GPUTexture
    :members:

.. autoclass:: wgpu.GPUTextureView
    :members:

.. autoclass:: wgpu.GPUSampler
    :members:


Bind groups
-----------

Shaders need access to resources like buffers, texture views, and samplers.
The access to these resources occurs via so called bindings. There are
integer slots, which you specify both via the API and in the shader, to
bind the resources to the shader.

Bindings are organized into bind groups, which are essentially a list
of bindings. E.g. in Python shaders the slot of each resource is specified
as a two-tuple (e.g. ``(1, 3)``) specifying the bind group and binding
slot respectively.

Further, in wgpu you need to specify a binding *layout*, providing
meta-information about the binding (type, texture dimension etc.).

One uses
:func:`device.create_bind_group() <wgpu.GPUDevice.create_bind_group>`
to create a group of bindings using the actual buffers/textures/samplers.

One uses
:func:`device.create_bind_group_layout() <wgpu.GPUDevice.create_bind_group_layout>`
to specify more information about these bindings, and
:func:`device.create_pipeline_layout() <wgpu.GPUDevice.create_pipeline_layout>`
to pack one or more bind group layouts together, into a complete layout
description for a pipeline.


.. autoclass:: wgpu.GPUBindGroupLayout
    :members:

.. autoclass:: wgpu.GPUBindGroup
    :members:

.. autoclass:: wgpu.GPUPipelineLayout
    :members:


Shaders and pipelines
---------------------

The wgpu API knows three kinds of shaders: compute, vertex and fragment.
Pipelines define how the shader is run, and with what resources.


.. autoclass:: wgpu.GPUShaderModule
    :members:

.. autoclass:: wgpu.GPUPipelineBase
    :members:

.. autoclass:: wgpu.GPUComputePipeline
    :members:

.. autoclass:: wgpu.GPURenderPipeline
    :members:


Command buffers and encoders
----------------------------

.. autoclass:: wgpu.GPUCommandBuffer
    :members:

.. autoclass:: wgpu.GPUCommandEncoder
    :members:

.. autoclass:: wgpu.GPUProgrammablePassEncoder
    :members:

.. autoclass:: wgpu.GPUComputePassEncoder
    :members:

.. autoclass:: wgpu.GPURenderEncoderBase
    :members:

.. autoclass:: wgpu.GPURenderPassEncoder
    :members:

.. autoclass:: wgpu.GPURenderBundle
    :members:

.. autoclass:: wgpu.GPURenderBundleEncoder
    :members:


Queue and swap chain
--------------------

.. autoclass:: wgpu.GPUQueue
    :members:

.. autoclass:: wgpu.GPUSwapChain
    :members:


Other
-----


.. autoclass:: wgpu.GPUCanvasContext
    :members:

.. autoclass:: wgpu.GPUQuerySet
    :members:

.. autoclass:: wgpu.GPUDeviceLostInfo
    :members:

.. autoclass:: wgpu.GPUOutOfMemoryError
    :members:

.. autoclass:: wgpu.GPUValidationError
    :members:

.. autoclass:: wgpu.GPUCompilationInfo
    :members:

.. autoclass:: wgpu.GPUCompilationMessage
    :members:

.. autoclass:: wgpu.GPUUncapturedErrorEvent
    :members:
