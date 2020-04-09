WGPU API
========

This document describes the wgpu API. It is basically a Pythonic version of the
`WebGPU API <https://gpuweb.github.io/gpuweb/>`_. It exposes an API
for performing operations, such as rendering and computation, on a
Graphics Processing Unit.

.. note::
    The WebGPU API is still being developed and occasionally there are backwards
    incompatible changes. Since we mostly follow the WebGPU API, there may be
    backwards incompatible changes to wgpu-py too. This will be so until
    the WebGPU settles as a standard.


How to read this API
--------------------

The classes in this API all start with "GPU", this helps discern them
from flags and enums. These classes are never instantiated directly;
new objects are returned by certain methods.

Most methods in this API have no positional arguments; each argument
must be referenced by name. Some argument values must be a dict, these
can be thought of as "nested" arguments.

Many arguments (and dict fields) must be a
:doc:`flags <reference_flags>` (integer bitmasks that can be *orred* together)
or an :doc:`enums <reference_enums>` (strings). Some arguments have a
default value. Most do not.


Selecting the backend
---------------------

Before you can use this API, you have to select a backend. Eventually
there may be multiple backends, but at the moment
there is only one backend, which is based on the Rust libary
`wgpu-native <https://github.com/gfx-rs/wgpu>`_. You select
the backend by importing it:


.. code-block:: py

    import wgpu.backends.rs


Adapter
-------

To start using the GPU for computations or rendering, we must obtain a
device object. But first, we request an adapter, which represens a GPU
implementation on the current system.

.. autofunction:: wgpu.request_adapter

.. autofunction:: wgpu.request_adapter_async

.. autoclass:: wgpu.base.GPUAdapter
    :members:


Device
------

The device is the central object; most other GPU objects are created
from it.

.. autoclass:: wgpu.base.GPUObject
    :members:


.. autoclass:: wgpu.base.GPUDevice
    :members:


Buffers and textures
--------------------

Buffers and textures are used to provide your shaders with data.

.. autoclass:: wgpu.base.GPUBuffer
    :members:

.. autoclass:: wgpu.base.GPUTexture
    :members:

.. autoclass:: wgpu.base.GPUTextureView
    :members:

.. autoclass:: wgpu.base.GPUSampler
    :members:


Bind groups
-----------

Shaders need access to resources like buffers, texture views, and samplers.
The access to these resources occurs via so called bindings. There are
integer slots, which you specify both via the API and in the shader, to
bind the resources to the shader.

Bindings are organized into bind groups, which essentially form a list
of bindings. E.g. in Python shaders the slot of each resource is specified
as a two-tuple (e.g. ``(1, 3)``) specifying the bind group and binding
slot respectively.

Further, in wgpu you need to specify a binding *layout*, providing
meta-information about the binding (type, texture dimension etc.).

One uses ``device.create_bind_group()`` to create a group of bindings
using the actual buffers/textures/samplers.

One uses ``device.create_bind_group_layout()`` to specify more information
about these bindings, and ``device.create_pipeline_layout()`` to pack
one or more bind group layouts together, into a complete layout description
for a pipeline.


.. autoclass:: wgpu.base.GPUBindGroupLayout
    :members:

.. autoclass:: wgpu.base.GPUBindGroup
    :members:

.. autoclass:: wgpu.base.GPUPipelineLayout
    :members:


Shaders and pipelines
---------------------

.. autoclass:: wgpu.base.GPUShaderModule
    :members:

.. autoclass:: wgpu.base.GPUComputePipeline
    :members:

.. autoclass:: wgpu.base.GPURenderPipeline
    :members:


Command buffers and encoders
----------------------------

.. autoclass:: wgpu.base.GPUCommandBuffer
    :members:

.. autoclass:: wgpu.base.GPUCommandEncoder
    :members:

.. autoclass:: wgpu.base.GPUProgrammablePassEncoder
    :members:

.. autoclass:: wgpu.base.GPUComputePassEncoder
    :members:

.. autoclass:: wgpu.base.GPURenderEncoderBase
    :members:

.. autoclass:: wgpu.base.GPURenderPassEncoder
    :members:

.. autoclass:: wgpu.base.GPURenderBundle
    :members:

.. autoclass:: wgpu.base.GPURenderBundleEncoder
    :members:


Swap chain
----------

.. autoclass:: wgpu.base.GPUSwapChain
