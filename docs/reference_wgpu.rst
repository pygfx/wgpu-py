WGPU API
========

.. currentmodule:: wgpu

This document describes the wgpu API. It is basically a Pythonic version of the
`WebGPU API <https://gpuweb.github.io/gpuweb/>`_. It exposes an API
for performing operations, such as rendering and computation, on a
Graphics Processing Unit.

*The WebGPU API is still being developed and occasionally there are backwards
incompatible changes. Since we mostly follow the WebGPU API, there may be
backwards incompatible changes to wgpu-py too. This will be so until
the WebGPU API settles as a standard.*


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


Each backend may also implement minor differences (usually additions)
from the base API. For the ``rs`` backend check ``print(wgpu.backends.rs.apidiff.__doc__)``.


Alphabetic list of GPU classes
------------------------------

* :class:`GPU`
* :class:`GPUAdapterInfo`
* :class:`GPUAdapter`
* :class:`GPUBindGroup`
* :class:`GPUBindGroupLayout`
* :class:`GPUBindingCommandsMixin`
* :class:`GPUBuffer`
* :class:`GPUCanvasContext`
* :class:`GPUCommandBuffer`
* :class:`GPUCommandEncoder`
* :class:`GPUCommandsMixin`
* :class:`GPUCompilationInfo`
* :class:`GPUCompilationMessage`
* :class:`GPUComputePassEncoder`
* :class:`GPUComputePipeline`
* :class:`GPUDebugCommandsMixin`
* :class:`GPUDevice`
* :class:`GPUDeviceLostInfo`
* :class:`GPUError`
* :class:`GPUExternalTexture`
* :class:`GPUInternalError`
* :class:`GPUObjectBase`
* :class:`GPUOutOfMemoryError`
* :class:`GPUPipelineBase`
* :class:`GPUPipelineError`
* :class:`GPUPipelineLayout`
* :class:`GPUQuerySet`
* :class:`GPUQueue`
* :class:`GPURenderBundle`
* :class:`GPURenderBundleEncoder`
* :class:`GPURenderCommandsMixin`
* :class:`GPURenderPassEncoder`
* :class:`GPURenderPipeline`
* :class:`GPUSampler`
* :class:`GPUShaderModule`
* :class:`GPUTexture`
* :class:`GPUTextureView`
* :class:`GPUUncapturedErrorEvent`
* :class:`GPUValidationError`


Overview
--------


Adapter, device and canvas
++++++++++++++++++++++++++

The :class:`GPU` represents the root namespace that contains the entrypoint to request an adapter.

The :class:`GPUAdapter` represents a hardware or software device, with specific
features, limits and properties. To actually start using that harware for computations or rendering, a :class:`GPUDevice` object must be requisted from the adapter. This is a logical unit
to control your hardware (or software).
The device is the central object; most other GPU objects are created from it.
Also see the convenience function :func:`wgpu.utils.get_default_device`.
Information on the adapter can be obtained using TODO in the form of a :class:`GPUAdapterInfo`.

A device is controlled with a specific backend API. By default one is selected automatically.
This can be overridden by setting the
`WGPU_BACKEND_TYPE` environment variable to "Vulkan", "Metal", "D3D12", "D3D11", or "OpenGL".

The device and all objects created from it inherit from :class:`GPUObjectBase` - they represent something on the GPU.

In most render use-cases you want the result to be presented to a canvas on the screen.
The :class:`GPUCanvasContext` is the bridge between wgpu and the underlying GUI backend.

Buffers and textures
++++++++++++++++++++

A :class:`GPUBuffer` can be created from a device. It is used to hold data, that can
be uploaded using it's API. From the shader's point of view, the buffer can be accessed
as a typed array.

A :class:`GPUTexture` is similar to a buffer, but has some image-specific features.
A texture can be 1D, 2D or 3D, can have multiple levels of detail (i.e. lod or mipmaps).
The texture itself represents the raw data, you can create one or more :class:`GPUTextureView` objects
for it, that can be attached to a shader.

To let a shader sample from a texture, you also need a :class:`GPUSampler` that
defines the filtering and sampling behavior beyond the edges.

WebGPU also defines the :class:`GPUExternalTexture`, but this is not (yet?) used in wgpu-py.

Bind groups
+++++++++++

Shaders need access to resources like buffers, texture views, and samplers.
The access to these resources occurs via so called bindings. There are
integer slots, which must be specifie both via the API, and in the shader.

Bindings are organized into :class:`GPUBindGroup` s, which are essentially a list
of :class:`GPUBinding` s.

Further, in wgpu you need to specify a :class:`GPUBindGroupLayout`, providing
meta-information about the binding (type, texture dimension etc.).

Multiple bind groups layouts are collected in a :class:`GPUPipelineLayout`,
which represents a complete layout description for a pipeline.

Shaders and pipelines
+++++++++++++++++++++

The wgpu API knows three kinds of shaders: compute, vertex and fragment.
Pipelines define how the shader is run, and with what resources.

Shaders are represented by a :class:`GPUShaderModule`.

Compute shaders are combined with a pipelinelayout into a :class:`GPUComputePipeline`.
Similarly, a vertex and (optional) fragment shader are combined with a pipelinelayout
into a :class:`GPURenderPipeline`. Both of these inherit from :class:`GPUPipelineBase`.

Command buffers and encoders
++++++++++++++++++++++++++++

The actual rendering occurs by recording a series of commands and then submitting these commands.

The root object to generate commands with is the :class:`GPUCommandEncoder`.
This class inherits from :class:`GPUCommandsMixin` (because it generates commands),
and :class:`GPUDebugCommandsMixin` (because it supports debugging).

Commands specific to compute and rendering are generated with a :class:`GPUComputePassEncoder` and :class:`GPURenderPassEncoder` respectively. You get these from the command encoder by the
corresponding ``begin_x_pass()`` method. These pass encoders inherit from
:class:`GPUBindingCommandsMixin` (because you associate a pipeline)
and the latter also from :class:`GPURenderCommandsMixin`.

When you're done generating commands, you call ``finish()`` and get the list of
commands as an opaque object: the :class:`GPUCommandBuffer`. You don't really use this object
except for submitting it to the :class:`GPUQueue`.

The command buffers are one-time use. The :class:`GPURenderBundle` and :class:`GPURenderBundleEncoder` can
be used to record commands to be used multiple times, but this is not yet
implememted in wgpu-py.

Error handling
++++++++++++++

Errors are caught and logged using the ``wgpu`` logger.

Todo: document the role of these classes:
:class:`GPUUncapturedErrorEvent`
:class:`GPUError`
:class:`GPUValidationError`
:class:`GPUOutOfMemoryError`
:class:`GPUInternalError`
:class:`GPUPipelineError`
:class:`GPUDeviceLostInfo`

TODO
++++

These classes are not supported and/or documented yet.
:class:`GPUCompilationMessage`
:class:`GPUCompilationInfo`
:class:`GPUQuerySet`
