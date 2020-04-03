WGPU API
========


Adapter
-------

To start using the GPU, we must obtain a device object. But first, we request
an adapter, which represens a GPU implementation on the current system.

.. autofunction:: wgpu.request_adapter

.. autofunction:: wgpu.request_adapter_async

.. autoclass:: wgpu.base.GPUAdapter
    :members:


Device
------

The device is the root object for all other GPU objects.

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

Bind groups are used to tell the GPU what the data looks like.

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
