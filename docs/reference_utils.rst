Utilities
=========

The wgpu library provides a few utilities. Note that the functions below need to be explictly imported.


.. autofunction:: wgpu.utils.get_default_device

.. autofunction:: wgpu.utils.compute_with_buffers


Shadertoy
--------------------

The Shadertoy class provides a "screen pixel shader programming interface" similar to `shadertoy <https://www.shadertoy.com/>`_,
Helps you research and quickly build or test shaders using WGSL via WGPU.


.. autoclass:: wgpu.utils.shadertoy.Shadertoy
    :members:
