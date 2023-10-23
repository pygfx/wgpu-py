Utils
=====

The wgpu library provides a few utilities. Note that most functions below need to be explictly imported.


Get default device
------------------

.. autofunction:: wgpu.utils.get_default_device


Compute with buffers
--------------------

.. code-block:: py

    from wgpu.utils.compute import compute_with_buffers

.. autofunction:: wgpu.utils.compute_with_buffers



Shadertoy
---------

.. code-block:: py

    from wgpu.utils.shadertoy import Shadertoy

.. autoclass:: wgpu.utils.shadertoy.Shadertoy
    :members:
