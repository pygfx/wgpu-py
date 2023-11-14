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


.. py:function:: wgpu.backends.wgpu_native.enumerate_adapters()

    Return a list of all available adapters on this system.

    :return: Adapters
    :rtype: list


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


The js_webgpu backend
---------------------

.. code-block:: py

    import wgpu.backends.js_webgpu


This backend calls into the JavaScript WebGPU API. For this, the Python code would need
access to JavaScript - this backend is intended for use-cases like `PScript <https://github.com/flexxui/pscript>`__ `PyScript <https://github.com/pyscript/pyscript>`__, and `RustPyhon <https://github.com/RustPython/RustPython>`__.

This backend is still a stub, see `issue #407 <https://github.com/pygfx/wgpu-py/issues/407>`__ for details.
