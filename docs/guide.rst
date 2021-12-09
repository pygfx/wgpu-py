-----
Guide
-----

Not a lot here yet. More will come over time.


A brief history of WGPU
-----------------------

For years, OpenGL has been the only cross-platform API to talk to the GPU.
But over time OpenGL has grown into an inconsistent and complex API ...

    *OpenGL is dying*
    --- Dzmitry Malyshau at `Fosdem 2020 <https://fosdem.org/2020/schedule/event/rust_webgpu/>`_

In recent years, modern API's have emerged that solve many of OpenGL's problems.
You may have heard of them: Vulkan, Metal, and DX12. These API's are
much closer to the hardware. Unfortunately, the huge amount of "knobs
to turn" makes them quite hard to work with for developers.

Therefore, people are working on a higher level API, that wraps Vulkan/Metal/DX12,
and uses the same principals, but is much easier to work with. This is the
`WebGPU spec <https://gpuweb.github.io/gpuweb/>`_. This is what future devs
will be using to write GPU code for the browser. And for desktop and mobile.

As WebGPU spec is being developed, a reference implementation is also
being build. It's written in Rust, and is powering the WebGPU implementation in Firefox.
This reference implementation, called `wgpu-native <https://github.com/gfx-rs/wgpu>`_,
also exposes a C-api, which means that it can be wrapped in Python. And this is what
wgpu-py does.

So in short, wgpu-py is a Python wrapper of wgpu-native, which is a wrapper
for Vulkan, Metal and DX12, which are low-level API's to talk to the GPU hardware.


Getting started with WGPU
-------------------------

For now, we'll direct you to some related tutorials:

* https://sotrh.github.io/learn-wgpu/
* https://rust-tutorials.github.io/learn-wgpu/


Coordinate system
-----------------

The Y-axis is up in normalized device coordinate (NDC): point(-1.0, -1.0)
in NDC is located at the bottom-left corner of NDC. In addition, x and
y in NDC should be between -1.0 and 1.0 inclusive, while z in NDC should
be between 0.0 and 1.0 inclusive. Vertices out of this range in NDC
will not introduce any errors, but they will be clipped.


Communicating array data
------------------------

The wgpu-py library makes no assumptions about how you store your data.
In places where you provide data to the API, it can consume any data
that supports the buffer protocol, which includes ``bytes``,
``bytearray``, ``memoryview``, ctypes arrays, and numpy arrays.

In places where data is returned, the API returns a ``memoryview``
object. These objects provide a quite versatile view on ndarray data:

.. code-block:: py

    # One could, for instance read the content of a buffer
    m = buffer.read_data()
    # Cast it to float32
    m = m.cast("f")
    # Index it
    m[0]
    # Show the content
    print(m.tolist())

Chances are that you prefer Numpy. Converting the ``memoryview`` to a
numpy array (without copying the data) is easy:

.. code-block:: py

    array = np.frombuffer(m, np.float32)


Debugging
---------

If the default wgpu-backend causes issues, or if you want to run on a
different backend for another reason, you can set the
`WGPU_BACKEND_TYPE` environment variable to "Vulkan", "Metal", "D3D12",
"D3D11", or "OpenGL".

The log messages produced (by Rust) in wgpu-native are captured and
injected into Python's "wgpu" logger. One can set the log level to
"INFO" or even "DEBUG" to get detailed logging information.

Many GPU objects can be given a string label. This label will be used
in Rust validation errors, and are also used in e.g. RenderDoc to
identify objects. Additionally, you can insert debug markers at the
render/compute pass object, which will then show up in RenderDoc.

Eventually, wgpu-native will fully validate API input. Until then, it
may be worthwhile to enable the Vulkan validation layers. To do so, run
a debug build of wgpu-native and make sure that the Lunar Vulkan SDK
is installed.

You can run your application via RenderDoc, which is able to capture a
frame, including all API calls, objects and the complete pipeline state,
and display all of that information within a nice UI.

You can use ``adapter.request_device_tracing()`` to provide a directory path
where a trace of all API calls will be written. This trace can then be used
to re-play your use-case elsewhere (it's cross-platform).

Also see wgpu-core's section on debugging:
https://github.com/gfx-rs/wgpu/wiki/Debugging-wgpu-Applications


Freezing apps with wgpu
-----------------------

Wgpu implements a hook for PyInstaller to help simplify the freezing process
(it e.g. ensures that the wgpu-native DLL is included). This hook requires
PyInstaller version 4+.


Examples
--------

Some examples with wgpu-py can be found here:

* https://github.com/pygfx/wgpu-py/tree/main/examples
