Guide
=====


This library (``wgpu``) presents a Pythonic API for the `WebGPU spec
<https://gpuweb.github.io/gpuweb/>`_. It is an API to control graphics
hardware. Like OpenGL but modern. Or like Vulkan but higher level.
GPU programming is a craft that requires knowledge of how GPU's work.


Getting started
---------------

Creating a canvas
+++++++++++++++++

If you want to render to the screen, you need a canvas. Multiple
GUI toolkits are supported, see the :doc:`gui`. In general, it's easiest to let ``wgpu`` select a GUI automatically:

.. code-block:: py

    from wgpu.gui.auto import WgpuCanvas, run

    canvas = WgpuCanvas(title="a wgpu example")


Next, we can setup the render context, which we will need later on.

.. code-block:: py

    present_context = canvas.get_context()
    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)


Obtaining a device
++++++++++++++++++

The next step is to obtain an adapter, which represents an abstract render device.
You can pass it the ``canvas`` that you just created, or pass ``None`` for the canvas
if you have none (e.g. for compute or offscreen rendering). From the adapter,
you can obtain a device.

.. code-block:: py

    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device = adapter.request_device()

The ``wgpu.gpu`` object is the API entrypoint (:class:`wgpu.GPU`). It contains just a handful of functions,
including ``request_adapter()``. The device is used to create most other GPU objects.


Creating buffers, textures shaders, etc.
++++++++++++++++++++++++++++++++++++++++

Using the device, you can create buffers, textures, write shader code, and put
these together into pipeline objects. How to do this depends a lot on what you
want to achieve, and is therefore out of scope for this guide. Have a look at the examples
or some of the tutorials that we link to below.

Setting up a draw function
++++++++++++++++++++++++++

Let's now define a function that will actually draw the stuff we put together in
the previous step.

.. code-block:: py

    def draw_frame():

        # We'll record commands that we do on a render pass object
        command_encoder = device.create_command_encoder()
        current_texture_view = present_context.get_current_texture()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
                    "resolve_target": None,
                    "clear_value": (1, 1, 1, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        # Perform commands, something like ...
        render_pass.set_pipeline(...)
        render_pass.set_index_buffer(...)
        render_pass.set_vertex_buffer(...)
        render_pass.set_bind_group(...)
        render_pass.draw_indexed(...)

        # When done, submit the commands to the device queue.
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

        # If you want to draw continuously, request a new draw right now
        canvas.request_draw()


Starting the event loop
+++++++++++++++++++++++


We can now pass the above render function to the canvas. The canvas will then
call the function whenever it (re)draws the window. And finally, we call ``run()`` to enter the mainloop.

.. code-block:: py

    canvas.request_draw(draw_frame)
    run()


Offscreen
+++++++++

If you render offscreen, or only do compute, you do not need a canvas. You also won't need a GUI toolkit, draw function or enter the event loop.
Instead, you will obtain a command encoder and submit it's records to the queue directly.


Examples and external resources
-------------------------------

Examples that show wgpu-py in action:

* https://github.com/pygfx/wgpu-py/tree/main/examples

.. note:: The examples in the main branch of the repository may not match the pip installable version.  Be sure to refer to the examples from the git tag that matches the version of wgpu you have installed.


External resources:

* https://webgpu.rocks/
* https://sotrh.github.io/learn-wgpu/
* https://rust-tutorials.github.io/learn-wgpu/


A brief history of WebGPU
-------------------------

For years, OpenGL has been the only cross-platform API to talk to the GPU.
But over time OpenGL has grown into an inconsistent and complex API ...

    *OpenGL is dying*
    --- Dzmitry Malyshau at `Fosdem 2020 <https://fosdem.org/2020/schedule/event/rust_webgpu/>`_

In recent years, modern API's have emerged that solve many of OpenGL's
problems. You may have heard of Vulkan, Metal, and DX12. These
API's are much closer to the hardware, which makes the drivers more
consistent and reliable. Unfortunately, the huge amount of "knobs to
turn" also makes them quite hard to work with for developers.

Therefore, higher level API are needed, which use the same concepts, but are much easier to work with.
The most notable one is the `WebGPU specification <https://gpuweb.github.io/gpuweb/>`_. This is what future devs
will be using to write GPU code for the browser. And for desktop and mobile as well.

As the WebGPU spec is being developed, a reference implementation is
also build. It's written in Rust and powers the WebGPU implementation in Firefox.
This reference implementation, called `wgpu <https://github.com/gfx-rs/wgpu>`__,
also exposes a C-api (via `wgpu-native <https://github.com/gfx-rs/wgpu-native>`__),
so that it can be wrapped in Python. And this is precisely what wgpu-py does.

So in short, wgpu-py is a Python wrapper of wgpu, which is an desktop
implementation of WebGPU, an API that wraps  Vulkan, Metal and DX12,
which talk to the GPU hardware.



Coordinate system
-----------------

In wgpu, the Y-axis is up in normalized device coordinate (NDC): point(-1.0, -1.0)
in NDC is located at the bottom-left corner of NDC. In addition, x and
y in NDC should be between -1.0 and 1.0 inclusive, while z in NDC should
be between 0.0 and 1.0 inclusive. Vertices out of this range in NDC
will not introduce any errors, but they will be clipped.


Array data
----------

The wgpu library makes no assumptions about how you store your data.
In places where you provide data to the API, it can consume any data
that supports the buffer protocol, which includes ``bytes``,
``bytearray``, ``memoryview``, ctypes arrays, and numpy arrays.

In places where data is returned, the API returns a ``memoryview``
object. These objects provide a quite versatile view on ndarray data:

.. code-block:: py

    # One could, for instance read the content of a buffer
    m = device.queue.read_buffer(buffer)
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


Freezing apps
-------------

In wgpu a PyInstaller-hook is provided to help simplify the freezing process
(it e.g. ensures that the wgpu-native DLL is included). This hook requires
PyInstaller version 4+.

Our hook also includes ``glfw`` when it is available, so code using ``wgpu.gui.auto``
should Just Work.

Note that PyInstaller needs ``wgpu`` to be installed in `site-packages` for
the hook to work (i.e. it seems not to work with a ``pip -e .`` dev install).
