GUI API
=======

You can use wgpu for compute tasks and to render offscreen. Rendering to
screen is also possible, but we need a *canvas* for that. Since the Python
ecosystem provides many different GUI toolkits, we need an interface.

For convenience, the wgpu library has builtin support for a few GUI
toolkits. At the moment these include GLFW, Qt, and wx (experimental).


The canvas interface
--------------------

To render to a window, an object is needed that implements the few
functions on the canvas interface, and provide that object to
:func:`request_adapter() <wgpu.request_adapter>`.
This is the minimal interface required to hook wgpu-py to any GUI that supports GPU rendering.

.. autoclass:: wgpu.gui.WgpuCanvasInterface
    :members:


The WgpuCanvas base class
-------------------------

For each supported GUI toolkit there are specific
``WgpuCanvas`` classes, which are detailed in the following sections.
These all derive from the same base class, which defines the common API.

.. autoclass:: wgpu.gui.WgpuCanvasBase
    :members:


Support for Qt
--------------

There is support fo PyQt4, PyQt5, PySide and PySide2. The wgpu library detects what
library you are using by looking what module has been imported.

.. code-block:: py

    # First import any of the Qt libraries
    from PyQt5 import QtWidgets

    # Then import the WgpuCanvas, which is a subclass of QWidget
    from wgpu.gui.qt import WgpuCanvas

    # Create a Qt app, as usual
    app = QtWidgets.QApplication([])

    # Instantiate the canvas
    canvas = WgpuCanvas(title="Example")

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    # Enter Qt's event loop, as usual
    app.exec_()

Also see the `Qt triangle example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_qt.py>`_.


Support for glfw
----------------

Glfw is a lightweight windowing toolkit. Install it with ``pip install glfw``.


.. code-block:: py

    # Import glfw itself
    glfw.init()

    # Then import the WgpuCanvas
    from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas

    # Initialize glfw, as usual
    import glfw

    # Instantiate the canvas
    canvas = WgpuCanvas(title="Example")

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    # Enter a main loop (this stops when all windows are closed)
    while update_glfw_canvasses():
        glfw.poll_events()


Also see the `GLFW triangle example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_glfw.py>`_
and the `async GLFW example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_glfw_asyncio.py>`_.


Offscreen canvases
------------------

A base class is provided to implement off-screen canvases. Note that you can
render to a texture without using any canvas object, but in some cases it's
convenient to do so with a canvas-like API.

.. autoclass:: wgpu.gui.WgpuOffscreenCanvas
    :members:


Support for Jupyter lab and notebook
------------------------------------

WGPU can be used in Jupyter lab and the Jupyter notebook. This canvas
is based on `jupyter_rfb <https://github.com/vispy/jupyter_rfb>`_ an ipywidget
subclass implementing a remote frame-buffer.

To implement interaction, create a subclass and overload the ``handle_event()``
method (and call ``super().handle_event(event)``).


.. code-block:: py

    from wgpu.gui.jupyter import WgpuCanvas

    canvas = WgpuCanvas()

    # ... wgpu code

    canvas  # Use as cell output
