GUI API
=======

You can use wgpu for compute tasks and to render offscreen. Rendering to
screen is also possible, but we need a *canvas* for that. Since the Python
ecosystem provides many different GUI toolkits, we need an interface.

For convenience, the wgpu library has builtin support for a few GUI
toolkits. At the moment these include GLFW, Jupyter, Qt, and wx.


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



Base offscreen class
--------------------

A base class is provided to implement off-screen canvases for different
purposes.

.. autoclass:: wgpu.gui.WgpuOffscreenCanvas
    :members:


The auto GUI backend
--------------------

The default approach for examples and small applications is to use
the automatically selected GUI backend.

.. code-block:: py

    from wgpu.gui.auto import WgpuCanvas, run, call_later

    canvas = WgpuCanvas(title="Example")
    canvas.request_draw(your_draw_function)

    run()

At the moment this selects either the GLFW or Jupyter backend, depending
on the enviornment. The ``WgpuCanvas`` has a ``handle_event()`` method
that can be overloaded (by subclassing ``WgpuCanvas``) to process user events.
See the `event spec <https://jupyter-rfb.readthedocs.io/en/latest/events.html>`_.


Support for Qt
--------------

There is support for PyQt5, PyQt6, PySide2 and PySide6. The wgpu library detects what
library you are using by looking what module has been imported.

.. code-block:: py

    # Import any of the Qt libraries before importing the WgpuCanvas.
    # This way wgpu knows which Qt library to use.
    from PySide6 import QtWidgets
    from wgpu.gui.qt import WgpuCanvas

    app = QtWidgets.QApplication([])

    # Instantiate the canvas
    canvas = WgpuCanvas(title="Example")

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    app.exec_()

For a toplevel widget, the ``WgpuCanvas`` class can be imported. If you want to
embed the canvas as a subwidget, use ``WgpuWidget`` instead.

Also see the `Qt triangle example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_qt.py>`_
and `Qt triangle embed example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_qt_embed.py>`_.


Support for wx
--------------

There is support for embedding a wgpu visualization in wxPython.

.. code-block:: py

    import wx
    from wgpu.gui.wx import WgpuCanvas

    app = wx.App()

    # Instantiate the canvas
    canvas = WgpuCanvas(title="Example")

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    app.MainLoop()

For a toplevel widget, the ``WgpuCanvas`` class can be imported. If you want to
embed the canvas as a subwidget, use ``WgpuWidget`` instead.


Also see the `wx triangle example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_wx.py>`_
and `wx triangle embed example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_wx_embed.py>`_.


Support for offscreen
---------------------

You can also use a "fake" canvas to draw offscreen and get the result as a numpy array.
Note that you can render to a texture without using any canvas
object, but in some cases it's convenient to do so with a canvas-like API.

.. code-block:: py

    from wgpu.gui.offscreen import WgpuCanvas

    # Instantiate the canvas
    canvas = WgpuCanvas(640, 480)

    # ...

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    # Perform a draw
    array = canvas.draw()


Support for GLFW
----------------

GLFW is a lightweight windowing toolkit. Install it with ``pip install glfw``.
The preferred approach is to use the auto backend, but you can replace ``from wgpu.gui.auto``
with ``from wgpu.gui.glfw`` to force using GLFW.

To implement interaction, create a subclass and overload the ``handle_event()``
method (and call ``super().handle_event(event)``).
See the `event spec <https://jupyter-rfb.readthedocs.io/en/latest/events.html>`_.


Support for Jupyter lab and notebook
------------------------------------

WGPU can be used in Jupyter lab and the Jupyter notebook. This canvas
is based on `jupyter_rfb <https://github.com/vispy/jupyter_rfb>`_ an ipywidget
subclass implementing a remote frame-buffer. There are also some `wgpu examples <https://jupyter-rfb.readthedocs.io/en/latest/examples/>`_.

To implement interaction, create a subclass and overload the ``handle_event()``
method (and call ``super().handle_event(event)``).
See the `event spec <https://jupyter-rfb.readthedocs.io/en/latest/events.html>`_.

.. code-block:: py

    # from wgpu.gui.jupyter import WgpuCanvas  # Direct approach
    from wgpu.gui.auto import WgpuCanvas  # Approach compatible with desktop usage

    canvas = WgpuCanvas()

    # ... wgpu code

    canvas  # Use as cell output
