gui API
=======

.. currentmodule:: wgpu.gui

You can use vanilla wgpu for compute tasks and to render offscreen. To
render to a window on screen we need a *canvas*. Since the Python
ecosystem provides many different GUI toolkits, wgpu implements a base
canvas class, and has builtin support for a few GUI toolkits. At the
moment these include GLFW, Jupyter, Qt, and wx.


The Canvas base classes
-----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated
    :template: wgpu_class_layout.rst

    ~WgpuCanvasInterface
    ~WgpuCanvasBase
    ~WgpuAutoGui
    ~WgpuOffscreenCanvasBase


For each supported GUI toolkit there is a module that implements a ``WgpuCanvas`` class,
which inherits from :class:`WgpuCanvasBase`, providing a common API.
The GLFW, Qt, and Jupyter backends also inherit from  :class:`WgpuAutoGui` to include
support for events (interactivity). In the next sections we demonstrates the different
canvas classes that you can use.


The auto GUI backend
--------------------

The default approach for examples and small applications is to use
the automatically selected GUI backend. At the moment this selects
either the GLFW, Qt, or Jupyter backend, depending on the environment.

To implement interaction, the ``canvas`` has a :func:`WgpuAutoGui.handle_event()` method
that can be overloaded. Alternatively you can use it's :func:`WgpuAutoGui.add_event_handler()`
method. See the `event spec <https://jupyter-rfb.readthedocs.io/en/stable/events.html>`_
for details about the event objects.


Also see the `triangle auto example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_auto.py>`_
and `cube example <https://github.com/pygfx/wgpu-py/blob/main/examples/cube.py>`_.

.. code-block:: py

    from wgpu.gui.auto import WgpuCanvas, run, call_later

    canvas = WgpuCanvas(title="Example")
    canvas.request_draw(your_draw_function)

    run()


Support for GLFW
----------------

`GLFW <https://github.com/FlorianRhiem/pyGLFW>`_ is a lightweight windowing toolkit.
Install it with ``pip install glfw``. The preferred approach is to use the auto backend,
but you can replace ``from wgpu.gui.auto`` with ``from wgpu.gui.glfw`` to force using GLFW.

.. code-block:: py

    from wgpu.gui.glfw import WgpuCanvas, run, call_later

    canvas = WgpuCanvas(title="Example")
    canvas.request_draw(your_draw_function)

    run()


Support for Qt
--------------

There is support for PyQt5, PyQt6, PySide2 and PySide6. The wgpu library detects what
library you are using by looking what module has been imported.
For a toplevel widget, the ``gui.qt.WgpuCanvas`` class can be imported. If you want to
embed the canvas as a subwidget, use ``gui.qt.WgpuWidget`` instead.

Also see the `Qt triangle example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_qt.py>`_
and `Qt triangle embed example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_qt_embed.py>`_.

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


Support for wx
--------------

There is support for embedding a wgpu visualization in wxPython.
For a toplevel widget, the ``gui.wx.WgpuCanvas`` class can be imported. If you want to
embed the canvas as a subwidget, use ``gui.wx.WgpuWidget`` instead.

Also see the `wx triangle example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_wx.py>`_
and `wx triangle embed example <https://github.com/pygfx/wgpu-py/blob/main/examples/triangle_wx_embed.py>`_.

.. code-block:: py

    import wx
    from wgpu.gui.wx import WgpuCanvas

    app = wx.App()

    # Instantiate the canvas
    canvas = WgpuCanvas(title="Example")

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    app.MainLoop()



Support for offscreen
---------------------

You can also use a "fake" canvas to draw offscreen and get the result as a numpy array.
Note that you can render to a texture without using any canvas
object, but in some cases it's convenient to do so with a canvas-like API.

.. code-block:: py

    from wgpu.gui.offscreen import WgpuCanvas

    # Instantiate the canvas
    canvas = WgpuCanvas(size=(500, 400), pixel_ratio=1)

    # ...

    # Tell the canvas what drawing function to call
    canvas.request_draw(your_draw_function)

    # Perform a draw
    array = canvas.draw()  # numpy array with shape (400, 500, 4)


Support for Jupyter lab and notebook
------------------------------------

WGPU can be used in Jupyter lab and the Jupyter notebook. This canvas
is based on `jupyter_rfb <https://github.com/vispy/jupyter_rfb>`_, an ipywidget
subclass implementing a remote frame-buffer. There are also some `wgpu examples <https://jupyter-rfb.readthedocs.io/en/stable/examples/>`_.

.. code-block:: py

    # from wgpu.gui.jupyter import WgpuCanvas  # Direct approach
    from wgpu.gui.auto import WgpuCanvas  # Approach compatible with desktop usage

    canvas = WgpuCanvas()

    # ... wgpu code

    canvas  # Use as cell output
