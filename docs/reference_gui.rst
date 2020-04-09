GUI API
=======

The wgpu library provides GUI integration for a few GUI toolkits. At the moment
these include GLFW and Qt .


The WgpuCanvas class
------------------------

For each supported toolkit, there is a ``WgpuCanvas`` class, which all
derive from the following class.

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

    # Tell the canvas what function to draw to update itself.
    # Alternatively you can create a subclass and implement draw_frame().
    canvas.draw_frame = your_draw_function

    # Enter Qt's event loop, as usual
    app.exec_()

Also see the `Qt triangle example <https://github.com/almarklein/wgpu-py/blob/master/examples/triangle_qt.py>`_.


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

    # Tell the canvas what function to draw to update itself.
    # Alternatively you can create a subclass and implement draw_frame().
    canvas.draw_frame = your_draw_function

    # Enter a main loop (this stops when all windows are closed)
    while update_glfw_canvasses():
        glfw.poll_events()


Also see the `GLFW triangle example <https://github.com/almarklein/wgpu-py/blob/master/examples/triangle_glfw.py>`_
and the `async GLFW example <https://github.com/almarklein/wgpu-py/blob/master/examples/triangle_glfw_asyncio.py>`_.
