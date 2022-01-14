"""
Automatic GUI backend selection.

Right now we only chose between GLFW, Qt and Jupyter. We might add suport
for e.g. wx later. Or we might decide to stick with these three.
"""

__all__ = ["WgpuCanvas", "run", "call_later"]

import sys


def is_jupyter():
    """Determine whether the user is executing in a Jupyter Notebook / Lab."""
    try:
        ip = get_ipython()
        if ip.has_trait("kernel"):
            return True
        else:
            return False
    except NameError:
        return False


if is_jupyter():
    from .jupyter import WgpuCanvas, run, call_later  # noqa
else:
    try:
        from .glfw import WgpuCanvas, run, call_later  # noqa
    except ImportError as glfw_err:
        try:
            from .qt import WgpuCanvas, QtWidgets, QtCore

            # When using Qt, there needs to be an
            # application before any widget is created
            app = QtWidgets.QApplication([])

            def run():
                app.exec() if hasattr(app, "exec") else app.exec_()

            def call_later(delay, callback, *args):
                QtCore.QTimer.singleShot(delay * 1000, lambda: callback(*args))

        except ImportError as qt_err:
            msg = str(glfw_err)
            msg += "\n" + str(qt_err)
            msg += "\n\n  Could not find either glfw or Qt framework."
            msg += "\n  Install glfw using e.g. ``pip install -U glfw``,"
            msg += (
                "\n  or install a qt framework using e.g. ``pip install -U pyside6``."
            )
            if sys.platform.startswith("linux"):
                msg += "\n  You may also need to run the equivalent of ``apt install libglfw3``."
            raise ImportError(msg) from None
