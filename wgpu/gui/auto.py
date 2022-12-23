"""
Automatic GUI backend selection.

Right now we only chose between GLFW, Qt and Jupyter. We might add support
for e.g. wx later. Or we might decide to stick with these three.
"""

__all__ = ["WgpuCanvas", "run", "call_later"]

import importlib
import os
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


if os.environ.get("WGPU_FORCE_OFFSCREEN") == "true":
    from .offscreen import WgpuCanvas, run, call_later  # noqa
elif is_jupyter():
    from .jupyter import WgpuCanvas, run, call_later  # noqa
else:
    try:
        from .glfw import WgpuCanvas, run, call_later  # noqa
    except ImportError as glfw_err:
        qt_backends = ("PySide6", "PyQt6", "PySide2", "PyQt5")
        for backend in qt_backends:
            if backend in sys.modules:
                break
        else:
            for libname in qt_backends:
                try:
                    importlib.import_module(libname)
                    break
                except ModuleNotFoundError:
                    pass
            else:
                msg = str(glfw_err)
                msg += "\n\n  Could not find either glfw or Qt framework."
                msg += "\n  Install glfw using e.g. ``pip install -U glfw``,"
                msg += "\n  or install a qt framework using e.g. ``pip install -U pyside6``."
                if sys.platform.startswith("linux"):
                    msg += "\n  You may also need to run the equivalent of ``apt install libglfw3``."
                raise ImportError(msg) from None

        from .qt import WgpuCanvas, run, call_later
