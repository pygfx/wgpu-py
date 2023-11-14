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


def _load_backend(backend_name):
    """Load a gui backend by name."""
    if backend_name == "glfw":
        from . import glfw as module  # noqa
    elif backend_name == "qt":
        from . import qt as module  # noqa
    elif backend_name == "jupyter":
        from . import jupyter as module  # noqa
    elif backend_name == "wx":
        from . import wx as module  # noqa
    elif backend_name == "offscreen":
        from . import offscreen as module  # noqa
    else:  # no-cover
        raise ImportError("Unknown wgpu gui backend: '{backend_name}'")
    return module


def _auto_load_backend():
    """Decide on the gui backend automatically."""

    # Backends to auto load, ordered by preference. Maps libname -> backend_name
    gui_backends = {
        "glfw": "glfw",
        "PySide6": "qt",
        "PyQt6": "qt",
        "PySide2": "qt",
        "PyQt5": "qt",
    }

    # The module that we try to find
    module = None

    # Any errors we come accross as we try to import the gui backends
    errors = []

    # Prefer a backend for which the lib is already imported
    imported = [libname for libname in gui_backends if libname in sys.modules]
    for libname in imported:
        try:
            module = _load_backend(gui_backends[libname])
            break
        except Exception as err:
            errors.append(err)

    # If no module found yet, try importing the lib, then import the backend
    if not module:
        for libname in gui_backends:
            try:
                importlib.import_module(libname)
            except ModuleNotFoundError:
                continue
            try:
                module = _load_backend(gui_backends[libname])
                break
            except Exception as err:
                errors.append(err)

    # If still nothing found, raise a useful error
    if not module:
        msg = "\n".join(str(err) for err in errors)
        msg += "\n\n  Could not find either glfw or Qt framework."
        msg += "\n  Install glfw using e.g. ``pip install -U glfw``,"
        msg += "\n  or install a qt framework using e.g. ``pip install -U pyside6``."
        if sys.platform.startswith("linux"):
            msg += "\n  You may also need to run the equivalent of ``apt install libglfw3``."
        raise ImportError(msg) from None

    return module


# Triage
if os.environ.get("WGPU_FORCE_OFFSCREEN") == "true":
    module = _load_backend("offscreen")
elif is_jupyter():
    module = _load_backend("jupyter")
else:
    module = _auto_load_backend()


WgpuCanvas, run, call_later = module.WgpuCanvas, module.run, module.call_later
