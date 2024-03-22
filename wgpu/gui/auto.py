"""
Automatic GUI backend selection.

Right now we only chose between GLFW, Qt and Jupyter. We might add support
for e.g. wx later. Or we might decide to stick with these three.
"""

__all__ = ["WgpuCanvas", "run", "call_later"]

import os
import sys
import importlib
from ._gui_utils import logger, QT_MODULE_NAMES, get_imported_qt_lib, asyncio_is_running


# Note that wx is not in here, because it does not (yet) implement base.WgpuAutoGui
WGPU_GUI_BACKEND_NAMES = ["glfw", "qt", "jupyter", "offscreen"]


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


def select_backend():
    """Select a backend using a careful multi-stage selection process."""

    module = None
    failed_backends = {}  # name -> error

    for backend_name, reason in backends_generator():
        if "force" in reason.lower():
            return _load_backend(backend_name)
        if backend_name in failed_backends:
            continue
        try:
            module = _load_backend(backend_name)
            break
        except Exception as err:
            failed_backends[backend_name] = str(err)

    # Always report failed backeds, because we only try them when it looks like we can.
    if failed_backends:
        msg = "WGPU could not load some backends:"
        for key, val in failed_backends.items():
            msg += f"\n{key}: {val}"
        logger.warning(msg)

    # Return or raise
    if module is not None:
        log = logger.warning if failed_backends else logger.info
        log(f"WGPU selected {backend_name} gui because {reason}.")
        return module
    else:
        msg = "WGPU Could not load any of the supported GUI backends."
        if "jupyter" in failed_backends:
            msg += "\n  You may need to ``pip install -U jupyter_rfb``."
        else:
            msg += "\n  Install glfw using e.g. ``pip install -U glfw``,"
            msg += (
                "\n  or install a qt framework using e.g. ``pip install -U pyside6``."
            )
        raise ImportError(msg) from None


def backends_generator():
    """Generator that iterates over all sub-generators."""
    for gen in [
        backends_by_env_vars,
        backends_by_jupyter,
        backends_by_imported_modules,
        backends_by_trying_in_order,
    ]:
        yield from gen()


def backends_by_env_vars():
    """Generate backend names set via one the supported environment variables."""
    # Env var intended for testing, overrules everything else
    if os.environ.get("WGPU_FORCE_OFFSCREEN", "").lower() in ("1", "true", "yes"):
        yield "offscreen", "WGPU_FORCE_OFFSCREEN is set"
    # Env var to force a backend for general use
    backend_name = os.getenv("WGPU_GUI_BACKEND", "").lower().strip() or None
    if backend_name:
        if backend_name not in WGPU_GUI_BACKEND_NAMES:
            logger.warning(
                f"Ignoring invalid WGPU_GUI_BACKEND '{backend_name}', must be one of {WGPU_GUI_BACKEND_NAMES}"
            )
            backend_name = None
    if backend_name:
        yield backend_name, "WGPU_GUI_BACKEND is set"


def backends_by_jupyter():
    """Generate backend names that are appropriate for the current Jupyter session (if any)."""
    try:
        ip = get_ipython()
    except NameError:
        return
    if not ip.has_trait("kernel"):
        # probably old-school ipython, we follow the normal selection process
        return

    # We're in a Jupyter kernel. This might be a notebook, jupyter lab, jupyter
    # console, qtconsole, etc. In the latter cases we cannot render ipywidgets.
    # Unfortunately, there does not seem to be a (reasonable) way to detect
    # whether we're in a console or notebook. Technically this kernel could be
    # connected to a client of each. So we assume that ipywidgets can be used.
    # User on jupyter console (or similar) should ``%gui qt`` or set
    # WGPU_GUI_BACKEND to 'glfw'.

    # If GUI integration is enabled, we select the corresponding backend instead of jupyter
    app = getattr(ip.kernel, "app", None)
    if app:
        gui_module_name = app.__class__.__module__.split(".")[0]
        if gui_module_name in QT_MODULE_NAMES:
            yield "qt", "running on Jupyter with qt gui"
        # elif "wx" in app.__class__.__name__.lower() == "wx":
        #     yield "wx", "running on Hupyter with wx gui"

    yield "jupyter", "running on Jupyter"


def backends_by_imported_modules():
    """Generate backend names based on what modules are currently imported."""

    # Get some info on loaded backends, and available apps/loops
    qtlib, has_qt_app = get_imported_qt_lib()
    has_asyncio_loop = asyncio_is_running()

    # If there is a qt app instance, chances are high that the user wants to run in Qt.
    # More so than with asyncio, because asyncio may just be used by the runtime.
    if has_qt_app:
        yield "qt", "Qt app is running"

    # If there is an asyncio loop, we can nicely run glfw, if glfw is available.
    if has_asyncio_loop:
        try:
            importlib.import_module("glfw")
        except ModuleNotFoundError:
            pass
        else:
            yield "glfw", "asyncio loop is running"

    # The rest is just "is the corresponding lib imported?"

    if "glfw" in sys.modules:
        yield "glfw", "glfw is imported"

    if qtlib:
        yield "qt", "qt is imported"

    # if "wx" in sys.modules:
    #     yield "wx", "wx is imported"


def backends_by_trying_in_order():
    """Generate backend names by trying to import the GUI lib in order. This is the final fallback."""

    gui_lib_to_backend = {
        "glfw": "glfw",
        "PySide6": "qt",
        "PyQt6": "qt",
        "PySide2": "qt",
        "PyQt5": "qt",
        # "wx": "wx",
    }

    for libname, backend_name in gui_lib_to_backend.items():
        try:
            importlib.import_module(libname)
        except ModuleNotFoundError:
            continue
        yield backend_name, f"{libname} can be imported"


# Load!
module = select_backend()
WgpuCanvas, run, call_later = module.WgpuCanvas, module.run, module.call_later
