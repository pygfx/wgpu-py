# The default backend is wgpu-native. The Dawn backend can be selected with the
# WGPUPY_BACKEND environment variable, and is the default in Pyodide when the
# compiled wgpu_dawn package is available.
import os
import sys
import importlib.util


def _load_backend(backend_name):
    """Load a wgpu backend by name."""

    if backend_name == "wgpu_native":
        from . import wgpu_native as module
    elif backend_name == "dawn":
        from . import dawn as module
    elif backend_name == "js_webgpu":
        from . import js_webgpu as module
    else:  # no-cover
        raise ImportError(f"Unknown wgpu backend: '{backend_name}'")

    return module.gpu


def _auto_load_backend():
    """Decide on the backend automatically."""

    backend_name = os.getenv("WGPUPY_BACKEND", "").strip().lower()
    if backend_name:
        return _load_backend(backend_name)
    elif sys.platform == "emscripten":
        if importlib.util.find_spec("wgpu_dawn") is not None:
            return _load_backend("dawn")
        return _load_backend("js_webgpu")
    else:
        return _load_backend("wgpu_native")


gpu = _auto_load_backend()
