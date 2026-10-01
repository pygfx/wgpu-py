# The default backend is wgpu-native, but this may change in the future.
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

    # If a backend has already been loaded (e.g. by importing wgpu.backends.dawn), use that
    current_gpu = sys.modules["wgpu"].gpu
    if type(current_gpu).__module__ != "wgpu._classes":
        return current_gpu

    # The WGPUPY_BACKEND env var can be used to select a backend, e.g. for testing
    backend_name = os.getenv("WGPUPY_BACKEND", "").strip().lower().replace("-", "_")
    if backend_name:
        return _load_backend(backend_name)

    if sys.platform == "emscripten":
        # In Pyodide, prefer the Dawn backend when its compiled bindings are installed
        if importlib.util.find_spec("wgpu_dawn") is not None:
            return _load_backend("dawn")
        return _load_backend("js_webgpu")
    else:
        return _load_backend("wgpu_native")


gpu = _auto_load_backend()
