# The auto/default/only backend is wgpu-native, but this may change in the future.
import sys


def _load_backend(backend_name):
    """Load a wgpu backend by name."""

    if backend_name == "wgpu_native":
        from . import wgpu_native as module  # noqa: F401,F403
    elif backend_name == "js_webgpu":
        from . import js_webgpu as module  # noqa: F401,F403
    else:  # no-cover
        raise ImportError(f"Unknown wgpu backend: '{backend_name}'")

    return module.gpu


def _auto_load_backend():
    """Decide on the backend automatically."""

    if sys.platform == "emscripten":
        return _load_backend("js_webgpu")
    else:
        return _load_backend("wgpu_native")


gpu = _auto_load_backend()
