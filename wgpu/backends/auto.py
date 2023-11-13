# The auto/default/only backend is wgpu-native, but this may change in the future.


def _load_backend(backend_name):
    """Load a wgpu backend by name."""
    if backend_name == "wgpu_native":
        from . import wgpu_native as module  # noqa: F401,F403
    else:  # no-cover
        raise ImportError("Unknown wgpu backend: '{backend_name}'")

    globals()["GPU"] = module.GPU


_load_backend("wgpu_native")  # we make the import dynamic from the start
