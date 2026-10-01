"""Load the compiled Dawn bindings (the ``wgpu_dawn`` package).

The binding layer is a cffi module in API (out-of-line) mode, compiled from a
cdef that is generated from Dawn's webgpu.h. The exact same module source is
compiled for two targets:

* native: linked against Dawn (``libwebgpu_dawn``);
* Pyodide: linked with Emdawnwebgpu, Dawn's implementation of webgpu.h on top
  of the browser's WebGPU (``navigator.gpu``), so the GPU work is done by the
  browser, hardware accelerated.

Callbacks are not created at runtime (``ffi.callback`` needs runtime code
generation, which WebAssembly does not allow). Instead, the bindings define one
static entry point per callback type (cffi ``extern "Python"``); the userdata1
pointer carries an id that maps to the Python function to call.
"""

import sys
import logging
import threading

try:
    import wgpu_dawn
except ImportError as err:  # no-cover
    raise ImportError(
        "The dawn backend needs the compiled 'wgpu_dawn' package. "
        "See dawn_bindings/README.md in the wgpu-py repository."
    ) from err

ffi = wgpu_dawn.ffi
lib = wgpu_dawn.lib
lib_path = wgpu_dawn._wgpu_dawn.__file__
lib_version_info = tuple(int(x) for x in wgpu_dawn.dawn_version.lstrip("v").split("."))

IS_WEB = sys.platform == "emscripten"

logger = logging.getLogger("wgpu")

# In the browser, callbacks are called "spontaneously": directly from the JS
# promise resolution, i.e. from the JS event loop. Natively, they are called
# when we call wgpuInstanceProcessEvents(), so that they always run in a thread
# we control.
CALLBACK_MODE = (
    lib.WGPUCallbackMode_AllowSpontaneous
    if IS_WEB
    else lib.WGPUCallbackMode_AllowProcessEvents
)


class CallbackRegistry:
    """Map the userdata1 pointer of webgpu.h callbacks to Python callables."""

    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0
        self._callbacks = {}

    def register(self, func, *, once=True):
        """Register a Python callable, returns the userdata1 pointer to pass to C."""
        with self._lock:
            self._count += 1
            key = self._count
            self._callbacks[key] = (func, once)
        return ffi.cast("void *", key)

    def unregister(self, userdata):
        self._callbacks.pop(int(ffi.cast("uintptr_t", userdata)), None)

    def call(self, userdata, *args):
        key = int(ffi.cast("uintptr_t", userdata))
        try:
            func, once = self._callbacks[key]
        except KeyError:
            return  # e.g. after unregister
        if once:
            self._callbacks.pop(key, None)
        try:
            func(*args)
        except Exception as err:  # Don't let errors propagate into C
            logger.error(f"Error in wgpu callback {func}: {err}")


callbacks = CallbackRegistry()


def _make_trampoline(name):
    def trampoline(*args):
        # The signature of all webgpu.h callbacks ends with (userdata1, userdata2)
        callbacks.call(args[-2], *args)

    trampoline.__name__ = f"_wgpu_dawn_{name}"
    ffi.def_extern(name=f"_wgpu_dawn_{name}")(trampoline)


for _name in dir(lib):
    if _name.startswith("_wgpu_dawn_WGPU") and _name.endswith("Callback"):
        _make_trampoline(_name[len("_wgpu_dawn_") :])


def c_callback(callback_type):
    """Get the C function pointer that dispatches to registered Python functions."""
    return getattr(lib, f"_wgpu_dawn_{callback_type}")
