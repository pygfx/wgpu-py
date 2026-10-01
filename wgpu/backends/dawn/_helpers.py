"""Utilities used in the Dawn backend."""

import sys
import time
import types
import threading
import ctypes
import inspect

from ._ffi import ffi, lib, lib_path, IS_WEB
from ..._async import get_backoff_time_generator
from ...classes import (
    GPUError,
    GPUInternalError,
    GPUOutOfMemoryError,
    GPUPipelineError,
    GPUValidationError,
)

ERROR_TYPES = {
    "": GPUError,
    "OutOfMemory": GPUOutOfMemoryError,
    "Validation": GPUValidationError,
    "Pipeline": GPUPipelineError,
    "Internal": GPUInternalError,
}

if sys.platform.startswith("darwin"):
    from rubicon.objc.api import ObjCInstance, ObjCClass


def get_memoryview_and_address(data):
    """Get a memoryview and memory-address for the given data.
    The data object must support the buffer protocol, and be contiguous.
    """

    # To get the address from a memoryview, there are multiple options.
    # The most obvious is using ctypes:
    #
    #   c_array = (ctypes.c_uint8 * nbytes).from_buffer(m)
    #   address = ctypes.addressof(c_array)
    #
    # Unfortunately, this call fails if the memoryview is readonly, e.g. if
    # the data is a bytes object or readonly numpy array. One could then
    # use from_buffer_copy(), but that introduces an extra data copy, which
    # can hurt performance when the data is large.
    #
    # Another alternative that can be used for objects implementing the array
    # interface (like numpy arrays) is to directly read the address:
    #
    #   address = data.__array_interface__["data"][0]
    #
    # But what seems to work best (at the moment) is using cffi.

    # Convert data to a memoryview. That way we have something consistent
    # to work with, which supports all objects implementing the buffer protocol.
    m = memoryview(data)

    # Test that the data is contiguous.
    # In most cases we'd want c_contiguous data, but the user may be
    # playing fancy tricks so we check for general contiguous-ness only.
    # Note that pypy does not have the contiguous attribute, so we assume it is.
    if not getattr(m, "contiguous", True):
        raise ValueError("The given data is not contiguous")

    # Get the address via ffi. In contrast to ctypes, this also
    # works for readonly data (e.g. bytes)
    c_data = ffi.from_buffer("uint8_t []", m)
    address = int(ffi.cast("uintptr_t", c_data))

    return m, address


def get_memoryview_from_address(address, nbytes, format="B"):
    """Get a memoryview from an int memory address and a byte count,"""
    buf = ffi.buffer(ffi.cast("uint8_t *", address), nbytes)
    return memoryview(buf).cast(format, shape=(nbytes,))


_the_instance = None


def get_wgpu_instance(extras=None):
    """Get the global wgpu instance."""
    # Note, we could also use wgpuInstanceRelease,
    # but we keep a global instance, so we don't have to.
    global _the_instance
    if _the_instance is not None and extras is not None:
        # For cases where it might be required, like testing to request new extras for the instance,
        # it is possible to delete the existing instance using `del _the_instance` or `lib.wgpuInstanceRelease(_the_instance)`.
        # However this means existing adapters/devices will most likely not work with new surfaces.
        raise RuntimeError(
            "Instance already exists. Please call `set_instance_extras` before the instance is created (calls to `request_adapter` or `enumerate_adapters`)"
        )

    if _the_instance is None:
        # H: nextInChain: WGPUChainedStruct *
        struct = ffi.new("WGPUInstanceDescriptor *")
        if extras is not None:
            c_instance_next_in_chain = ffi.cast("WGPUChainedStruct *", extras)
            struct.nextInChain = c_instance_next_in_chain
        if not IS_WEB:
            # Allow SPIR-V shaders natively (not possible in the browser)
            # and use wgpuInstanceWaitAny() with a timeout for sync waits.
            features = ffi.new(
                "WGPUInstanceFeatureName[]",
                [
                    lib.WGPUInstanceFeatureName_ShaderSourceSPIRV,
                    lib.WGPUInstanceFeatureName_TimedWaitAny,
                ],
            )
            struct.requiredFeatureCount = len(features)
            struct.requiredFeatures = features
        _the_instance = lib.wgpuCreateInstance(struct)
    return _the_instance


_canvas_count = 0
_keepalive = []


def get_surface_id_from_info(present_info):
    """Get an id representing the surface to render to. The way to
    obtain this id differs per platform and GUI toolkit.
    """

    if IS_WEB:
        # Emdawnwebgpu: the surface is identified by a CSS selector for the <canvas>
        canvas = present_info["window"]
        if not canvas.id:
            global _canvas_count
            _canvas_count += 1
            canvas.id = f"wgpu-dawn-canvas-{_canvas_count}"
        struct = ffi.new("WGPUEmscriptenSurfaceSourceCanvasHTMLSelector *")
        selector = ffi.new("char[]", f"#{canvas.id}".encode())
        struct.selector.data = selector
        struct.selector.length = len(selector) - 1
        struct.chain.sType = lib.WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector
        _keepalive.append(selector)

    elif sys.platform.startswith("win"):  # no-cover
        GetModuleHandle = ctypes.windll.kernel32.GetModuleHandleW  # noqa: N806
        struct = ffi.new("WGPUSurfaceSourceWindowsHWND *")
        struct.hinstance = ffi.cast("void *", GetModuleHandle(lib_path))
        struct.hwnd = ffi.cast("void *", int(present_info["window"]))
        struct.chain.sType = lib.WGPUSType_SurfaceSourceWindowsHWND

    elif sys.platform.startswith("darwin"):  # no-cover
        # This is what the triangle example from wgpu-native does:
        # if WGPU_TARGET == WGPU_TARGET_MACOS
        # {
        #     id metal_layer = NULL;
        #     NSWindow *ns_window = glfwGetCocoaWindow(window);
        #     [ns_window.contentView setWantsLayer:YES];
        #     metal_layer = [CAMetalLayer layer];
        #     [ns_window.contentView setLayer:metal_layer];
        #     surface = wgpu_create_surface_from_metal_layer(metal_layer);
        # }
        window = ctypes.c_void_p(present_info["window"])

        cw = ObjCInstance(window)
        try:
            cv = cw.contentView
        except AttributeError:
            # With wxPython, ObjCInstance is actually already a wxNSView and
            # not a NSWindow so no need to get the contentView (which is a
            # NSWindow method)
            wx_view = ObjCInstance(window)
            # Creating a metal layer directly in the wxNSView does not seem to
            # work, so instead add a subview with the same bounds that resizes
            # with the wxNSView and add a metal layer to that
            if not len(wx_view.subviews):
                new_view = ObjCClass("NSView").alloc().initWithFrame(wx_view.bounds)
                # typedef NS_OPTIONS(NSUInteger, NSAutoresizingMaskOptions) {
                #     ...
                #     NSViewWidthSizable          =  2,
                #     NSViewHeightSizable         = 16,
                #     ...
                # };
                # Make subview resize with superview by combining
                # NSViewHeightSizable and NSViewWidthSizable
                new_view.setAutoresizingMask(18)
                wx_view.setAutoresizesSubviews(True)
                wx_view.addSubview(new_view)
            cv = wx_view.subviews[0]

        if cv.layer and cv.layer.isKindOfClass(ObjCClass("CAMetalLayer")):
            # No need to create a metal layer again
            metal_layer = cv.layer
        else:
            metal_layer = ObjCClass("CAMetalLayer").layer()
            cv.setLayer(metal_layer)
            cv.setWantsLayer(True)

        struct = ffi.new("WGPUSurfaceSourceMetalLayer *")
        struct.layer = ffi.cast("void *", metal_layer.ptr.value)
        struct.chain.sType = lib.WGPUSType_SurfaceSourceMetalLayer

    elif sys.platform.startswith("linux"):  # no-cover
        platform = present_info.get("platform", "x11")
        if platform == "x11":
            struct = ffi.new("WGPUSurfaceSourceXlibWindow *")
            struct.display = ffi.cast("void *", present_info["display"])
            struct.window = int(present_info["window"])
            struct.chain.sType = lib.WGPUSType_SurfaceSourceXlibWindow
        elif platform == "wayland":
            struct = ffi.new("WGPUSurfaceSourceWaylandSurface *")
            struct.display = ffi.cast("void *", present_info["display"])
            struct.surface = ffi.cast("void *", present_info["window"])
            struct.chain.sType = lib.WGPUSType_SurfaceSourceWaylandSurface
        elif platform == "xcb":
            # todo: xcb untested
            struct = ffi.new("WGPUSurfaceSourceXCBWindow *")
            struct.connection = ffi.cast("void *", present_info["connection"])  # ??
            struct.window = int(present_info["window"])
            struct.chain.sType = lib.WGPUSType_SurfaceSourceXCBWindow
        else:
            raise RuntimeError("Unexpected Linux surface platform '{platform}'.")

    else:  # no-cover
        raise RuntimeError("Cannot get surface id: unsupported platform.")

    surface_descriptor = ffi.new("WGPUSurfaceDescriptor *")
    surface_descriptor.label.data = ffi.NULL  # not setting label for now
    surface_descriptor.nextInChain = ffi.cast("WGPUChainedStruct *", struct)

    return lib.wgpuInstanceCreateSurface(get_wgpu_instance(), surface_descriptor)


# The functions below are copied from codegen/utils.py - let's keep these in sync!


def to_snake_case(name, separator="_"):
    """Convert a name from camelCase to snake_case. Names that already are
    snake_case remain the same.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0:
            prev = name2[-1]
            if c2 == "d" and prev in "123":
                name2 = name2[:-1] + separator + prev
            elif prev != separator:
                name2 += separator
        name2 += c2
    return name2


def to_camel_case(name):
    """Convert a name from snake_case to camelCase. Names that already are
    camelCase remain the same.
    """
    is_capital = False
    name2 = ""
    for c in name:
        if c in "_-" and name2:
            is_capital = True
        elif is_capital:
            name2 += c.upper()
            is_capital = False
        else:
            name2 += c
    if name2.endswith(("1d", "2d", "3d")):
        name2 = name2[:-1] + "D"
    return name2


class ErrorHandler:
    """Object that turns errors reported by Dawn into Python exceptions, or logs them.

    Natively, Dawn reports a validation error synchronously via the
    uncaptured-error callback, while the offending API call is running. The
    callback stores the error, and the wrapper of that call (see
    ``SafeLibCalls``) raises it as a Python exception at the call site. Checking
    for a pending error after each call is a cheap list truthiness test.

    In the browser (Pyodide), errors usually arrive asynchronously, from the JS
    event loop, so they cannot be attributed to a call; these are logged. Some
    WebGPU implementations (e.g. Dawn's Node.js bindings) do report errors
    during the call; these are raised as well.

    Calls on pass/bundle encoders are made without a check, because Dawn
    defers errors in these to ``GPUCommandEncoder.finish()`` (or
    ``GPURenderBundleEncoder.finish()``) anyway.
    """

    def __init__(self, logger):
        self._logger = logger
        self._error_message_counts = {}
        # A list, so that ``if pending:`` is as cheap as it can be
        self.pending = []
        # In the browser, the number of (checked) calls that are running
        self.web_calls_running = 0

    def handle_error(self, error_type: str, message: str):
        """Handle an error message (called from the uncaptured-error callback)."""
        if IS_WEB and not self.web_calls_running:
            self.log_error(message)  # cannot attribute it to a call
        else:
            self.pending.append((error_type, message))

    def raise_pending(self, frame_depth=1):
        """Raise the first pending error, and log any others."""
        errors = self.pending[:]
        self.pending.clear()
        if not errors:
            return
        for _, message in errors[1:]:
            self.log_error(message)
        error_type, message = errors[0]
        cls = ERROR_TYPES.get(error_type, GPUError)
        wgpu_error = cls(message)
        # Select the traceback object matching the call that raised the error. The
        # traceback will still actually show the line where we raise below, but the
        # bottommost line (which ppl look at first) will be correct.
        f = inspect.currentframe()
        for _ in range(frame_depth):
            f = f.f_back if f.f_back is not None else f
        tb = types.TracebackType(None, f, f.f_lasti, f.f_lineno)
        raise wgpu_error.with_traceback(tb)

    def log_pending(self):
        """Log pending errors instead of raising them, e.g. during cleanup."""
        errors = self.pending[:]
        self.pending.clear()
        for _, message in errors:
            self.log_error(message)

    def log_error(self, message):
        """Handle an error message by logging it, bypassing any capturing."""
        # Get count for this message. Use a hash that does not use the
        # digits in the message, because of id's getting renewed on
        # each draw.
        h = hash("".join(c for c in message if not c.isdigit()))
        error_message_counts = self._error_message_counts
        count = error_message_counts.get(h, 0) + 1
        error_message_counts[h] = count

        # Decide what to do
        if count == 1:
            self._logger.error(message)
        elif count < 10:
            self._logger.error(message.splitlines()[0] + f" ({count})")
        elif count == 10:
            self._logger.error(message.splitlines()[0] + " (hiding from now)")


class EventPump:
    """Calls ``process_events()`` on the event loop's thread while promises are pending (native only).

    This makes ``promise.then()`` work natively, also when nothing awaits the
    promise. Dawn objects are not thread-safe (unless the
    ImplicitDeviceSynchronization feature is used), so we never call into Dawn
    from a background thread. Instead, a lightweight thread periodically
    schedules a call to ``process_events()`` in the event loop, using its
    ``call_soon_threadsafe()``.
    """

    def __init__(self, process_events):
        self._process_events = process_events
        self._lock = threading.Lock()
        self._promises = set()
        self._thread = None

    def add(self, promise):
        if promise._call_soon_threadsafe is None:
            return
        with self._lock:
            self._promises.add(promise)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def _run(self):
        sleep_gen = get_backoff_time_generator()
        while True:
            with self._lock:
                self._promises = {p for p in self._promises if p._state == "pending"}
                if not self._promises:
                    self._thread = None
                    return
                call_soons = {p._call_soon_threadsafe for p in self._promises}
            for call_soon in call_soons:
                try:
                    call_soon(self._process_events)
                except Exception:
                    pass  # e.g. loop is closed
            time.sleep(max(0.001, next(sleep_gen)))


class SafeLibCalls:
    """Object that copies all library functions, but wrapped in such
    a way that errors occurring in that call are raised as exceptions.
    """

    def __init__(self, lib, error_handler):
        self._error_handler = error_handler
        self._make_function_copies(lib)

    def _make_function_copies(self, lib):
        for name in dir(lib):
            if name.startswith("wgpu"):
                ob = getattr(lib, name)
                if callable(ob):
                    setattr(self, name, self._make_proxy_func(name, ob))

    def _make_proxy_func(self, name, ob):
        error_handler = self._error_handler
        pending = error_handler.pending
        raise_pending = error_handler.raise_pending

        if IS_WEB:

            def proxy_func(*args):
                error_handler.web_calls_running += 1
                try:
                    result = ob(*args)
                except TypeError:
                    if any(arg is None for arg in args):
                        raise RuntimeError(
                            f"{name}() was called with a released object."
                        ) from None
                    raise
                finally:
                    error_handler.web_calls_running -= 1
                if pending:
                    raise_pending(2)
                return result

            proxy_func.__name__ = name
            return proxy_func

        def proxy_func(*args):
            try:
                result = ob(*args)
            except TypeError:
                # cffi raises TypeError for None, which is what _internal is
                # set to when an object is released.
                if any(arg is None for arg in args):
                    raise RuntimeError(
                        f"{name}() was called with a released object."
                    ) from None
                raise
            if pending:
                raise_pending(2)
            return result

        proxy_func.__name__ = name
        return proxy_func
