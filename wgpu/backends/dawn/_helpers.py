"""Utilities used in the Dawn backend."""

import sys
import types
import ctypes
import inspect
import threading
from queue import deque

from ._ffi import ffi, lib, lib_path, IS_WEB
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
            features = ffi.new(
                "WGPUInstanceFeatureName[]",
                [lib.WGPUInstanceFeatureName_ShaderSourceSPIRV],
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


class ErrorSlot:
    __slot__ = ["name", "type", "message"]

    def __init__(self, name):
        self.name = name
        self.type = type
        self.message = None


class ErrorHandler:
    """Object that logs errors, with the option to collect incoming
    errors elsewhere.
    """

    def __init__(self, logger):
        self._logger = logger
        # threadlocal -> deque -> ErrorSlot
        self._per_thread_data = threading.local()

    def _get_proxy_stack(self):
        try:
            return self._per_thread_data.stack
        except AttributeError:
            stack = deque()
            self._per_thread_data.stack = stack
            self._per_thread_data.error_message_counts = {}
            return stack

    def capture(self, name):
        """Capture incoming error messages instead of logging them directly."""
        # This codepath must be as fast as it can be
        self._get_proxy_stack().append(ErrorSlot(name))

    def release(self, name):
        """Release the given name, returning the last captured error."""
        # This codepath, with matching name, must be as fast as it can be

        proxy_stack = self._get_proxy_stack()
        try:
            error_slot = proxy_stack.pop()
        except IndexError:
            error_slot = ErrorSlot("notavalidname")

        if error_slot.name == name:
            if error_slot.message is None:
                return None
            else:
                return error_slot.type, error_slot.message
        else:
            # This should never happen, but if it does, we want to know.
            self._logger.error("ErrorHandler capture/release out of sync")
            if error_slot.message:
                self.log_error(error_slot.message)
            while proxy_stack:
                es = proxy_stack.pop()
                if es.message:
                    self.log_error(es.message)
            return None

    def handle_error(self, error_type: str, message: str):
        """Handle an error message."""
        proxy_stack = self._get_proxy_stack()
        if proxy_stack:
            error_slot = proxy_stack[-1]
            if error_slot.message:
                self.log_error(error_slot.message)
            error_slot.type = error_type
            error_slot.message = message
        else:
            self.log_error(message)

    def log_error(self, message):
        """Handle an error message by logging it, bypassing any capturing."""
        # Get count for this message. Use a hash that does not use the
        # digits in the message, because of id's getting renewed on
        # each draw.
        h = hash("".join(c for c in message if not c.isdigit()))
        self._get_proxy_stack()  # make sure the error_message_counts attribute exists
        error_message_counts = self._per_thread_data.error_message_counts
        count = error_message_counts.get(h, 0) + 1
        error_message_counts[h] = count

        # Decide what to do
        if count == 1:
            self._logger.error(message)
        elif count < 10:
            self._logger.error(message.splitlines()[0] + f" ({count})")
        elif count == 10:
            self._logger.error(message.splitlines()[0] + " (hiding from now)")


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

        def proxy_func(*args):
            # Make the call, with error capturing on
            error_handler.capture(name)
            try:
                result = ob(*args)
            finally:
                error_type_msg = error_handler.release(name)

            # Handle the error.
            if error_type_msg is not None:
                error_type, message = error_type_msg
                cls = ERROR_TYPES.get(error_type, GPUError)
                wgpu_error = cls(message)
                # Select the traceback object matching the call that raised the error. The
                # traceback will still actually show the line where we raise below, but the
                # bottommost line (which ppl look at first) will be correct.
                f = inspect.currentframe()
                f = f.f_back
                tb = types.TracebackType(None, f, f.f_lasti, f.f_lineno)
                # Raise message with alt traceback
                wgpu_error = wgpu_error.with_traceback(tb)
                raise wgpu_error
            return result

        proxy_func.__name__ = name
        return proxy_func
