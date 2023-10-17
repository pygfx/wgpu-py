"""Utilities used in rs.py.
"""

import os
import sys
import ctypes

from .rs_ffi import ffi, lib
from ..base import (
    GPUError,
    GPUOutOfMemoryError,
    GPUValidationError,
    GPUPipelineError,
    GPUInternalError,
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
    """Get a memoryview for the given data and its memory address.
    The data object must support the buffer protocol.
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
    # Note that pypy does not have the contiguous attribute, so we assume it is.
    if not getattr(m, "contiguous", True):
        raise ValueError("The given texture data is not contiguous")

    # Get the address via ffi. In contrast to ctypes, this also
    # works for readonly data (e.g. bytes)
    c_data = ffi.from_buffer("uint8_t []", m)
    address = int(ffi.cast("uintptr_t", c_data))

    return m, address


def get_memoryview_from_address(address, nbytes, format="B"):
    """Get a memoryview from an int memory address and a byte count,"""
    # The default format is "<B", which seems to confuse some memoryview
    # operations, so we always cast it.
    c_array = (ctypes.c_uint8 * nbytes).from_address(address)
    return memoryview(c_array).cast(format, shape=(nbytes,))


_the_instance = None


def get_wgpu_instance():
    """Get the global wgpu instance."""
    # Note, we could also use wgpuInstanceRelease,
    # but we keep a global instance, so we don't have to.
    global _the_instance
    if _the_instance is None:
        # H: nextInChain: WGPUChainedStruct *
        struct = ffi.new("WGPUInstanceDescriptor *")
        _the_instance = lib.wgpuCreateInstance(struct)
    return _the_instance


def get_surface_id_from_canvas(canvas):
    """Get an id representing the surface to render to. The way to
    obtain this id differs per platform and GUI toolkit.
    """
    win_id = canvas.get_window_id()

    if sys.platform.startswith("win"):  # no-cover
        struct = ffi.new("WGPUSurfaceDescriptorFromWindowsHWND *")
        struct.hinstance = ffi.NULL
        struct.hwnd = ffi.cast("void *", int(win_id))
        struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromWindowsHWND

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
        window = ctypes.c_void_p(win_id)

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

        struct = ffi.new("WGPUSurfaceDescriptorFromMetalLayer *")
        struct.layer = ffi.cast("void *", metal_layer.ptr.value)
        struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromMetalLayer

    elif sys.platform.startswith("linux"):  # no-cover
        display_id = canvas.get_display_id()
        is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
        is_xcb = False
        if is_wayland:
            # todo: wayland seems to be broken right now
            struct = ffi.new("WGPUSurfaceDescriptorFromWaylandSurface *")
            struct.display = ffi.cast("void *", display_id)
            struct.surface = ffi.cast("void *", win_id)
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromWaylandSurface
        elif is_xcb:
            # todo: xcb untested
            struct = ffi.new("WGPUSurfaceDescriptorFromXcbWindow *")
            struct.connection = ffi.NULL  # ?? ffi.cast("void *", display_id)
            struct.window = int(win_id)
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlibWindow
        else:
            struct = ffi.new("WGPUSurfaceDescriptorFromXlibWindow *")
            struct.display = ffi.cast("void *", display_id)
            struct.window = int(win_id)
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlibWindow

    else:  # no-cover
        raise RuntimeError("Cannot get surface id: unsupported platform.")

    surface_descriptor = ffi.new("WGPUSurfaceDescriptor *")
    surface_descriptor.label = ffi.NULL
    surface_descriptor.nextInChain = ffi.cast("WGPUChainedStruct *", struct)

    return lib.wgpuInstanceCreateSurface(get_wgpu_instance(), surface_descriptor)


# The functions below are copied from codegen/utils.py


def to_snake_case(name):
    """Convert a name from camelCase to snake_case. Names that already are
    snake_case remain the same.
    """
    name2 = ""
    for c in name:
        c2 = c.lower()
        if c2 != c and len(name2) > 0 and name2[-1] not in "_123":
            name2 += "_"
        name2 += c2
    return name2


def to_camel_case(name):
    """Convert a name from snake_case to camelCase. Names that already are
    camelCase remain the same.
    """
    is_capital = False
    name2 = ""
    for c in name:
        if c == "_" and name2:
            is_capital = True
        elif is_capital:
            name2 += c.upper()
            is_capital = False
        else:
            name2 += c
    if name2.endswith(("1d", "2d", "3d")):
        name2 = name2[:-1] + "D"
    return name2


class DelayedReleaser:
    """Helps release objects at a later time."""

    # I found that when wgpuDeviceRelease() was called in Device._destroy,
    # the tests would hang. I found that the release call was done around
    # the time when another device was used (e.g. to create a buffer
    # or shader module). For some reason, the delay in destruction (by
    # Python's CG) causes a deadlock or something. We seem to be able
    # to fix this by doing the actual release later - e.g. when the
    # user creates a new device. Seems to be the same for the adapter.
    def __init__(self):
        self._things_to_release = []

    def release_soon(self, fun, i):
        self._things_to_release.append((fun, i))

    def release_all_pending(self):
        while self._things_to_release:
            fun, i = self._things_to_release.pop(0)
            release_func = getattr(lib, fun)
            release_func(i)


class ErrorHandler:
    """Object that logs errors, with the option to collect incoming
    errors elsewhere.
    """

    def __init__(self, logger):
        self._logger = logger
        self._proxy_stack = []
        self._error_message_counts = {}

    def capture(self, func):
        """Send incoming error messages to the given func instead of logging them."""
        self._proxy_stack.append(func)

    def release(self, func):
        """Release the given func."""
        f = self._proxy_stack.pop(-1)
        if f is not func:
            self._proxy_stack.clear()
            self._logger.warning("ErrorHandler capture/release out of sync")

    def handle_error(self, error_type: str, message: str):
        """Handle an error message."""
        if self._proxy_stack:
            self._proxy_stack[-1](error_type, message)
        else:
            self.log_error(message)

    def log_error(self, message):
        """Hanle an error message by logging it, bypassing any capturing."""
        # Get count for this message. Use a hash that does not use the
        # digits in the message, because of id's getting renewed on
        # each draw.
        h = hash("".join(c for c in message if not c.isdigit()))
        count = self._error_message_counts.get(h, 0) + 1
        self._error_message_counts[h] = count

        # Decide what to do
        if count == 1:
            self._logger.error(message)
        elif count < 10:
            self._logger.error(message.splitlines()[0] + f" ({count})")
        elif count == 10:
            self._logger.error(message.splitlines()[0] + " (hiding from now)")


class SafeLibCalls:
    """Object that copies all library functions, but wrapped in such
    a way that errors occuring in that call are raised as exceptions.
    """

    def __init__(self, lib, error_handler):
        self._error_handler = error_handler
        self._error_message = None
        self._make_function_copies(lib)

    def _make_function_copies(self, lib):
        for name in dir(lib):
            if name.startswith("wgpu"):
                ob = getattr(lib, name)
                if callable(ob):
                    setattr(self, name, self._make_proxy_func(name, ob))

    def _handle_error(self, error_type, message):
        # If we already had an error, we log the earlier one now
        if self._error_message:
            self._error_handler.log_error(self._error_message[1])
        # Store new error
        self._error_message = (error_type, message)

    def _make_proxy_func(self, name, ob):
        def proxy_func(*args):
            # Make the call, with error capturing on
            handle_error = self._handle_error
            self._error_handler.capture(handle_error)
            try:
                result = ob(*args)
            finally:
                self._error_handler.release(handle_error)

            # Handle the error.
            if self._error_message:
                error_type, message = self._error_message
                self._error_message = None
                cls = ERROR_TYPES.get(error_type, GPUError)
                wgpu_error = cls(message)
                # The line below will be the bottom line in the traceback,
                # so better make it informative! As far as I know there is
                # no way to exclude this frame from the traceback.
                raise wgpu_error  # the frame above is more interesting ↑↑
            return result

        proxy_func.__name__ = name
        return proxy_func
