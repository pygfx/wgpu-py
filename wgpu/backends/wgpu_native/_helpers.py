"""Utilities used in the wgpu-native backend."""

import sys
import time
import types
import ctypes
import inspect
import threading
from queue import deque
from collections.abc import Generator

import sniffio

from ._ffi import ffi, lib, lib_path
from ..._diagnostics import DiagnosticsBase
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


def get_surface_id_from_info(present_info):
    """Get an id representing the surface to render to. The way to
    obtain this id differs per platform and GUI toolkit.
    """

    if sys.platform.startswith("win"):  # no-cover
        GetModuleHandle = ctypes.windll.kernel32.GetModuleHandleW  # noqa: N806
        struct = ffi.new("WGPUSurfaceDescriptorFromWindowsHWND *")
        struct.hinstance = ffi.cast("void *", GetModuleHandle(lib_path))
        struct.hwnd = ffi.cast("void *", int(present_info["window"]))
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

        struct = ffi.new("WGPUSurfaceDescriptorFromMetalLayer *")
        struct.layer = ffi.cast("void *", metal_layer.ptr.value)
        struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromMetalLayer

    elif sys.platform.startswith("linux"):  # no-cover
        platform = present_info.get("platform", "x11")
        if platform == "x11":
            struct = ffi.new("WGPUSurfaceDescriptorFromXlibWindow *")
            struct.display = ffi.cast("void *", present_info["display"])
            struct.window = int(present_info["window"])
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlibWindow
        elif platform == "wayland":
            struct = ffi.new("WGPUSurfaceDescriptorFromWaylandSurface *")
            struct.display = ffi.cast("void *", present_info["display"])
            struct.surface = ffi.cast("void *", present_info["window"])
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromWaylandSurface
        elif platform == "xcb":
            # todo: xcb untested
            struct = ffi.new("WGPUSurfaceDescriptorFromXcbWindow *")
            struct.connection = ffi.cast("void *", present_info["connection"])  # ??
            struct.window = int(present_info["window"])
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlibWindow
        else:
            raise RuntimeError("Unexpected Linux surface platform '{platform}'.")

    else:  # no-cover
        raise RuntimeError("Cannot get surface id: unsupported platform.")

    surface_descriptor = ffi.new("WGPUSurfaceDescriptor *")
    surface_descriptor.label = ffi.NULL
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


async def async_sleep(delay):
    """Async sleep that uses sniffio to be compatible with asyncio, trio, rendercanvas.utils.asyncadapter, and possibly more."""
    libname = sniffio.current_async_library()
    sleep = sys.modules[libname].sleep
    await sleep(delay)


class WgpuAwaitable:
    """An object that can be waited for, either synchronously using sync_wait() or asynchronously using await.

    The purpose of this class is to implememt the asynchronous methods in a
    truely async manner, as well as to support a synchronous version of them.
    """

    def __init__(self, title, callback, finalizer, poll_function=None):
        self.title = title  # for context in error messages
        self.callback = callback  # only used to prevent it from being gc'd
        self.finalizer = finalizer  # function to finish the result
        self.poll_function = poll_function  # call this to poll wgpu
        self.result = None

    def set_result(self, result):
        self.result = (result, None)

    def set_error(self, error):
        self.result = (None, error)

    def _finish(self):
        try:
            result, error = self.result
            if error:
                raise RuntimeError(error)
            else:
                return self.finalizer(result)
        finally:
            # Reset attrs to prevent potential memory leaks
            self.callback = self.finalizer = self.poll_function = self.result = None

    def sync_wait(self):
        if self.result is not None:
            pass
        elif not self.poll_function:
            raise RuntimeError("Expected callback to have already happened")
        else:
            backoff_time_generator = self._get_backoff_time_generator()
            while True:
                self.poll_function()
                if self.result is not None:
                    break
                time.sleep(next(backoff_time_generator))
                # We check the result after sleeping just in case another thread
                # causes the callback to happen
                if self.result is not None:
                    break

        return self._finish()

    def __await__(self):
        # There is no documentation on what __await__() is supposed to return, but we
        # can certainly copy from a function that *does* know what to return.
        # It would also be nice if wait_for_callback and sync_wait() could be merged,
        # but Python has no wait of combining them.
        async def wait_for_callback():
            if self.result is not None:
                pass
            elif not self.poll_function:
                raise RuntimeError("Expected callback to have already happened")
            else:
                backoff_time_generator = self._get_backoff_time_generator()
                while True:
                    self.poll_function()
                    if self.result is not None:
                        break
                    await async_sleep(next(backoff_time_generator))
                    # We check the result after sleeping just in case another
                    # flow of control causes the callback to happen
                    if self.result is not None:
                        break
            return self._finish()

        return (yield from wait_for_callback().__await__())

    def _get_backoff_time_generator(self) -> Generator[float, None, None]:
        for _ in range(5):
            yield 0
        for i in range(1, 20):
            yield i / 2000.0  # ramp up from 0ms to 10ms
        while True:
            yield 0.01


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


def generate_report():
    """Get a report similar to the one produced by wgpuGenerateReport(),
    but in the form of a Python dict.
    """

    # H: surfaces: WGPURegistryReport, backendType: WGPUBackendType, vulkan: WGPUHubReport, metal: WGPUHubReport, dx12: WGPUHubReport, gl: WGPUHubReport
    struct = ffi.new("WGPUGlobalReport *")

    # H: void f(WGPUInstance instance, WGPUGlobalReport * report)
    lib.wgpuGenerateReport(get_wgpu_instance(), struct)

    report = {}

    report["surfaces"] = {
        "allocated": struct.surfaces.numAllocated,
        "kept": struct.surfaces.numKeptFromUser,
        "released": struct.surfaces.numReleasedFromUser,
        "error": struct.surfaces.numError,
        "element_size": struct.surfaces.elementSize,
    }

    for backend in ("vulkan", "metal", "dx12", "gl"):
        c_hub_report = getattr(struct, backend)
        report[backend] = {}
        for key in dir(c_hub_report):
            c_registry_report = getattr(c_hub_report, key)
            registry_report = {
                "allocated": c_registry_report.numAllocated,
                "kept": c_registry_report.numKeptFromUser,
                "released": c_registry_report.numReleasedFromUser,
                "error": c_registry_report.numError,
                "element_size": c_registry_report.elementSize,
            }
            # if any(x!=0 for x in registry_report.values()):
            report[backend][key] = registry_report

    return report


class WgpuNativeCountsDiagnostics(DiagnosticsBase):
    def get_subscript(self):
        text = ""
        text += "    * The a, k, r, e are allocated, kept, released, and error, respectively.\n"
        text += "    * Reported memory does not include buffer/texture data.\n"
        return text

    def get_dict(self):
        result = {}
        native_report = generate_report()

        # Names in the root of the report (backend-less)
        root_names = ["surfaces"]

        # Get per-backend names and a list of backends
        names = list(native_report["vulkan"].keys())
        backends = [name for name in native_report.keys() if name not in root_names]

        # Get a mapping from native names to wgpu-py names
        name_map = {"surfaces": "CanvasContext"}
        for name in names:
            if name not in name_map:
                name_map[name] = name[0].upper() + name[1:-1]

        # Initialize the result dict (sorted)
        for report_name in sorted(name_map[name] for name in names + root_names):
            result[report_name] = {"count": 0, "mem": 0}

        # The field names to add together to obtain a representation for
        # the number of objects "allocated" by wgpu-core. In practice,
        # wgpu-core can keep objects around for re-use, which is why "allocated"
        # and released" are not in this equation.
        fields_to_add = ["kept", "error"]

        # Establish what backends are active
        active_backends = []
        for backend in backends:
            total = 0
            for name in names:
                d = native_report[backend][name]
                total += sum(d[k] for k in fields_to_add)
            if total > 0:
                active_backends.append(backend)

        # Process names in the root
        for name in root_names:
            d = native_report[name]
            subtotal_count = sum(d[k] for k in fields_to_add)
            impl = {
                "a": d["allocated"],
                "k": d["kept"],
                "r": d["released"],
                "e": d["error"],
                "el_size": d["element_size"],
            }
            # Store in report
            report_name = name_map[name]
            result[report_name]["count"] = subtotal_count
            result[report_name]["mem"] = subtotal_count * d["element_size"]
            result[report_name]["backend"] = {"": impl}

        # Iterate over backends
        for name in names:
            total_count = 0
            total_mem = 0
            implementations = {}
            for backend in active_backends:
                d = native_report[backend][name]
                subtotal_count = sum(d[k] for k in fields_to_add)
                subtotal_mem = subtotal_count * d["element_size"]
                impl = {
                    "a": d["allocated"],
                    "k": d["kept"],
                    "r": d["released"],
                    "e": d["error"],
                    "el_size": d["element_size"],
                }
                total_count += subtotal_count
                total_mem += subtotal_mem
                implementations[backend] = impl
            # Store in report
            report_name = name_map[name]
            result[report_name]["count"] = total_count
            result[report_name]["mem"] = total_mem
            result[report_name]["backend"] = implementations

        # Add totals
        totals = {}
        for key in ("count", "mem"):
            totals[key] = sum(v.get(key, 0) for v in result.values())
        result["total"] = totals

        return result


diagnostics = WgpuNativeCountsDiagnostics("wgpu_native_counts")
