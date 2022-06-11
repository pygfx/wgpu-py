import os
import sys
import ctypes
import re

from .rs_ffi import ffi, lib

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

    instance_id = ffi.NULL
    return lib.wgpuInstanceCreateSurface(instance_id, surface_descriptor)


# The function below are copied from "https://github.com/django/django/blob/main/django/core/management/color.py"


def _terminal_supports_colors():
    """
    Return True if the running system's terminal supports color,
    and False otherwise.
    """

    def vt_codes_enabled_in_windows_registry():
        """
        Check the Windows Registry to see if VT code handling has been enabled
        by default, see https://superuser.com/a/1300251/447564.
        """
        try:
            # winreg is only available on Windows.
            import winreg
        except ImportError:
            return False
        else:
            try:
                reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Console")
                reg_key_value, _ = winreg.QueryValueEx(reg_key, "VirtualTerminalLevel")
            except FileNotFoundError:
                return False
            else:
                return reg_key_value == 1

    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    try:
        import colorama

        colorama.init()
    except (ImportError, OSError):
        has_colorama = False
    else:
        has_colorama = True

    return is_a_tty and (
        sys.platform != "win32"
        or has_colorama
        or "ANSICON" in os.environ
        or
        # Windows Terminal supports VT codes.
        "WT_SESSION" in os.environ
        or
        # Microsoft Visual Studio Code's built-in terminal supports colors.
        os.environ.get("TERM_PROGRAM") == "vscode"
        or vt_codes_enabled_in_windows_registry()
    )


__terminal_supports_colors = _terminal_supports_colors()


def _color_string(color, string):
    """color: ANSI color code. 33 is yellow, 36 is cyan, etc."""

    if not __terminal_supports_colors:
        return string
    return f"\033[{color}m{string}\033[0m"


__shader_error_tmpl = re.compile(
    r'(Parsing|Validation)\(ShaderError { source: "(.*)", label:(.*), inner: (ParseError|WithSpan) (.*) }\)',
    re.M | re.S,
)
__inner_parsing_error_tmpl = re.compile(
    r'{ message: "(.*)", labels: \[\((\d+)\.\.(\d+), "(.*)"\)\], notes: \[(.*)\] }',
    re.M | re.S,
)
__inner_validation_error_tmpl = re.compile(
    r'{ inner: (.*) { (.*), name: "(.*)", error: (.*) }, spans: \[\(Span { start: (\d+), end: (\d+) }, (.*)\)\] }',
    re.M | re.S,
)  # TODO: simple error message

__inner_validation_error_info = re.compile(
    r", error: (.*) [}|,]",
    re.M | re.S,
)  # TODO: simple error message


def parse_wgpu_shader_error(message):
    """Parse a WGPU shader error message, give an easy-to-understand error prompt."""

    err_msg = ["\n"]

    match = __shader_error_tmpl.match(message)

    if match:
        error_type = match.group(1)
        source = match.group(2)
        source = source.replace("\\t", " ")
        label = match.group(3)
        # inner_error_type = match.group(4)
        inner_error = match.group(5)
        err_msg.append(_color_string(33, f"Shader error: label: {label}"))

        if error_type == "Parsing":
            match2 = __inner_parsing_error_tmpl.match(inner_error)
            if match2:
                err_msg.append(_color_string(33, f"Parsing error: {match2.group(1)}"))
                start = int(match2.group(2))
                end = int(match2.group(3))
                label = match2.group(4)
                note = match2.group(5)

        elif error_type == "Validation":
            match2 = __inner_validation_error_tmpl.match(inner_error)
            if match2:
                error = match2.group(4)
                err_msg.append(_color_string(33, f"Validation error: {error}"))
                start = int(match2.group(5))
                end = int(match2.group(6))
                error_match = __inner_validation_error_info.search(error)
                label = error_match.group(1) if error_match else error
                note = ""
        else:
            return "\n".join(err_msg)

        next_n = source.index("\n", end)
        s = source[:next_n]
        lines = s.splitlines(True)
        line_num = len(lines)
        line = lines[-1]
        line_pos = start - (next_n - len(line))

        def pad_str(s):
            pad = len(str(line_num))
            return f"{' '*pad} {s}"

        err_msg.append("\n")
        err_msg.append(
            pad_str(_color_string(36, "┌─")) + f" wgsl:{line_num}:{line_pos}"
        )
        err_msg.append(pad_str(_color_string(36, "│")))
        err_msg.append(_color_string(36, f"{line_num} │") + f" {line}")
        err_msg.append(
            pad_str(_color_string(36, "│"))
            + _color_string(33, f" {' '*line_pos + '^'*(end-start)} {label}")
        )
        err_msg.append(pad_str(_color_string(36, "│")))
        err_msg.append(pad_str(_color_string(36, f"= note: {note}")))
        err_msg.append("\n\n")

        return "\n".join(err_msg)

    else:
        return None


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


class DeviceDropper:
    """Helps drop devices at a good time."""

    # I found that when wgpuDeviceDrop() was called in Device._destroy,
    # the tests would hang. I found that the drop call was done around
    # the time when another device was used (e.g. to create a buffer
    # or shader module). For some reason, the delay in destruction (by
    # Python's CG) causes a deadlock or something. We seem to be able
    # to fix this by doing the actual dropping later - e.g. when the
    # user creates a new device.
    def __init__(self):
        self._devices_to_drop = []

    def drop_soon(self, internal):
        self._devices_to_drop.append(internal)

    def drop_all_pending(self):
        while self._devices_to_drop:
            internal = self._devices_to_drop.pop(0)
            # H: void f(WGPUDevice device)
            lib.wgpuDeviceDrop(internal)


device_dropper = DeviceDropper()
