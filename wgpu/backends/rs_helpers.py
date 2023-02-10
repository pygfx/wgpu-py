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


_the_instance = None


def get_wgpu_instance():
    """Get the global wgpu instance."""
    # Note, we could also use wgpuInstanceDrop,
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


def color_string(color, string):
    """color: ANSI color code. 33 is yellow, 36 is cyan, etc."""

    if not __terminal_supports_colors:
        return string
    return f"\033[{color}m{string}\033[0m"


_wgsl_error_tmpl = re.compile(
    r'(Parsing|Validation)\(ShaderError { source: "(.*)", label:(.*), inner: (ParseError|WithSpan) (.*) }\)',
    re.M | re.S,
)

_wgsl_inner_parsing_error_tmpl = re.compile(
    r'{ message: "(.*)", labels: \[\((\d+)\.\.(\d+), "(.*)"\)\], notes: \[(.*)\] }',
    re.M | re.S,
)

_wgsl_inner_validation_error_tmpl = re.compile(
    r'{ inner: (.*) { (.*), name: "(.*)", error: (.*) }, spans: \[\(Span { start: (\d+), end: (\d+) }, (.*)\)\] }',
    re.M | re.S,
)  # TODO: simple error message

_wgsl_inner_validation_error_info_tmpl = re.compile(
    r", error: (.*) [}|,]",
    re.M | re.S,
)  # TODO: simple error message


def parse_wgsl_error(message):
    """Parse a WGPU shader error message, give an easy-to-understand error prompt."""

    err_msg = ["\n"]

    match = _wgsl_error_tmpl.match(message)

    if match:
        error_type = match.group(1)
        source = match.group(2)
        source = source.replace("\\t", " ")
        label = match.group(3)
        # inner_error_type = match.group(4)
        inner_error = match.group(5)
        err_msg.append(color_string(33, f"Shader error: label: {label}"))

        if error_type and inner_error:
            if error_type == "Parsing":
                match2 = _wgsl_inner_parsing_error_tmpl.match(inner_error)
                if match2:
                    err_msg.append(
                        color_string(33, f"Parsing error: {match2.group(1)}")
                    )
                    start = int(match2.group(2))
                    end = int(match2.group(3))
                    label = match2.group(4)
                    note = match2.group(5)
                    err_msg += _wgsl_parse_extract_line(source, start, end, label, note)
                else:
                    err_msg += [color_string(33, inner_error)]

            elif error_type == "Validation":
                match2 = _wgsl_inner_validation_error_tmpl.match(inner_error)
                if match2:
                    error = match2.group(4)
                    err_msg.append(color_string(33, f"Validation error: {error}"))
                    start = int(match2.group(5))
                    end = int(match2.group(6))
                    error_match = _wgsl_inner_validation_error_info_tmpl.search(error)
                    label = error_match.group(1) if error_match else error
                    note = ""
                    err_msg += _wgsl_parse_extract_line(source, start, end, label, note)
                else:
                    err_msg += [color_string(33, inner_error)]

            return "\n".join(err_msg)

    return None  # Does not look like a shader error


def _wgsl_parse_extract_line(source, start, end, label, note):
    # Find next newline after the end pos
    try:
        next_n = source.index("\n", end)
    except ValueError:
        next_n = len(source)

    # Truncate and convert to lines
    lines = source[:next_n].splitlines(True)
    line_num = len(lines)

    # Collect the lines relevant to this error
    error_lines = []
    line_pos = start - next_n
    while line_pos < 0:
        line = lines[line_num - 1]
        line_length = len(line)
        line_pos += line_length

        start_pos = line_pos
        if start_pos < 0:
            start_pos = 0
        end_pos = line_length - (next_n - end)
        if end_pos > line_length:
            end_pos = line_length

        error_lines.insert(0, (line_num, line, start_pos, end_pos))

        next_n -= line_length
        line_num -= 1

    def pad_str(s, line_num=None):
        pad = len(str(len(lines)))
        if line_num is not None:
            pad -= len(str(line_num))
            return f"{' '*pad}{line_num} {s}".rstrip()
        else:
            return f"{' '*pad} {s}".rstrip()

    err_msg = [""]

    # Show header
    if len(error_lines) == 1:
        prefix = pad_str(color_string(36, "┌─"))
        err_msg.append(prefix + f" wgsl:{len(lines)}:{line_pos}")
    else:
        prefix = pad_str(color_string(36, "┌─"))
        err_msg.append(prefix + f" wgsl:{line_num+1}--{len(lines)}")

    # Add lines
    err_msg.append(pad_str(color_string(36, "│")))
    for line_num, line, _, _ in error_lines:
        prefix = color_string(36, pad_str("│", line_num))
        err_msg.append(prefix + f" {line}".rstrip())

    # Show annotation
    if len(error_lines) == 1:
        prefix = pad_str(color_string(36, "│"))
        annotation = f" {' '*error_lines[0][2] + '^'*(end-start)} {label}"
        err_msg.append(prefix + color_string(33, annotation))
    else:
        prefix = pad_str(color_string(36, "│"))
        annotation = color_string(33, f" ^^^{label}")
        err_msg.append(prefix + annotation)

    err_msg.append(pad_str(color_string(36, "│")))
    err_msg.append(pad_str(color_string(36, f"= note: {note}".rstrip())))
    err_msg.append("\n")

    return err_msg


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


class DelayedDropper:
    """Helps drop objects at a later time."""

    # I found that when wgpuDeviceDrop() was called in Device._destroy,
    # the tests would hang. I found that the drop call was done around
    # the time when another device was used (e.g. to create a buffer
    # or shader module). For some reason, the delay in destruction (by
    # Python's CG) causes a deadlock or something. We seem to be able
    # to fix this by doing the actual dropping later - e.g. when the
    # user creates a new device. Seems to be the same for the adapter.
    def __init__(self):
        self._things_to_drop = []

    def drop_soon(self, fun, i):
        self._things_to_drop.append((fun, i))

    def drop_all_pending(self):
        while self._things_to_drop:
            fun, i = self._things_to_drop.pop(0)
            drop_fun = getattr(lib, fun)
            drop_fun(i)


delayed_dropper = DelayedDropper()
