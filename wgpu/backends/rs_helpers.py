import os
import sys
import ctypes

from .rs_ffi import ffi, lib


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
        # #if WGPU_TARGET == WGPU_TARGET_MACOS
        #     {
        #         id metal_layer = NULL;
        #         NSWindow *ns_window = glfwGetCocoaWindow(window);
        #         [ns_window.contentView setWantsLayer:YES];
        #         metal_layer = [CAMetalLayer layer];
        #         [ns_window.contentView setLayer:metal_layer];
        #         surface = wgpu_create_surface_from_metal_layer(metal_layer);
        #     }
        window = ctypes.c_void_p(win_id)

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        content_view_sel = objc.sel_registerName(b"contentView")
        set_wants_layer_sel = objc.sel_registerName(b"setWantsLayer:")
        responds_to_sel_sel = objc.sel_registerName(b"respondsToSelector:")
        layer_sel = objc.sel_registerName(b"layer")
        set_layer_sel = objc.sel_registerName(b"setLayer:")

        # Try some duck typing to see what kind of object the window pointer points to
        # Qt doesn't return a NSWindow, but a QNSView instead, which is subclass of NSView.
        if objc.objc_msgSend(
            window, responds_to_sel_sel, ctypes.c_void_p(content_view_sel)
        ):
            # NSWindow instances respond to contentView selector
            content_view = objc.objc_msgSend(window, content_view_sel)
        elif objc.objc_msgSend(window, responds_to_sel_sel, ctypes.c_void_p(layer_sel)):
            # NSView instances respond to layer selector
            # Let's assume that the given window pointer is actually the content view
            content_view = window
        else:
            # If the code reaches this part, we know that `window` is an
            # objective-c object but the type is neither NSView or NSWindow.
            raise RuntimeError("Received unidentified objective-c object.")

        # [ns_window.contentView setWantsLayer:YES]
        objc.objc_msgSend(content_view, set_wants_layer_sel, True)

        # metal_layer = [CAMetalLayer layer];
        ca_metal_layer_class = objc.objc_getClass(b"CAMetalLayer")
        metal_layer = objc.objc_msgSend(ca_metal_layer_class, layer_sel)

        # [ns_window.content_view setLayer:metal_layer];
        objc.objc_msgSend(content_view, set_layer_sel, ctypes.c_void_p(metal_layer))

        struct = ffi.new("WGPUSurfaceDescriptorFromMetalLayer *")
        struct.layer = ffi.cast("void *", metal_layer)
        struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromMetalLayer

    elif sys.platform.startswith("linux"):  # no-cover
        display_id = canvas.get_display_id()
        is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
        if is_wayland:
            # todo: probably does not work since we dont have WGPUSurfaceDescriptorFromWayland
            struct = ffi.new("WGPUSurfaceDescriptorFromXlib *")
            struct.display = ffi.cast("void *", display_id)
            struct.window = int(win_id)
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlib
        else:
            struct = ffi.new("WGPUSurfaceDescriptorFromXlib *")
            struct.display = ffi.cast("void *", display_id)
            struct.window = int(win_id)
            struct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlib

    else:  # no-cover
        raise RuntimeError("Cannot get surface id: unsupported platform.")

    surface_descriptor = ffi.new("WGPUSurfaceDescriptor *")
    surface_descriptor.label = ffi.NULL
    surface_descriptor.nextInChain = ffi.cast("WGPUChainedStruct *", struct)

    instance_id = ffi.NULL
    return lib.wgpuInstanceCreateSurface(instance_id, surface_descriptor)
