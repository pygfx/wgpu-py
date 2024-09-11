"""
Support to render in a glfw window. The advantage of glfw is that it's
very lightweight.

Install pyGLFW using ``pip install glfw``. On Windows this is enough.
On Linux, install the glfw lib using ``sudo apt install libglfw3``,
or ``sudo apt install libglfw3-wayland`` when using Wayland.
"""

import sys
import time
import atexit
import weakref
import asyncio

import glfw

from .base import WgpuCanvasBase, WgpuAutoGui
from ._gui_utils import SYSTEM_IS_WAYLAND, weakbind, logger


# Make sure that glfw is new enough
glfw_version_info = tuple(int(i) for i in glfw.__version__.split(".")[:2])
if glfw_version_info < (1, 9):
    raise ImportError("wgpu-py requires glfw 1.9 or higher.")

# Do checks to prevent pitfalls on hybrid Xorg/Wayland systems
is_wayland = False
if sys.platform.startswith("linux") and SYSTEM_IS_WAYLAND:
    if not hasattr(glfw, "get_x11_window"):
        # Probably glfw was imported before we wgpu was, so we missed our chance
        # to set the env var to make glfw use x11.
        is_wayland = True
        logger.warning("Using GLFW with Wayland, which is experimental.")


# Some glfw functions are not always available
set_window_content_scale_callback = lambda *args: None  # noqa: E731
set_window_maximize_callback = lambda *args: None  # noqa: E731
get_window_content_scale = lambda *args: (1, 1)  # noqa: E731

if hasattr(glfw, "set_window_content_scale_callback"):
    set_window_content_scale_callback = glfw.set_window_content_scale_callback
if hasattr(glfw, "set_window_maximize_callback"):
    set_window_maximize_callback = glfw.set_window_maximize_callback
if hasattr(glfw, "get_window_content_scale"):
    get_window_content_scale = glfw.get_window_content_scale


# Map keys to JS key definitions
# https://www.glfw.org/docs/3.3/group__keys.html
# https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values
KEY_MAP = {
    glfw.KEY_DOWN: "ArrowDown",
    glfw.KEY_UP: "ArrowUp",
    glfw.KEY_LEFT: "ArrowLeft",
    glfw.KEY_RIGHT: "ArrowRight",
    glfw.KEY_BACKSPACE: "Backspace",
    glfw.KEY_CAPS_LOCK: "CapsLock",
    glfw.KEY_DELETE: "Delete",
    glfw.KEY_END: "End",
    glfw.KEY_ENTER: "Enter",  # aka return
    glfw.KEY_ESCAPE: "Escape",
    glfw.KEY_F1: "F1",
    glfw.KEY_F2: "F2",
    glfw.KEY_F3: "F3",
    glfw.KEY_F4: "F4",
    glfw.KEY_F5: "F5",
    glfw.KEY_F6: "F6",
    glfw.KEY_F7: "F7",
    glfw.KEY_F8: "F8",
    glfw.KEY_F9: "F9",
    glfw.KEY_F10: "F10",
    glfw.KEY_F11: "F11",
    glfw.KEY_F12: "F12",
    glfw.KEY_HOME: "Home",
    glfw.KEY_INSERT: "Insert",
    glfw.KEY_LEFT_ALT: "Alt",
    glfw.KEY_LEFT_CONTROL: "Control",
    glfw.KEY_LEFT_SHIFT: "Shift",
    glfw.KEY_LEFT_SUPER: "Meta",  # in glfw super means Windows or MacOS-command
    glfw.KEY_NUM_LOCK: "NumLock",
    glfw.KEY_PAGE_DOWN: "PageDown",
    glfw.KEY_PAGE_UP: "Pageup",
    glfw.KEY_PAUSE: "Pause",
    glfw.KEY_PRINT_SCREEN: "PrintScreen",
    glfw.KEY_RIGHT_ALT: "Alt",
    glfw.KEY_RIGHT_CONTROL: "Control",
    glfw.KEY_RIGHT_SHIFT: "Shift",
    glfw.KEY_RIGHT_SUPER: "Meta",
    glfw.KEY_SCROLL_LOCK: "ScrollLock",
    glfw.KEY_TAB: "Tab",
}

KEY_MAP_MOD = {
    glfw.KEY_LEFT_SHIFT: "Shift",
    glfw.KEY_RIGHT_SHIFT: "Shift",
    glfw.KEY_LEFT_CONTROL: "Control",
    glfw.KEY_RIGHT_CONTROL: "Control",
    glfw.KEY_LEFT_ALT: "Alt",
    glfw.KEY_RIGHT_ALT: "Alt",
    glfw.KEY_LEFT_SUPER: "Meta",
    glfw.KEY_RIGHT_SUPER: "Meta",
}


def get_surface_info(window):
    if sys.platform.startswith("win"):
        return {
            "platform": "windows",
            "window": int(glfw.get_win32_window(window)),
        }
    elif sys.platform.startswith("darwin"):
        return {
            "platform": "cocoa",
            "window": int(glfw.get_cocoa_window(window)),
        }
    elif sys.platform.startswith("linux"):
        if is_wayland:
            return {
                "platform": "wayland",
                "window": int(glfw.get_wayland_window(window)),
                "display": int(glfw.get_wayland_display()),
            }
        else:
            return {
                "platform": "x11",
                "window": int(glfw.get_x11_window(window)),
                "display": int(glfw.get_x11_display()),
            }
    else:
        raise RuntimeError(f"Cannot get GLFW surafce info on {sys.platform}.")


def get_physical_size(window):
    psize = glfw.get_framebuffer_size(window)
    return int(psize[0]), int(psize[1])


class GlfwWgpuCanvas(WgpuAutoGui, WgpuCanvasBase):
    """A glfw window providing a wgpu canvas."""

    # See https://www.glfw.org/docs/latest/group__window.html

    def __init__(self, *, size=None, title=None, **kwargs):
        app.init_glfw()
        super().__init__(**kwargs)

        # Handle inputs
        if not size:
            size = 640, 480
        title = str(title or "glfw wgpu canvas")

        # Set window hints
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)

        # Create the window (the initial size may not be in logical pixels)
        self._window = glfw.create_window(int(size[0]), int(size[1]), title, None, None)

        # Other internal variables
        self._need_draw = False
        self._request_draw_timer_running = False
        self._changing_pixel_ratio = False
        self._is_minimized = False

        # Register ourselves
        app.all_glfw_canvases.add(self)

        # Register callbacks. We may get notified too often, but that's
        # ok, they'll result in a single draw.
        glfw.set_framebuffer_size_callback(self._window, weakbind(self._on_size_change))
        glfw.set_window_close_callback(self._window, weakbind(self._check_close))
        glfw.set_window_refresh_callback(self._window, weakbind(self._on_window_dirty))
        glfw.set_window_focus_callback(self._window, weakbind(self._on_window_dirty))
        set_window_content_scale_callback(
            self._window, weakbind(self._on_pixelratio_change)
        )
        set_window_maximize_callback(self._window, weakbind(self._on_window_dirty))
        glfw.set_window_iconify_callback(self._window, weakbind(self._on_iconify))

        # User input
        self._key_modifiers = ()
        self._pointer_buttons = ()
        self._pointer_pos = 0, 0
        self._double_click_state = {"clicks": 0}
        glfw.set_mouse_button_callback(self._window, weakbind(self._on_mouse_button))
        glfw.set_cursor_pos_callback(self._window, weakbind(self._on_cursor_pos))
        glfw.set_scroll_callback(self._window, weakbind(self._on_scroll))
        glfw.set_key_callback(self._window, weakbind(self._on_key))
        glfw.set_char_callback(self._window, weakbind(self._on_char))

        # Initialize the size
        self._pixel_ratio = -1
        self._screen_size_is_logical = False
        self.set_logical_size(*size)
        self._request_draw()

    # Callbacks to provide a minimal working canvas for wgpu

    def _on_pixelratio_change(self, *args):
        if self._changing_pixel_ratio:
            return
        self._changing_pixel_ratio = True  # prevent recursion (on Wayland)
        try:
            self._set_logical_size(self._logical_size)
        finally:
            self._changing_pixel_ratio = False
        self._request_draw()

    def _on_size_change(self, *args):
        self._determine_size()
        self._request_draw()

    def _check_close(self, *args):
        # Follow the close flow that glfw intended.
        # This method can be overloaded and the close-flag can be set to False
        # using set_window_should_close() if now is not a good time to close.
        if self._window is not None and glfw.window_should_close(self._window):
            self._on_close()

    def _on_close(self, *args):
        app.all_glfw_canvases.discard(self)
        if self._window is not None:
            glfw.destroy_window(self._window)  # not just glfw.hide_window
            self._window = None
            self._handle_event_and_flush({"event_type": "close"})

    def _on_window_dirty(self, *args):
        self._request_draw()

    def _on_iconify(self, window, iconified):
        self._is_minimized = bool(iconified)

    # helpers

    def _mark_ready_for_draw(self):
        self._request_draw_timer_running = False
        self._need_draw = True  # The event loop looks at this flag
        glfw.post_empty_event()  # Awake the event loop, if it's in wait-mode

    def _determine_size(self):
        if self._window is None:
            return
        # Because the value of get_window_size is in physical-pixels
        # on some systems and in logical-pixels on other, we use the
        # framebuffer size and pixel ratio to derive the logical size.
        pixel_ratio = get_window_content_scale(self._window)[0]
        psize = get_physical_size(self._window)

        self._pixel_ratio = pixel_ratio
        self._physical_size = psize
        self._logical_size = psize[0] / pixel_ratio, psize[1] / pixel_ratio

        ev = {
            "event_type": "resize",
            "width": self._logical_size[0],
            "height": self._logical_size[1],
            "pixel_ratio": self._pixel_ratio,
        }
        self._handle_event_and_flush(ev)

    def _set_logical_size(self, new_logical_size):
        if self._window is None:
            return
        # There is unclarity about the window size in "screen pixels".
        # It appears that on Windows and X11 its the same as the
        # framebuffer size, and on macOS it's logical pixels.
        # See https://github.com/glfw/glfw/issues/845
        # Here, we simply do a quick test so we can compensate.

        # The current screen size and physical size, and its ratio
        pixel_ratio = get_window_content_scale(self._window)[0]
        ssize = glfw.get_window_size(self._window)
        psize = glfw.get_framebuffer_size(self._window)

        # Apply
        if is_wayland:
            # Not sure why, but on Wayland things work differently
            screen_ratio = ssize[0] / new_logical_size[0]
            glfw.set_window_size(
                self._window,
                int(new_logical_size[0] / screen_ratio),
                int(new_logical_size[1] / screen_ratio),
            )
        else:
            screen_ratio = ssize[0] / psize[0]
            glfw.set_window_size(
                self._window,
                int(new_logical_size[0] * pixel_ratio * screen_ratio),
                int(new_logical_size[1] * pixel_ratio * screen_ratio),
            )
        self._screen_size_is_logical = screen_ratio != 1
        # If this causes the widget size to change, then _on_size_change will
        # be called, but we may want force redetermining the size.
        if pixel_ratio != self._pixel_ratio:
            self._determine_size()

    # API

    def get_surface_info(self):
        return get_surface_info(self._window)

    def get_pixel_ratio(self):
        return self._pixel_ratio

    def get_logical_size(self):
        return self._logical_size

    def get_physical_size(self):
        return self._physical_size

    def set_logical_size(self, width, height):
        if width < 0 or height < 0:
            raise ValueError("Window width and height must not be negative")
        self._set_logical_size((float(width), float(height)))

    def set_title(self, title):
        glfw.set_window_title(self._window, title)

    def _request_draw(self):
        if not self._request_draw_timer_running:
            self._request_draw_timer_running = True
            call_later(self._get_draw_wait_time(), self._mark_ready_for_draw)

    def close(self):
        if self._window is not None:
            glfw.set_window_should_close(self._window, True)
            self._check_close()

    def is_closed(self):
        return self._window is None

    # User events

    def _on_mouse_button(self, window, but, action, mods):
        # Map button being changed, which we use to update self._pointer_buttons.
        button_map = {
            glfw.MOUSE_BUTTON_1: 1,  # == MOUSE_BUTTON_LEFT
            glfw.MOUSE_BUTTON_2: 2,  # == MOUSE_BUTTON_RIGHT
            glfw.MOUSE_BUTTON_3: 3,  # == MOUSE_BUTTON_MIDDLE
            glfw.MOUSE_BUTTON_4: 4,
            glfw.MOUSE_BUTTON_5: 5,
            glfw.MOUSE_BUTTON_6: 6,
            glfw.MOUSE_BUTTON_7: 7,
            glfw.MOUSE_BUTTON_8: 8,
        }
        button = button_map.get(but, 0)

        if action == glfw.PRESS:
            event_type = "pointer_down"
            buttons = set(self._pointer_buttons)
            buttons.add(button)
            self._pointer_buttons = tuple(sorted(buttons))
        elif action == glfw.RELEASE:
            event_type = "pointer_up"
            buttons = set(self._pointer_buttons)
            buttons.discard(button)
            self._pointer_buttons = tuple(sorted(buttons))
        else:
            return

        ev = {
            "event_type": event_type,
            "x": self._pointer_pos[0],
            "y": self._pointer_pos[1],
            "button": button,
            "buttons": tuple(self._pointer_buttons),
            "modifiers": tuple(self._key_modifiers),
            "ntouches": 0,  # glfw does not have touch support
            "touches": {},
        }

        # Emit the current event
        self._handle_event_and_flush(ev)

        # Maybe emit a double-click event
        self._follow_double_click(action, button)

    def _follow_double_click(self, action, button):
        # If a sequence of down-up-down-up is made in nearly the same
        # spot, and within a short time, we emit the double-click event.

        x, y = self._pointer_pos[0], self._pointer_pos[1]
        state = self._double_click_state

        timeout = 0.25
        distance = 5

        # Clear the state if it does no longer match
        if state["clicks"] > 0:
            d = ((x - state["x"]) ** 2 + (y - state["y"]) ** 2) ** 0.5
            if (
                d > distance
                or time.perf_counter() - state["time"] > timeout
                or button != state["button"]
            ):
                self._double_click_state = {"clicks": 0}

        clicks = self._double_click_state["clicks"]

        # Check and update order. Emit event if we make it to the final step
        if clicks == 0 and action == glfw.PRESS:
            self._double_click_state = {
                "clicks": 1,
                "button": button,
                "time": time.perf_counter(),
                "x": x,
                "y": y,
            }
        elif clicks == 1 and action == glfw.RELEASE:
            self._double_click_state["clicks"] = 2
        elif clicks == 2 and action == glfw.PRESS:
            self._double_click_state["clicks"] = 3
        elif clicks == 3 and action == glfw.RELEASE:
            self._double_click_state = {"clicks": 0}
            ev = {
                "event_type": "double_click",
                "x": self._pointer_pos[0],
                "y": self._pointer_pos[1],
                "button": button,
                "buttons": tuple(self._pointer_buttons),
                "modifiers": tuple(self._key_modifiers),
                "ntouches": 0,  # glfw does not have touch support
                "touches": {},
            }
            self._handle_event_and_flush(ev)

    def _on_cursor_pos(self, window, x, y):
        # Store pointer position in logical coordinates
        if self._screen_size_is_logical:
            self._pointer_pos = x, y
        else:
            self._pointer_pos = x / self._pixel_ratio, y / self._pixel_ratio

        ev = {
            "event_type": "pointer_move",
            "x": self._pointer_pos[0],
            "y": self._pointer_pos[1],
            "button": 0,
            "buttons": tuple(self._pointer_buttons),
            "modifiers": tuple(self._key_modifiers),
            "ntouches": 0,  # glfw does not have touch support
            "touches": {},
        }

        match_keys = {"buttons", "modifiers", "ntouches"}
        accum_keys = {}
        self._handle_event_rate_limited(ev, call_later, match_keys, accum_keys)

    def _on_scroll(self, window, dx, dy):
        # wheel is 1 or -1 in glfw, in jupyter_rfb this is ~100
        ev = {
            "event_type": "wheel",
            "dx": 100.0 * dx,
            "dy": -100.0 * dy,
            "x": self._pointer_pos[0],
            "y": self._pointer_pos[1],
            "buttons": tuple(self._pointer_buttons),
            "modifiers": tuple(self._key_modifiers),
        }
        match_keys = {"modifiers"}
        accum_keys = {"dx", "dy"}
        self._handle_event_rate_limited(ev, call_later, match_keys, accum_keys)

    def _on_key(self, window, key, scancode, action, mods):
        modifier = KEY_MAP_MOD.get(key, None)

        if action == glfw.PRESS:
            event_type = "key_down"
            if modifier:
                modifiers = set(self._key_modifiers)
                modifiers.add(modifier)
                self._key_modifiers = tuple(sorted(modifiers))
        elif action == glfw.RELEASE:
            event_type = "key_up"
            if modifier:
                modifiers = set(self._key_modifiers)
                modifiers.discard(modifier)
                self._key_modifiers = tuple(sorted(modifiers))
        else:  # glfw.REPEAT
            return

        # Note that if the user holds shift while pressing "5", will result in "5",
        # and not in the "%" that you'd expect on a US keyboard. Glfw wants us to
        # use set_char_callback for text input, but then we'd only get an event for
        # key presses (down followed by up). So we accept that GLFW is less complete
        # in this respect.
        if key in KEY_MAP:
            keyname = KEY_MAP[key]
        else:
            try:
                keyname = chr(key)
            except ValueError:
                return  # Probably a special key that we don't have in our KEY_MAP
            if "Shift" not in self._key_modifiers:
                keyname = keyname.lower()

        ev = {
            "event_type": event_type,
            "key": keyname,
            "modifiers": tuple(self._key_modifiers),
        }
        self._handle_event_and_flush(ev)

    def _on_char(self, window, char):
        # Undocumented char event to make imgui work, see https://github.com/pygfx/wgpu-py/issues/530
        ev = {
            "event_type": "char",
            "char_str": chr(char),
            "modifiers": tuple(self._key_modifiers),
        }
        self._handle_event_and_flush(ev)


# Make available under a name that is the same for all gui backends
WgpuCanvas = GlfwWgpuCanvas


class AppState:
    """Little container for state about the loop and glfw."""

    def __init__(self):
        self.all_glfw_canvases = weakref.WeakSet()
        self._loop = None
        self.stop_if_no_more_canvases = False
        self._glfw_initialized = False

    def init_glfw(self):
        glfw.init()  # Safe to call multiple times
        if not self._glfw_initialized:
            self._glfw_initialized = True
            atexit.register(glfw.terminate)

    def get_loop(self):
        if self._loop is None:
            self._loop = self._get_loop()
            self._loop.create_task(keep_glfw_alive())
        return self._loop

    def _get_loop(self):
        try:
            return asyncio.get_running_loop()
        except Exception:
            pass
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


app = AppState()


def update_glfw_canvasses():
    """Call this in your glfw event loop to draw each canvas that needs
    an update. Returns the number of visible canvases.
    """
    # Note that _draw_frame_and_present already catches errors, it can
    # only raise errors if the logging system fails.
    canvases = tuple(app.all_glfw_canvases)
    for canvas in canvases:
        if canvas._need_draw and not canvas._is_minimized:
            canvas._need_draw = False
            canvas._draw_frame_and_present()
    return len(canvases)


async def keep_glfw_alive():
    """Co-routine that lives forever, keeping glfw going.

    Although it stops the event-loop if there are no more canvases (and we're
    running the loop), this task stays active and continues when the loop is
    restarted.
    """
    # TODO: this is not particularly pretty. It'd be better to use normal asyncio to
    # schedule draws and then also process events. But let's address when we do #355 / #391
    while True:
        await asyncio.sleep(0.001)
        glfw.poll_events()
        n = update_glfw_canvasses()
        if app.stop_if_no_more_canvases and not n:
            loop = asyncio.get_running_loop()
            loop.stop()


def call_later(delay, callback, *args):
    loop = app.get_loop()
    loop.call_later(delay, callback, *args)


def run():
    loop = app.get_loop()
    if loop.is_running():
        return  # Interactive mode!

    app.stop_if_no_more_canvases = True
    loop.run_forever()
    app.stop_if_no_more_canvases = False
