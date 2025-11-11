"""
Direct integration of sdl3 and wgpu-py without using the RenderCanvas library.

Demonstration for hardcore users that need total low-level control.

"""

# run_example = false

import time
import atexit

import sdl3
import ctypes
import wgpu
# from wgpu.utils.glfw_present_info import get_glfw_present_info

# from triangle import setup_drawing_sync
from cube import setup_drawing_sync


sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO)
atexit.register(sdl3.SDL_Quit)

window = sdl3.SDL_CreateWindow(b"Hello World", 640, 480, 0)

win_props_id = sdl3.SDL_GetWindowProperties(window)
hwnd = sdl3.SDL_GetPointerProperty(win_props_id, sdl3.SDL_PROP_WINDOW_WIN32_HWND_POINTER, ctypes.c_voidp())
import pdb; pdb.set_trace()


# # Setup glfw
# glfw.init()
# atexit.register(glfw.terminate)

# # disable automatic API selection, we are not using opengl
# glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
# glfw.window_hint(glfw.RESIZABLE, True)


# title = "wgpu glfw direct"
# window = glfw.create_window(640, 480, title, None, None)
# present_info = get_glfw_present_info(window)

# context = wgpu.gpu.get_canvas_context(present_info)

# # Initialize physical size once. For robust apps update this on resize events.
# context.set_physical_size(*glfw.get_framebuffer_size(window))


def main():
#     draw_frame = setup_drawing_sync(context)

    last_frame_time = time.perf_counter()
    frame_count = 0

    # render loop
    quit_requested = False
    event = sdl3.SDL_Event()
    while not quit_requested:
        # process inputs
        while sdl3.SDL_PollEvent(ctypes.byref(event)):
            if event.type == sdl3.SDL_QuitEvent:
                quit_requested = True
            elif event.type == sdl3.SDL_EVENT_KEY_DOWN:
                if event.key.key == sdl3.SDLK_ESCAPE:
                    quit_requested = True

#         # resize handling
#         context.set_physical_size(*glfw.get_framebuffer_size(window))

#         # draw a frame
#         draw_frame()
#         # present the frame to the screen
#         context.present()

        # stats
        frame_count += 1
        etime = time.perf_counter() - last_frame_time
        if etime > 1:
            print(f"{frame_count / etime:0.1f} FPS")
            last_frame_time, frame_count = time.perf_counter(), 0

    # dispose resources
    # context.unconfigure()
    sdl3.SDL_DestroyWindow(window)


if __name__ == "__main__":
    main()
