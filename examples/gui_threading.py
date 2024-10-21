"""
Example that renders frames in a separate thread.

This uses an offscreen canvas, the result is only used to print the
frame shape. But one can see how one can e.g. render a movie this way.

Threaded rendering using a real GUI is not supported right now, since
this is tricky to do with both Qt and glfw. Plus in general its a bad
idea to run your UI in anything other than the main thread. In other
words, you should probably only use threaded rendering for off-screen
stuff.

"""

# test_example = true

import time
import threading

from wgpu.gui.offscreen import WgpuCanvas

from cube import setup_drawing_sync


# create canvas
canvas = WgpuCanvas()
draw_frame = setup_drawing_sync(canvas)


def main():
    frame_count = 0
    canvas.request_draw(draw_frame)

    while not canvas.is_closed():
        image = canvas.draw()
        frame_count += 1
        print(f"Rendered {frame_count} frames, last shape is {image.shape}")


if __name__ == "__main__":
    t1 = threading.Thread(target=main)
    t1.start()

    # In the main thread, we wait a little
    time.sleep(1)

    # ... then change the canvas size, and wait some more
    canvas.set_logical_size(200, 200)
    time.sleep(1)

    # Close the canvas to stop the tread
    canvas.close()
