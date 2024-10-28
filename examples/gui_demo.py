"""
An example that uses events to trigger some canvas functionality.

A nice demo, and very convenient to test the different backends.

* Can be closed with Escape or by pressing the window close button.
* In both cases, it should print "Close detected" exactly once.
* Hit space to spend 2 seconds doing direct draws.

"""

import time

from wgpu.gui.auto import WgpuCanvas, loop

from cube import setup_drawing_sync


canvas = WgpuCanvas(
    size=(640, 480),
    title="Canvas events on $backend - $fps fps",
    max_fps=10,
    update_mode="continuous",
    present_method="",
)


draw_frame = setup_drawing_sync(canvas)
canvas.request_draw(lambda: (draw_frame(), canvas.request_draw()))


@canvas.add_event_handler("*")
def process_event(event):
    if event["event_type"] not in ["pointer_move", "before_draw", "animate"]:
        print(event)

    if event["event_type"] == "key_down":
        if event["key"] == "Escape":
            canvas.close()
        elif event["key"] == " ":
            etime = time.time() + 2
            i = 0
            while time.time() < etime:
                i += 1
                canvas.force_draw()
            print(f"force-drawed {i} frames in 2s.")
    elif event["event_type"] == "close":
        # Should see this exactly once, either when pressing escape, or
        # when pressing the window close button.
        print("Close detected!")
        assert canvas.is_closed()


if __name__ == "__main__":
    loop.run()
