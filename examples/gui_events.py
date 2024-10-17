"""
A simple example to demonstrate events.

Also serves as a test-app for the canvas backends.
"""

from wgpu.gui.auto import WgpuCanvas, loop


canvas = WgpuCanvas(size=(640, 480), title="wgpu events")


@canvas.add_event_handler("*")
def process_event(event):
    if event["event_type"] not in ["pointer_move", "before_draw", "animate"]:
        print(event)

    if event["event_type"] == "key_down":
        if event["key"] == "Escape":
            canvas.close()
    elif event["event_type"] == "close":
        # Should see this exactly once, either when pressing escape, or
        # when pressing the window close button.
        print("Close detected!")
        assert canvas.is_closed()


if __name__ == "__main__":
    loop.run()
