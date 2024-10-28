"""
A simple example to demonstrate events.
"""

from wgpu.gui.auto import WgpuCanvas, loop

from cube import setup_drawing_sync


canvas = WgpuCanvas(
    size=(640, 480), title="Canvas events on $backend", update_mode="continuous"
)


draw_frame = setup_drawing_sync(canvas)
canvas.request_draw(draw_frame)


@canvas.add_event_handler("*")
def process_event(event):
    if event["event_type"] not in ["pointer_move", "before_draw", "animate"]:
        print(event)


if __name__ == "__main__":
    loop.run()
