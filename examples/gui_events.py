"""
A simple example to demonstrate events.
"""

from rendercanvas.auto import RenderCanvas, loop


canvas = RenderCanvas(size=(640, 480), title="wgpu events")


@canvas.add_event_handler("*")
def process_event(event):
    if event["event_type"] != "pointer_move":
        print(event)


if __name__ == "__main__":
    loop.run()
