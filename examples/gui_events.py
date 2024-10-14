"""
A simple example to demonstrate events.
"""

from wgpu.gui.auto import WgpuCanvas, loop


canvas = WgpuCanvas(size=(640, 480), title="wgpu events")


@canvas.events.add_handler("*")
def process_event(event):
    if event["event_type"] not in ["pointer_move", "before_draw"]:
        print(event)


if __name__ == "__main__":
    loop.run()
