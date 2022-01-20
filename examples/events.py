"""
A simple example to demonstrate events.
"""
from wgpu.gui.auto import WgpuCanvas, run, call_later


class MyCanvas(WgpuCanvas):
    def handle_event(self, event):
        if event["event_type"] != "pointer_move":
            print(event)


if __name__ == "__main__":
    canvas = MyCanvas(size=(640, 480), title="wgpu events")

    def send_message(message):
        print(f"Message: {message}")

    call_later(2, send_message, "hello")

    run()
