"""
A simple example to demonstrate events.
"""
try:
    # Try importing PySide6 to use PySide6 for
    # auto gui. If PySide6 can't be imported
    # then glfw is used as fallback
    import PySide6  # noqa
except ImportError:
    pass
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
