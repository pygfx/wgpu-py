"""
Test the base canvas class.
"""

import sys
import subprocess

import numpy as np
import wgpu.gui
from testutils import run_tests, can_use_wgpu_lib
from pytest import mark, raises


def test_base_canvas_context():
    assert not issubclass(wgpu.gui.WgpuCanvasInterface, wgpu.GPUCanvasContext)
    assert hasattr(wgpu.gui.WgpuCanvasInterface, "get_context")


def test_base_canvas_cannot_use_context():
    canvas = wgpu.gui.WgpuCanvasInterface()
    with raises(NotImplementedError):
        wgpu.GPUCanvasContext(canvas)

    canvas = wgpu.gui.WgpuCanvasBase()
    with raises(NotImplementedError):
        canvas.get_context()


def test_canvas_get_context_needs_backend_to_be_selected():
    code = "from wgpu.gui import WgpuCanvasBase; canvas = WgpuCanvasBase(); canvas.get_context()"

    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    out = result.stdout.rstrip()

    assert "RuntimeError" in out
    assert "backend must be selected" in out.lower()
    assert "canvas.get_context" in out.lower()


class CanvasThatRaisesErrorsDuringDrawing(wgpu.gui.WgpuCanvasBase):
    def __init__(self):
        super().__init__()
        self._count = 0

    def _draw_frame(self):
        self._count += 1
        if self._count <= 4:
            self.foo_method()
        else:
            self.spam_method()

    def foo_method(self):
        self.bar_method()

    def bar_method(self):
        raise Exception("call-failed-" + "but-test-passed")

    def spam_method(self):
        msg = "intended-fail"  # avoid line with the message to show in the tb
        raise Exception(msg)


def test_canvas_logging(caplog):
    """As we attempt to draw, the canvas will error, which are logged.
    Each first occurrence is logged with a traceback. Subsequent same errors
    are much shorter and have a counter.
    """

    canvas = CanvasThatRaisesErrorsDuringDrawing()

    canvas._draw_frame_and_present()  # prints traceback
    canvas._draw_frame_and_present()  # prints short logs ...
    canvas._draw_frame_and_present()
    canvas._draw_frame_and_present()

    text = caplog.text
    assert text.count("bar_method") == 2  # one traceback => 2 mentions
    assert text.count("foo_method") == 2
    assert text.count("call-failed-but-test-passed") == 4
    assert text.count("(4)") == 1
    assert text.count("(5)") == 0

    assert text.count("spam_method") == 0
    assert text.count("intended-fail") == 0

    canvas._draw_frame_and_present()  # prints traceback
    canvas._draw_frame_and_present()  # prints short logs ...
    canvas._draw_frame_and_present()
    canvas._draw_frame_and_present()

    text = caplog.text
    assert text.count("bar_method") == 2  # one traceback => 2 mentions
    assert text.count("foo_method") == 2
    assert text.count("call-failed-but-test-passed") == 4

    assert text.count("spam_method") == 2
    assert text.count("intended-fail") == 4


class MyOffscreenCanvas(wgpu.gui.WgpuCanvasBase):
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.physical_size = 100, 100

    def get_present_info(self):
        return {
            "method": "image",
            "formats": ["rgba8unorm-srgb"],
        }

    def present_image(self, image, **kwargs):
        self.frame_count += 1
        self.array = np.frombuffer(image, np.uint8).reshape(image.shape)

    def get_pixel_ratio(self):
        return 1

    def get_logical_size(self):
        return self.get_physical_size()

    def get_physical_size(self):
        return self.physical_size


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_run_bare_canvas():
    """Test that a bare canvas does not error."""

    # This is (more or less) the equivalent of:
    #
    #     from wgpu.gui.auto import WgpuCanvas, loop
    #     canvas = WgpuCanvas()
    #     loop.run()
    #
    # Note: loop.run() calls _draw_frame_and_present() in event loop.

    canvas = MyOffscreenCanvas()
    canvas._draw_frame_and_present()


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_simpple_offscreen_canvas():
    canvas = MyOffscreenCanvas()
    device = wgpu.utils.get_default_device()
    present_context = canvas.get_context()
    present_context.configure(device=device, format=None)

    def draw_frame():
        # Note: we deliberately obtain the texture, and only create the view
        # where the dict is constructed below. This covers the case where
        # begin_render_pass() has to prevent the texture-view-object from being
        # deleted before its native handle is passed to wgpu-native.
        current_texture = present_context.get_current_texture()
        command_encoder = device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 1, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )
        render_pass.end()
        device.queue.submit([command_encoder.finish()])

    assert canvas.frame_count == 0

    canvas.request_draw(draw_frame)

    # Draw 1
    canvas.force_draw()
    assert canvas.array.shape == (100, 100, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # Draw 2
    canvas.force_draw()
    assert canvas.array.shape == (100, 100, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # Change resolution
    canvas.physical_size = 120, 100

    # Draw 3
    canvas.force_draw()
    assert canvas.array.shape == (100, 120, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # Change resolution
    canvas.physical_size = 120, 140

    # Draw 4
    canvas.force_draw()
    assert canvas.array.shape == (140, 120, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # We now have four unique texture objects
    assert canvas.frame_count == 4


def test_canvas_base_events():
    c = wgpu.gui.WgpuCanvasBase()

    # We test events extensively in another test module. This is just
    # to make sure that events are working for the base canvas.

    events = []

    def handler(event):
        events.append(event["value"])

    c.add_event_handler(handler, "key_down")
    c.submit_event({"event_type": "key_down", "value": 1})
    c.submit_event({"event_type": "key_down", "value": 2})
    c._events.flush()
    assert events == [1, 2]


if __name__ == "__main__":
    run_tests(globals())
