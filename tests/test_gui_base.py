"""
Test the canvas basics.
"""

import gc
import sys
import subprocess

import numpy as np
import wgpu.gui  # noqa
from testutils import run_tests, can_use_wgpu_lib, is_pypy
from pytest import mark


class TheTestCanvas(wgpu.gui.WgpuCanvasBase):
    def __init__(self):
        super().__init__()
        self._count = 0

    def draw_frame(self):
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
        1 / 0


def test_base_canvas_context():
    assert not issubclass(wgpu.gui.WgpuCanvasInterface, wgpu.GPUCanvasContext)
    assert hasattr(wgpu.gui.WgpuCanvasInterface, "get_context")
    # Provides good default already
    canvas = wgpu.gui.WgpuCanvasInterface()
    ctx = wgpu.GPUCanvasContext(canvas)
    assert ctx.get_preferred_format(None) == "bgra8unorm-srgb"


def test_canvas_logging(caplog):
    """As we attempt to draw, the canvas will error, which are logged.
    Each first occurrence is logged with a traceback. Subsequent same errors
    are much shorter and have a counter.
    """

    canvas = TheTestCanvas()

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
    assert text.count("division by zero") == 0

    canvas._draw_frame_and_present()  # prints traceback
    canvas._draw_frame_and_present()  # prints short logs ...
    canvas._draw_frame_and_present()
    canvas._draw_frame_and_present()

    text = caplog.text
    assert text.count("bar_method") == 2  # one traceback => 2 mentions
    assert text.count("foo_method") == 2
    assert text.count("call-failed-but-test-passed") == 4

    assert text.count("spam_method") == 2
    assert text.count("division by zero") == 4


class MyOffscreenCanvas(wgpu.gui.WgpuOffscreenCanvasBase):
    def __init__(self):
        super().__init__()
        self.textures = []
        self.physical_size = 100, 100

    def get_pixel_ratio(self):
        return 1

    def get_logical_size(self):
        return self.get_physical_size()

    def get_physical_size(self):
        return self.physical_size

    def _request_draw(self):
        # Note: this would normally schedule a call in a later event loop iteration
        self._draw_frame_and_present()

    def present(self, texture):
        self.textures.append(texture)
        device = texture._device
        size = texture.size
        bytes_per_pixel = 4
        data = device.queue.read_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )
        self.array = np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_run_bare_canvas():
    """Test that a bare canvas does not error."""

    # This is (more or less) the equivalent of:
    #
    #     from wgpu.gui.auto import WgpuCanvas, run
    #     canvas = WgpuCanvas()
    #     run()
    #
    # Note: run() calls _draw_frame_and_present() in event loop.

    canvas = MyOffscreenCanvas()
    canvas._draw_frame_and_present()


def test_canvas_context_not_base():
    """Check that it is prevented that canvas context is instance of base context class."""
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


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_offscreen_canvas():
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

    assert len(canvas.textures) == 0

    # Draw 1
    canvas.request_draw(draw_frame)
    assert canvas.array.shape == (100, 100, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # Draw 2
    canvas.request_draw(draw_frame)
    assert canvas.array.shape == (100, 100, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # Change resolution
    canvas.physical_size = 120, 100

    # Draw 3
    canvas.request_draw(draw_frame)
    assert canvas.array.shape == (100, 120, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # Change resolution
    canvas.physical_size = 120, 140

    # Draw 4
    canvas.request_draw(draw_frame)
    assert canvas.array.shape == (140, 120, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)

    # We now have four unique texture objects
    assert len(canvas.textures) == 4
    assert len(set(canvas.textures)) == 4


def test_autogui_mixin():
    c = wgpu.gui.WgpuAutoGui()

    # It's a mixin
    assert not isinstance(c, wgpu.gui.WgpuCanvasBase)

    # It's event handling mechanism should be fully functional

    events = []

    def handler(event):
        events.append(event["value"])

    c.add_event_handler(handler, "foo", "bar")
    c.handle_event({"event_type": "foo", "value": 1})
    c.handle_event({"event_type": "bar", "value": 2})
    c.handle_event({"event_type": "spam", "value": 3})
    c.remove_event_handler(handler, "foo")
    c.handle_event({"event_type": "foo", "value": 4})
    c.handle_event({"event_type": "bar", "value": 5})
    c.handle_event({"event_type": "spam", "value": 6})
    c.remove_event_handler(handler, "bar")
    c.handle_event({"event_type": "foo", "value": 7})
    c.handle_event({"event_type": "bar", "value": 8})
    c.handle_event({"event_type": "spam", "value": 9})

    assert events == [1, 2, 5]


def test_weakbind():
    weakbind = wgpu.gui._gui_utils.weakbind

    xx = []

    class Foo:
        def bar(self):
            xx.append(1)

    f1 = Foo()
    f2 = Foo()

    b1 = f1.bar
    b2 = weakbind(f2.bar)

    assert len(xx) == 0
    b1()
    assert len(xx) == 1
    b2()
    assert len(xx) == 2

    del f1
    del f2

    if is_pypy:
        gc.collect()

    assert len(xx) == 2
    b1()
    assert len(xx) == 3  # f1 still exists
    b2()
    assert len(xx) == 3  # f2 is gone!


if __name__ == "__main__":
    run_tests(globals())
