"""
Test the canvas basics.
"""

import numpy as np
import wgpu.gui  # noqa
from testutils import run_tests, can_use_wgpu_lib
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
    assert not issubclass(
        wgpu.gui.WgpuCanvasInterface, wgpu.base.GPUPresentationContext
    )
    assert hasattr(wgpu.gui.WgpuCanvasInterface, "get_context")
    # Provides good default already
    canvas = wgpu.gui.WgpuCanvasInterface()
    ctx = wgpu.GPUPresentationContext(canvas)
    assert ctx.get_preferred_format(None) == "bgra8unorm-srgb"


def test_canvas_logging(caplog):
    """As we attempt to draw, the canvas will error, which are logged.
    Each first occurance is logged with a traceback. Subsequent same errors
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


class MyOffscreenCanvas(wgpu.gui.WgpuOffscreenCanvas):
    def get_pixel_ratio(self):
        return 1

    def get_logical_size(self):
        return self.get_physical_size()

    def get_physical_size(self):
        return 100, 100

    def _request_draw(self):
        # Note: this would normaly schedule a call in a later event loop iteration
        self._draw_frame_and_present()

    def present(self, texture_view):
        device = texture_view._device
        size = texture_view.size
        bytes_per_pixel = 4
        data = device.queue.read_texture(
            {
                "texture": texture_view.texture,
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
def test_offscreen_canvas():

    canvas = MyOffscreenCanvas()
    device = wgpu.utils.get_default_device()
    present_context = canvas.get_context()
    present_context.configure(device=device, format=None)

    @canvas.request_draw
    def draw_frame():
        current_texture_view = present_context.get_current_texture()
        command_encoder = device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
                    "resolve_target": None,
                    "load_value": (0, 1, 0, 1),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )
        render_pass.end_pass()
        device.queue.submit([command_encoder.finish()])

    assert canvas.array.shape == (100, 100, 4)
    assert np.all(canvas.array[:, :, 0] == 0)
    assert np.all(canvas.array[:, :, 1] == 255)


if __name__ == "__main__":
    run_tests(globals())
