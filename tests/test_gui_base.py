"""
Test the canvas basics.
"""

import wgpu.gui  # noqa


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
    assert issubclass(wgpu.gui.WgpuCanvasInterface, wgpu.base.GPUCanvasContext)
    # Provides good default already
    ctx = wgpu.GPUCanvasContext()
    assert ctx.get_swap_chain_preferred_format(None) == "bgra8unorm-srgb"


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
