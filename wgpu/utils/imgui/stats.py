import time

from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui
import numpy as np


class Stats:
    """A Stats helper which displays performance statistics such
    as FPS and draw time on the screen.

    """

    def __init__(
        self,
        device,
        canvas,
    ):
        self._foreground = imgui.ImVec4(0, 1, 0, 1)
        self._background = imgui.ImVec4(0, 0.2, 0, 0.5)

        self._renderer = ImguiRenderer(device, canvas)

        self._renderer.set_gui(self._draw_imgui)

        canvas.add_event_handler(self._on_mouse, "pointer_down", order=-100)

        # flag used to skip the first frame
        # which typically has all the CPU->GPU transfer and
        # shader compilation overhead
        self._init = False

        # performance trackers
        self._tmin = 1e10
        self._tmax = 0
        self._tbegin = None
        self._tprev = self._tbegin
        self._frames = 0
        self._fmin = 1e10
        self._fmax = 0
        # Sentinel value of None indicates that the fps has never been computed
        self._fps = None

        self._fps_samples = np.zeros(100, dtype=np.float32)
        self._ms_samples = np.zeros(100, dtype=np.float32)

        self._mode = 0

    def _draw_imgui(self):
        imgui.new_frame()

        imgui.set_next_window_size((150, 0), imgui.Cond_.always)
        imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

        imgui.push_style_color(imgui.Col_.window_bg, self._background)

        imgui.begin(
            "stats",
            True,
            flags=imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_title_bar,
        )

        imgui.push_style_color(imgui.Col_.text, self._foreground)
        if self._mode == 0:
            if self._fps is not None:
                ms = self._ms_samples[-1]
                text = f"{ms} ms ({self._tmin}-{self._tmax})"
                text += f"\n{self._fps} fps ({self._fmin}-{self._fmax})"
                imgui.text(text)

        elif self._mode == 1:
            imgui.text(f"{self._fps} fps ({self._fmin}-{self._fmax})")
            imgui.plot_histogram("##", self._fps_samples, graph_size=(120, 30))
        elif self._mode == 2:
            ms = self._ms_samples[-1]
            imgui.text(f"{ms} ms ({self._tmin}-{self._tmax})")
            imgui.plot_lines("##", self._ms_samples, graph_size=(120, 30))

        imgui.pop_style_color()

        imgui.end()

        imgui.pop_style_color()

        imgui.end_frame()
        imgui.render()
        return imgui.get_draw_data()

    def _on_mouse(self, event):
        self._mode = (self._mode + 1) % 3

    def start(self):
        if not self._init:
            return

        self._tbegin = time.perf_counter_ns()
        if self._tprev is None:
            self._tprev = self._tbegin

    def stop(self):
        if not self._init:
            self._init = True
            return

        t = time.perf_counter_ns()
        self._frames += 1

        delta = round((t - self._tbegin) / 1_000_000)
        self._tmin = min(self._tmin, delta)
        self._tmax = max(self._tmax, delta)

        if t >= self._tprev + 1_000_000_000:
            # update FPS counter whenever a second has passed
            fps = round(self._frames / ((t - self._tprev) / 1_000_000_000))
            self._tprev = t
            self._frames = 0
            self._fmin = min(self._fmin, fps)
            self._fmax = max(self._fmax, fps)
            self._fps = fps

            # update fps samples, remove last element
            self._fps_samples = np.roll(self._fps_samples, -1)
            self._fps_samples[-1] = fps

        self._ms_samples = np.roll(self._ms_samples, -1)
        self._ms_samples[-1] = delta

    def render(self):
        self._renderer.render()

    def __enter__(self):
        self.start()

    def __exit__(self, *exc):
        self.stop()
