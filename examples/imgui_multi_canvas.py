"""
Example showing how to use multiple imgui contexts to draw to multiple canvases

# run_example = false
"""

import wgpu
from imgui_bundle import imgui
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer

# Create a canvas to render to
canvas1 = WgpuCanvas(title="imgui", size=(512, 256))
canvas2 = WgpuCanvas(title="imgui", size=(512, 256))
canvas3 = WgpuCanvas(title="imgui", size=(512, 256))

canvases = [canvas1, canvas2, canvas3]

# Create a wgpu device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

# create a imgui renderer for each canvas
imgui_renderer1 = ImguiRenderer(device, canvas1)
imgui_renderer2 = ImguiRenderer(device, canvas2)
imgui_renderer3 = ImguiRenderer(device, canvas3)


# Separate GUIs that are drawn to each canvas
def update_gui1():
    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.appearing)
    imgui.set_next_window_pos((0, 20), imgui.Cond_.appearing)

    imgui.begin("window1", None)
    imgui.button("b1")

    imgui.end()

    imgui.end_frame()
    imgui.render()

    return imgui.get_draw_data()


def update_gui2():
    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.appearing)
    imgui.set_next_window_pos((0, 20), imgui.Cond_.appearing)

    imgui.begin("window2", None)
    imgui.button("b2")

    imgui.end()

    imgui.end_frame()
    imgui.render()

    return imgui.get_draw_data()


def update_gui3():
    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.appearing)
    imgui.set_next_window_pos((0, 20), imgui.Cond_.appearing)

    imgui.begin("window3", None)
    imgui.button("b3")

    imgui.end()

    imgui.end_frame()
    imgui.render()

    return imgui.get_draw_data()


# give the corresponding gui updater functions to the imgui renderers
imgui_renderer1.set_gui(update_gui1)
imgui_renderer2.set_gui(update_gui2)
imgui_renderer3.set_gui(update_gui3)


# draw function for each canvas
def draw1():
    imgui_renderer1.render()
    canvas1.request_draw()


def draw2():
    imgui_renderer2.render()
    canvas2.request_draw()


def draw3():
    imgui_renderer3.render()
    canvas3.request_draw()


if __name__ == "__main__":
    canvas1.request_draw(draw1)
    canvas2.request_draw(draw2)
    canvas3.request_draw(draw3)

    run()
