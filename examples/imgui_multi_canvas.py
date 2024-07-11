"""
Example showing how to use multiple imgui contexts to draw to multiple canvases

# run_example = false
"""

import wgpu
from imgui_bundle import imgui
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer

# Create a canvas to render to
canvas = WgpuCanvas(title="imgui", size=(512, 256))
canvas2 = WgpuCanvas(title="imgui", size=(512, 256))
canvas3 = WgpuCanvas(title="imgui", size=(512, 256))

# Create a wgpu device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

# create a imgui renderer for each canvas
imgui_renderer = ImguiRenderer(device, canvas)
imgui_renderer2 = ImguiRenderer(device, canvas2)
imgui_renderer3 = ImguiRenderer(device, canvas3)


# Separate GUIs that are drawn to each canvas
def update_gui():
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


def draw_frame():
    # set corresponding imgui context before rendering
    imgui.set_current_context(imgui_renderer.imgui_context)
    imgui_renderer.render(update_gui())

    imgui.set_current_context(imgui_renderer2.imgui_context)
    imgui_renderer2.render(update_gui2())

    imgui.set_current_context(imgui_renderer3.imgui_context)
    imgui_renderer3.render(update_gui3())

    # done! request draw
    canvas.request_draw()
    canvas2.request_draw()
    canvas3.request_draw()


if __name__ == "__main__":
    canvas.request_draw(draw_frame)
    run()
