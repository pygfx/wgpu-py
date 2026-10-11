"""
ImGui run_sync test
-------------------

trying to minimize the problem with fpl
"""
from wgpu.utils.device import get_default_device
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui
from rendercanvas.auto import RenderCanvas, loop
import asyncio

from pyodide.ffi import run_sync

canvas = RenderCanvas(title="imgui run_sync test", size=(640, 480), max_fps=60, update_mode="continuous")
device = get_default_device()
imgui_renderer = ImguiRenderer(device, canvas)

async def noop():
    await asyncio.sleep(0.2) # leads to a SystemError: <built-in function run_sync> returned NULL without setting an exception
    print("slept in noop")

def update_gui():
    imgui.begin("window text")
    if imgui.button("danger"):
        run_sync(noop())

    imgui.end()

imgui_renderer.set_gui(update_gui)


if __name__ == "__main__":
    canvas.request_draw(imgui_renderer.render)
    loop.run()
