"""
Extras Debug
------------

Basic example of how to use wgpu-native instance extras to enable debug symbols and labels in the shader compiler.
As debugger we will use RenderDoc (https://renderdoc.org/) - other tools will require a similar setup.
While RenderDoc doesn't fully support WebGPU - it can still be useful for inspecting the render pipeline.
RenderDoc also doesn't support WGSL, so it will work off the naga translated shaders for debug stepping and editing.
Using DX12 (HLSL) or OpenGL (GLSL) gives a better decompilation experience compared to Vulkan (SPIR-V).
The Vulkan research structure most closely matches WebGPU.
"""

# run_example = false

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from rendercanvas.auto import RenderCanvas, loop

# this script has two behaviours: launch RenderDoc and run the example.


def renderdoc_launcher():
    """
    This writes a temporary .cap file which contains all the renderdoc capture setup.
    Then launches the gui.
    """
    # see https://renderdoc.org/docs/window/capture_attach.html for explanation
    # and https://renderdoc.org/docs/python_api/qrenderdoc/main.html#qrenderdoc.CaptureSettings for details
    # the following settings work for me, although variations should mostly work.
    cap_settings = {
        "rdocCaptureSettings": 1,
        "settings": {
            "autoStart": "true",
            "commandLine": str(Path(__file__).name)
            + " example",  # by adding an argument we signal that the script should run the example
            "environment": [
                {
                    "separator": "Platform style",
                    "type": "Set",
                    "value": "Vulkan",  # change the graphics Backend(API) here: "D3D12", "OpenGL", ...
                    "variable": "WGPU_BACKEND_TYPE",
                },
                # this env var uses the debug build of wgpu-native, it's not required here as this will just add symbols for rust panic backtraces for example
                # {
                #     "separator": "Platform style",
                #     "type": "Set",
                #     "value": 1,
                #     "variable": "WGPU_DEBUG",
                # },
            ],
            "executable": str(sys.executable),
            "inject": "false",
            "numQueuedFrames": 1,
            "options": {
                "allowFullscreen": "true",
                "allowVSync": "true",
                "apiValidation": "false",
                "captureAllCmdLists": "false",
                "captureCallstacks": "false",
                "captureCallstacksOnlyDraws": "false",
                "debugOutputMute": "true",
                "delayForDebugger": 0,
                "hookIntoChildren": "true",
                "refAllResources": "false",
                "softMemoryLimit": 0,
                "verifyBufferAccess": "false",
            },
            "queuedFrameCap": 50,
            "workingDir": str(Path(__file__).parent),
        },
    }

    cap_str = json.dumps(cap_settings, indent=4)
    cap_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cap", delete=False, encoding="utf-8"
    )
    with open(cap_file.name, "w") as f:
        f.write(cap_str)
    cap_file.close()  # doesn't the contextmanager make sure the file is closed and not blocked?

    # relies on having associated the .cap file with RenderDoc
    # TODO: other OS - untested
    if sys.platform.startswith("win"):
        # apparently this should be none blocking and safer
        os.startfile(cap_file.name)
    elif sys.platform.startswith("darwin"):  # macOS
        subprocess.Popen(["open", cap_file.name])
    else:  # likely Linux
        subprocess.Popen(["xdg-open", cap_file.name])
    # TODO: cleanup tempfiles?


def setup_demo():
    """
    this is inside a function so it's only called later. Similar to other examples
    """
    # before we enumerate or request a device, we need to set instance extras
    # as we import from the base examples, those do the request_device call
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    # this communicates with the compiler to enable debug symbols.
    # I have confirmed this works for Vulkan and Dx12, however it seems like it's always enabled for Fxc and doesn't work for Dxc
    # OpenGL sorta works, but even GLSL shader code gets translated by naga, so the code is messed up but symbols are part way still there.
    # TODO can someone test this on Metal?
    set_instance_extras(
        flags=["Debug"]  # an additional option here is "Validation".
    )

    try:
        from .cube import setup_drawing_sync
    except ImportError:
        from cube import setup_drawing_sync

    canvas = RenderCanvas(title="Cube example with debug symbols")
    draw_frame = setup_drawing_sync(canvas.get_wgpu_context())

    # we set the auto capture on frame 50, so after 100 frames gui should exit
    # this will lead to RenderDoc automatically opening the capture!
    frame_count = 0

    @canvas.request_draw
    def animate():
        nonlocal frame_count
        frame_count += 1
        draw_frame()
        canvas.request_draw()
        if frame_count > 100:
            print("Stopping the loop after 100 frames")
            canvas.close()


if __name__ == "__main__":
    # if the script is run by a user, we write the tempfile to then run the launcher with auto capture
    # to know if the script is launched by the launcher, we call it with an argument
    if len(sys.argv) == 1:  # essentially means no arguments provided
        renderdoc_launcher()
        print("Should have opened the RenderDoc GUI, python process should close")
    else:
        setup_demo()
        loop.run()

# The capture should have opened now, on the left hand side in the Event Browser find the `vkCmdDrawIndexed` event. Inside the render pass inside the debug group.
# Clicking the timer button at the top of the Event Browser helps to locate interesting events. https://renderdoc.org/docs/window/event_browser.html#timing-actions
# Now the Pipeline State tab should let you chose either Vertex Shader or Fragment Shader and see a button called "View" next to the ShaderModule.
# If you see the source code from the example including symbols and comments then it worked!

# known issues:
# Other backends don't always capture automatically. Make sure to press F11 to cycle if the overlays says so.
# If the child process isn't automatically hooked, select the `python.exe` from the Child Processes list manually.
# the auto capture and auto exit might be too quick, so increase those values don't capture manually.
