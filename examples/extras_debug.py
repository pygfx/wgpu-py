"""
Basic example of how to use wgpu-native instance extras to enable debug symbols.
As debugger we will use RenderDoc (https://renderdoc.org/) - other tools will require a similar setup.
While RenderDoc doesn't support WebGPU - it still works to inspect the Pipeline or with translated shaders.
"""

# run_example = false

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from rendercanvas.auto import RenderCanvas, loop

# this skript has two behaviours, either be an example - or launch RenderDoc.


def setup_demo():
    """
    this is inside a function so it's only called later.
    """
    # before we enumerate or request a device, we need to set the instance extras
    # as we import from the base examples, those do the request_device call
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    # this communicates with the compiler to enable debug symbols.
    # I have confirmed this works for Vulkan and Dx12, however it seems like it's always enabled for Fxc and doesn't work for Dxc
    # OpenGL sorta works, but even GLSL shader code gets translated by naga, so the code is messed up but symbols are part way still there.
    # TODO can someone test this on Metal?
    set_instance_extras(
        flags=["Debug"]  # an additional option here is "Validation".
    )

    # TODO: replace the default examples by including additional debug markers using Encoder.inser_debug_marker(label)
    try:
        from .cube import setup_drawing_sync
    except ImportError:
        from cube import setup_drawing_sync

    canvas = RenderCanvas(title="Cube example with debug symbols")
    draw_frame = setup_drawing_sync(canvas)

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


def renderdoc_launcher():
    """
    This writes a temporary .cap file which contains all the renderdoc capture setup.
    Then launches the gui.
    """
    cap_settings = {
        "rdocCaptureSettings": 1,
        "settings": {
            "autoStart": "true",
            "commandLine": str(Path(__file__).name),
            "environment": [
                {
                    "separator": "Platform style",
                    "type": "Set",
                    "value": "Vulkan",  # not required but you can set something else here!
                    "variable": "WGPU_BACKEND_TYPE",
                },
                {
                    "separator": "Platform style",
                    "type": "Set",
                    "value": 1,
                    "variable": "RENDERDOC_CAPTURE",  # this is used specifically for this example to avoid a fork bomb.
                },
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

    # relies on having associated the .cap file with RenderDoc in Windows?
    # TODO: other OS???
    # is this now a child process or can the python script end?
    subprocess.run(["start", cap_file.name], shell=True)
    # TODO: cleanup tempfiles?


if __name__ == "__main__":
    # awful hack: if the script is run by a user, we write the tempfile to then run the launcher and auto catpure
    # while the capture itself has an envvar to launch the gui instead.
    is_renderdoc = os.environ.get("RENDERDOC_CAPTURE", "0") == "1"
    if is_renderdoc:
        setup_demo()
        loop.run()
    else:
        renderdoc_launcher()
        print("Should have opened the RenderDoc GUI, python process should close")

# The capture should have opened now, on the left hand side in the Event Browser find the `vkCmdDrawIndexed` event. Click to open this event.
# Now the Pipeline State should let you chose either Vertex Shader or Fragment Shader and see a button called "View" next to the ShaderModule.
# If you see the source code from the example including symbols and comments then it worked!
# Note that using Dx12 will not show the the same source, as naga translated the shader to HLSL.
# as RenderDoc doesn't support WGSL, you won't be able to edit or step the source directly with Vulkan.
