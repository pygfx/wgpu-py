"""
Render offscreen using a 16bit (HDR) render target
--------------------------------------------------

Render a wgpu example offscreen, safe to 16bit PNG, and show the result.

"""

import os
import tempfile
import webbrowser

import numpy as np
import png  # provided by the pypng package

# from rendercanvas.offscreen import RenderCanvas
from wgpu.gui.offscreen import WgpuCanvas as RenderCanvas
from triangle import setup_drawing_sync


canvas = RenderCanvas(size=(640, 480), pixel_ratio=2)
draw_frame = setup_drawing_sync(canvas, format="rgba16float")
canvas.request_draw(draw_frame)

image = canvas.draw()
image = np.asarray(image)

# Convert to RGB
image = image[:, :, :3]

# Convert float16 to uint16
image = (image.astype(np.float32) * 65535).astype("uint16")


# Save with pypng. It's API is not great, but imageio/pillow cannot do 16bit png
filename = os.path.join(tempfile.gettempdir(), "wgpuexample.png")
with open(filename, "wb") as f:
    writer = png.Writer(
        width=image.shape[1], height=image.shape[0], bitdepth=16, greyscale=False
    )
    writer.write(f, image.reshape(image.shape[0], -1).tolist())

# Show the written file
webbrowser.open("file://" + filename)
