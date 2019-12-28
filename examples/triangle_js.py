"""
Hypothetical example for a visualization to be converted to JS.

DOES NOT WORK YET. THIS IS MOSTLY TO GET AN IMPRESSION OF HOW IT COULD WORK.

This example uses Flexx to collect and compile the Python code to JS modules,
and provide a HTML5 canvas without having to write HTML.
"""

from flexx import flx
from wgpu.gui.flexx import WgpuCanvas  # WgpuCanvas is a flx.Canvas subclass
import wgpu.backend.js  # noqa: F401, Select JS backend

# Import the (async) function that we must call to run the visualization
import triangle


# todo: how to serialize the shaders? base64 or via a custom hook?
triangle.vertex_shader = "something that flexx can serialize"
triangle.fragment_shader = "something that flexx can serialize"


class Example(flx.Widget):
    def init(self):
        # All of this gets executed in JS
        super().init()
        with flx.HBox():
            self.canvas = WgpuCanvas()
        triangle.main(self.canvas)


if __name__ == "__main__":
    m = flx.launch(Example, "chrome-browser")
    flx.run()
