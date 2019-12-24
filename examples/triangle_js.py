"""
Hypothetical example for a visualization to be converted to JS.

DOES NOT WORK YET. THIS IS MOSTLY TO GET AN IMPRESSION OF HOW IT COULD WORK.

This example uses Flexx to collect and compile the Python code to JS modules,
and provide a HTML5 canvas without having to write HTML.
"""


# %% Shaders

# This is not yet a public library
from py2spirv import python2spirv, i32, vec2, vec3, vec4


@python2spirv
def vertex_shader(input, output):
    input.define("index", "VertexId", i32)
    output.define("pos", "Position", vec4)
    output.define("color", 0, vec3)

    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]

    p = positions[input.index]
    output.pos = vec4(p, 0.0, 1.0)
    output.color = vec3(p, 0.5)


@python2spirv
def fragment_shader(input, output):
    input.define("color", 0, vec3)
    output.define("color", 0, vec4)

    output.color = vec4(input.color, 1.0)


# todo: how to serialize the shaders? base64 or via a custom hook?
vertex_shader = "something that flexx can serialize"  # noqa: F811
fragment_shader = "something that flexx can serialize"  # noqa: F811


# %% The wgpu calls - exact same code as in triangle1.py


async def main(canvas):

    adapter = await wgpu.requestAdapter(powerPreference="high-performance")
    device = await adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())

    vshader = device.createShaderModule(code=vertex_shader.to_bytes())
    fshader = device.createShaderModule(code=fragment_shader.to_bytes())

    bind_group_layout = device.createBindGroupLayout(bindings=[])  # zero bindings
    bind_group = device.createBindGroup(layout=bind_group_layout, bindings=[])

    pipeline_layout = device.createPipelineLayout(bindGroupLayouts=[bind_group_layout])

    render_pipeline = device.createRenderPipeline(
        layout=pipeline_layout,
        vertexStage={"module": vshader, "entryPoint": "main"},
        fragmentStage={"module": fshader, "entryPoint": "main"},
        primitiveTopology=wgpu.PrimitiveTopology.triangle_list,
        rasterizationState={
            "frontFace": wgpu.FrontFace.ccw,
            "cullMode": wgpu.CullMode.none,
            "depthBias": 0,
            "depthBiasSlopeScale": 0.0,
            "depthBiasClamp": 0.0,
        },
        colorStates=[
            {
                "format": wgpu.TextureFormat.bgra8unorm_srgb,
                "alphaBlend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "colorBlend": (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.zero,
                    wgpu.BlendOperation.add,
                ),
                "writeMask": wgpu.ColorWrite.ALL,
            }
        ],
        depthStencilState=None,
        vertexState={"indexFormat": wgpu.IndexFormat.uint32, "vertexBuffers": []},
        sampleCount=1,
        sampleMask=0xFFFFFFFF,
        alphaToCoverageEnabled=False,
    )

    swap_chain = canvas.configureSwapChain(
        device, wgpu.TextureFormat.bgra8unorm_srgb, wgpu.TextureUsage.OUTPUT_ATTACHMENT
    )

    def drawFrame():
        current_texture_view = swap_chain.getCurrentTextureView()
        command_encoder = device.createCommandEncoder()

        render_pass = command_encoder.beginRenderPass(
            colorAttachments=[
                {
                    "attachment": current_texture_view,
                    "resolveTarget": None,
                    "loadValue": (0, 0, 0, 1),  # LoadOp.load or color
                    "storeOp": wgpu.StoreOp.store,
                }
            ],
            depthStencilAttachment=None,
        )

        render_pass.setPipeline(render_pipeline)
        render_pass.setBindGroup(
            0, bind_group, [], 0, 999999
        )  # last 2 elements not used
        render_pass.draw(3, 1, 0, 0)
        render_pass.endPass()
        device.defaultQueue.submit([command_encoder.finish()])

    canvas.drawFrame = drawFrame


# %% Create the canvas and run - JS backend

from flexx import flx  # noqa: E402
from wgpu.gui.flexx import WgpuCanvas  # noqa: E402, WgpuCanvas is a flx.Canvas subclass
import wgpu.backend.js  # noqa: E402, select JS backend


class Example(flx.Widget):
    def init(self):
        # All of this gets executed in JS
        super().init()
        with flx.HBox():
            self.canvas = WgpuCanvas()
        main(self.canvas)


if __name__ == "__main__":
    m = flx.launch(Example, "chrome-browser")
    flx.run()
