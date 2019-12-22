"""
Example use of webgpu API to draw a triangle.

Ported from:

https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/main.rs
https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c

For reference, the same kind of example using the Vulkan API is 700 lines:
https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

"""

import asyncio

import wgpu.rs
from py2spirv import python2spirv  # This is not yet a public library

# Pick either PyQt5 or Pyside2
from PyQt5 import QtWidgets

# from PySide2 import QtWidgets

app = QtWidgets.QApplication([])

widget = QtWidgets.QWidget(None)
widget.resize(640, 480)
widget.setWindowTitle("Python wgpu triangle")
widget.show()


# %% Shaders


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


# %% The wgpu calls

adapter = wgpu.requestAdapterSync(powerPreference="high-performance")
device = adapter.requestDeviceSync(extensions=[], limits=wgpu.GPULimits())

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
    # sampleCount=1,
    # sampleMask: int=0xFFFFFFFF,
    # alphaToCoverageEnabled=False,
)

# in IDL: canvas.configureSwapChain(device, ...)

# Likewise we can have configureSwapChainSDL2 and  configureSwapChainCanvas for JS
swap_chain = device.configureSwapChainQt(
    surface=widget,
    format=wgpu.TextureFormat.bgra8unorm_srgb,
    usage=wgpu.TextureUsage.OUTPUT_ATTACHMENT,
)


def drawFrame():
    # wgpu.requestAnimationFrame(drawFrame)

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
    render_pass.setBindGroup(0, bind_group, [], 0, 999999)  # last 2 elements not used
    render_pass.draw(3, 1, 0, 0)
    render_pass.endPass()

    command_buffer = command_encoder.finish()
    device.defaultQueue.submit([command_buffer])

    swap_chain._gui_present()


# wgpu.requestAnimationFrame(drawFrame)


async def drawer():
    while True:
        await asyncio.sleep(0.1)
        # print("draw")
        drawFrame()


asyncio.get_event_loop().create_task(drawer())
