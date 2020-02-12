"""
Example use of webgpu API to draw a triangle. See the triangle_glfw.py
script (and related scripts) for actually running this.

Similar example in other languages / API's:

* Rust wgpu:
  https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/main.rs
* C wgpu:
  https://github.com/gfx-rs/wgpu/blob/master/examples/triangle/main.c
* Python Vulkan:
  https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

"""

import wgpu
from python_shader import python2shader
from python_shader import RES_INPUT, RES_OUTPUT
from python_shader import vec2, vec3, vec4, i32


# %% Shaders


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", i32),
    pos: (RES_OUTPUT, "Position", vec4),
    color: (RES_OUTPUT, 0, vec3),
):
    positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
    p = positions[index]
    pos = vec4(p, 0.0, 1.0)  # noqa
    color = vec3(p, 0.5)  # noqa


@python2shader
def fragment_shader(
    in_color: (RES_INPUT, 0, vec3), out_color: (RES_OUTPUT, 0, vec4),
):
    out_color = vec4(in_color, 1.0)  # noqa


# %% The wgpu calls


def main(canvas):
    """ Regular function to setup a viz on the given canvas.
    """
    adapter = wgpu.requestAdapter(powerPreference="high-performance")
    device = adapter.requestDevice(extensions=[], limits=wgpu.GPULimits())
    return _main(canvas, device)


async def mainAsync(canvas):
    """ Async function to setup a viz on the given canvas.
    """
    adapter = await wgpu.requestAdapterAsync(powerPreference="high-performance")
    device = await adapter.requestDeviceAsync(extensions=[], limits=wgpu.GPULimits())
    return _main(canvas, device)


def _main(canvas, device):

    vshader = device.createShaderModule(code=vertex_shader)
    fshader = device.createShaderModule(code=fragment_shader)

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

    swap_chain = canvas.configure_swap_chain(
        device, wgpu.TextureFormat.bgra8unorm_srgb, wgpu.TextureUsage.OUTPUT_ATTACHMENT
    )

    def drawFrame():
        current_texture_view = swap_chain.get_current_texture_view()
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
