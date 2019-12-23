[![Build Status](https://dev.azure.com/almarklein/wgpu-py/_apis/build/status/almarklein.wgpu-py?branchName=master)](https://dev.azure.com/almarklein/wgpu-py/_build/latest?definitionId=1&branchName=master)

# wgpu-py

Next generation GPU API for Python

This is experimental, work in progress, you probably don't want to use this just yet!


## Introduction

In short, this is a Python lib wrapping the Rust wgpu lib and exposing
it with a Pythonic API similar to WebGPU.

The OpenGL API is old and showing it's cracks. New API's like Vulkan,
Metal and DX12 provide a modern way to control the GPU, but these API's
are too low-level. The WebGPU API follows the same concepts, but with
a simpler (higher level) spelling. The Python `wgpu` library brings the
WebGPU API to Python. Based on [wgpu-native](https://github.com/gfx-rs/wgpu).


## Installation

```
pip install wgpu
```

This library does not have any dependencies on other Python libraries.
Though if you want to render to the screen you need a GUI toolkit.
Currently supported are:

* PySide2
* PyQt5

This library will eventually include the required Rust library, but for
now, you have to bring it yourself. Tell where it is by setting the
environment variable `WGPU_LIB_PATH`.


## Usage

The full API is accessable via the main namespace:
```py
import wgpu
```

But to use it, you need to select a backend first. You do this by importing it.
There is currently only one backend:
```py
import wgpu.backend.rs
```


To give an idea of what this API looks like, here's the API code from the triangle example:
```py
# ... code to create shaders and GUI are omitted for brevity

import wgpu

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

    canvas.setDrawFunction(drawFrame)
```


## License

This code is distributed under the 2-clause BSD license.


## Developers

* Clone the repo and run `python setup.py develop`, or simply add the root dir to your `PYTHONPATH`.
* Point the `WGPU_LIB_PATH` environment variable to the dynamic library created by `wgpu-native`.
* Use `black .` to apply autoformatting.
* Use `pytest .` to run the tests.
