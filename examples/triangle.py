"""
Example use of the wgpu API to draw a triangle.

The triangle is a classic example representing the simplest possible
visualisation because it does not need buffers or textures. The same
example in other languages / API's:

* JavaScript WebGPU:
  https://webgpu.github.io/webgpu-samples/?sample=helloTriangle
* Rust wgpu:
  https://github.com/gfx-rs/wgpu/blob/trunk/examples/features/src/hello_triangle/mod.rs
* C wgpu:
  https://github.com/gfx-rs/wgpu-native/blob/trunk/examples/triangle/main.c
* Python Vulkan:
  https://github.com/realitix/vulkan/blob/master/example/contribs/example_glfw.py

This example is meant as a standalone starting point. And is therefore as minimal as possible.
"""

import wgpu

from rendercanvas.auto import RenderCanvas, loop


# the shader code is provided as a string literal for protability
wgsl_shader_source = """
struct VertexOutput {
    @location(0) color : vec4f,
    @builtin(position) pos: vec4f,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2(0.0, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.75),
    );
    // vertex attributes are interpolated in the fragment shader.
    var colors = array<vec3f, 3>(  // srgb colors
        vec3(1.0, 1.0, 0.0),
        vec3(1.0, 0.0, 1.0),
        vec3(0.0, 1.0, 1.0),
    );
    var out: VertexOutput;
    out.pos = vec4(positions[index], 0.0, 1.0);
    out.color = vec4(colors[index], 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let physical_color = pow(in.color.rgb, vec3(2.2));  // gamma correct
    return vec4(physical_color, in.color.a);
}
"""


# adapter provides allows us to create a single device, which is the general entrypoint to create most wgpu objects.
# for convenience and interoperability `wgpu.utils.get_default_device()` and associated configuration are provided.
adapter = wgpu.gpu.request_adapter_sync()
device = adapter.request_device_sync()

# setting up a canvas, so we can see what we draw
canvas = RenderCanvas(size=(640, 480), title="wgpu triangle example")
context = canvas.get_wgpu_context()
render_texture_format = context.get_preferred_format(device.adapter)
context.configure(device=device, format=render_texture_format)

# creating the shader module compiles the shader code for your GPU.
shader = device.create_shader_module(code=wgsl_shader_source)

# in wgpu-py, methods that take descriptors will take the keyword arguments instead.
# descriptors and other structs can still be accessed via wgpu.structs or top level wgpu.
render_pipeline = device.create_render_pipeline(
    **wgpu.RenderPipelineDescriptor(
        layout=wgpu.AutoLayoutMode.auto,
        vertex=wgpu.VertexState(module=shader),
        depth_stencil=None,
        multisample=None,
        fragment=wgpu.FragmentState(
            module=shader,
            targets=[wgpu.ColorTargetState(format=render_texture_format)],
        ),
    )
)

# this function gets called for every frame. It ends with submitting a buffer of work onto the GPU queue.
def drawing_function():
    command_encoder = device.create_command_encoder()
    current_texture_view = context.get_current_texture().create_view()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=current_texture_view,
                clear_value=(0, 1, 0, 1),  # a green background
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
            )
        ],
    )
    render_pass.set_pipeline(render_pipeline)
    render_pass.draw(3)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])


if __name__ == "__main__":
    canvas.request_draw(drawing_function)
    loop.run()
