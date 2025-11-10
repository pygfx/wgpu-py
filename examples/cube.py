"""
Example that renders a textured rotating cube.

This example is a bit more interesting (and larger) than the triangle,
because it adds buffers and textures.

This example is set up so it can be run with any canvas. Running this file
as a script will use rendercanvas with the auto-backend.
"""

# test_example = true

import time
from typing import Callable

import wgpu
import numpy as np


from rendercanvas.auto import RenderCanvas, loop


# %% Entrypoints (sync and async)


def setup_drawing_sync(
    context, power_preference="high-performance", limits=None
) -> Callable:
    """Setup to draw a rotating cube on the given context.

    Returns the draw function.
    """

    adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
    device = adapter.request_device_sync(
        label="Cube Example device",
        required_limits=limits,
    )

    pipeline_layout, uniform_buffer, bind_group = create_pipeline_layout(device)
    pipeline_kwargs = get_render_pipeline_kwargs(context, device, pipeline_layout)

    render_pipeline = device.create_render_pipeline(**pipeline_kwargs)

    return get_draw_function(
        context, device, render_pipeline, uniform_buffer, bind_group
    )


async def setup_drawing_async(context, limits=None):
    """Setup to async-draw a rotating cube on the given context.

    Returns the draw function.
    """
    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")

    device = await adapter.request_device_async(
        label="Cube Example async device", required_limits=limits
    )

    pipeline_layout, uniform_buffer, bind_group = create_pipeline_layout(device)
    pipeline_kwargs = get_render_pipeline_kwargs(context, device, pipeline_layout)

    render_pipeline = await device.create_render_pipeline_async(**pipeline_kwargs)

    return get_draw_function(
        context, device, render_pipeline, uniform_buffer, bind_group
    )


def get_drawing_func(context, device):
    pipeline_layout, uniform_buffer, bind_group = create_pipeline_layout(device)
    pipeline_kwargs = get_render_pipeline_kwargs(context, device, pipeline_layout)

    render_pipeline = device.create_render_pipeline(**pipeline_kwargs)
    # render_pipeline = device.create_render_pipeline(**pipeline_kwargs)

    return get_draw_function(
        context, device, render_pipeline, uniform_buffer, bind_group
    )


# %% Functions to create wgpu objects


def get_render_pipeline_kwargs(
    context, device: wgpu.GPUDevice, pipeline_layout: wgpu.GPUPipelineLayout
) -> wgpu.RenderPipelineDescriptor:
    render_texture_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=render_texture_format)

    shader = device.create_shader_module(
        code=shader_source, label="Cube Example shader module"
    )

    # wgpu.RenderPipelineDescriptor
    return wgpu.RenderPipelineDescriptor(
        label="Cube Example render pipeline",
        layout=pipeline_layout,
        vertex=wgpu.VertexState(
            module=shader,
            entry_point="vs_main",
            buffers=[
                wgpu.VertexBufferLayout(
                    array_stride=4 * 6,
                    step_mode="vertex",
                    attributes=[
                        wgpu.VertexAttribute(
                            format="float32x4",
                            offset=0,
                            shader_location=0,
                        ),
                        wgpu.VertexAttribute(
                            format="float32x2",
                            offset=4 * 4,
                            shader_location=1,
                        ),
                    ],
                ),
            ],
        ),
        primitive=wgpu.PrimitiveState(
            topology="triangle-list",
            front_face="ccw",
            cull_mode="back",
        ),
        fragment=wgpu.FragmentState(
            module=shader,
            entry_point="fs_main",
            targets=[
                wgpu.ColorTargetState(
                    format=render_texture_format,
                    blend={"alpha": {}, "color": {}},
                )
            ],
        ),
    )


def create_pipeline_layout(device: wgpu.GPUDevice):
    # Create uniform buffer - data is uploaded each frame
    uniform_buffer = device.create_buffer(
        size=uniform_data.nbytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        label="Cube Example uniform buffer",
    )

    # Create another buffer to copy data to it (by mapping it and then copying the data)
    uniform_buffer.copy_buffer = device.create_buffer(
        size=uniform_data.nbytes,
        usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        label="Cube Example uniform buffer copy buffer",
    )

    # Create texture, and upload data
    texture = device.create_texture(
        size=texture_size,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension="2d",
        format="r8unorm",
        mip_level_count=1,
        sample_count=1,
        label="Cube Example texture",
    )
    texture_view = texture.create_view(label="Cube Example texture view")

    device.queue.write_texture(
        wgpu.TexelCopyTextureInfo(
            texture=texture,
            mip_level=0,
            origin=(0, 0, 0),
        ),
        texture_data,
        wgpu.TexelCopyBufferLayout(
            offset=0,
            bytes_per_row=texture_data.strides[0],
        ),
        texture_size,
    )

    # Create a sampler
    sampler = device.create_sampler(label="Cube Example sampler")

    # Create bind group layouts for our resources
    # We will use a single bind group with three resources
    bind_group_entries = []
    bind_group_layout_entries = []

    bind_group_entries.append(
        wgpu.BindGroupEntry(
            binding=0,
            resource=wgpu.BufferBinding(
                buffer=uniform_buffer, offset=0, size=uniform_buffer.size
            ),
        )
    )
    bind_group_layout_entries.append(
        wgpu.BindGroupLayoutEntry(
            binding=0,
            visibility=wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
            buffer={},
        )
    )

    bind_group_entries.append(
        wgpu.BindGroupEntry(
            binding=1,
            resource=texture_view,
        )
    )
    bind_group_layout_entries.append(
        wgpu.BindGroupLayoutEntry(
            binding=1,
            visibility=wgpu.ShaderStage.FRAGMENT,
            texture={},
        )
    )

    bind_group_entries.append(
        wgpu.BindGroupEntry(
            binding=2,
            resource=sampler,
        )
    )
    bind_group_layout_entries.append(
        wgpu.BindGroupLayoutEntry(
            binding=2, visibility=wgpu.ShaderStage.FRAGMENT, sampler={}
        )
    )

    # Create the wgpu binding objects
    bind_group_layout = device.create_bind_group_layout(
        entries=bind_group_layout_entries,
        label="Cube Example bind group layout",
    )
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=bind_group_entries,
        label="Cube Example bind group",
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout], label="Cube Example pipeline layout"
    )

    return pipeline_layout, uniform_buffer, bind_group


def get_draw_function(
    context,
    device: wgpu.GPUDevice,
    render_pipeline: wgpu.GPURenderPipeline,
    uniform_buffer: wgpu.GPUBuffer,
    bind_group: wgpu.GPUBindGroup,
):
    # Create vertex buffer, and upload data
    vertex_buffer = device.create_buffer_with_data(
        data=vertex_data,
        usage=wgpu.BufferUsage.VERTEX,
        label="Cube Example vertex buffer",
    )

    # Create index buffer, and upload data
    index_buffer = device.create_buffer_with_data(
        data=index_data, usage=wgpu.BufferUsage.INDEX, label="Cube Example index buffer"
    )

    def update_transform():
        # Update uniform transform
        a1 = -0.3
        a2 = time.time()
        s = 0.6
        ortho = np.array(
            [
                [s, 0, 0, 0],
                [0, s, 0, 0],
                [0, 0, s, 0],
                [0, 0, 0, 1],
            ],
        )
        rot1 = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(a1), -np.sin(a1), 0],
                [0, np.sin(a1), +np.cos(a1), 0],
                [0, 0, 0, 1],
            ],
        )
        rot2 = np.array(
            [
                [np.cos(a2), 0, np.sin(a2), 0],
                [0, 1, 0, 0],
                [-np.sin(a2), 0, np.cos(a2), 0],
                [0, 0, 0, 1],
            ],
        )
        uniform_data["transform"] = rot2 @ rot1 @ ortho

    def upload_uniform_buffer():
        device.queue.write_buffer(uniform_buffer, 0, uniform_data)

    def draw_frame():
        current_texture_view: wgpu.GPUTextureView = (
            context.get_current_texture().create_view(
                label="Cube Example current surface texture view"
            )
        )
        command_encoder = device.create_command_encoder(
            label="Cube Example render pass command encoder"
        )
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=current_texture_view,
                    clear_value=(0, 0, 0, 1),
                    load_op="clear",
                    store_op="store",
                )
            ],
            label="Cube Example render pass",
        )

        # debug groups and markers can optionally be added to help debugging.
        render_pass.push_debug_group("Cube Example Debug Group")
        render_pass.set_pipeline(render_pipeline)
        render_pass.set_index_buffer(index_buffer, "uint32")
        render_pass.set_vertex_buffer(0, vertex_buffer)
        render_pass.set_bind_group(0, bind_group)
        render_pass.insert_debug_marker("Cube Example draw call")
        render_pass.draw_indexed(index_data.size, 1, 0, 0, 0)
        render_pass.pop_debug_group()
        render_pass.end()

        device.queue.submit(
            [command_encoder.finish(label="Cube Example render pass command buffer")]
        )

    def draw_func():
        update_transform()
        upload_uniform_buffer()
        draw_frame()

    return draw_func


# %% WGSL


shader_source = """
struct Locals {
    transform: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> r_locals: Locals;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texcoord: vec2<f32>,
};
struct VertexOutput {
    @location(0) texcoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
};
struct FragmentOutput {
    @location(0) color : vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let ndc: vec4<f32> = r_locals.transform * in.pos;
    let xy_ratio = 0.75;  // hardcoded for 640x480 canvas size
    var out: VertexOutput;
    out.pos = vec4<f32>(ndc.x * xy_ratio, ndc.y, 0.0, 1.0);
    out.texcoord = in.texcoord;
    return out;
}

@group(0) @binding(1)
var r_tex: texture_2d<f32>;

@group(0) @binding(2)
var r_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let value = textureSample(r_tex, r_sampler, in.texcoord).r;
    let physical_color = vec3<f32>(pow(value, 2.2));  // gamma correct
    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color.rgb, 1.0);
    return out;
}
"""


# %% Data


# pos         texcoord
# x, y, z, w, u, v
vertex_data = np.array(
    [
        # top (0, 0, 1)
        [-1, -1, 1, 1, 0, 0],
        [1, -1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [-1, 1, 1, 1, 0, 1],
        # bottom (0, 0, -1)
        [-1, 1, -1, 1, 1, 0],
        [1, 1, -1, 1, 0, 0],
        [1, -1, -1, 1, 0, 1],
        [-1, -1, -1, 1, 1, 1],
        # right (1, 0, 0)
        [1, -1, -1, 1, 0, 0],
        [1, 1, -1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, 0, 1],
        # left (-1, 0, 0)
        [-1, -1, 1, 1, 1, 0],
        [-1, 1, 1, 1, 0, 0],
        [-1, 1, -1, 1, 0, 1],
        [-1, -1, -1, 1, 1, 1],
        # front (0, 1, 0)
        [1, 1, -1, 1, 1, 0],
        [-1, 1, -1, 1, 0, 0],
        [-1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
        # back (0, -1, 0)
        [1, -1, 1, 1, 0, 0],
        [-1, -1, 1, 1, 1, 0],
        [-1, -1, -1, 1, 1, 1],
        [1, -1, -1, 1, 0, 1],
    ],
    dtype=np.float32,
)

index_data = np.array(
    [
        [0, 1, 2, 2, 3, 0],  # top
        [4, 5, 6, 6, 7, 4],  # bottom
        [8, 9, 10, 10, 11, 8],  # right
        [12, 13, 14, 14, 15, 12],  # left
        [16, 17, 18, 18, 19, 16],  # front
        [20, 21, 22, 22, 23, 20],  # back
    ],
    dtype=np.uint32,
).flatten()


# Create texture data (srgb gray values)
texture_data = np.array(
    [
        [50, 100, 150, 200],
        [100, 150, 200, 50],
        [150, 200, 50, 100],
        [200, 50, 100, 150],
    ],
    dtype=np.uint8,
)
texture_data = np.repeat(texture_data, 64, 0)
texture_data = np.repeat(texture_data, 64, 1)
texture_size = texture_data.shape[1], texture_data.shape[0], 1

# Use numpy to create a struct for the uniform
uniform_dtype = [("transform", "float32", (4, 4))]
uniform_data = np.zeros((), dtype=uniform_dtype)

print("Available adapters on this system:")
for a in wgpu.gpu.enumerate_adapters_sync():
    print(a.summary)


if __name__ == "__main__":
    canvas = RenderCanvas(
        size=(640, 480),
        title="wgpu cube example at $fps using $backend",
        update_mode="continuous",
        max_fps=60,
        vsync=True,
    )
    context = canvas.get_wgpu_context()

    # Pick one

    if True:
        # Async
        @loop.add_task
        async def init():
            draw_frame = await setup_drawing_async(context)
            canvas.request_draw(draw_frame)
    else:
        # Sync
        draw_frame = setup_drawing_sync(context)
        canvas.request_draw(draw_frame)

    # loop.add_task(poller)
    loop.run()
