"""
Example that renders a textured rotating cube.

This example is a bit more interesting (and larger) than the triangle,
because it adds buffers and textures.

This example is set up so it can be run with any canvas. Running this file
as a script will use the auto-backend.
"""

# test_example = true

import time

import wgpu
import numpy as np


from rendercanvas.auto import RenderCanvas, loop


# %% Entrypoints (sync and async)


def setup_drawing_sync(canvas, power_preference="high-performance", limits={}):
    """Setup to draw a rotating cube on the given canvas.

    The given canvas must implement WgpuCanvasInterface, but nothing more.
    Returns the draw function.
    """

    adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
    device = adapter.request_device_sync(required_limits=limits)

    pipeline_layout, uniform_buffer, bind_groups = create_pipeline_layout(device)
    pipeline_kwargs = get_render_pipeline_kwargs(canvas, device, pipeline_layout)

    render_pipeline = device.create_render_pipeline(**pipeline_kwargs)

    return get_draw_function(
        canvas, device, render_pipeline, uniform_buffer, bind_groups, asynchronous=False
    )


async def setup_drawing_async(canvas, limits={}):
    """Setup to async-draw a rotating cube on the given canvas.

    The given canvas must implement WgpuCanvasInterface, but nothing more.
    Returns the draw function.
    """

    adapter = await wgpu.gpu.request_adapter_async(power_preference="high-performance")
    device = await adapter.request_device_async(required_limits=limits)

    pipeline_layout, uniform_buffer, bind_groups = create_pipeline_layout(device)
    pipeline_kwargs = get_render_pipeline_kwargs(canvas, device, pipeline_layout)

    render_pipeline = await device.create_render_pipeline_async(**pipeline_kwargs)

    return get_draw_function(
        canvas, device, render_pipeline, uniform_buffer, bind_groups, asynchronous=True
    )


# %% Functions to create wgpu objects


def get_render_pipeline_kwargs(canvas, device, pipeline_layout):
    context = canvas.get_context("wgpu")
    render_texture_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=render_texture_format)

    shader = device.create_shader_module(code=shader_source)

    return dict(
        layout=pipeline_layout,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
            "buffers": [
                {
                    "array_stride": 4 * 6,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {
                            "format": wgpu.VertexFormat.float32x4,
                            "offset": 0,
                            "shader_location": 0,
                        },
                        {
                            "format": wgpu.VertexFormat.float32x2,
                            "offset": 4 * 4,
                            "shader_location": 1,
                        },
                    ],
                },
            ],
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.back,
        },
        depth_stencil=None,
        multisample=None,
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [
                {
                    "format": render_texture_format,
                    "blend": {
                        "alpha": {},
                        "color": {},
                    },
                }
            ],
        },
    )


def create_pipeline_layout(device):
    # Create uniform buffer - data is uploaded each frame
    uniform_buffer = device.create_buffer(
        size=uniform_data.nbytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # Create another buffer to copy data to it (by mapping it and then copying the data)
    uniform_buffer.copy_buffer = device.create_buffer(
        size=uniform_data.nbytes,
        usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
    )

    # Create texture, and upload data
    texture = device.create_texture(
        size=texture_size,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.r8unorm,
        mip_level_count=1,
        sample_count=1,
    )
    texture_view = texture.create_view()

    device.queue.write_texture(
        {
            "texture": texture,
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        texture_data,
        {
            "offset": 0,
            "bytes_per_row": texture_data.strides[0],
        },
        texture_size,
    )

    # Create a sampler
    sampler = device.create_sampler()

    # We always have two bind groups, so we can play distributing our
    # resources over these two groups in different configurations.
    bind_groups_entries = [[]]
    bind_groups_layout_entries = [[]]

    bind_groups_entries[0].append(
        {
            "binding": 0,
            "resource": {
                "buffer": uniform_buffer,
                "offset": 0,
                "size": uniform_buffer.size,
            },
        }
    )
    bind_groups_layout_entries[0].append(
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
            "buffer": {},
        }
    )

    bind_groups_entries[0].append({"binding": 1, "resource": texture_view})
    bind_groups_layout_entries[0].append(
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "texture": {},
        }
    )

    bind_groups_entries[0].append({"binding": 2, "resource": sampler})
    bind_groups_layout_entries[0].append(
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "sampler": {},
        }
    )

    # Create the wgpu binding objects
    bind_group_layouts = []
    bind_groups = []

    for entries, layout_entries in zip(bind_groups_entries, bind_groups_layout_entries):
        bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
        bind_group_layouts.append(bind_group_layout)
        bind_groups.append(
            device.create_bind_group(layout=bind_group_layout, entries=entries)
        )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=bind_group_layouts
    )

    return pipeline_layout, uniform_buffer, bind_groups


def get_draw_function(
    canvas, device, render_pipeline, uniform_buffer, bind_groups, *, asynchronous
):
    # Create vertex buffer, and upload data
    vertex_buffer = device.create_buffer_with_data(
        data=vertex_data, usage=wgpu.BufferUsage.VERTEX
    )

    # Create index buffer, and upload data
    index_buffer = device.create_buffer_with_data(
        data=index_data, usage=wgpu.BufferUsage.INDEX
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

    def upload_uniform_buffer_sync():
        if True:
            tmp_buffer = uniform_buffer.copy_buffer
            tmp_buffer.map_sync(wgpu.MapMode.WRITE)
            tmp_buffer.write_mapped(uniform_data)
            tmp_buffer.unmap()
        else:
            tmp_buffer = device.create_buffer_with_data(
                data=uniform_data, usage=wgpu.BufferUsage.COPY_SRC
            )
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(
            tmp_buffer, 0, uniform_buffer, 0, uniform_data.nbytes
        )
        device.queue.submit([command_encoder.finish()])

    async def upload_uniform_buffer_async():
        tmp_buffer = uniform_buffer.copy_buffer
        await tmp_buffer.map_async(wgpu.MapMode.WRITE)
        tmp_buffer.write_mapped(uniform_data)
        tmp_buffer.unmap()
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(
            tmp_buffer, 0, uniform_buffer, 0, uniform_data.nbytes
        )
        device.queue.submit([command_encoder.finish()])

    def draw_frame():
        current_texture_view = (
            canvas.get_context("wgpu").get_current_texture().create_view()
        )
        command_encoder = device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        for bind_group_id, bind_group in enumerate(bind_groups):
            render_pass.set_bind_group(bind_group_id, bind_group)
        render_pass.draw_indexed(index_data.size, 1, 0, 0, 0)
        render_pass.end()

        device.queue.submit([command_encoder.finish()])

    def draw_frame_sync():
        update_transform()
        upload_uniform_buffer_sync()
        draw_frame()

    async def draw_frame_async():
        update_transform()
        await upload_uniform_buffer_async()
        draw_frame()

    if asynchronous:
        return draw_frame_async
    else:
        return draw_frame_sync


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
    draw_frame = setup_drawing_sync(canvas)
    canvas.request_draw(draw_frame)
    loop.run()
