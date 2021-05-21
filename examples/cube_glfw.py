"""
This example renders a simple textured rotating cube.
"""

import time

import glfw
import wgpu
from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend
import numpy as np


# %% Create canvas and device

# Create a canvas to render to
glfw.init()
glfw.ERROR_REPORTING = "warn"
canvas = WgpuCanvas(title="wgpu cube with GLFW")

# Create a wgpu device
adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
device = adapter.request_device()


# %% Generate data

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


# %% Create resource objects (buffers, textures, samplers)

# Create vertex buffer, and upload data
vertex_buffer = device.create_buffer_with_data(
    data=vertex_data, usage=wgpu.BufferUsage.VERTEX
)

# Create index buffer, and upload data
index_buffer = device.create_buffer_with_data(
    data=index_data, usage=wgpu.BufferUsage.INDEX
)

# Create uniform buffer - data is uploaded each frame
uniform_buffer = device.create_buffer(
    size=uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

# Create texture, and upload data
texture = device.create_texture(
    size=texture_size,
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.SAMPLED,
    dimension=wgpu.TextureDimension.d2,
    format=wgpu.TextureFormat.r8unorm,
    mip_level_count=1,
    sample_count=1,
)
texture_view = texture.create_view()
tmp_buffer = device.create_buffer_with_data(
    data=texture_data, usage=wgpu.BufferUsage.COPY_SRC
)

command_encoder = device.create_command_encoder()
command_encoder.copy_buffer_to_texture(
    {
        "buffer": tmp_buffer,
        "offset": 0,
        "bytes_per_row": texture_data.strides[0],
        "rows_per_image": 0,
    },
    {
        "texture": texture,
        "mip_level": 0,
        "origin": (0, 0, 0),
    },
    copy_size=texture_size,
)
device.queue.submit([command_encoder.finish()])


# Create a sampler
sampler = device.create_sampler()


# %% The shaders


shader_source = """
[[block]]
struct Locals {
    transform: mat4x4<f32>;
};
[[group(0), binding(0)]]
var r_locals: Locals;

struct VertexInput {
    [[location(0)]] pos : vec4<f32>;
    [[location(1)]] texcoord: vec2<f32>;
};
struct VertexOutput {
    [[location(0)]] texcoord: vec2<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {
    let ndc: vec4<f32> = r_locals.transform * in.pos;
    var out: VertexOutput;
    out.pos = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    out.texcoord = in.texcoord;
    return out;
}

[[group(0), binding(1)]]
var r_tex: texture_2d<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let value = textureSample(r_tex, r_sampler, in.texcoord).r;;
    return vec4<f32>(value, value, value, 1.0);
}
"""

shader = device.create_shader_module(code=shader_source)


# %% The bind groups

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
        "buffer": {"type": wgpu.BufferBindingType.uniform},
    }
)

bind_groups_entries[0].append({"binding": 1, "resource": texture_view})
bind_groups_layout_entries[0].append(
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.FRAGMENT,
        "texture": {
            "sample_type": wgpu.TextureSampleType.float,
            "view_dimension": wgpu.TextureViewDimension.d2,
        },
    }
)

bind_groups_entries[0].append({"binding": 2, "resource": sampler})
bind_groups_layout_entries[0].append(
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.FRAGMENT,
        "sampler": {"type": wgpu.SamplerBindingType.filtering},
    }
)


# Create the wgou binding objects
bind_group_layouts = []
bind_groups = []

for entries, layout_entries in zip(bind_groups_entries, bind_groups_layout_entries):
    bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
    bind_group_layouts.append(bind_group_layout)
    bind_groups.append(
        device.create_bind_group(layout=bind_group_layout, entries=entries)
    )

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)


# %% The render pipeline

render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": shader,
        "entry_point": "vs_main",
        "buffers": [
            {
                "array_stride": 4 * 6,
                "step_mode": wgpu.InputStepMode.vertex,
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
                "format": wgpu.TextureFormat.bgra8unorm_srgb,
                "blend": {
                    "alpha": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "color": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                },
            }
        ],
    },
)


# %% Setup the render function

swap_chain = canvas.configure_swap_chain(device=device)


def draw_frame():

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

    # Upload the uniform struct
    tmp_buffer = device.create_buffer_with_data(
        data=uniform_data, usage=wgpu.BufferUsage.COPY_SRC
    )

    with swap_chain as current_texture_view:
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(
            tmp_buffer, 0, uniform_buffer, 0, uniform_data.nbytes
        )

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
                    "resolve_target": None,
                    "load_value": (0.1, 0.3, 0.2, 1),
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        for bind_group_id, bind_group in enumerate(bind_groups):
            render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
        render_pass.draw_indexed(index_data.size, 1, 0, 0, 0)

        render_pass.end_pass()
        device.queue.submit([command_encoder.finish()])

    canvas.request_draw()


canvas.request_draw(draw_frame)


# %% Run the event loop


def simple_event_loop():
    """A real simple event loop, but it keeps the CPU busy."""
    while update_glfw_canvasses():
        glfw.poll_events()


simple_event_loop()
glfw.terminate()
