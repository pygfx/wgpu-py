"""
This example renders a simple textured rotating cube.
"""

# test_example = true

import time

from wgpu.gui.auto import WgpuCanvas, run
import wgpu
import numpy as np


print("Available adapters on this system:")
for a in wgpu.gpu.enumerate_adapters():
    print(a.summary)


# %% Create canvas and device

# Create a canvas to render to
canvas = WgpuCanvas(title="wgpu cube", size=(640, 480))

# Create a wgpu device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapter.request_device()

# Prepare present context
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)


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


# %% The shaders


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


# %% Setup the render function


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

    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(
        tmp_buffer, 0, uniform_buffer, 0, uniform_data.nbytes
    )

    current_texture_view = present_context.get_current_texture().create_view()
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

    canvas.request_draw()


canvas.request_draw(draw_frame)

if __name__ == "__main__":
    run()
