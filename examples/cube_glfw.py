"""
This example renders a simple textured rotating cube.
"""

import time
import ctypes

import glfw
import wgpu
from wgpu.gui.glfw import update_glfw_canvasses, WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend
import numpy as np
from pyshader import python2shader, shadertype_as_ctype
from pyshader import Struct, mat4, vec4, vec2


# %% Create canvas and device

# Create a canvas to render to
glfw.init()
canvas = WgpuCanvas(title="wgpu cube with GLFW")

# Create a wgpu device
adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
device = adapter.request_device(extensions=[], limits={})


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
texture_size = texture_data.shape[1], texture_data.shape[0], 1


uniform_type = Struct(transform=mat4)
uniform_data = np.asarray(shadertype_as_ctype(uniform_type)())


# %% Create resource objects (buffers, textures, samplers)

# Create vertex buffer, and upload data
vertex_buffer = device.create_buffer_mapped(
    size=vertex_data.nbytes, usage=wgpu.BufferUsage.VERTEX
)
ctypes.memmove(
    ctypes.addressof(vertex_buffer.mapping),
    vertex_data.ctypes.data,
    vertex_data.nbytes,
)
vertex_buffer.unmap()


# Create index buffer, and upload data
index_buffer = device.create_buffer_mapped(
    size=index_data.nbytes, usage=wgpu.BufferUsage.INDEX
)
ctypes.memmove(
    ctypes.addressof(index_buffer.mapping), index_data.ctypes.data, index_data.nbytes,
)
index_buffer.unmap()


# Create uniform buffer - data is uploaded each frame
uniform_buffer = device.create_buffer(
    size=uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)


# Create texture, and upload data
texture = device.create_texture(
    size=texture_size,
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.SAMPLED,
    dimension=wgpu.TextureDimension.d2,
    format=wgpu.TextureFormat.r8uint,
    mip_level_count=1,
    sample_count=1,
)
texture_view = texture.create_view()
tmp_buffer = device.create_buffer_mapped(
    size=texture_data.nbytes, usage=wgpu.BufferUsage.COPY_SRC
)
ctypes.memmove(
    ctypes.addressof(tmp_buffer.mapping), texture_data.ctypes.data, texture_data.nbytes,
)
vertex_buffer.unmap()
command_encoder = device.create_command_encoder()
command_encoder.copy_buffer_to_texture(
    {
        "buffer": tmp_buffer,
        "offset": 0,
        "bytes_per_row": texture_data.strides[0],
        "rows_per_image": 0,
    },
    {"texture": texture_view, "mip_level": 0, "origin": (0, 0, 0),},
    copy_size=texture_size,
)
device.default_queue.submit([command_encoder.finish()])


# Create a sampler
sampler = device.create_sampler()


# %% The shaders

# Define the bindings (bind_group, slot). These are used in the shader
# creating, and further down where the bind groups and bind group
# layouts are created.
UNIFORM_BINDING = 0, 0
SAMPLER_BINDING = 0, 1
TEXTURE_BINDING = 0, 2


@python2shader
def vertex_shader(
    in_pos: ("input", 0, vec4),
    in_texcoord: ("input", 1, vec2),
    out_pos: ("output", "Position", vec4),
    v_texcoord: ("output", 0, vec2),
    u_locals: ("uniform", UNIFORM_BINDING, uniform_type),
):
    ndc = u_locals.transform * in_pos
    out_pos = vec4(ndc.xy, 0, 1)  # noqa - shader output
    v_texcoord = in_texcoord  # noqa - shader output


@python2shader
def fragment_shader(
    v_texcoord: ("input", 0, vec2),
    s_sam: ("sampler", SAMPLER_BINDING, ""),
    t_tex: ("texture", TEXTURE_BINDING, "2d i32"),
    out_color: ("output", 0, vec4),
):
    value = f32(t_tex.sample(s_sam, v_texcoord).r) / 255.0
    out_color = vec4(value, value, value, 1.0)  # noqa - shader output


vshader = device.create_shader_module(code=vertex_shader)
fshader = device.create_shader_module(code=fragment_shader)


# %% The bind groups

# We always have two bind groups, so we can play distributing our
# resources over these two groups in different configurations.
bind_groups_entries = [], []
bind_groups_layout_entries = [], []

bind_groups_entries[UNIFORM_BINDING[0]].append(
    {
        "binding": UNIFORM_BINDING[1],
        "resource": {
            "buffer": uniform_buffer,
            "offset": 0,
            "size": uniform_buffer.size,
        },
    }
)
bind_groups_layout_entries[UNIFORM_BINDING[0]].append(
    {
        "binding": UNIFORM_BINDING[1],
        "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
        "type": wgpu.BindingType.uniform_buffer,
    }
)

bind_groups_entries[SAMPLER_BINDING[0]].append(
    {"binding": SAMPLER_BINDING[1], "resource": sampler}
)
bind_groups_layout_entries[SAMPLER_BINDING[0]].append(
    {
        "binding": SAMPLER_BINDING[1],
        "visibility": wgpu.ShaderStage.FRAGMENT,
        "type": wgpu.BindingType.sampler,
    }
)

bind_groups_entries[TEXTURE_BINDING[0]].append(
    {"binding": TEXTURE_BINDING[1], "resource": texture_view}
)
bind_groups_layout_entries[TEXTURE_BINDING[0]].append(
    {
        "binding": TEXTURE_BINDING[1],
        "visibility": wgpu.ShaderStage.FRAGMENT,
        "type": wgpu.BindingType.sampled_texture,
        "view_dimension": wgpu.TextureViewDimension.d2,
        "texture_component_type": wgpu.TextureComponentType.uint,
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
    vertex_stage={"module": vshader, "entry_point": "main"},
    fragment_stage={"module": fshader, "entry_point": "main"},
    primitive_topology=wgpu.PrimitiveTopology.triangle_list,
    rasterization_state={
        "front_face": wgpu.FrontFace.ccw,
        "cull_mode": wgpu.CullMode.back,
        "depth_bias": 0,
        "depth_bias_slope_scale": 0.0,
        "depth_bias_clamp": 0.0,
    },
    color_states=[
        {
            "format": wgpu.TextureFormat.bgra8unorm_srgb,
            "alpha_blend": (
                wgpu.BlendFactor.one,
                wgpu.BlendFactor.zero,
                wgpu.BlendOperation.add,
            ),
            "color_blend": (
                wgpu.BlendFactor.one,
                wgpu.BlendFactor.zero,
                wgpu.BlendOperation.add,
            ),
        }
    ],
    vertex_state={
        "index_format": wgpu.IndexFormat.uint32,
        "vertex_buffers": [
            {
                "array_stride": 4 * 6,
                "step_mode": wgpu.InputStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float4,
                        "offset": 0,
                        "shader_location": 0,
                    },
                    {
                        "format": wgpu.VertexFormat.float2,
                        "offset": 4 * 4,
                        "shader_location": 1,
                    },
                ],
            },
        ],
    },
    sample_count=1,
    sample_mask=0xFFFFFFFF,
    alpha_to_coverage_enabled=False,
)


# %% Setup the render function


swap_chain = device.configure_swap_chain(
    canvas,
    device.get_swap_chain_preferred_format(canvas),
    wgpu.TextureUsage.OUTPUT_ATTACHMENT,
)


def draw_frame():

    # Update uniform transform
    a1 = -0.3
    a2 = time.time()
    s = 0.6
    ortho = np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1],],)
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
    uniform_data["transform"] = (rot2 @ rot1 @ ortho).flat

    # Upload the uniform struct
    uniform_nbytes = uniform_data.nbytes
    tmp_buffer = device.create_buffer_mapped(
        size=uniform_nbytes, usage=wgpu.BufferUsage.COPY_SRC
    )
    ctypes.memmove(
        ctypes.addressof(tmp_buffer.mapping), uniform_data.ctypes.data, uniform_nbytes
    )
    tmp_buffer.unmap()

    with swap_chain as current_texture_view:
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(
            tmp_buffer, 0, uniform_buffer, 0, uniform_nbytes
        )

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "attachment": current_texture_view,
                    "resolve_target": None,
                    "load_value": (0.1, 0.3, 0.2, 1),
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(render_pipeline)
        render_pass.set_index_buffer(index_buffer)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        for bind_group_id, bind_group in enumerate(bind_groups):
            render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
        render_pass.draw_indexed(index_data.size, 1, 0, 0, 0)

        render_pass.end_pass()
        device.default_queue.submit([command_encoder.finish()])

    canvas.request_draw()


canvas.draw_frame = draw_frame


# %% Run the event loop


def simple_event_loop():
    """ A real simple event loop, but it keeps the CPU busy. """
    while update_glfw_canvasses():
        glfw.poll_events()


simple_event_loop()
glfw.terminate()
