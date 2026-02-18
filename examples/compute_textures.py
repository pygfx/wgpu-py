"""
Compute Textures
----------------

Example that shows how to use textures in a compute shader to convert an RGBA image to YCbCr.

The shader uses workgroups to processes non-overlapping 8x8 blocks of the input rgba texture.
"""

# run_example = false
#
# Note: This example segfaults on CI (as of 08-02-2026 or so) even though it has
# worked before. It seems to be due to a regression in Lavapipe. Investigation
# showed that the segfault happened in queue.submit() for the part where the
# compute shader is run. The compute shader does not seem to have any anomalies.
# It writes to two textures. When the writing to either one is disabled
# (commenting the respective for-loop) and the corresponding texture removed
# from the bind group, the example runs fine.

import numpy as np
import wgpu
import imageio.v3 as iio


def size_from_array(data, dim):
    # copied from pygfx
    # Check if shape matches dimension
    shape = data.shape

    if len(shape) not in (dim, dim + 1):
        raise ValueError(
            f"Can't map shape {shape} on {dim}D tex. Maybe also specify size?"
        )
    # Determine size based on dim and shape
    if dim == 1:
        return shape[0], 1, 1
    elif dim == 2:
        return shape[1], shape[0], 1
    else:  # dim == 3:
        return shape[2], shape[1], shape[0]


# get example image, add alpha channel of all ones
# image = iio.imread("imageio:astronaut.png")
# for pyodide compatibility right now.
from pyodide.http import pyfetch
from io import BytesIO
response = await pyfetch("https://raw.githubusercontent.com/imageio/imageio-binaries/master/images/astronaut.png")

image = iio.imread(BytesIO(await response.bytes()))

image_rgba = np.zeros((*image.shape[:-1], 4), dtype=np.uint8)
image_rgba[..., :-1] = image
image_rgba[..., -1] = 255

# wgpu texture size is (width, height) instead of (rows, cols) for whatever reason
rgba_size = size_from_array(image_rgba, dim=2)

# output
y_size = size_from_array(image_rgba[:, :, 0], dim=2)
cbcr_size = size_from_array(image_rgba[::2, ::2, 0], dim=2)

# create device
device: wgpu.GPUDevice = wgpu.utils.get_default_device()

# create texture for input rgba image
texture_rgb = device.create_texture(
    label="rgba",
    size=rgba_size,
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    format=wgpu.TextureFormat.rgba8unorm,
    mip_level_count=1,
    sample_count=1,
)

# write input texture to device queue
device.queue.write_texture(
    {
        "texture": texture_rgb,
        "mip_level": 0,
        "origin": (0, 0, 0),
    },
    image_rgba,
    {
        "offset": 0,
        "bytes_per_row": image.shape[0] * 4,
    },
    rgba_size,
)

# texture for Y channel output
texture_y = device.create_texture(
    label="y",
    size=y_size,
    # use as storage texture since we do not need to sample it
    # COPY_SRC so we can copy the texture back from the gpu
    usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    # NOTE: we cannot use r8unorm for storage textures!!
    format=wgpu.TextureFormat.r32float,
    mip_level_count=1,
    sample_count=1,
)

# sample we will use to generate the CbCr channels
chroma_sampler = device.create_sampler(
    # I don't think min filtering actually occurs for chroma sampling
    # since we are always sampling from the center of 4 pixels to create 1 subsampled new CbCr pixel
    min_filter=wgpu.FilterMode.linear,
    mag_filter=wgpu.FilterMode.linear,
)

# texture for CbCr channels
texture_cbcr = device.create_texture(
    label="uv",
    size=cbcr_size,
    # use as storage texture since we do not need to sample it
    usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    # we will use rg32float so we can store a pair of 2D textures for the Cb and Cr channels
    format=wgpu.TextureFormat.rg32float,
    mip_level_count=1,
    sample_count=1,
)

shader_src = """
@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

// note that we use r32float and write for the output textures
@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var tex_cbcr: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var chroma_sampler: sampler;

// block size
override group_size_x: u32;
override group_size_y: u32;


@compute @workgroup_size(group_size_x, group_size_y)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    // wid.xy is the workgroup invocation ID
    // to get the starting (x, y) texture coordinate for a given (8, 8) block we must multiply by workgroup size
    // Example:
    //  workgroup invocation id (0, 0) becomes texture coord (0, 0)
    //  workgroup invocation id (1, 0) becomes texture coord (8, 0) (block size is 8x8)
    //  workgroup invocation id (1, 1) becomes texture coord (8, 8)
    // We can iterate through pixels within this block by just adding to this starting (x, y) position
    // upto the max position which is (start_x + group_size_x, start_y + group_size_y)

    // start and stop indices for this block
    let start = wid.xy * vec2u(group_size_x, group_size_y);
    let stop = start + vec2u(group_size_x, group_size_y);

    // write luminance for each pixel in this block
    for (var x: u32 = start.x; x < stop.x; x++) {
        for (var y: u32 = start.y; y < stop.y; y++) {
            let pos = vec2u(x, y);

            // read array element i.e. "pixel" value
            var px: vec4f = textureLoad(tex_rgba, pos, 0);

            // create luminance channel by converting to grayscale
            var L: f32 = (0.299 * px.r + 0.587 * px.g + 0.114 * px.b);

            // store luma channel
            textureStore(tex_y, pos, vec4<f32>(L, 0, 0, 0));
        }
    }

    // chroma subsampling
    for (var x: u32 = start.x; x < stop.x; x += 2) {
        for (var y: u32 = start.y; y < stop.y; y += 2) {
            // convert to normalized uv coords for sampler
            let coords_sample: vec2f = (vec2f(f32(x), f32(y)) + 0.5) / vec2f(textureDimensions(tex_rgba).xy);

            var px_sample: vec4f = textureSampleLevel(tex_rgba, chroma_sampler, coords_sample, 0.0);

            // create cb, cr channels
            var cb: f32 = (-0.1687 * px_sample.r - 0.3313 * px_sample.g + 0.5 * px_sample.b) + 0.5;
            var cr: f32 = (0.5 * px_sample.r - 0.4187 * px_sample.g - 0.0813 * px_sample.b) + 0.5;
            let pos_out: vec2u = vec2u(x / 2, y / 2);
            textureStore(tex_cbcr, pos_out.xy, vec4<f32>(cb, cr, 0, 0));
        }
    }
}
"""

shader_module = device.create_shader_module(code=shader_src)

# compute in 8 x 8 blocks
workgroup_size = 8

workgroup_size_constants = {
    "group_size_x": workgroup_size,
    "group_size_y": workgroup_size,
}

# create compute pipeline
pipeline: wgpu.GPUComputePipeline = device.create_compute_pipeline(
    layout=wgpu.AutoLayoutMode.auto,
    compute={
        "module": shader_module,
        "entry_point": "main",
        "constants": workgroup_size_constants,
    },
)

# create bindings for the texture resources and sampler
bindings = [
    {"binding": 0, "resource": texture_rgb.create_view()},
    {"binding": 1, "resource": texture_y.create_view()},
    {"binding": 2, "resource": texture_cbcr.create_view()},
    {
        "binding": 3,
        "resource": chroma_sampler,
    },
]

# set layout
layout = pipeline.get_bind_group_layout(0)
bind_group = device.create_bind_group(layout=layout, entries=bindings)

# make sure we have enough workgroups to process all blocks of the input image
# each workgroup will process the pixels within one 8x8 block
# the blocks are non-overlapping
workgroups = np.ceil(np.asarray(image.shape[:2]) / workgroup_size).astype(int)

# encode, submit
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(pipeline)
compute_pass.set_bind_group(0, bind_group)
compute_pass.dispatch_workgroups(*workgroups, 1)
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# read luminance output
buffer_y = device.queue.read_texture(
    source={
        "texture": texture_y,
        "origin": (0, 0, 0),
        "mip_level": 0,
    },
    data_layout={
        "offset": 0,
        "bytes_per_row": image.shape[1] * 4,
    },
    size=size_from_array(image[:, :, 0], dim=2),
).cast("f")

# read CbCr output
buffer_cbcr = device.queue.read_texture(
    source={
        "texture": texture_cbcr,
        "origin": (0, 0, 0),
        "mip_level": 0,
    },
    data_layout={
        "offset": 0,
        "bytes_per_row": image.shape[1] * 4,
    },
    size=size_from_array(image[::2, ::2, :2], dim=2),
).cast("f")

# create numpy arrays
Y = np.frombuffer(buffer_y, dtype=np.float32).reshape(image.shape[:2])
CbCr = np.frombuffer(buffer_cbcr, dtype=np.float32).reshape(*image[::2, ::2, :2].shape)

# quick rendercanvas based result visualization (round trip to bitmap instead of using existing texture...)
from rendercanvas.auto import RenderCanvas

channels = [Y, CbCr[..., 0], CbCr[..., 1], image_rgba/255]
channel_idx = 0
canvas = RenderCanvas(present_method="bitmap")
context = canvas.get_bitmap_context()
context.set_bitmap((channels[channel_idx]*255).astype(np.uint8))
canvas.request_draw()

@canvas.add_event_handler("pointer_down")
def on_pointer_down(event):
    global channel_idx
    channel_idx = (channel_idx + 1) % len(channels)
    context.set_bitmap((channels[channel_idx]*255).astype(np.uint8))
    canvas.request_draw()


# view result with fastplotlib ImageWidget
# import fastplotlib as fpl
#
# iw = fpl.ImageWidget(
#     data=[Y, CbCr[..., 0], CbCr[..., 1],],
#     names=["Y", "Cb", "Cr"],
#     figure_shape=(1, 3),
#     figure_kwargs={"size": (1000, 400), "controller_ids": None},
#     cmap="viridis"
# )
#
# iw.show()
# fpl.loop.run()
