import time

import numpy as np

import wgpu
from wgpu.gui.auto import WgpuCanvas, run


vertex_code = """

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
    var out: Varyings;
    if (index == u32(0)) {
        out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, 1.0);
    } else if (index == u32(1)) {
        out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(2.0, 1.0);
    } else {
        out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, -1.0);
    }
    return out;

}
"""

builtin_variables = """

var<private> i_time: f32;
var<private> i_resolution: vec2<f32>;
var<private> i_time_delta: f32;
var<private> i_mouse: vec2<f32>;
var<private> i_frame: u32;

// TODO: more global variables
// var<private> i_frag_coord: vec2<f32>;

"""

fragment_code = """

struct ShadertoyInput {
    resolution: vec2<f32>,
    mouse: vec2<f32>,
    time: f32,
    time_delta: f32,
    frame: u32,
};

@group(0) @binding(0)
var<uniform> input: ShadertoyInput;


@fragment
fn fs_main(in: Varyings) -> @location(0) vec4<f32> {

    i_time = input.time;
    i_resolution = input.resolution;
    i_time_delta = input.time_delta;
    i_mouse = input.mouse;
    i_mouse.y = i_resolution.y - i_mouse.y;
    i_frame = input.frame;


    let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    let frag_coord = uv * i_resolution;

    return shader_main(frag_coord);
}

 """

binding_layout = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.FRAGMENT,
        "buffer": {"type": wgpu.BufferBindingType.uniform},
    }
]

uniform_dtype = [
    ("resolution", "float32", (2)),
    ("mouse", "float32", (2)),
    ("time", "float32"),
    ("time_delta", "float32"),
    ("frame", "uint32"),
    ("__padding", "uint8", (4)),  # padding to 32 bytes
]


class Shadertoy:
    """Provides a "screen pixel shader programming interface" similar to `shadertoy <https://www.shadertoy.com/>`_.

    It helps you research and quickly build or test shaders using WGSL via WGPU.

    Parameters:
        shader_code (str): The shader code to use.
        resolution (tuple): The resolution of the shadertoy.

    The shader code must contain a entry point function:

    ``fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{}``

    It has a parameter ``frag_coord`` which is the current pixel coordinate (in range 0..resolution, origin is bottom-left),
    and it must return a vec4<f32> color, which is the color of the pixel at that coordinate.

    some built-in variables are available in the shader:

    * ``i_time``: the global time in seconds
    * ``i_time_delta``: the time since last frame in seconds
    * ``i_frame``: the frame number
    * ``i_resolution``: the resolution of the shadertoy
    * ``i_mouse``: the mouse position in pixels

    """

    # todo: add more built-in variables
    # todo: support input textures
    # todo: support multiple render passes (`i_channel0`, `i_channel1`, etc.)

    def __init__(self, shader_code, resolution=(800, 450)) -> None:
        self._uniform_data = np.zeros((), dtype=uniform_dtype)

        self._shader_code = shader_code
        self._uniform_data["resolution"] = resolution

        self._prepare_render()
        self._bind_events()

    @property
    def resolution(self):
        """The resolution of the shadertoy as a tuple (width, height) in pixels."""
        return tuple(self._uniform_data["resolution"])

    @property
    def shader_code(self):
        """The shader code to use."""
        return self._shader_code

    def _prepare_render(self):
        import wgpu.backends.rs  # noqa

        self._canvas = WgpuCanvas(title="Shadertoy", size=self.resolution, max_fps=60)

        adapter = wgpu.request_adapter(
            canvas=self._canvas, power_preference="high-performance"
        )
        self._device = adapter.request_device()

        self._present_context = self._canvas.get_context()

        # We use "bgra8unorm" not "bgra8unorm-srgb" here because we want to let the shader fully control the color-space.
        self._present_context.configure(
            device=self._device, format=wgpu.TextureFormat.bgra8unorm
        )

        shader_code = vertex_code + builtin_variables + self.shader_code + fragment_code
        shader_program = self._device.create_shader_module(code=shader_code)

        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        bind_group_layout = self._device.create_bind_group_layout(
            entries=binding_layout
        )

        self._bind_group = self._device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniform_buffer,
                        "offset": 0,
                        "size": self._uniform_data.nbytes,
                    },
                },
            ],
        )

        self._render_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[bind_group_layout]
            ),
            vertex={
                "module": shader_program,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": shader_program,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.bgra8unorm,
                        "blend": {
                            "color": (
                                wgpu.BlendFactor.one,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                            "alpha": (
                                wgpu.BlendFactor.one,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                        },
                    },
                ],
            },
        )

    def _bind_events(self):
        def on_resize(event):
            w, h = event["width"], event["height"]
            self._uniform_data["resolution"] = (w, h)

        def on_mouse_move(event):
            xy = event["x"], event["y"]
            if event["button"] == 1 or 1 in event["buttons"]:
                self._uniform_data["mouse"] = xy

        self._canvas.add_event_handler(on_resize, "resize")
        self._canvas.add_event_handler(on_mouse_move, "pointer_move", "pointer_down")

    def _update(self):
        now = time.perf_counter()
        if not hasattr(self, "_last_time"):
            self._last_time = now

        time_delta = now - self._last_time
        self._uniform_data["time_delta"] = time_delta
        self._last_time = now
        self._uniform_data["time"] += time_delta

        if not hasattr(self, "_frame"):
            self._frame = 0

        self._uniform_data["frame"] = self._frame
        self._frame += 1

    def _draw_frame(self):
        # Update uniform buffer
        self._update()
        self._device.queue.write_buffer(
            self._uniform_buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

        command_encoder = self._device.create_command_encoder()

        current_texture_view = self._present_context.get_current_texture()

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

        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()

        self._device.queue.submit([command_encoder.finish()])

        self._canvas.request_draw()

    def show(self):
        self._canvas.request_draw(self._draw_frame)
        run()


if __name__ == "__main__":
    shader = Shadertoy(
        """
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
        let uv = frag_coord / i_resolution;

        if ( length(frag_coord - i_mouse) < 20.0 ) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }else{
            return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
        }

    }
    """
    )

    shader.show()
