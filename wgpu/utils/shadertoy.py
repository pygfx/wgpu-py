import time
import ctypes

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas, run as run_offscreen

vertex_code_glsl = """
#version 450 core

layout(location = 0) out vec2 uv;

void main(void){
    int index = int(gl_VertexID);
    if (index == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
        uv = vec2(0.0, 1.0);
    } else if (index == 1) {
        gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
        uv = vec2(2.0, 1.0);
    } else {
        gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
        uv = vec2(0.0, -1.0);
    }
}
"""


builtin_variables_glsl = """
#version 450 core

vec3 i_resolution;
vec4 i_mouse;
float i_time;
float i_time_delta;
int i_frame;

// Shadertoy compatibility, see we can use the same code copied from shadertoy website

#define iTime i_time
#define iResolution i_resolution
#define iTimeDelta i_time_delta
#define iMouse i_mouse
#define iFrame i_frame

#define mainImage shader_main
"""


fragment_code_glsl = """
layout(location = 0) in vec2 uv;

struct ShadertoyInput {
    vec4 mouse;
    vec3 resolution;
    float time;
    float time_delta;
    int frame;
};

layout(binding = 0) uniform ShadertoyInput input;
out vec4 FragColor;
void main(){

    i_time = input.time;
    i_resolution = input.resolution;
    i_time_delta = input.time_delta;
    i_mouse = input.mouse;
    i_frame = input.frame;


    vec2 uv = vec2(uv.x, 1.0 - uv.y);
    vec2 frag_coord = uv * i_resolution.xy;

    shader_main(FragColor, frag_coord);

}

"""


vertex_code_wgsl = """

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) index: u32) -> Varyings {
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


builtin_variables_wgsl = """

var<private> i_resolution: vec3<f32>;
var<private> i_mouse: vec4<f32>;
var<private> i_time_delta: f32;
var<private> i_time: f32;
var<private> i_frame: u32;

// TODO: more global variables
// var<private> i_frag_coord: vec2<f32>;

"""


fragment_code_wgsl = """

struct ShadertoyInput {
    mouse: vec4<f32>,
    resolution: vec3<f32>,
    time: f32,
    time_delta: f32,
    frame: u32,
};

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@group(0) @binding(0)
var<uniform> input: ShadertoyInput;


@fragment
fn main(in: Varyings) -> @location(0) vec4<f32> {

    i_time = input.time;
    i_resolution = input.resolution;
    i_time_delta = input.time_delta;
    i_mouse = input.mouse;
    i_frame = input.frame;


    let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    let frag_coord = uv * i_resolution.xy;

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


class UniformArray:
    """Convenience class to create a uniform array.

    Maybe we can make it a public util at some point.
    """

    def __init__(self, *args):
        # Analyse incoming fields
        fields = []
        byte_offet = 0
        for name, format, n in args:
            assert format in ("f", "i", "I")
            field = name, format, byte_offet, byte_offet + n * 4
            fields.append(field)
            byte_offet += n * 4
        # Get padding
        nbytes = byte_offet
        while nbytes % 16:
            nbytes += 1
        # Construct memoryview object and a view for each field
        self._mem = memoryview((ctypes.c_uint8 * nbytes)()).cast("B")
        self._views = {}
        for name, format, i1, i2 in fields:
            self._views[name] = self._mem[i1:i2].cast(format)

    @property
    def mem(self):
        return self._mem

    @property
    def nbytes(self):
        return self._mem.nbytes

    def __getitem__(self, key):
        v = self._views[key].tolist()
        return v[0] if len(v) == 1 else v

    def __setitem__(self, key, val):
        m = self._views[key]
        n = m.shape[0]
        if n == 1:
            assert isinstance(val, (float, int))
            m[0] = val
        else:
            assert isinstance(val, (tuple, list))
            for i in range(n):
                m[i] = val[i]


class Shadertoy:
    """Provides a "screen pixel shader programming interface" similar to `shadertoy <https://www.shadertoy.com/>`_.

    It helps you research and quickly build or test shaders using `WGSL` or `GLSL` via WGPU.

    Parameters:
        shader_code (str): The shader code to use.
        resolution (tuple): The resolution of the shadertoy.
        offscreen (bool): Whether to render offscreen. Default is False.

    The shader code must contain a entry point function:

    WGSL: ``fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{}``
    GLSL: ``void shader_main(out vec4 frag_color, in vec2 frag_coord){}``

    It has a parameter ``frag_coord`` which is the current pixel coordinate (in range 0..resolution, origin is bottom-left),
    and it must return a vec4<f32> color (for GLSL, it's the ``out vec4 frag_color`` parameter), which is the color of the pixel at that coordinate.

    some built-in variables are available in the shader:

    * ``i_time``: the global time in seconds
    * ``i_time_delta``: the time since last frame in seconds
    * ``i_frame``: the frame number
    * ``i_resolution``: the resolution of the shadertoy
    * ``i_mouse``: the mouse position in pixels

    For GLSL, you can also use the aliases ``iTime``, ``iTimeDelta``, ``iFrame``, ``iResolution``, and ``iMouse`` of these built-in variables,
    the entry point function also has an alias ``mainImage``, so you can use the shader code copied from shadertoy website without making any changes.
    """

    # todo: add more built-in variables
    # todo: support input textures
    # todo: support multiple render passes (`i_channel0`, `i_channel1`, etc.)

    def __init__(self, shader_code, resolution=(800, 450), offscreen=False) -> None:
        self._uniform_data = UniformArray(
            ("mouse", "f", 4),
            ("resolution", "f", 3),
            ("time", "f", 1),
            ("time_delta", "f", 1),
            ("frame", "I", 1),
        )

        self._shader_code = shader_code
        self._uniform_data["resolution"] = resolution + (1,)

        self._offscreen = offscreen

        self._prepare_render()
        self._bind_events()

    @property
    def resolution(self):
        """The resolution of the shadertoy as a tuple (width, height) in pixels."""
        return tuple(self._uniform_data["resolution"])[:2]

    @property
    def shader_code(self):
        """The shader code to use."""
        return self._shader_code

    @property
    def shader_type(self):
        """The shader type, automatically detected from the shader code, can be "wgsl" or "glsl"."""
        if "fn shader_main" in self.shader_code:
            return "wgsl"
        elif (
            "void shader_main" in self.shader_code
            or "void mainImage" in self.shader_code
        ):
            return "glsl"
        else:
            raise ValueError("Invalid shader code.")

    def _prepare_render(self):
        import wgpu.backends.auto  # noqa

        if self._offscreen:
            self._canvas = OffscreenCanvas(
                title="Shadertoy", size=self.resolution, max_fps=60
            )
        else:
            self._canvas = WgpuCanvas(
                title="Shadertoy", size=self.resolution, max_fps=60
            )

        self._device = wgpu.utils.device.get_default_device()

        self._present_context = self._canvas.get_context()

        # We use "bgra8unorm" not "bgra8unorm-srgb" here because we want to let the shader fully control the color-space.
        self._present_context.configure(
            device=self._device, format=wgpu.TextureFormat.bgra8unorm
        )

        shader_type = self.shader_type
        if shader_type == "glsl":
            vertex_shader_code = vertex_code_glsl
            frag_shader_code = (
                builtin_variables_glsl + self.shader_code + fragment_code_glsl
            )
        elif shader_type == "wgsl":
            vertex_shader_code = vertex_code_wgsl
            frag_shader_code = (
                builtin_variables_wgsl + self.shader_code + fragment_code_wgsl
            )

        vertex_shader_program = self._device.create_shader_module(
            label="triangle_vert", code=vertex_shader_code
        )
        frag_shader_program = self._device.create_shader_module(
            label="triangle_frag", code=frag_shader_code
        )

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
                "module": vertex_shader_program,
                "entry_point": "main",
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
                "module": frag_shader_program,
                "entry_point": "main",
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
            self._uniform_data["resolution"] = (w, h, 1)

        def on_mouse_move(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                _, _, x2, y2 = self._uniform_data["mouse"]
                x1, y1 = event["x"], self.resolution[1] - event["y"]
                self._uniform_data["mouse"] = x1, y1, x2, y2

        def on_mouse_down(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                x, y = event["x"], self.resolution[1] - event["y"]
                self._uniform_data["mouse"] = (x, y, x, -y)

        def on_mouse_up(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                x1, y1, x2, y2 = self._uniform_data["mouse"]
                self._uniform_data["mouse"] = x1, y1, abs(x2), y2

        self._canvas.add_event_handler(on_resize, "resize")
        self._canvas.add_event_handler(on_mouse_move, "pointer_move")
        self._canvas.add_event_handler(on_mouse_down, "pointer_down")
        self._canvas.add_event_handler(on_mouse_up, "pointer_up")

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
            self._uniform_buffer,
            0,
            self._uniform_data.mem,
            0,
            self._uniform_data.nbytes,
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
        if self._offscreen:
            run_offscreen()
        else:
            run()

    def snapshot(self, time_float: float = 0.0, mouse_pos: tuple = (0, 0, 0, 0)):
        """
        Returns an image of the specified time. (Only available when ``offscreen=True``)

        Parameters:
            time_float (float): The time to snapshot. It essentially sets ``i_time`` to a specific number. (Default is 0.0)
            mouse_pos (tuple): The mouse position in pixels in the snapshot. It essentially sets ``i_mouse`` to a 4-tuple. (Default is (0,0,0,0))
        Returns:
            frame (memoryview): snapshot with transparancy. This object can be converted to a numpy array (without copying data)
        using ``np.asarray(arr)``
        """
        if not self._offscreen:
            raise NotImplementedError("Snapshot is only available in offscreen mode.")

        if hasattr(self, "_last_time"):
            self.__delattr__("_last_time")
        self._uniform_data["time"] = time_float
        self._uniform_data["mouse"] = mouse_pos
        self._canvas.request_draw(self._draw_frame)
        frame = self._canvas.draw()
        return frame


if __name__ == "__main__":
    shader = Shadertoy(
        """
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
        let uv = frag_coord / i_resolution.xy;

        if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }else{
            return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
        }

    }
    """
    )

    shader.show()
