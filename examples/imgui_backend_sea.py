"""
An example demonstrating a wgpu app with imgui backend.

# run_example = false
"""

from wgpu.gui.auto import WgpuCanvas, run
import wgpu
import time
import numpy as np
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiWgpuBackend

# Create a canvas to render to
canvas = WgpuCanvas(title="imgui_sea", size=(800, 450), max_fps=60)

# Create a wgpu device
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")

device = adapter.request_device_sync()

# Prepare present context
present_context = canvas.get_context("wgpu")
render_texture_format = wgpu.TextureFormat.bgra8unorm
present_context.configure(device=device, format=render_texture_format)

module = device.create_shader_module(
    code="""
    struct UniformInput {
        resolution: vec2<f32>,
        time: f32,
        SEA_HEIGHT: f32,
        SEA_BASE: vec3<f32>,
        SEA_CHOPPY: f32,
        SEA_WATER_COLOR: vec3<f32>,
        SEA_SPEED: f32,
        SEA_FREQ: f32,
    };

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

    @group(0) @binding(0)
    var<uniform> input: UniformInput;

    const NUM_STEPS = 8;
    const PI = 3.141592;
    const EPSILON = 0.001;
    const ITER_GEOMETRY = 3;
    const ITER_FRAGMENT = 5;
    // const SEA_HEIGHT = 0.6;
    // const SEA_CHOPPY = 4.0;
    // const SEA_SPEED = 0.8;
    // const SEA_FREQ = 0.16;
    // const SEA_BASE = vec3<f32>(0.0,0.09,0.18);
    // const SEA_WATER_COLOR = vec3<f32>(0.48, 0.54, 0.36);

    fn hash( p : vec2<f32> ) -> f32 {

        let h = dot(p,vec2<f32>(1.0,113.0));
        return fract(sin(h)*43758.5453123);
    }

    // Perlin noise, TODO: try simplex noise
    fn noise( p : vec2<f32> ) -> f32 {
        let i = floor(p);
        let f = fract(p);
        let u = f * f * (3.0 - 2.0 * f);
        let mix1 = mix( hash( i + vec2<f32>(0.0,0.0) ), hash( i + vec2<f32>(1.0,0.0) ), u.x);
        let mix2 = mix( hash( i + vec2<f32>(0.0,1.0) ), hash( i + vec2<f32>(1.0,1.0) ), u.x);
        let mix3 = mix(mix1, mix2, u.y);
        return -1.0 + 2.0 * mix3;
    }
    // lighting
    fn diffuse( n : vec3<f32>, l : vec3<f32>, p : f32 ) -> f32 {
        return pow(dot(n,l) * 0.4 + 0.6, p);
    }
    fn specular( n : vec3<f32>, l : vec3<f32>, e : vec3<f32>, s : f32 ) -> f32 {
        let nrm = (s + 8.0) / (PI * 8.0);
        return pow(max(dot(reflect(e,n),l),0.0),s) * nrm;
    }
    // sky
    fn getSkyColor( _e : vec3<f32> ) -> vec3<f32> {
        var e = _e;
        e.y = (max(e.y,0.0) * 0.8 + 0.2) * 0.8;
        return vec3<f32>(pow(1.0-e.y, 2.0), 1.0-e.y, 0.6+(1.0-e.y)*0.4) * 1.1;
    }
    // sea
    fn sea_octave( _uv : vec2<f32>, choppy : f32 ) -> f32 {
        let uv = _uv + noise(_uv);
        var wv = 1.0-abs(sin(uv));
        let swv = abs(cos(uv));
        wv = mix(wv,swv,wv);
        return pow(1.0-pow(wv.x * wv.y,0.65),choppy);
    }
    fn _map( p : vec3<f32>, iter: i32 ) -> f32 {
        var freq = input.SEA_FREQ;
        var amp = input.SEA_HEIGHT;
        var choppy = input.SEA_CHOPPY;
        let sea_time = 1.0 + input.time * input.SEA_SPEED;
        var uv = p.xz;
        uv.x *= 0.75;
        var d = 0.0;
        var h = 0.0;
        for (var i = 0; i < iter; i+=1) {
            d = sea_octave((uv+sea_time)*freq, choppy);
            d += sea_octave((uv-sea_time)*freq, choppy);
            h += d * amp;
            uv *= mat2x2<f32>(1.6, 1.2, -1.2, 1.6);
            freq *= 1.9;
            amp *= 0.22;
            choppy = mix(choppy,1.0,0.2);
        }
        return p.y - h;
    }
    fn map( p : vec3<f32>) -> f32 {
        return _map(p, ITER_GEOMETRY);
    }
    fn map_detailed( p : vec3<f32> ) -> f32 {
        return _map(p, ITER_FRAGMENT);
    }
    fn getSeaColor( p : vec3<f32>, n : vec3<f32>, l : vec3<f32>, eye : vec3<f32>, dist : vec3<f32> ) -> vec3<f32> {
        var fresnel = clamp(1.0 - dot(n,-eye), 0.0, 1.0);
        fresnel = pow(fresnel,3.0) * 0.5;
        let reflected = getSkyColor(reflect(eye,n));
        let refracted = input.SEA_BASE + diffuse(n,l,80.0) * input.SEA_WATER_COLOR * 0.12;
        var color = mix(refracted,reflected,fresnel);
        let atten = max(1.0 - dot(dist,dist) * 0.001, 0.0);
        color += input.SEA_WATER_COLOR * (p.y - input.SEA_HEIGHT) * 0.18 * atten;
        color += vec3<f32>(specular(n, l, eye, 60.0));
        return color;
    }
    // tracing
    fn getNormal( p : vec3<f32>, eps : f32 ) -> vec3<f32> {
        var n : vec3<f32>;
        n.y = map_detailed(p);
        n.x = map_detailed(vec3<f32>(p.x+eps,p.y,p.z)) - n.y;
        n.z = map_detailed(vec3<f32>(p.x,p.y,p.z+eps)) - n.y;
        n.y = eps;
        return normalize(n);
    }
    fn heightMapTracing( ori : vec3<f32>, dir : vec3<f32> ) -> vec3<f32> {
        var tm = 0.0;
        var tx = 1000.0;
        var hx = map(ori + dir * tx);
        var p : vec3<f32>;
        if (hx > 0.0){
            p = ori + dir * tx;
            return p;
        }
        var hm = map(ori + dir * tm);
        var tmid = 0.0;
        for (var i = 0; i < NUM_STEPS; i+=1) {
            tmid = mix(tm,tx, hm/(hm-hx));
            p = ori + dir * tmid;
            let hmid = map(p);
            if (hmid < 0.0) {
                tx = tmid;
                hx = hmid;
            } else {
                tm = tmid;
                hm = hmid;
            }
        }
        return p;
    }
    fn getPixel( coord: vec2<f32>, time: f32 ) -> vec3<f32> {
        var uv = coord / input.resolution.xy;
        uv = uv * 2.0 - 1.0;
        uv.x *= input.resolution.x / input.resolution.y;
        // ray
        let ori = vec3<f32>(0.0,3.5,time*5.0);
        let dir = normalize(vec3<f32>(uv.xy,-2.0));
        // tracing
        var p = heightMapTracing(ori, dir);
        let dist = p - ori;
        let n = getNormal(p, dot(dist,dist) * (0.1/input.resolution.x));
        let light = normalize(vec3<f32>(0.0,1.0,0.8));
        // color
        return mix(
            getSkyColor(dir),
            getSeaColor(p,n,light,dir,dist),
            pow(smoothstep(0.0,-0.02,dir.y),0.2)
        );
    }
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
        let time = input.time * 0.3;
        var color: vec3<f32>;
        for (var i = -1; i <=1; i+=1) {
            for (var j =-1; j <=1; j+=1) {
                let uv = frag_coord + vec2<f32>(f32(i),f32(j)) / 3.0;
                color += getPixel(uv, time);
            }
        }
        color = color / 9.0;
        // let color = getPixel(frag_coord, time);
        return vec4<f32>(pow(color, vec3<f32>(0.65)), 1.0);
    }

    @fragment
    fn fs_main(in: Varyings) -> @location(0) vec4<f32> {
        var uv = in.uv;
        uv.y = 1.0 - uv.y;
        let frag_coord = uv * input.resolution.xy;
        let time = input.time * 0.3;
        var color: vec3<f32>;
        for (var i = -1; i <=1; i+=1) {
            for (var j =-1; j <=1; j+=1) {
                let uv = frag_coord + vec2<f32>(f32(i),f32(j)) / 3.0;
                color += getPixel(uv, time);
            }
        }
        color = color / 9.0;
        // let color = getPixel(frag_coord, time);
        return vec4<f32>(pow(color, vec3<f32>(0.65)), 1.0);
    }
"""
)

pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": module,
        "entry_point": "vs_main",
    },
    fragment={
        "module": module,
        "entry_point": "fs_main",
        "targets": [{"format": render_texture_format}],
    },
)


uniform_data = np.zeros(
    (),
    dtype=[
        ("resolution", "float32", (2)),
        ("time", "float32"),
        ("SEA_HEIGHT", "float32"),
        ("SEA_BASE", "float32", (3)),
        ("SEA_CHOPPY", "float32"),
        ("SEA_WATER_COLOR", "float32", (3)),
        ("SEA_SPEED", "float32"),
        ("SEA_FREQ", "float32"),
        ("__padding", "uint32", (3)),  # padding to 64 bytes
    ],
)
uniform_buffer = device.create_buffer(
    size=uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {
            "binding": 0,
            "resource": {
                "buffer": uniform_buffer,
                "offset": 0,
                "size": uniform_data.nbytes,
            },
        },
    ],
)

render_pass_descriptor = {
    "color_attachments": [
        {
            "clear_value": (0.0, 0.0, 0.0, 1),
            "load_op": "clear",
            "store_op": "store",
        }
    ]
}

app_state = {
    "SEA_HEIGHT": 0.6,
    "SEA_CHOPPY": 4.0,
    "SEA_BASE": (0.0, 0.09, 0.18),
    "SEA_WATER_COLOR": (0.48, 0.54, 0.36),
    "SEA_SPEED": 0.8,
    "SEA_FREQ": 0.16,
}


def gui(app_state):
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0), imgui.Cond_.appearing)
    imgui.set_next_window_size((400, 0), imgui.Cond_.appearing)
    imgui.begin("Shader parameters", None)

    imgui.text('Shader: "Seascape" by Alexander Alekseev aka TDM - 2014\n')
    if imgui.is_item_hovered():
        imgui.set_tooltip("https://www.shadertoy.com/view/Ms2SD1")

    _, app_state["SEA_HEIGHT"] = imgui.slider_float(
        "SEA_HEIGHT", app_state["SEA_HEIGHT"], 0.1, 2.1
    )
    _, app_state["SEA_CHOPPY"] = imgui.slider_float(
        "SEA_CHOPPY", app_state["SEA_CHOPPY"], 0.1, 10.0
    )

    _, app_state["SEA_BASE"] = imgui.color_edit3("SEA_BASE", app_state["SEA_BASE"])
    _, app_state["SEA_WATER_COLOR"] = imgui.color_edit3(
        "SEA_WATER_COLOR", app_state["SEA_WATER_COLOR"]
    )

    _, app_state["SEA_SPEED"] = imgui.slider_float(
        "SEA_SPEED", app_state["SEA_SPEED"], 0.1, 3.0
    )
    _, app_state["SEA_FREQ"] = imgui.slider_float(
        "SEA_FREQ", app_state["SEA_FREQ"], 0.01, 0.5
    )

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


# init imgui backend
imgui.create_context()
imgui_backend = ImguiWgpuBackend(device, render_texture_format)
imgui_backend.io.display_size = canvas.get_logical_size()
imgui_backend.io.display_framebuffer_scale = (
    canvas.get_pixel_ratio(),
    canvas.get_pixel_ratio(),
)


# register event handlers
def on_resize(event):
    imgui_backend.io.display_size = (event["width"], event["height"])


canvas.add_event_handler(on_resize, "resize")


def on_mouse_move(event):
    imgui_backend.io.add_mouse_pos_event(event["x"], event["y"])


canvas.add_event_handler(on_mouse_move, "pointer_move")


def on_mouse(event):
    event_type = event["event_type"]
    down = event_type == "pointer_down"
    imgui_backend.io.add_mouse_button_event(event["button"] - 1, down)


canvas.add_event_handler(on_mouse, "pointer_up", "pointer_down")


global_time = time.perf_counter()


def render():
    global global_time
    current_time = time.perf_counter()
    delta_time = current_time - global_time
    global_time = current_time

    canvas_texture = present_context.get_current_texture()
    ca0 = render_pass_descriptor["color_attachments"][0]
    ca0["view"] = canvas_texture.create_view()

    # Update uniform buffer
    uniform_data["resolution"] = (canvas_texture.size[0], canvas_texture.size[1])
    uniform_data["time"] += delta_time
    uniform_data["SEA_HEIGHT"] = app_state["SEA_HEIGHT"]
    uniform_data["SEA_CHOPPY"] = app_state["SEA_CHOPPY"]
    uniform_data["SEA_BASE"] = app_state["SEA_BASE"]
    uniform_data["SEA_WATER_COLOR"] = app_state["SEA_WATER_COLOR"]
    uniform_data["SEA_SPEED"] = app_state["SEA_SPEED"]
    uniform_data["SEA_FREQ"] = app_state["SEA_FREQ"]

    device.queue.write_buffer(uniform_buffer, 0, uniform_data.tobytes())

    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(**render_pass_descriptor)
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bind_group)
    render_pass.draw(3, 1)

    # draw imgui
    psize = canvas.get_physical_size()
    imgui_data = gui(app_state)
    imgui_backend.render(imgui_data, render_pass, psize)

    render_pass.end()

    device.queue.submit([command_encoder.finish()])


def loop():
    render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(loop)
    run()
