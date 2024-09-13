import numpy as np
import pytest
import numpy.testing as npt


import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import GPUValidationError, TextureFormat

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


"""
This code is an amazingly slow way of adding together two 10-element arrays of 32-bit
integers defined by push constants and store them into an output buffer.


The source code assumes the topology is POINT-LIST, so that each call to vertexMain
corresponds with one call to fragmentMain. We draw exactly one point.
"""

SHADER_SOURCE = """
    override a: i32 = 1;
    override b: u32 = 2u;
    override c: f32 = 3.0;
    override d: bool;   // We must specify this!
    @id(1) override aa: i32 = 10;
    @id(2) override bb: u32 = 20u;
    @id(3) override cc: f32 = 30.0;
    @id(4) override dd: bool = false;

    // Put the results here
    @group(0) @binding(0) var<storage, read_write> data: array<u32>;

    struct VertexOutput {
        @location(0) results1: vec4u,
        @location(1) results2: vec4u,
        @builtin(position) position: vec4f,
    }

    @vertex
    fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
        var output: VertexOutput;
        output.position = vec4f(0, 0, 0, 1);
        output.results1 = vec4u(u32(a), u32(b), u32(c), u32(d));
        output.results2 = vec4u(u32(aa), u32(bb), u32(cc), u32(dd));
        return output;
    }

    @fragment
    fn fragment(output: VertexOutput) -> @location(0) vec4f {
        var i: u32;
        let results1 = vec4u(u32(a), u32(b), u32(c), u32(d));
        let results2 = vec4u(u32(aa), u32(bb), u32(cc), u32(dd));
        write_results(results1, results2);

        // write_results(output.results1, output.results2);
        return vec4f();
    }

    @compute @workgroup_size(1)
    fn computeMain() {
        let results1 = vec4u(u32(a), u32(b), u32(c), u32(d));
        let results2 = vec4u(u32(aa), u32(bb), u32(cc), u32(dd));
        write_results(results1, results2);
    }
    
    fn write_results(results1: vec4u, results2: vec4u) {
        for (var i = 0; i < 4; i++) { 
            data[i] = results1[i];
            data[i+4] = results2[i];
        }
    }
"""

BIND_GROUP_ENTRIES = [
    {"binding": 0, "visibility": "FRAGMENT|COMPUTE", "buffer": {"type": "storage"}},
]


def run_override_constant_test(
    use_render=True,
    *,
    compute_constants=None,
    vertex_constants=None,
    fragment_constants=None
):
    device = wgpu.utils.get_default_device()
    output_texture = device.create_texture(
        # Actual size is immaterial.  Could just be 1x1
        size=[128, 128],
        format=TextureFormat.rgba8unorm,
        usage="RENDER_ATTACHMENT|COPY_SRC",
    )
    shader = device.create_shader_module(code=SHADER_SOURCE)
    bind_group_layout = device.create_bind_group_layout(entries=BIND_GROUP_ENTRIES)
    render_pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout],
    )
    if use_render:
        pipeline = device.create_render_pipeline(
            layout=render_pipeline_layout,
            vertex={
                "module": shader,
                "constants": vertex_constants,
            },
            fragment={
                "module": shader,
                "targets": [{"format": output_texture.format}],
                "constants": fragment_constants,
            },
            primitive={
                "topology": "point-list",
            },
        )
        render_pass_descriptor = {
            "color_attachments": [
                {
                    "clear_value": (0, 0, 0, 0),  # only first value matters
                    "load_op": "clear",
                    "store_op": "store",
                    "view": output_texture.create_view(),
                }
            ],
        }
    else:
        pipeline = device.create_compute_pipeline(
            layout=render_pipeline_layout,
            compute={
                "module": shader,
                "constants": compute_constants,
            },
        )
        render_pass_descriptor = {}

    output_buffer = device.create_buffer(size=8 * 4, usage="STORAGE|COPY_SRC")
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": output_buffer}},
        ],
    )

    encoder = device.create_command_encoder()
    if use_render:
        this_pass = encoder.begin_render_pass(**render_pass_descriptor)
    else:
        this_pass = encoder.begin_compute_pass()
    this_pass.set_pipeline(pipeline)
    this_pass.set_bind_group(0, bind_group)
    if use_render:
        this_pass.draw(1)
    else:
        this_pass.dispatch_workgroups(1)
    this_pass.end()
    device.queue.submit([encoder.finish()])
    info_view = device.queue.read_buffer(output_buffer)
    result = np.frombuffer(info_view, dtype=np.uint32)
    print(result)
    return list(result)


def test_no_constants():
    with pytest.raises(GPUValidationError):
        run_override_constant_test(use_render=True)

    with pytest.raises(GPUValidationError):
        run_override_constant_test(use_render=False)


def test_with_minimal_constants():
    run_override_constant_test(
        use_render=True, vertex_constants={"d": 1}, fragment_constants={"d": 0}
    )
    run_override_constant_test(use_render=False, compute_constants={"d": 1})
    pytest.fail("hello")


if __name__ == "__main__":
    run_tests(globals())
