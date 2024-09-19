import pytest

import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import GPUValidationError, TextureFormat

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


"""
The vertex shader should be called exactly once, which then calls the fragment shader
exactly once. Alternatively, we call the compute shader exactly once

This copies the values of the four variables a, b, c, and d as seen by each of the shaders
and writes it into a buffer.  We can then examine that buffer to see the values of the
constants.

This code is also showing that you no longer need to include the name of a shader when
it is the only shader of that type.
"""

SHADER_SOURCE = """
    override a: i32 = 1;
    override b: u32 = 2u;
    @id(1) override c: f32 = 3.0;
    @id(2) override d: bool = false;

    // Put the results here
    @group(0) @binding(0) var<storage, read_write> data: array<u32>;

    struct VertexOutput {
        @location(0) values: vec4u,
        @builtin(position) position: vec4f,
    }

    @vertex
    fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
        var output: VertexOutput;
        output.position = vec4f(0, 0, 0, 1);
        output.values = vec4u(u32(a), u32(b), u32(c), u32(d));
        return output;
    }

    @fragment
    fn fragment(output: VertexOutput) -> @location(0) vec4f {
        let values1 = output.values;
        let values2 = vec4u(u32(a), u32(b), u32(c), u32(d));
        write_results(values1, values2);
        return vec4f();
    }

    @compute @workgroup_size(1)
    fn computeMain() {
        let results = vec4u(u32(a), u32(b), u32(c), u32(d));
        write_results(results, results);
    }

    fn write_results(results1: vec4u, results2: vec4u) {
        for (var i = 0; i < 4; i++) {
            data[i] = results1[i];
            data[i + 4] = results2[i];
        }
    }
"""

BIND_GROUP_ENTRIES = [
    {"binding": 0, "visibility": "FRAGMENT|COMPUTE", "buffer": {"type": "storage"}},
]


class Runner:
    def __init__(self):
        self.device = device = wgpu.utils.get_default_device()
        self.output_texture = device.create_texture(
            # Actual size is immaterial.  Could just be 1x1
            size=[128, 128],
            format=TextureFormat.rgba8unorm,
            usage="RENDER_ATTACHMENT|COPY_SRC",
        )
        self.shader = device.create_shader_module(code=SHADER_SOURCE)
        bind_group_layout = device.create_bind_group_layout(entries=BIND_GROUP_ENTRIES)
        self.render_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout],
        )

        self.output_buffer = device.create_buffer(size=8 * 4, usage="STORAGE|COPY_SRC")
        self.bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.output_buffer}},
            ],
        )

        self.color_attachment = {
            "clear_value": (0, 0, 0, 0),  # only first value matters
            "load_op": "clear",
            "store_op": "store",
            "view": self.output_texture.create_view(),
        }

    def create_render_pipeline(self, vertex_constants, fragment_constants):
        return self.device.create_render_pipeline(
            layout=self.render_pipeline_layout,
            vertex={
                "module": self.shader,
                "constants": vertex_constants,
            },
            fragment={
                "module": self.shader,
                "targets": [{"format": self.output_texture.format}],
                "constants": fragment_constants,
            },
            primitive={
                "topology": "point-list",
            },
        )

    def create_compute_pipeline(self, constants):
        return self.device.create_compute_pipeline(
            layout=self.render_pipeline_layout,
            compute={
                "module": self.shader,
                "constants": constants,
            },
        )

    def run_test(
        self,
        *,
        render: bool = False,
        compute: bool = False,
        vertex_constants=None,
        fragment_constants=None,
        compute_constants=None
    ):
        assert render + compute == 1
        device = self.device
        encoder = device.create_command_encoder()
        if render:
            this_pass = encoder.begin_render_pass(
                color_attachments=[self.color_attachment]
            )
            pipeline = self.create_render_pipeline(vertex_constants, fragment_constants)
        else:
            this_pass = encoder.begin_compute_pass()
            pipeline = self.create_compute_pipeline(compute_constants)
        this_pass.set_bind_group(0, self.bind_group)
        this_pass.set_pipeline(pipeline)
        if render:
            this_pass.draw(1)
        else:
            this_pass.dispatch_workgroups(1)
        this_pass.end()
        device.queue.submit([encoder.finish()])
        result = device.queue.read_buffer(self.output_buffer).cast("I").tolist()
        if compute:
            result = result[:4]
        print(result)
        return result


@pytest.fixture(scope="module")
def runner():
    return Runner()


def test_no_overridden_constants_render(runner):
    assert runner.run_test(render=True) == [1, 2, 3, 0, 1, 2, 3, 0]


def test_no_constants_compute(runner):
    runner.run_test(compute=True) == [1, 2, 3, 0]


def test_override_vertex_constants(runner):
    # Note that setting "d" to any non-zero value is setting it to True
    overrides = {"a": 21, "b": 22, 1: 23, 2: 24}
    assert [21, 22, 23, 1, 1, 2, 3, 0] == runner.run_test(
        render=True, vertex_constants=overrides
    )


def test_override_fragment_constants(runner):
    # Note that setting "d" to any non-zero value is setting it to True
    overrides = {"a": 21, "b": 22, 1: 23, 2: -1}
    assert [1, 2, 3, 0, 21, 22, 23, 1] == runner.run_test(
        render=True, fragment_constants=overrides
    )


def test_override_compute_constants(runner):
    # Note that setting "d" to any non-zero value is setting it to True
    overrides = {"a": 21, "b": 22, 1: 23, 2: 24}
    assert [21, 22, 23, 1] == runner.run_test(compute=True, compute_constants=overrides)


def test_numbered_constants_must_be_overridden_by_number(runner):
    overrides = {"c": 23, "d": 24}
    try:
        # In naga, the bad constant is ignored.
        # In the JS implementation, this throws an exception, which I think is the
        # correct behavior.  So just in case this ever gets fixed, we accept either.
        result = runner.run_test(
            render=True, vertex_constants=overrides, fragment_constants=overrides
        )
    except GPUValidationError:
        return
    assert [1, 2, 3, 0, 1, 2, 3, 0] == result


if __name__ == "__main__":
    run_tests(globals())
