import itertools

import numpy as np
import pytest
import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import TextureFormat
from wgpu.backends.wgpu_native.extras import (
    multi_draw_indexed_indirect,
    multi_draw_indirect,
)

REQUIRED_FEATURES = ["multi-draw-indirect", "indirect-first-instance"]

MAX_INFO = 100

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def has_required_features():
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    return set(REQUIRED_FEATURES) <= adapter.features


if not has_required_features():
    pytest.skip("Tests don't have all required features", allow_module_level=True)

SHADER_SOURCE = (
    f"""
    const MAX_INFO: u32 = {MAX_INFO}u;
    """
    """
    @group(0) @binding(0) var<storage, read_write> data: array<vec2u, MAX_INFO>;
    @group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) info: vec2u
    }

    const POSITION: vec4f = vec4f(0, 0, 0, 1);

    @vertex
    fn vertexMain(
        @builtin(vertex_index) vertexIndex: u32,
        @builtin(instance_index) instanceIndex: u32
    ) -> VertexOutput {
        let info = vec2u(vertexIndex, instanceIndex);
        return VertexOutput(POSITION, info);
    }

    @fragment
    fn fragmentMain(@location(0) info: vec2u) -> @location(0) vec4f {
        let index = atomicAdd(&counter, 1u);
        data[index % MAX_INFO] = info;
        return vec4f();
    }
"""
)

BIND_GROUP_ENTRIES = [
    {"binding": 0, "visibility": "FRAGMENT", "buffer": {"type": "storage"}},
    {"binding": 1, "visibility": "FRAGMENT", "buffer": {"type": "storage"}},
]


class Runner:
    def __init__(self):
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self.device = adapter.request_device(required_features=REQUIRED_FEATURES)
        self.output_texture = self.device.create_texture(
            # Actual size is immaterial.  Could just be 1x1
            size=[128, 128],
            format=TextureFormat.rgba8unorm,
            usage="RENDER_ATTACHMENT|COPY_SRC",
        )
        shader = self.device.create_shader_module(code=SHADER_SOURCE)
        bind_group_layout = self.device.create_bind_group_layout(
            entries=BIND_GROUP_ENTRIES
        )
        render_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        self.pipeline = self.device.create_render_pipeline(
            layout=render_pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vertexMain",
            },
            fragment={
                "module": shader,
                "entry_point": "fragmentMain",
                "targets": [{"format": self.output_texture.format}],
            },
            primitive={
                "topology": "point-list",
            },
        )

        self.vertex_call_buffer = self.device.create_buffer(
            size=MAX_INFO * 2 * 4, usage="STORAGE|COPY_SRC"
        )
        self.counter_buffer = self.device.create_buffer(
            size=4, usage="STORAGE|COPY_SRC|COPY_DST"
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.vertex_call_buffer}},
                {"binding": 1, "resource": {"buffer": self.counter_buffer}},
            ],
        )
        self.render_pass_descriptor = {
            "color_attachments": [
                {
                    "clear_value": (0, 0, 0, 0),  # only first value matters
                    "load_op": "clear",
                    "store_op": "store",
                    "view": self.output_texture.create_view(),
                }
            ],
        }
        self.draw_args1 = [2, 3, 100, 10]
        self.draw_args2 = [1, 1, 30, 50]
        self.expected_result_draw = set(itertools.product((100, 101), (10, 11, 12))) | {
            (30, 50)
        }
        self.draw_data_buffer = self.device.create_buffer_with_data(
            data=np.uint32([0, 0, *self.draw_args1, *self.draw_args2]), usage="INDIRECT"
        )

        self.draw_indexed_args1 = (4, 2, 1, 100, 1000)
        self.draw_indexed_args2 = (1, 1, 7, 200, 2000)
        self.expected_result_draw_indexed = set(
            itertools.product((103, 105, 107, 111), (1000, 1001))
        )
        self.expected_result_draw_indexed.add((219, 2000))
        self.index_buffer = self.device.create_buffer_with_data(
            data=(np.uint32((2, 3, 5, 7, 11, 13, 17, 19))), usage="INDEX"
        )
        self.draw_data_buffer_indexed = self.device.create_buffer_with_data(
            data=np.uint32([0, 0, *self.draw_indexed_args1, *self.draw_indexed_args2]),
            usage="INDIRECT",
        )

    def create_render_bundle_encoder(self, draw_function):
        render_bundle_encoder = self.device.create_render_bundle_encoder(
            color_formats=[self.output_texture.format]
        )
        render_bundle_encoder.set_pipeline(self.pipeline)
        render_bundle_encoder.set_bind_group(0, self.bind_group)
        render_bundle_encoder.set_index_buffer(self.index_buffer, "uint32")

        draw_function(render_bundle_encoder)
        return render_bundle_encoder.finish()

    def run_function(self, expected_result, draw_function):
        encoder = self.device.create_command_encoder()
        encoder.clear_buffer(self.counter_buffer)
        this_pass = encoder.begin_render_pass(**self.render_pass_descriptor)
        this_pass.set_pipeline(self.pipeline)
        this_pass.set_bind_group(0, self.bind_group)
        this_pass.set_index_buffer(self.index_buffer, "uint32")
        draw_function(this_pass)
        this_pass.end()
        self.device.queue.submit([encoder.finish()])
        counter_buffer_view = self.device.queue.read_buffer(self.counter_buffer)
        count = np.frombuffer(counter_buffer_view, dtype=np.uint32)[0]
        if count > MAX_INFO:
            pytest.fail("Too many data points written to output buffer")
        info_view = self.device.queue.read_buffer(
            self.vertex_call_buffer, size=count * 2 * 4
        )
        info = np.frombuffer(info_view, dtype=np.uint32).reshape(-1, 2)
        info = [tuple(info[i]) for i in range(len(info))]
        info_set = set(info)
        assert len(info) == len(info_set)
        assert info_set == expected_result

    def run_functions(self, expected_result, functions):
        for function in functions:
            self.run_function(expected_result, function)


@pytest.fixture(scope="module")
def runner():
    return Runner()


def test_draw(runner):
    def draw(encoder):
        encoder.draw(*runner.draw_args1)
        encoder.draw(*runner.draw_args2)

    runner.run_function(runner.expected_result_draw, draw)


def test_draw_indirect(runner):
    def draw(encoder):
        encoder.draw_indirect(runner.draw_data_buffer, 8)
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    runner.run_function(runner.expected_result_draw, draw)


def test_draw_mixed(runner):
    def draw(encoder):
        encoder.draw(*runner.draw_args1)
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    runner.run_function(runner.expected_result_draw, draw)


def test_multi_draw_indirect(runner):
    def draw(encoder):
        multi_draw_indirect(encoder, runner.draw_data_buffer, offset=8, count=2)

    runner.run_function(runner.expected_result_draw, draw)


def test_draw_via_encoder(runner):
    def draw(encoder):
        encoder.draw(*runner.draw_args1)
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    render_bundle_encoder = runner.create_render_bundle_encoder(draw)
    for _ in range(2):
        runner.run_function(
            runner.expected_result_draw,
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]),
        )


def test_draw_indexed(runner):
    def draw(encoder):
        encoder.draw_indexed(*runner.draw_indexed_args1)
        encoder.draw_indexed(*runner.draw_indexed_args2)

    runner.run_function(runner.expected_result_draw_indexed, draw)


def test_draw_indexed_indirect(runner):
    def draw(encoder):
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8)
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8 + 20)

    runner.run_function(runner.expected_result_draw_indexed, draw)


def test_draw_indexed_mixed(runner):
    def draw(encoder):
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8)
        encoder.draw_indexed(*runner.draw_indexed_args2)

    runner.run_function(runner.expected_result_draw_indexed, draw)


def test_multi_draw_indexed_indirect(runner):
    def draw(encoder):
        multi_draw_indexed_indirect(
            encoder, runner.draw_data_buffer_indexed, offset=8, count=2
        )

    runner.run_function(runner.expected_result_draw_indexed, draw)


def test_draw_indexed_via_encoder(runner):
    def draw(encoder):
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8)
        encoder.draw_indexed(*runner.draw_indexed_args2)

    render_bundle_encoder = runner.create_render_bundle_encoder(draw)
    for _ in range(2):
        runner.run_function(
            runner.expected_result_draw_indexed,
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]),
        )


if __name__ == "__main__":
    run_tests(globals())
