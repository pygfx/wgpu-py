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

MAX_INFO = 100

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


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
    def __init__(self, use_multidraw_if_available: bool = True):
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        features = []
        if use_multidraw_if_available and "multi-draw-indirect" in adapter.features:
            features.append("multi-draw-indirect")
        self.device = adapter.request_device(required_features=features)
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

    def create_render_bundle_encoder(self, draw_function):
        render_bundle_encoder = self.device.create_render_bundle_encoder(
            color_formats=[self.output_texture.format]
        )
        render_bundle_encoder.set_pipeline(self.pipeline)
        render_bundle_encoder.set_bind_group(0, self.bind_group)
        draw_function(render_bundle_encoder)
        return render_bundle_encoder.finish()

    def run_function(self, expected_result, draw_function):
        encoder = self.device.create_command_encoder()
        encoder.clear_buffer(self.counter_buffer)
        this_pass = encoder.begin_render_pass(**self.render_pass_descriptor)
        this_pass.set_pipeline(self.pipeline)
        this_pass.set_bind_group(0, self.bind_group)
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


def test_draw_no_index():
    runner = Runner()

    # vertex_count, index_count, first_vertex, first_index
    draw_args1 = [2, 3, 100, 10]
    draw_args2 = [1, 1, 30, 50]
    expected_result = set(itertools.product((100, 101), (10, 11, 12))) | {(30, 50)}

    draw_data_info = np.uint32([0, 0] + draw_args1 + draw_args2)
    draw_data_buffer = runner.device.create_buffer_with_data(
        data=draw_data_info, usage="INDIRECT"
    )

    def draw_direct(encoder):
        encoder.draw(*draw_args1)
        encoder.draw(*draw_args2)

    def draw_indirect(encoder):
        encoder.draw_indirect(draw_data_buffer, 8)
        encoder.draw_indirect(draw_data_buffer, 8 + 16)

    def draw_mixed(encoder):
        encoder.draw(*draw_args1)
        encoder.draw_indirect(draw_data_buffer, 8 + 16)

    def draw_indirect_multi(encoder):
        multi_draw_indirect(encoder, draw_data_buffer, offset=8, count=2)

    render_bundle_encoder = runner.create_render_bundle_encoder(draw_mixed)

    has_multi_draw_indirect = "multi-draw-indirect" in runner.device.features
    runner.run_functions(
        expected_result,
        [
            draw_direct,
            draw_indirect,
            draw_mixed,
            *([draw_indirect_multi] if has_multi_draw_indirect else []),
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]),
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]),
        ],
    )


def test_draw_indexed():
    runner = Runner()

    # index_count, instance_count, first_index, base_vertex, first_intance
    draw_args1 = (4, 2, 1, 100, 1000)
    draw_args2 = (1, 1, 7, 200, 2000)
    index_buffer_data = (2, 3, 5, 7, 11, 13, 17, 19)
    expected_result = set(itertools.product((103, 105, 107, 111), (1000, 1001)))
    expected_result.add((219, 2000))

    index_buffer_data = np.uint32(index_buffer_data)
    index_buffer = runner.device.create_buffer_with_data(
        data=index_buffer_data, usage="INDEX"
    )

    draw_data = np.uint32([0, 0] + list(draw_args1) + list(draw_args2))
    draw_data_buffer = runner.device.create_buffer_with_data(
        data=draw_data, usage="INDIRECT"
    )

    def draw_direct(encoder):
        encoder.set_index_buffer(index_buffer, "uint32")
        encoder.draw_indexed(*draw_args1)
        encoder.draw_indexed(*draw_args2)

    def draw_indirect(encoder):
        encoder.set_index_buffer(index_buffer, "uint32")
        encoder.draw_indexed_indirect(draw_data_buffer, 8)
        encoder.draw_indexed_indirect(draw_data_buffer, 8 + 20)

    def draw_mixed(encoder):
        encoder.set_index_buffer(index_buffer, "uint32")
        encoder.draw_indexed(*draw_args1)
        encoder.draw_indexed_indirect(draw_data_buffer, 8 + 20)

    def draw_indirect_multi(encoder):
        encoder.set_index_buffer(index_buffer, "uint32")
        multi_draw_indexed_indirect(encoder, draw_data_buffer, offset=8, count=2)

    render_bundle_encoder = runner.create_command_encoder(draw_mixed)

    has_multi_draw_indirect = "multi-draw-indirect" in runner.device.features
    runner.run_functions(
        expected_result,
        [
            draw_direct,
            draw_indirect,
            draw_mixed,
            *([draw_indirect_multi] if has_multi_draw_indirect else []),
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]),
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]),
        ],
    )


if __name__ == "__main__":
    run_tests(globals())
