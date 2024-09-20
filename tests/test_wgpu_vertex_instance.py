import itertools

import numpy as np
import pytest
import wgpu.utils
from tests.testutils import can_use_wgpu_lib, run_tests
from wgpu import TextureFormat
from wgpu.backends.wgpu_native.extras import (
    multi_draw_indexed_indirect,
    multi_draw_indirect,
    multi_draw_indirect_count,
    multi_draw_indexed_indirect_count,
)

MAX_INFO = 100

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


"""
The fundamental information about any of the many draw commands is the
<vertex_instance, instance_index> pair that is passed to the vertex shader. By using
point-list topology, each call to the vertex shader turns into a single call to the
fragment shader, where the pair is recorded.

(To modify a buffer in the vertex shader requires the feature vertex-writable-storage)

We call various combinations of draw functions and verify that they generate precisely
the pairs (those possibly in a different order) that we expect.
"""
SHADER_SOURCE = (
    f"""
    const MAX_INFO: u32 = {MAX_INFO}u;
    """
    """
    @group(0) @binding(0) var<storage, read_write> data: array<vec2u>;
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
    REQUIRED_FEATURES = ["indirect-first-instance"]
    OPTIONAL_FEATURES = ["multi-draw-indirect", "multi-draw-indirect-count"]

    @classmethod
    def is_usable(cls):
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        return set(cls.REQUIRED_FEATURES) <= adapter.features

    def __init__(self):
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        features = [
            *self.REQUIRED_FEATURES,
            *[x for x in self.OPTIONAL_FEATURES if x in adapter.features],
        ]
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

        self.data_buffer = self.device.create_buffer(
            size=MAX_INFO * 2 * 4, usage="STORAGE|COPY_SRC"
        )
        self.counter_buffer = self.device.create_buffer(
            size=4, usage="STORAGE|COPY_SRC|COPY_DST"
        )
        self.bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.data_buffer}},
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
        # Args are [vertex_count, instant_count, first_vertex, first_instance]
        self.draw_args1 = [2, 3, 100, 10]
        self.draw_args2 = [1, 1, 30, 50]
        expected_draw_args1 = set(itertools.product((100, 101), (10, 11, 12)))
        expected_draw_args2 = {(30, 50)}
        self.expected_result_draw = expected_draw_args1 | expected_draw_args2

        # Args are [vertex_count, instance_count, index_buffer_offset, vertex_offset, first_instance]
        self.draw_indexed_args1 = [4, 2, 1, 100, 1000]
        self.draw_indexed_args2 = [1, 1, 7, 200, 2000]
        self.expected_result_draw_indexed = set(
            itertools.product((103, 105, 107, 111), (1000, 1001))
        )
        self.expected_result_draw_indexed.add((219, 2000))

        indices = (2, 3, 5, 7, 11, 13, 17, 19)
        self.draw_indexed_args1 = (4, 2, 1, 100, 1000)
        self.draw_indexed_args2 = (1, 1, 7, 200, 2000)
        expected_draw_indexed_args1 = set(
            itertools.product((103, 105, 107, 111), (1000, 1001))
        )
        expected_draw_indexed_args2 = {(219, 2000)}
        self.expected_result_draw_indexed = (
            expected_draw_indexed_args1 | expected_draw_indexed_args2
        )

        # We're going to want to try calling these draw functions from a buffer, and it
        # would be nice to test that these buffers have an offset
        self.draw_data_buffer = self.device.create_buffer_with_data(
            data=np.uint32([10, 2, *self.draw_args1, *self.draw_args2, *([0] * 50)]),
            usage="INDIRECT",
        )
        self.draw_data_buffer_indexed = self.device.create_buffer_with_data(
            data=np.uint32(
                [10, 2, *self.draw_indexed_args1, *self.draw_indexed_args2, *([0] * 50)]
            ),
            usage="INDIRECT",
        )

        # And let's not forget our index buffer.
        self.index_buffer = self.device.create_buffer_with_data(
            data=(np.uint32(indices)), usage="INDEX"
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

    def run_draw_test(self, draw_function, indexed, *, expected_result=None):
        encoder = self.device.create_command_encoder()
        encoder.clear_buffer(self.counter_buffer)
        this_pass = encoder.begin_render_pass(**self.render_pass_descriptor)
        this_pass.set_pipeline(self.pipeline)
        this_pass.set_bind_group(0, self.bind_group)
        this_pass.set_index_buffer(self.index_buffer, "uint32")
        draw_function(this_pass)
        this_pass.end()
        self.device.queue.submit([encoder.finish()])
        count = self.device.queue.read_buffer(self.counter_buffer).cast("i")[0]
        if count > MAX_INFO:
            pytest.fail("Too many data points written to output buffer")
        if count == 0:
            pytest.fail("No data points written to output buffer")
        # Get the result as a series of tuples
        info_view = self.device.queue.read_buffer(self.data_buffer, size=count * 2 * 4)
        info = info_view.cast("I", (count, 2)).tolist()
        info_set = set(tuple(x) for x in info)
        if expected_result is None:
            if indexed:
                expected_result = self.expected_result_draw_indexed
            else:
                expected_result = self.expected_result_draw
        if info_set != expected_result:
            print(f"{sorted(info_set)=}")
            print(f"{sorted(expected_result)=}")
            assert False


if not Runner.is_usable():
    pytest.skip("Runner don't have all required features", allow_module_level=True)


@pytest.fixture(scope="module")
def runner():
    return Runner()


def test_draw(runner):
    def draw(encoder):
        encoder.draw(*runner.draw_args1)
        encoder.draw(*runner.draw_args2)

    runner.run_draw_test(draw, False)


def test_draw_indirect(runner):
    def draw(encoder):
        encoder.draw_indirect(runner.draw_data_buffer, 8)
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    runner.run_draw_test(draw, False)


def test_draw_mixed(runner):
    def draw(encoder):
        encoder.draw(*runner.draw_args1)
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    runner.run_draw_test(draw, False)


def test_multi_draw_indirect(runner):
    if "multi-draw-indirect" not in runner.device.features:
        pytest.skip("Must have 'multi-draw-indirect' to run")

    def draw(encoder):
        multi_draw_indirect(encoder, runner.draw_data_buffer, offset=8, count=2)

    runner.run_draw_test(draw, False)


def test_draw_via_encoder(runner):
    def draw(encoder):
        encoder.draw(*runner.draw_args1)
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    render_bundle_encoder = runner.create_render_bundle_encoder(draw)
    for _ in range(2):
        # We run this test twice to verify that encoders are reusable.
        runner.run_draw_test(
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]), False
        )


def test_draw_via_multiple_encoders(runner):
    # Make sure that execute_bundles() works with multiple encoders.
    def draw1(encoder):
        encoder.draw(*runner.draw_args1)

    def draw2(encoder):
        encoder.draw_indirect(runner.draw_data_buffer, 8 + 16)

    render_bundle_encoder1 = runner.create_render_bundle_encoder(draw1)
    render_bundle_encoder2 = runner.create_render_bundle_encoder(draw2)

    runner.run_draw_test(
        lambda encoder: encoder.execute_bundles(
            [render_bundle_encoder1, render_bundle_encoder2]
        ),
        False,
    )


def test_draw_indexed(runner):
    def draw(encoder):
        encoder.draw_indexed(*runner.draw_indexed_args1)
        encoder.draw_indexed(*runner.draw_indexed_args2)

    runner.run_draw_test(draw, True)


def test_draw_indexed_indirect(runner):
    def draw(encoder):
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8)
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8 + 20)

    runner.run_draw_test(draw, True)


def test_draw_indexed_mixed(runner):
    def draw(encoder):
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8)
        encoder.draw_indexed(*runner.draw_indexed_args2)

    runner.run_draw_test(draw, True)


def test_multi_draw_indexed_indirect(runner):
    if "multi-draw-indirect" not in runner.device.features:
        pytest.skip("Must have 'multi-draw-indirect' to run")

    def draw(encoder):
        multi_draw_indexed_indirect(
            encoder, runner.draw_data_buffer_indexed, offset=8, count=2
        )

    runner.run_draw_test(draw, True)


def test_draw_indexed_via_encoder(runner):
    def draw(encoder):
        encoder.draw_indexed_indirect(runner.draw_data_buffer_indexed, 8)
        encoder.draw_indexed(*runner.draw_indexed_args2)

    render_bundle_encoder = runner.create_render_bundle_encoder(draw)
    for _ in range(2):
        runner.run_draw_test(
            lambda encoder: encoder.execute_bundles([render_bundle_encoder]), True
        )


@pytest.mark.parametrize("test_max_count", [False])
@pytest.mark.parametrize("indexed", [False])
def test_multi_draw_indirect_count(runner, test_max_count, indexed):
    if "multi-draw-indirect-count" not in runner.device.features:
        pytest.skip("Must have 'multi-draw-indirect-count' to run")

    print(f"{test_max_count=}, {indexed=}")

    count_buffer = runner.device.create_buffer_with_data(
        data=(np.int32([10, 2])), usage="INDIRECT"
    )

    def draw(encoder):
        if indexed:
            function = multi_draw_indexed_indirect_count
            buffer = runner.draw_data_buffer_indexed
        else:
            function = multi_draw_indirect_count
            data = np.uint32(range(3, 70))
            print(f"{data[0:12]=}")
            buffer = runner.device.create_buffer_with_data(data=data, usage="INDIRECT")
        if test_max_count:
            # We pull a count of 10, but we're limited by max_count to 2
            count_buffer_offset, max_count = 0, 2
        else:
            # We pull a count of 2, and set the max_count to something bigger
            count_buffer_offset, max_count = 4, 10
        function(
            encoder,
            buffer,
            offset=8,
            count_buffer=count_buffer,
            count_buffer_offset=count_buffer_offset,
            max_count=max_count,
        )

    runner.run_draw_test(draw, indexed)


if __name__ == "__main__":
    run_tests(globals())
