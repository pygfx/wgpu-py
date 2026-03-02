import gc
import sys

import numpy as np
import wgpu.utils
from wgpu import TextureFormat
from wgpu.backends.wgpu_native.extras import write_timestamp

import pytest
from testutils import run_tests, can_use_wgpu_lib, is_pypy


if not can_use_wgpu_lib:
    pytest.mark.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


SHADER_SOURCE = """
    @group(0) @binding(0)
    var<storage,read_write> data2: array<f32>;

    struct VertexOutput {
        @location(0) index: u32,
        @builtin(position) position: vec4f,
    }

    @vertex
    fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
        var output: VertexOutput;
        output.position = vec4f(0, 0, 0, 1);
        output.index = index;
        return output;
    }

    @fragment
    fn fragment(output: VertexOutput) -> @location(0) vec4f {
        let i = output.index;
        data2[i] = f32(i) / 2.0;
        return vec4f();
    }

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {
        let i: u32 = index.x;
        data2[i] = f32(i) / 2.0;
    }
"""


BINDING_LAYOUTS = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.FRAGMENT,
        "buffer": {"type": wgpu.BufferBindingType.storage},
    },
]

REQUIRED_FEATURES = ["timestamp-query"]
OPTIONAL_FEATURES = ["timestamp-query-inside-passes", "timestamp-query-inside-encoders"]


def get_device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    features = [
        *REQUIRED_FEATURES,
        *[x for x in OPTIONAL_FEATURES if x in adapter.features],
    ]
    return adapter.request_device_sync(required_features=features)


def test_create_query_set():
    device = get_device()
    query_count = 7
    query_type = wgpu.QueryType.timestamp
    query_label = "div_by_2"
    query_set = device.create_query_set(
        label=query_label, type=query_type, count=query_count
    )
    assert query_set.count == query_count
    assert query_set.type == query_type
    assert query_set._device == device
    assert query_set.label == query_label


def test_create_query_buffer():
    device = wgpu.utils.get_default_device()
    query_count = 7
    query_buf_size = 8 * query_count
    query_usage = wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC
    query_buf = device.create_buffer(size=query_buf_size, usage=query_usage)
    assert query_buf.size == query_buf_size
    assert query_buf.usage == query_usage


class Runner:
    def __init__(self):
        if is_pypy:
            gc.collect()  # avoid a panic here when using pypy

        self.count = 1024  # number of elements in the array
        self.device = device = get_device()
        self.output_texture = device.create_texture(
            # Actual size is immaterial.  Could just be 1x1
            size=[16, 16],
            format=TextureFormat.rgba8unorm,
            usage="RENDER_ATTACHMENT|COPY_SRC",
        )
        self.shader = device.create_shader_module(code=SHADER_SOURCE)

        # Create buffer objects, input buffer is mapped.
        self.expected_output = np.arange(self.count, dtype=np.float32) / 2
        self.output_buffer = device.create_buffer(
            size=self.expected_output.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        bind_group_layout = device.create_bind_group_layout(entries=BINDING_LAYOUTS)
        self.pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        bindings = [
            {"binding": 0, "resource": {"buffer": self.output_buffer}},
        ]
        self.bind_group = device.create_bind_group(
            layout=bind_group_layout, entries=bindings
        )

        self.query_set = device.create_query_set(type="timestamp", count=10)

        query_usage = wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC
        self.query_buffer = device.create_buffer(
            size=(8 * self.query_set.count), usage=query_usage
        )
        self.color_attachment = {
            "clear_value": (0, 0, 0, 0),  # only first value matters
            "load_op": "clear",
            "store_op": "store",
            "view": self.output_texture.create_view(),
        }

    def create_render_pipeline(self):
        return self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex={"module": self.shader},
            fragment={
                "module": self.shader,
                "targets": [{"format": self.output_texture.format}],
            },
            primitive={"topology": "point-list"},
        )

    def create_compute_pipeline(self):
        return self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={
                "module": self.shader,
            },
        )

    def check_gpu_results(self):
        # Make sure we've done the right operation by comparing the results we got from
        # the gpu with the results from performing the same operation on the CPU.
        result_buffer = self.device.queue.read_buffer(self.output_buffer)
        result_gpu = np.frombuffer(result_buffer, dtype=np.float32)
        assert (result_gpu == self.expected_output).all()

    def get_timestamp_results(self, query_count):
        device = self.device
        command_encoder = device.create_command_encoder()
        command_encoder.resolve_query_set(
            self.query_set, 0, query_count, self.query_buffer, 0
        )
        self.device.queue.submit([command_encoder.finish()])
        timestamp_buffer = device.queue.read_buffer(
            self.query_buffer, size=8 * query_count
        )
        timestamps = timestamp_buffer.cast("Q").tolist()
        return timestamps

    def run_write_timestamp_test(self, render=False, compute=False):
        # At least one of them needs to be set
        assert render + compute == 1
        device = self.device
        pipeline = (
            self.create_compute_pipeline() if compute else self.create_render_pipeline()
        )
        command_encoder = device.create_command_encoder()

        def do_one_pass(*, start_index=None, end_index=None):
            timestamp_writes = {"query_set": self.query_set}
            if start_index is not None:
                timestamp_writes["beginning_of_pass_write_index"] = start_index
            if end_index is not None:
                timestamp_writes["end_of_pass_write_index"] = end_index

            if compute:
                this_pass = command_encoder.begin_compute_pass(
                    timestamp_writes=timestamp_writes
                )
            else:
                this_pass = command_encoder.begin_render_pass(
                    timestamp_writes=timestamp_writes,
                    color_attachments=[self.color_attachment],
                )
            this_pass.set_pipeline(pipeline)
            this_pass.set_bind_group(0, self.bind_group)
            if compute:
                this_pass.dispatch_workgroups(self.count, 1, 1)  # x y z
            else:
                this_pass.draw(self.count)
            this_pass.end()

        do_one_pass(start_index=0, end_index=1)
        do_one_pass(start_index=2)
        do_one_pass(end_index=3)

        device.queue.submit([command_encoder.finish()])

        self.check_gpu_results()

        start1, end1, start2, end3 = self.get_timestamp_results(query_count=4)
        # GPUs are asynchronous, so we really don't know the order operations will happen.
        assert 0 < start1 < end1
        assert 0 < start2
        assert 0 < end3

    def run_write_timestamp_inside_passes_test(self, render=False, compute=False):
        assert render + compute == 1  # set at least one of them
        device = self.device
        command_encoder = device.create_command_encoder()
        if compute:
            pipeline = self.create_compute_pipeline()
            this_pass = command_encoder.begin_compute_pass()
        else:
            pipeline = self.create_render_pipeline()
            this_pass = command_encoder.begin_render_pass(
                color_attachments=[self.color_attachment],
            )

        write_timestamp(this_pass, self.query_set, 0)
        this_pass.set_pipeline(pipeline)
        this_pass.set_bind_group(0, self.bind_group)
        if compute:
            this_pass.dispatch_workgroups(self.count, 1, 1)  # x y z
        else:
            this_pass.draw(self.count)
        write_timestamp(this_pass, self.query_set, 1)
        this_pass.end()
        device.queue.submit([command_encoder.finish()])

        self.check_gpu_results()

        start1, end1 = self.get_timestamp_results(query_count=2)
        assert 0 < start1 < end1

    def run_write_timestamp_inside_encoders_test(self):
        # At least one of them needs to be set
        device = self.device
        command_encoder = device.create_command_encoder()
        write_timestamp(command_encoder, self.query_set, 0)

        pipeline = self.create_compute_pipeline()
        this_pass = command_encoder.begin_compute_pass()
        this_pass.set_pipeline(pipeline)
        this_pass.set_bind_group(0, self.bind_group)
        this_pass.dispatch_workgroups(self.count, 1, 1)  # x y z
        this_pass.end()

        write_timestamp(command_encoder, self.query_set, 1)

        device.queue.submit([command_encoder.finish()])

        self.check_gpu_results()

        start1, end1 = self.get_timestamp_results(query_count=2)
        assert 0 < start1 < end1


def test_compute_timestamps():
    Runner().run_write_timestamp_test(compute=True)


def test_render_timestamps():
    Runner().run_write_timestamp_test(render=True)


def test_compute_timestamps_inside_passes():
    runner = Runner()
    if "timestamp-query-inside-passes" not in runner.device.features:
        pytest.skip("Must have 'timestamp-query-inside-passes' to run")
    runner.run_write_timestamp_inside_passes_test(compute=True)


def test_render_timestamps_inside_passes():
    runner = Runner()
    if "timestamp-query-inside-passes" not in runner.device.features:
        pytest.skip("Must have 'timestamp-query-inside-passes' to run")
    runner.run_write_timestamp_inside_passes_test(render=True)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Known to currently fail (and sometimes hang) on MacOS",
)
def test_render_timestamps_inside_encoder():
    runner = Runner()
    if "timestamp-query-inside-encoders" not in runner.device.features:
        pytest.skip("Must have 'timestamp-query-inside-encoders' to run")
    runner.run_write_timestamp_inside_encoders_test()


if __name__ == "__main__":
    run_tests(globals())
