"""
Test all the wgpu objects.
"""

import pytest
import testutils  # noqa
from testutils import can_use_wgpu_lib, create_and_release


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need wgpu lib", allow_module_level=True)


import wgpu

DEVICE = wgpu.utils.get_default_device()


@create_and_release
def test_release_adapter(n):
    yield {}
    for i in range(n):
        yield wgpu.gpu.request_adapter(power_preference="high-performance")


@create_and_release
def test_release_device(n):
    pytest.skip("XFAIL")
    # todo: XFAIL: Device object seem not to be cleaned up at wgpu-native.

    # Note: the WebGPU spec says:
    # [request_device()] is a one-time action: if a device is returned successfully, the adapter becomes invalid.

    yield {
        "expected_counts_after_create": {"Device": (n, n), "Queue": (n, 0)},
    }
    adapter = DEVICE.adapter
    for i in range(n):
        d = adapter.request_device()
        # d.queue._destroy()
        # d._queue = None
        yield d


@create_and_release
def test_release_bind_group(n):
    buffer1 = DEVICE.create_buffer(size=128, usage=wgpu.BufferUsage.STORAGE)

    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
    ]

    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
        },
    ]

    bind_group_layout = DEVICE.create_bind_group_layout(entries=binding_layouts)

    yield {}

    for i in range(n):
        yield DEVICE.create_bind_group(layout=bind_group_layout, entries=bindings)


_bind_group_layout_binding = 10


@create_and_release
def test_release_bind_group_layout(n):
    # Note: when we use the same binding layout descriptor, wgpu-native
    # re-uses the BindGroupLayout object.

    global _bind_group_layout_binding
    _bind_group_layout_binding += 1

    yield {
        "expected_counts_after_create": {"BindGroupLayout": (n, 1)},
    }

    binding_layouts = [
        {
            "binding": _bind_group_layout_binding,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
    ]

    for i in range(n):
        # binding_layouts[0]["binding"] = i  # force unique objects
        yield DEVICE.create_bind_group_layout(entries=binding_layouts)


@create_and_release
def test_release_buffer(n):
    yield {}
    for i in range(n):
        yield DEVICE.create_buffer(size=128, usage=wgpu.BufferUsage.COPY_DST)


@create_and_release
def test_release_command_buffer(n):
    # Note: a command encoder can only be used once (it gets destroyed on finish())
    yield {
        "expected_counts_after_create": {
            "CommandEncoder": (n, 0),
            "CommandBuffer": (n, n),
        },
    }

    for i in range(n):
        command_encoder = DEVICE.create_command_encoder()
        yield command_encoder.finish()


@create_and_release
def test_release_command_encoder(n):
    # Note: a CommandEncoder does not exist in wgpu-core, but we do
    # observe its internal CommandBuffer.
    yield {
        "expected_counts_after_create": {
            "CommandEncoder": (n, 0),
            "CommandBuffer": (0, n),
        },
    }

    for i in range(n):
        yield DEVICE.create_command_encoder()


@create_and_release
def test_release_compute_pass_encoder(n):
    # Note: ComputePassEncoder does not really exist in wgpu-core
    # -> Check gpu.diagnostics.wgpu_native_counts.print_report(), nothing there that ends with "Encoder".
    command_encoder = DEVICE.create_command_encoder()

    yield {
        "expected_counts_after_create": {
            "ComputePassEncoder": (n, 0),
        },
    }

    for i in range(n):
        yield command_encoder.begin_compute_pass()


@create_and_release
def test_release_compute_pipeline(n):
    code = """
        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) index: vec3<u32>) {
            let i: u32 = index.x;
        }
    """
    shader = DEVICE.create_shader_module(code=code)

    binding_layouts = []
    pipeline_layout = DEVICE.create_pipeline_layout(bind_group_layouts=binding_layouts)

    yield {}

    for i in range(n):
        yield DEVICE.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader, "entry_point": "main"},
        )


@create_and_release
def test_release_pipeline_layout(n):
    yield {}
    for i in range(n):
        yield DEVICE.create_pipeline_layout(bind_group_layouts=[])


@create_and_release
def test_release_query_set(n):
    yield {}
    for i in range(n):
        yield DEVICE.create_query_set(type=wgpu.QueryType.occlusion, count=2)


@create_and_release
def test_release_queue(n):
    pytest.skip("XFAIL")
    # todo: XFAIL: the device and queue are kinda one, and the former won't release at wgpu-native.
    yield {}
    adapter = DEVICE.adapter
    for i in range(n):
        d = adapter.request_device()
        q = d.queue
        d._queue = None  # detach
        yield q


@create_and_release
def test_release_render_bundle(n):
    # todo: implement this when we do support them
    pytest.skip("Render bundle not implemented")


@create_and_release
def test_release_render_bundle_encoder(n):
    pytest.skip("Render bundle not implemented")


@create_and_release
def test_release_render_pass_encoder(n):
    # Note: RenderPassEncoder does not really exist in wgpu-core
    # -> Check gpu.diagnostics.wgpu_native_counts.print_report(), nothing there that ends with "Encoder".
    command_encoder = DEVICE.create_command_encoder()

    yield {
        "expected_counts_after_create": {
            "RenderPassEncoder": (n, 0),
        },
    }

    for i in range(n):
        yield command_encoder.begin_render_pass(color_attachments=[])


@create_and_release
def test_release_render_pipeline(n):
    code = """
        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };
        struct VertexOutput {
            @location(0) color : vec4<f32>,
            @builtin(position) pos: vec4<f32>,
        };

        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(0.0, -0.5),
                vec2<f32>(0.5, 0.5),
                vec2<f32>(-0.5, 0.75),
            );
            var colors = array<vec3<f32>, 3>(  // srgb colors
                vec3<f32>(1.0, 1.0, 0.0),
                vec3<f32>(1.0, 0.0, 1.0),
                vec3<f32>(0.0, 1.0, 1.0),
            );
            let index = i32(in.vertex_index);
            var out: VertexOutput;
            out.pos = vec4<f32>(positions[index], 0.0, 1.0);
            out.color = vec4<f32>(colors[index], 1.0);
            return out;
        }

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            let physical_color = pow(in.color.rgb, vec3<f32>(2.2));  // gamma correct
            return vec4<f32>(physical_color, in.color.a);
        }
    """
    shader = DEVICE.create_shader_module(code=code)

    binding_layouts = []
    pipeline_layout = DEVICE.create_pipeline_layout(bind_group_layouts=binding_layouts)

    yield {}

    for i in range(n):
        yield DEVICE.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
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
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": "bgra8unorm-srgb",
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


@create_and_release
def test_release_sampler(n):
    yield {}
    for i in range(n):
        yield DEVICE.create_sampler()


@create_and_release
def test_release_shader_module(n):
    yield {}

    code = """
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
           return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    """

    for i in range(n):
        yield DEVICE.create_shader_module(code=code)


@create_and_release
def test_release_texture(n):
    yield {}
    for i in range(n):
        yield DEVICE.create_texture(
            size=(16, 16, 16),
            usage=wgpu.TextureUsage.TEXTURE_BINDING,
            format="rgba8unorm",
        )


@create_and_release
def test_release_texture_view(n):
    texture = DEVICE.create_texture(
        size=(16, 16, 16), usage=wgpu.TextureUsage.TEXTURE_BINDING, format="rgba8unorm"
    )
    yield {}
    for i in range(n):
        yield texture.create_view()


# %% The end


TEST_FUNCS = [
    ob
    for name, ob in list(globals().items())
    if name.startswith("test_") and callable(ob)
]

if __name__ == "__main__":
    # testutils.TEST_ITERS = 40  # Uncomment for a mem-usage test run

    for func in TEST_FUNCS:
        print(func.__name__ + " ...")
        try:
            func()
        except pytest.skip.Exception:
            print("  skipped")
    print("done")
