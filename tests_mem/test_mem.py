import gc
import asyncio

import wgpu.backends.rs

import pytest
from testutils import can_use_glfw, can_use_wgpu_lib, can_use_pyside6
from testutils import create_and_release, get_counts, ob_name_from_test_func

if not can_use_wgpu_lib:
    pytest.skip(
        "Skipping tests that need a window or the wgpu lib", allow_module_level=True
    )


# Create the default device beforehand
DEVICE = wgpu.utils.get_default_device()


async def stub_event_loop():
    pass


def make_draw_func_for_canvas(canvas):
    """Create a draw function for the given canvas,
    so that we can really present something to a canvas being tested.
    """
    ctx = canvas.get_context()
    ctx.configure(device=DEVICE, format="bgra8unorm-srgb")

    def draw():
        ctx = canvas.get_context()
        command_encoder = DEVICE.create_command_encoder()
        current_texture_view = ctx.get_current_texture()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture_view,
                    "resolve_target": None,
                    "clear_value": (1, 1, 1, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )
        render_pass.end()
        DEVICE.queue.submit([command_encoder.finish()])
        ctx.present()

    return draw


# %% Meta tests


def test_meta_all_objects_covered():
    """Test that we have a test_release test function for each known object."""

    ref_obnames = set(key for key in get_counts().keys())
    func_obnames = set(ob_name_from_test_func(func) for func in RELEASE_TEST_FUNCS)

    missing = ref_obnames - func_obnames
    extra = func_obnames - ref_obnames
    assert not missing
    assert not extra


def test_meta_all_functions_solid():
    """Test that all funcs starting with "test_release_" are decorated appropriately."""
    for func in RELEASE_TEST_FUNCS:
        is_decorated = func.__code__.co_name == "core_test_func"
        assert is_decorated, func.__name__ + " not decorated"


def test_meta_buffers_1():
    """Making sure that the test indeed fails, when holding onto the objects."""

    lock = []

    @create_and_release
    def test_release_buffer(n):
        yield {}
        for i in range(n):
            b = DEVICE.create_buffer(size=128, usage=wgpu.BufferUsage.COPY_DST)
            lock.append(b)
            yield b

    with pytest.raises(AssertionError):
        test_release_buffer()


def test_meta_buffers_2():
    """Making sure that the test indeed fails, by disabling the release call."""

    ori = wgpu.backends.rs.GPUBuffer._destroy
    wgpu.backends.rs.GPUBuffer._destroy = lambda self: None

    try:
        with pytest.raises(AssertionError):
            test_release_buffer()

    finally:
        wgpu.backends.rs.GPUBuffer._destroy = ori


# %% The actual tests

# These tests need to do one thing: generate n objects of the correct type.


@create_and_release
def test_release_adapter(n):
    yield {}
    for i in range(n):
        yield wgpu.request_adapter(canvas=None, power_preference="high-performance")


@create_and_release
def test_release_device(n):
    pytest.skip("XFAIL")
    # todo: XFAIL: Device object seem not to be cleaned up at wgpu-native.

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


@create_and_release
def test_release_bind_group_layout(n):
    # Note: when we use the same binding layout descriptor, wgpu-native
    # re-uses the BindGroupLayout object. On the other hand, it also
    # does not seem to clean them up. Perhaps it just caches them? There
    # are only so many possible combinations, and its just 152 bytes
    # (on Metal) per object.

    # todo: do we want similar behavior for *our* BindGroupLayout object?

    yield {
        "expected_counts_after_create": {"BindGroupLayout": (n, 1)},
        "expected_counts_after_release": {"BindGroupLayout": (0, 1)},
    }

    binding_layouts = [
        {
            "binding": 0,
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
def test_release_canvas_context_1(n):
    # Test with offscreen canvases. A context is created, but not a wgpu-native surface.

    # Note: the offscreen canvas keeps the render-texture-view alive, since it
    # is used to e.g. download the resulting image. That's why we also see
    # Textures and TextureViews in the counts.

    from wgpu.gui.offscreen import WgpuCanvas

    if hasattr(wgpu.gui.offscreen, "asyncio"):
        pytest.skip("#404 is not yet merged")  # todo: remove this

    yield {
        "expected_counts_after_create": {
            "CanvasContext": (n, 0),
            "Texture": (n, n),
            "TextureView": (n, n),
        },
    }

    for i in range(n):
        c = WgpuCanvas()
        c.request_draw(make_draw_func_for_canvas(c))
        c.draw()
        yield c.get_context()


@create_and_release
def test_release_canvas_context_2(n):
    # Test with GLFW canvases.

    # Note: in a draw, the textureview is obtained (thus creating a
    # Texture and a TextureView, but these are released in present(),
    # so we don't see them in the counts.

    loop = asyncio.get_event_loop_policy().get_event_loop()

    if loop.is_running():
        pytest.skip("Cannot run this test when asyncio loop is running")
    if not can_use_glfw:
        pytest.skip("Need glfw for this test")

    from wgpu.gui.glfw import WgpuCanvas  # noqa

    yield {}

    for i in range(n):
        c = WgpuCanvas()
        c.request_draw(make_draw_func_for_canvas(c))
        loop.run_until_complete(stub_event_loop())
        yield c.get_context()

    # Need some shakes to get all canvas refs gone
    del c
    loop.run_until_complete(stub_event_loop())
    gc.collect()
    loop.run_until_complete(stub_event_loop())


@create_and_release
def test_release_canvas_context_3(n):
    # Test with PySide canvases.

    # Note: in a draw, the textureview is obtained (thus creating a
    # Texture and a TextureView, but these are released in present(),
    # so we don't see them in the counts.

    if not can_use_pyside6:
        pytest.skip("Need pyside6 for this test")

    import PySide6  # noqa
    from wgpu.gui.qt import WgpuCanvas  # noqa

    app = PySide6.QtWidgets.QApplication.instance()
    if app is None:
        app = PySide6.QtWidgets.QApplication([""])

    yield {}

    for i in range(n):
        c = WgpuCanvas()
        c.request_draw(make_draw_func_for_canvas(c))
        app.processEvents()
        yield c.get_context()

    # Need some shakes to get all canvas refs gone
    del c
    gc.collect()
    app.processEvents()


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
    # -> Check gpu.diagnostics.rs_counts.print_report(), nothing there that ends with "Encoder".
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
    # todo: implement this when we do support them
    pytest.skip("Query set not implemented")


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
    # -> Check gpu.diagnostics.rs_counts.print_report(), nothing there that ends with "Encoder".
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


ALL_TEST_FUNCS = [
    ob
    for name, ob in list(globals().items())
    if name.startswith("test_") and callable(ob)
]
RELEASE_TEST_FUNCS = [
    func for func in ALL_TEST_FUNCS if func.__name__.startswith("test_release_")
]


if __name__ == "__main__":
    for func in ALL_TEST_FUNCS:
        print(func.__name__ + " ...")
        try:
            func()
        except pytest.skip.Exception:
            print("  skipped")
    print("done")
