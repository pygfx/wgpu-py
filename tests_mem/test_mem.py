import gc
import asyncio

import wgpu.backends.rs

import pytest
from testutils import can_use_glfw, can_use_wgpu_lib, can_use_pyside6


if not can_use_wgpu_lib:
    pytest.skip(
        "Skipping tests that need a window or the wgpu lib", allow_module_level=True
    )


# Create the default device beforehand
DEVICE = wgpu.utils.get_default_device()


# %% The logic


def get_counts():
    """Get a dict that maps object names to a 2-tuple represening
    the counts in py and wgpu-native.
    """
    counts_py = wgpu.diagnostics.object_counts.get_dict()
    counts_native = wgpu.diagnostics.rs_counts.get_dict()

    all_keys = set(counts_py) | set(counts_native)

    default = {"count": -1}

    counts = {}
    for key in sorted(all_keys):
        counts[key] = (
            counts_py.get(key, default)["count"],
            counts_native.get(key, default)["count"],
        )
    counts.pop("total")

    return counts


def get_excess_counts(counts1, counts2):
    """Compare two counts dicts, and return a new dict with the fields
    that have increased counts.
    """
    more = {}
    for name in counts1:
        c1 = counts1[name][0]
        c2 = counts2[name][0]
        more_py = 0
        if c2 > c1:
            more_py = c2 - c1
        c1 = counts1[name][1]
        c2 = counts2[name][1]
        more_native = 0
        if c2 > c1:
            more_native = c2 - c1
        if more_py or more_native:
            more[name] = more_py, more_native
    return more


def ob_name_from_test_func(func):
    """Translate test_release_bind_group() to "BindGroup"."""
    func_name = func.__name__
    prefix = "test_release_"
    assert func_name.startswith(prefix)
    words = func_name[len(prefix) :].split("_")
    if words[-1].isnumeric():
        words.pop(-1)
    return "".join(word.capitalize() for word in words)


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


def create_and_release(create_objects_func):
    """Decorator."""

    def core_test_func():
        """The core function that does the testing."""

        n = 32

        generator = create_objects_func(n)
        ob_name = ob_name_from_test_func(create_objects_func)

        # ----- Collect options

        options = {
            "expected_counts_after_create": {ob_name: (32, 32)},
            "expected_counts_after_release": {},
        }

        func_options = next(generator)
        assert isinstance(func_options, dict), "First yield must be an options dict"
        options.update(func_options)

        # Measure baseline object counts
        counts1 = get_counts()

        # ----- Create

        # Create objects
        objects = list(generator)

        # Test the count
        assert len(objects) == n

        # Test that all objects are of the same class.
        # (this for-loop is a bit weird, but its to avoid leaking refs to objects)
        cls = objects[0].__class__
        assert all(isinstance(objects[i], cls) for i in range(len(objects)))

        # Test that class matches function name (should prevent a group of copy-paste errors)
        assert ob_name == cls.__name__[3:]

        # Measure peak object counts
        counts2 = get_counts()
        more2 = get_excess_counts(counts1, counts2)
        print("  more after create:", more2)

        # Make sure the actual object has increased
        assert more2  # not empty
        assert more2 == options["expected_counts_after_create"]

        # It's ok if other objects are created too ...

        # ----- Release

        # Delete objects
        del objects
        gc.collect()

        # Measure after-release object counts
        counts3 = get_counts()
        more3 = get_excess_counts(counts1, counts3)
        print("  more after release:", more3)

        # Check!
        assert more3 == options["expected_counts_after_release"]

    core_test_func.__name__ = create_objects_func.__name__
    return core_test_func


async def stub_event_loop():
    pass


# %% Meta tests


# todo: enable this when done
def xfail_test_meta_all_objects_covered():
    """Test that we have a test_release test function for each known object."""

    ref_obnames = set(key.lower() for key in get_counts().keys())
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
def xfail_test_release_device(n):
    # XFAIL: Device object seem not to be cleaned up at wgpu-native

    yield {}
    adapter = DEVICE.adapter
    for i in range(n):
        yield adapter.request_device()


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

    from wgpu.gui.glfw import WgpuCanvas, glfw  # noqa

    yield {}

    for i in range(n):
        c = WgpuCanvas()
        c.request_draw(make_draw_func_for_canvas(c))
        loop.run_until_complete(stub_event_loop())
        yield c.get_context()
        del c

    # Need some shakes to get all canvas refs gone
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

    import PySide6
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
        del c

    # Need some shakes to get all canvas refs gone
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


# @create_and_release
# def test_release_compute_pass_encoder(n):
#     pass


# @create_and_release
# def test_release_compute_pipeline(n):
#     pass


# @create_and_release
# def test_release_pipeline_layout(n):
#     pass


# @create_and_release
# def test_release_query_set(n):
#     pass


# @create_and_release
# def test_release_queue(n):
#     pass


# @create_and_release
# def test_release_render_bundle(n):
#     pass


# @create_and_release
# def test_release_render_bundle_encoder(n):
#     pass


# @create_and_release
# def test_release_render_pass_encoder(n):
#     pass


# @create_and_release
# def test_release_render_pipeline(n):
#     pass


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
            pass
    print("done")
