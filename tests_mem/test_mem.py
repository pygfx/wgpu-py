import gc

import wgpu.backends.rs
import wgpu.gui.offscreen

import pytest
from testutils import can_use_glfw, can_use_wgpu_lib


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


def create_and_release(create_objects_func):
    """Decorator."""

    def core_test_func():
        """The core function that does the testing."""

        n = 32

        generator = create_objects_func(n)

        # ----- Collect options

        options = {
            "expected_native_high_count": n,
            "allowed_native_low_count": 0,
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
        ob_name = ob_name_from_test_func(create_objects_func)
        assert ob_name == cls.__name__[3:]

        # Measure peak object counts
        counts2 = get_counts()
        more2 = get_excess_counts(counts1, counts2)
        print("  more after create:", more2)

        # Make sure the actual object has increased
        expected_high_count = n, options["expected_native_high_count"]
        assert ob_name in more2
        assert more2[ob_name] == expected_high_count

        # It's ok if other objects are created too ...

        # ----- Release

        # Delete objects
        del objects
        gc.collect()

        # Measure after-release object counts
        counts3 = get_counts()
        more3 = get_excess_counts(counts1, counts3)
        print("  more after release:", more3)

        # If no excess objects, all is well!
        if not more3:
            return

        # Otherwise, look deeper
        expected_low_count = 0, options["allowed_native_low_count"]
        more3.setdefault(ob_name, (0, 0))
        assert more3[ob_name] == expected_low_count

        # Nothing left other than that?
        more4 = more3.copy()
        more4.pop(ob_name)
        assert not more4

    core_test_func.__name__ = create_objects_func.__name__
    return core_test_func


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
    yield {}

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

    yield {"expected_native_high_count": 1, "allowed_native_low_count": 1}

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

    yield {"expected_native_high_count": 0}

    for i in range(n):
        c = wgpu.gui.offscreen.WgpuCanvas()
        ctx = c.get_context()
        ctx.configure(device=DEVICE, format="bgra8unorm-srgb")
        ctx.get_current_texture()  # forces creating a surface
        yield ctx


@create_and_release
def xfail_test_release_canvas_context_2(n):
    # Test with GLFW canvases.

    # XFAIL: The CanvasContext objects (both the Py side and the native surface) are not properly released,
    # but they are when I check right afterwards, maybe it needs an event loop iter or glfw poll?

    # todo: revisit this with the new upcoming APIs?

    if not can_use_glfw:
        pytest.skip("Need glfw for this test")

    from wgpu.gui.glfw import WgpuCanvas, glfw  # noqa

    yield {}

    for i in range(n):
        c = WgpuCanvas()
        ctx = c.get_context()
        ctx.configure(device=DEVICE, format="bgra8unorm-srgb")
        ctx.get_current_texture()  # forces creating a surface
        yield ctx


# @create_and_release
# def test_release_command_buffer(n):
#     pass


# @create_and_release
# def test_release_command_encoder(n):
#     pass


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


# @create_and_release
# def test_release_shader_module(n):
#     pass


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
    yield {}
    texture = DEVICE.create_texture(
        size=(16, 16, 16), usage=wgpu.TextureUsage.TEXTURE_BINDING, format="rgba8unorm"
    )
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
        func()
    print("done")
