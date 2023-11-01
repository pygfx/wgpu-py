import gc

import wgpu.backends.rs
import pytest


# Create the default device beforehand
wgpu.utils.get_default_device()


# %% The logic


def get_counts():
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


def compare_counts(counts1, counts2):
    more_py = {}
    more_native = {}
    for name in counts1:
        c1 = counts1[name][0]
        c2 = counts2[name][0]
        if c2 > c1:
            more_py[name] = c2 - c1
        c1 = counts1[name][1]
        c2 = counts2[name][1]
        if c2 > c1:
            more_native[name] = c2 - c1
    return more_py, more_native


def obname_from_test_func_name(func):
    obname = func.__name__
    prefix = "test_release_"
    assert obname.startswith(prefix)
    return obname[len(prefix) :].replace("_", "").lower()


def create_and_release(create_objects_func):
    def _create_and_release():
        n = 32

        # Before ...
        counts1 = get_counts()

        # Create objects
        objects = list(create_objects_func(n))

        # Test the count and that all are of the same class
        assert len(objects) == n
        assert all(
            isinstance(objects[i], objects[0].__class__) for i in range(len(objects))
        )
        class_name = objects[0].__class__.__name__[3:].lower()

        # Test that cass matches function name (should prevent a group of copy-paste errors)
        obname = obname_from_test_func_name(create_objects_func)
        assert obname == class_name

        counts2 = get_counts()
        more_py, more_native = compare_counts(counts1, counts2)

        print("  more_py", more_py)
        print("  more_native", more_native)

        # Make sure we do indeed have n more
        max_excess_py, max_excess_native = max(more_py.values()), max(
            more_native.values()
        )
        assert max_excess_py == n
        assert max_excess_native == n

        # Delete objects
        print("  release")
        del objects
        gc.collect()

        counts3 = get_counts()
        more_py, more_native = compare_counts(counts1, counts3)
        max_excess_py, max_excess_native = max(more_py.values(), default=0), max(
            more_native.values(), default=0
        )

        print("  more_py", more_py)
        print("  more_native", more_native)

        assert max_excess_py == 0
        assert max_excess_native <= 1
        # assert not more_py
        # assert not more_native

    _create_and_release.__name__ = create_objects_func.__name__
    return _create_and_release


# %% Meta tests


# todo: enable this when done
def xfail_test_all_objects_covered():
    """Test that we have a test_release test function for each known object."""

    ref_obnames = set(key.lower() for key in get_counts().keys())
    func_obnames = set(obname_from_test_func_name(func) for func in RELEASE_TEST_FUNCS)

    missing = ref_obnames - func_obnames
    extra = func_obnames - ref_obnames
    assert not missing
    assert not extra


def test_all_functions_solid():
    """Test that all funcs starting with test_release are decorated appropriately."""
    for func in RELEASE_TEST_FUNCS:
        is_decorated = func.__code__.co_name == "_create_and_release"
        assert is_decorated, func.__name__ + " not decorated"


def test_buffers_meta1():
    """Making sure that the test indeed fails, when holding onto the objects."""

    lock = []

    @create_and_release
    def test_buffer(n):
        device = wgpu.utils.get_default_device()
        for i in range(n):
            b = device.create_buffer(size=128, usage=wgpu.BufferUsage.COPY_DST)
            lock.append(b)
            yield b

    with pytest.raises(AssertionError):
        test_buffer()


def test_buffers_meta2():
    """Making sure that the test indeed fails, by disabling the release call."""

    ori = wgpu.backends.rs.GPUBuffer._destroy
    wgpu.backends.rs.GPUBuffer._destroy = lambda self: None

    try:
        with pytest.raises(AssertionError):
            test_release_buffer()

    finally:
        wgpu.backends.rs.GPUBuffer._destroy = ori


# %% The actual tests


@create_and_release
def test_release_adapter(n):
    for i in range(n):
        yield wgpu.request_adapter(canvas=None, power_preference="high-performance")


@create_and_release
def xfail_test_release_device(n):
    # Device object seem not to be cleaned up at wgpu-native. Wazzup with that?

    adapter = wgpu.utils.get_default_device().adapter
    for i in range(n):
        yield adapter.request_device()


@create_and_release
def test_release_bind_group(n):
    device = wgpu.utils.get_default_device()

    buffer1 = device.create_buffer(size=128, usage=wgpu.BufferUsage.STORAGE)

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

    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)

    for i in range(n):
        yield device.create_bind_group(layout=bind_group_layout, entries=bindings)


@create_and_release
def xfail_test_release_bind_group_layout(n):
    # When we use the same binding layout descriptor, wgpu-native
    # re-uses the BindGroupLayout object. On the other hand, it also
    # does not seem to clean them up. Perhaps it just caches them? There
    # are only so many possible combinations, and its just 152 bytes
    # (on Metal) per object.
    #
    # * We should somehow allow this case in the test.
    # * Should we do something similar with *our* BindGroupLayout object?

    device = wgpu.utils.get_default_device()

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
        binding_layouts[0]["binding"] = i
        yield device.create_bind_group_layout(entries=binding_layouts)


@create_and_release
def test_release_buffer(n):
    device = wgpu.utils.get_default_device()
    for i in range(n):
        yield device.create_buffer(size=128, usage=wgpu.BufferUsage.COPY_DST)


def test_CanvasContext(n):
    pass


def test_CommandBuffer(n):
    pass


def test_CommandEncoder(n):
    pass


def test_ComputePassEncoder(n):
    pass


def test_ComputePipeline(n):
    pass


def test_PipelineLayout(n):
    pass


def test_QuerySet(n):
    pass


def test_Queue(n):
    pass


def test_RenderBundle(n):
    pass


def test_RenderBundleEncoder(n):
    pass


def test_RenderPassEncoder(n):
    pass


def test_RenderPipeline(n):
    pass


def test_Sampler(n):
    pass


def test_ShaderModule(n):
    pass


def test_Texture(n):
    pass


def test_TextureView(n):
    pass


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
