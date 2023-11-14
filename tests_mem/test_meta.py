"""
Some tests to confirm that the test mechanism is sound, and that tests
indeed fail under the right circumstances.
"""

import wgpu

import pytest
from testutils import can_use_wgpu_lib, create_and_release
from testutils import get_counts, ob_name_from_test_func
from test_objects import TEST_FUNCS as OBJECT_TEST_FUNCS
from test_gui_offscreen import TEST_FUNCS as GUI_TEST_FUNCS


ALL_TEST_FUNCS = OBJECT_TEST_FUNCS + GUI_TEST_FUNCS


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need wgpu lib", allow_module_level=True)


DEVICE = wgpu.utils.get_default_device()


def test_meta_all_objects_covered():
    """Test that we have a test_release test function for each known object."""

    ref_obnames = set(key for key in get_counts().keys())
    func_obnames = set(ob_name_from_test_func(func) for func in ALL_TEST_FUNCS)

    missing = ref_obnames - func_obnames
    extra = func_obnames - ref_obnames
    assert not missing
    assert not extra


def test_meta_all_functions_solid():
    """Test that all funcs starting with "test_release_" are decorated appropriately."""
    for func in ALL_TEST_FUNCS:
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

    ori = wgpu.backends.wgpu_native.GPUBuffer._destroy
    wgpu.backends.wgpu_native.GPUBuffer._destroy = lambda self: None

    from test_objects import test_release_buffer  # noqa

    try:
        with pytest.raises(AssertionError):
            test_release_buffer()

    finally:
        wgpu.backends.wgpu_native.GPUBuffer._destroy = ori


if __name__ == "__main__":
    test_meta_all_objects_covered()
    test_meta_all_functions_solid()
    test_meta_buffers_1()
    test_meta_buffers_2()
