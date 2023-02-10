import sys
import logging
import subprocess

import wgpu

from pytest import raises
from testutils import run_tests


def test_basic_api():
    import wgpu  # noqa: F401

    assert isinstance(wgpu.__version__, str)
    assert isinstance(wgpu.version_info, tuple)
    assert wgpu.request_adapter
    assert wgpu.request_adapter_async
    assert (
        wgpu.base.GPU.request_adapter.__code__.co_varnames
        == wgpu.base.GPU.request_adapter_async.__code__.co_varnames
    )


def test_logging():
    level = [-1]

    def set_level_test(lvl):
        level[0] = lvl

    wgpu._coreutils.logger_set_level_callbacks.append(set_level_test)
    logger = logging.getLogger("wgpu")
    logger.setLevel("ERROR")
    assert level[0] == 40
    logger.setLevel("INFO")
    assert level[0] == 20
    logger.setLevel("DEBUG")
    assert level[0] == 10
    logger.setLevel(5)
    assert level[0] == 5  # "trace" in wgpu-native
    logger.setLevel(0)
    assert level[0] == 0  # off
    # Reset
    logger.setLevel("WARNING")
    assert level[0] == 30


def test_enums_and_flags_and_structs():
    # Enums are str
    assert isinstance(wgpu.BufferBindingType.storage, str)

    # Enum groups show their values
    assert "storage" in repr(wgpu.BufferBindingType)

    # Flags are ints
    assert isinstance(wgpu.BufferUsage.STORAGE, int)

    # Flag groups show their field names (in uppercase)
    assert "STORAGE" in repr(wgpu.BufferUsage)

    # Structs are dict-like, their values str
    assert isinstance(wgpu.structs.DeviceDescriptor, wgpu.structs.Struct)
    assert isinstance(wgpu.structs.DeviceDescriptor.label, str)
    assert isinstance(wgpu.structs.DeviceDescriptor.required_features, str)

    # Structs show their field names
    for key in wgpu.structs.DeviceDescriptor:
        assert key in repr(wgpu.structs.DeviceDescriptor)


def test_base_wgpu_api():
    gpu = wgpu.base.GPU()
    with raises(RuntimeError) as error:
        gpu.request_adapter(canvas=None, power_preference="high-performance")
    assert "select a backend" in str(error.value).lower()

    # Fake a device and an adapter
    adapter = wgpu.base.GPUAdapter(None, set(), {}, {})
    queue = wgpu.GPUQueue("", None, None)
    device = wgpu.base.GPUDevice("device08", -1, adapter, {42, 43}, {}, queue)

    assert queue._device is device

    assert isinstance(adapter.features, set)
    assert adapter.features == set()
    assert isinstance(adapter.limits, dict)
    assert set(device.limits.keys()) == set()

    assert isinstance(device, wgpu.base.GPUObjectBase)
    assert device.label == "device08"
    assert device.features == {42, 43}
    assert hex(id(device)) in repr(device)
    assert device.label in repr(device)


def test_that_we_know_how_our_api_differs():
    doc = wgpu.base.apidiff.__doc__
    assert isinstance(doc, str)
    assert "GPUBuffer.get_mapped_range" in doc
    assert "GPUDevice.create_buffer_with_data" in doc


def test_that_all_docstrings_are_there():
    exclude = ["Union", "List", "Dict"]

    for cls in wgpu.base.__dict__.values():
        if not isinstance(cls, type):
            continue
        if cls.__name__.startswith("_") or cls.__name__ in exclude:
            continue
        assert cls.__doc__, f"No docstring on {cls.__name__}"
        for name, attr in cls.__dict__.items():
            if not (callable(attr) or isinstance(attr, property)):
                continue
            if name.startswith("_"):
                continue
            func = attr.fget if isinstance(attr, property) else attr
            assert func.__doc__, f"No docstring on {func.__name__}"


def get_output_from_subprocess(code):
    cmd = [sys.executable, "-c", code]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode(errors="ignore")


def test_do_not_import_utils_subpackage():
    # OK: use something from root package
    code = "import wgpu; print(wgpu.__version__)"
    out = get_output_from_subprocess(code)
    assert "Error" not in out
    assert wgpu.__version__ in out

    # OK: use something from utils if we import it first
    code = "import wgpu.utils; print(wgpu.utils.compute_with_buffers)"
    out = get_output_from_subprocess(code)
    assert "Error" not in out
    assert "function compute_with_buffers" in out

    # FAIL: use something from utils if we only import wgpu
    code = "import wgpu; print(wgpu.utils.compute_with_buffers)"
    out = get_output_from_subprocess(code)
    assert "Error" in out
    assert "has no attribute" in out and "utils" in out

    # Also, no numpy
    code = "import sys, wgpu.utils; print('numpy' in sys.modules)"
    out = get_output_from_subprocess(code)
    assert out.strip().endswith("False"), out


def test_register_backend_fails():
    class GPU:
        pass

    ori_GPU = wgpu.GPU  # noqa: N806
    try:
        wgpu.GPU = wgpu.base.GPU

        with raises(RuntimeError):
            wgpu._register_backend("foo")
        with raises(RuntimeError):
            wgpu._register_backend(GPU)

        GPU.request_adapter = lambda self: None
        with raises(RuntimeError):
            wgpu._register_backend(GPU)

        GPU.request_adapter_async = lambda self: None
        wgpu._register_backend(GPU)

        assert wgpu.GPU is GPU
        assert wgpu.request_adapter.__func__ is GPU.request_adapter
        assert wgpu.request_adapter_async.__func__ is GPU.request_adapter_async

        with raises(RuntimeError):
            wgpu._register_backend(GPU)  # Cannot register twice once wgpu.GPU is set

    finally:
        wgpu.GPU = wgpu.base.GPU
        wgpu._register_backend(ori_GPU)


if __name__ == "__main__":
    run_tests(globals())
