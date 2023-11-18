import sys
import logging
import subprocess

import wgpu

from pytest import raises, mark
from testutils import run_tests, can_use_wgpu_lib


def test_basic_api():
    import wgpu  # noqa: F401

    assert isinstance(wgpu.__version__, str)
    assert isinstance(wgpu.version_info, tuple)
    assert isinstance(wgpu.gpu, wgpu.GPU)

    # Entrypoint funcs
    assert wgpu.gpu.request_adapter
    assert wgpu.gpu.request_adapter_async

    code1 = wgpu.GPU.request_adapter.__code__
    code2 = wgpu.GPU.request_adapter_async.__code__
    nargs1 = code1.co_argcount + code1.co_kwonlyargcount
    assert code1.co_varnames[:nargs1] == code2.co_varnames

    assert repr(wgpu.classes.GPU()).startswith(
        "<wgpu.GPU "
    )  # does not include _classes


def test_api_subpackages_are_there():
    code = "import wgpu; x = [wgpu.resources, wgpu.utils, wgpu.backends, wgpu.gui]; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    out = result.stdout.rstrip()
    assert out.endswith("ok")
    assert "traceback" not in out.lower()


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
    # Fake a device and an adapter
    adapter = wgpu.GPUAdapter(None, set(), {}, {})
    queue = wgpu.GPUQueue("", None, None)
    device = wgpu.GPUDevice("device08", -1, adapter, {42, 43}, {}, queue)

    assert queue._device is device

    assert isinstance(adapter.features, set)
    assert adapter.features == set()
    assert isinstance(adapter.limits, dict)
    assert set(device.limits.keys()) == set()

    assert isinstance(device, wgpu.GPUObjectBase)
    assert device.label == "device08"
    assert device.features == {42, 43}
    assert hex(id(device)) in repr(device)
    assert device.label in repr(device)


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_backend_is_selected_automatically():
    # Test this in a subprocess to have a clean wgpu with no backend imported yet
    code = "import wgpu; print(wgpu.gpu.request_adapter())"
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    out = result.stdout.rstrip()
    assert "GPUAdapter object at" in out
    assert "traceback" not in out.lower()


def test_that_we_know_how_our_api_differs():
    doc = wgpu._classes.apidiff.__doc__
    assert isinstance(doc, str)
    assert "GPUBuffer.get_mapped_range" in doc
    assert "GPUDevice.create_buffer_with_data" in doc


def test_that_all_docstrings_are_there():
    for name, cls in wgpu.classes.__dict__.items():
        if name.startswith("_"):
            continue
        assert isinstance(cls, type)
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


def test_do_not_import_utils_submodules():
    # OK: use something from root package
    code = "import wgpu; print(wgpu.__version__)"
    out = get_output_from_subprocess(code)
    assert "Error" not in out
    assert wgpu.__version__ in out

    # OK: wgpu.utils itself is always there
    code = "import wgpu; print(wgpu.utils)"
    out = get_output_from_subprocess(code)
    assert "Error" not in out
    assert "module 'wgpu.utils' from " in out

    # OK: wgpu.utils itself is always there
    code = "import wgpu; print(wgpu.utils.get_default_device)"
    out = get_output_from_subprocess(code)
    assert "Error" not in out
    assert "function get_default_device" in out

    # FAIL: use something from utils that's not imported
    code = "import wgpu; print(wgpu.utils.compute.compute_with_buffers)"
    out = get_output_from_subprocess(code)
    assert "Error" in out
    assert "must be explicitly imported" in out and "utils" in out

    # Also, no numpy
    code = "import sys, wgpu.utils; print('numpy' in sys.modules)"
    out = get_output_from_subprocess(code)
    assert out.strip().endswith("False"), out


def test_register_backend_fails():
    class GPU:
        pass

    fake_gpu = GPU()

    ori_gpu = wgpu.gpu  # noqa: N806
    try:
        wgpu.gpu = wgpu.classes.GPU()

        with raises(RuntimeError):
            wgpu.backends._register_backend("foo")
        with raises(RuntimeError):
            wgpu.backends._register_backend(fake_gpu)

        fake_gpu.request_adapter = lambda: None
        with raises(RuntimeError):
            wgpu.backends._register_backend(fake_gpu)

        fake_gpu.request_adapter_async = lambda: None
        fake_gpu.wgsl_language_features = set()
        wgpu.backends._register_backend(fake_gpu)

        assert wgpu.gpu is fake_gpu

        # Cannot register twice once wgpu.GPU is set
        with raises(RuntimeError):
            wgpu.backends._register_backend(fake_gpu)

    finally:
        wgpu.gpu = ori_gpu
        wgpu.backends._register_backend(ori_gpu)


if __name__ == "__main__":
    run_tests(globals())
