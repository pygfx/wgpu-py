import sys
import subprocess

import wgpu


def test_stub():
    import wgpu  # noqa: F401

    assert wgpu.__version__
    assert wgpu.help
    assert wgpu.requestAdapter
    assert wgpu.requestAdapterAsync


def test_do_not_import_utils_subpackage():

    # OK: use something from root package
    code = "import wgpu; print(wgpu.__version__)"
    out = subprocess.getoutput([sys.executable, "-c", code])
    assert "Error" not in out
    assert wgpu.__version__ in out

    # OK: use something from utils if we import it first
    code = "import wgpu.utils; print(wgpu.utils.compute_with_buffers)"
    out = subprocess.getoutput([sys.executable, "-c", code,])
    assert "Error" not in out
    assert "function compute_with_buffers" in out

    # FAIL: use something from utils if we only import wgpu
    code = "import wgpu; print(wgpu.utils.compute_with_buffers)"
    out = subprocess.getoutput([sys.executable, "-c", code])
    assert "Error" in out
    assert "has no attribute" in out and "utils" in out

    # Also, no numpy
    code = "import sys, wgpu.utils; print('numpy' in sys.modules)"
    out = subprocess.getoutput([sys.executable, "-c", code])
    assert out.startswith("False")
