import sys
import subprocess

import wgpu


def test_stub():
    import wgpu  # noqa: F401

    assert wgpu.requestAdapter
    assert wgpu.requestAdapterAsync


def test_do_not_import_utils_subpackage():

    # OK: use something from root package
    out = subprocess.getoutput(
        [sys.executable, "-c", "import wgpu; print(wgpu.__version__)"]
    )
    assert "Error" not in out
    assert wgpu.__version__ in out

    # OK: use something from utils if we import it first
    out = subprocess.getoutput(
        [
            sys.executable,
            "-c",
            "import wgpu.utils; print(wgpu.utils.compute_with_buffers)",
        ]
    )
    assert "Error" not in out
    assert "function compute_with_buffers" in out

    # FAIL: use something from utils if we only import wgpu
    out = subprocess.getoutput(
        [sys.executable, "-c", "import wgpu; print(wgpu.utils.compute_with_buffers)"]
    )
    assert "Error" in out
    assert "has no attribute" in out and "utils" in out
