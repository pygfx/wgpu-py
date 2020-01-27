import sys
import subprocess


def iters_equal(iter1, iter2):
    iter1, iter2 = list(iter1), list(iter2)
    assert len(iter1) == len(iter2)
    assert all(iter1[i] == iter2[i] for i in range(len(iter1)))
    return True


def _determine_can_use_vulkan_sdk():
    try:
        subprocess.check_output(["spirv-val", "--version"])
    except Exception:
        return False
    else:
        return True


def _determine_can_use_wgpu_lib():
    code = "import wgpu.backend.rs;"
    code += "wgpu.requestAdapter(powerPreference='high-performance').requestDevice()"
    try:
        subprocess.check_output(
            [sys.executable, "-c", code,]
        )
    except Exception:
        return False
    else:
        return True


can_use_vulkan_sdk = _determine_can_use_vulkan_sdk()
can_use_wgpu_lib = _determine_can_use_wgpu_lib()
