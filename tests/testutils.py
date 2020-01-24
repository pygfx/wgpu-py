import sys
import subprocess


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
