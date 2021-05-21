import os
import sys
import subprocess

import wgpu.backends.rs  # noqa
from wgpu.utils import get_default_device  # noqa


def run_tests(scope):
    """Run all test functions in the given scope."""
    for func in list(scope.values()):
        if callable(func) and func.__name__.startswith("test_"):
            if func.__code__.co_argcount == 0:
                print(f"Running {func.__name__} ...")
                func()
            else:
                print(f"SKIPPING {func.__name__} because it needs args")
    print("Done")


def iters_equal(iter1, iter2):
    iter1, iter2 = list(iter1), list(iter2)
    if len(iter1) == len(iter2):
        if all(iter1[i] == iter2[i] for i in range(len(iter1))):
            return True
    return False


def _determine_can_use_vulkan_sdk():
    try:
        subprocess.check_output(["spirv-val", "--version"])
    except Exception:
        return False
    else:
        return True


def _determine_can_use_wgpu_lib():
    code = "import wgpu.utils; wgpu.utils.get_default_device()"
    try:
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                code,
            ]
        )
    except Exception:
        return False
    else:
        return True


can_use_vulkan_sdk = _determine_can_use_vulkan_sdk()
can_use_wgpu_lib = _determine_can_use_wgpu_lib()
is_ci = bool(os.getenv("CI", None))
