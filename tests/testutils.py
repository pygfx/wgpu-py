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
    # For some reason, since wgpu-native 5c304b5ea1b933574edb52d5de2d49ea04a053db
    # the process' exit code is not zero, so we test more pragmatically.
    code = "import wgpu.utils; wgpu.utils.get_default_device(); print('ok')"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
    )
    print("_determine_can_use_wgpu_lib() status code:", result.returncode)
    err = result.stderr.decode("utf-8")
    out = result.stdout.decode("utf-8")
    return out.strip().endswith("ok") and "traceback" not in err.lower()


can_use_vulkan_sdk = _determine_can_use_vulkan_sdk()
can_use_wgpu_lib = _determine_can_use_wgpu_lib()
is_ci = bool(os.getenv("CI", None))
