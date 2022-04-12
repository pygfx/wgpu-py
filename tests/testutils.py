import os
from pathlib import Path
import sys
import subprocess

import wgpu.backends.rs  # noqa
from wgpu.utils import get_default_device  # noqa


ROOT = Path(__file__).parent.parent  # repo root
examples_dir = ROOT / "examples"
screenshots_dir = examples_dir / "screenshots"
diffs_dir = screenshots_dir / "diffs"


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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    print("_determine_can_use_wgpu_lib() status code:", result.returncode)
    return (
        result.stdout.strip().endswith("ok")
        and "traceback" not in result.stderr.lower()
    )


def _determine_can_use_glfw():
    code = "import glfw;exit(0) if glfw.init() else exit(1)"
    try:
        subprocess.check_output([sys.executable, "-c", code])
    except Exception:
        return False
    else:
        return True


def wgpu_backend_endswith(query):
    """
    Query the configured wgpu backend driver.
    """
    code = "import wgpu.utils; d = wgpu.utils.get_default_device(); print(d.adapter.properties['adapterType'], d.adapter.properties['backendType'])"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=ROOT,
    )
    return (
        result.stdout.strip().endswith(query)
        and "traceback" not in result.stderr.lower()
    )


def find_examples(query=None, negative_query=None, return_stems=False):
    result = []
    for example_path in examples_dir.glob("*.py"):
        example_code = example_path.read_text()
        query_match = query is None or query in example_code
        negative_query_match = (
            negative_query is None or negative_query not in example_code
        )
        if query_match and negative_query_match:
            result.append(example_path)
    result = list(sorted(result))
    if return_stems:
        result = [r.stem for r in result]
    return result


can_use_wgpu_lib = _determine_can_use_wgpu_lib()
can_use_glfw = _determine_can_use_glfw()
is_ci = bool(os.getenv("CI", None))
is_lavapipe = wgpu_backend_endswith("CPU Vulkan")
