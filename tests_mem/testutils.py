import gc
import os
import sys
import time
import subprocess

import psutil
import wgpu
from wgpu._diagnostics import int_repr


p = psutil.Process()


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


def _determine_can_use_pyside6():
    code = "import PySide6.QtGui"
    try:
        subprocess.check_output([sys.executable, "-c", code])
    except Exception:
        return False
    else:
        return True


can_use_wgpu_lib = _determine_can_use_wgpu_lib()
can_use_glfw = _determine_can_use_glfw()
can_use_pyside6 = _determine_can_use_pyside6()
is_ci = bool(os.getenv("CI", None))
is_pypy = sys.implementation.name == "pypy"

TEST_ITERS = None


def get_memory_usage():
    """Get how much memory the process consumes right now."""
    # vms: total virtual memory. Seems not suitable, because it gets less but bigger differences.
    # rss: the part of the virtual memory that is not in swap, i.e. consumers ram.
    # uss: memory that would become available when the process is killed (excludes shared).
    # return p.memory_info().rss
    return p.memory_full_info().uss


def clear_mem():
    time.sleep(0.001)
    gc.collect()

    time.sleep(0.001)
    gc.collect()

    if is_pypy:
        gc.collect()

    device = wgpu.utils.get_default_device()
    device._poll()


def get_counts():
    """Get a dict that maps object names to a 2-tuple represening
    the counts in py and wgpu-native.
    """
    counts_py = wgpu.diagnostics.object_counts.get_dict()
    counts_native = wgpu.diagnostics.wgpu_native_counts.get_dict()

    all_keys = set(counts_py) | set(counts_native)

    default = {"count": -1}

    counts = {}
    for key in sorted(all_keys):
        counts[key] = (
            counts_py.get(key, default)["count"],
            counts_native.get(key, default)["count"],
        )
    counts.pop("total")

    return counts


def get_excess_counts(counts1, counts2):
    """Compare two counts dicts, and return a new dict with the fields
    that have increased counts.
    """
    more = {}
    for name in counts1:
        c1 = counts1[name][0]
        c2 = counts2[name][0]
        more_py = 0
        if c2 > c1:
            more_py = c2 - c1
        c1 = counts1[name][1]
        c2 = counts2[name][1]
        more_native = 0
        if c2 > c1:
            more_native = c2 - c1
        if more_py or more_native:
            more[name] = more_py, more_native
    return more


def ob_name_from_test_func(func):
    """Translate test_release_bind_group() to "BindGroup"."""
    func_name = func.__name__
    prefix = "test_release_"
    assert func_name.startswith(prefix)
    words = func_name[len(prefix) :].split("_")
    if words[-1].isnumeric():
        words.pop(-1)
    return "".join(word.capitalize() for word in words)


def create_and_release(create_objects_func):
    """Decorator."""

    def core_test_func():
        """The core function that does the testing."""

        if TEST_ITERS:
            n_objects_list = [8 for i in range(TEST_ITERS)]
        else:
            n_objects_list = [32, 17]

        # Init mem usage measurements
        clear_mem()
        mem3 = get_memory_usage()

        for iter, n_objects in enumerate(n_objects_list):
            generator = create_objects_func(n_objects)
            ob_name = ob_name_from_test_func(create_objects_func)

            # ----- Collect options

            options = {
                "expected_counts_after_create": {ob_name: (n_objects, n_objects)},
                "expected_counts_after_release": {},
            }

            func_options = next(generator)
            assert isinstance(func_options, dict), "First yield must be an options dict"
            options.update(func_options)

            # Measure baseline object counts
            clear_mem()
            counts1 = get_counts()

            # ----- Create

            # Create objects
            objects = list(generator)

            # Test the count
            assert len(objects) == n_objects

            # Test that all objects are of the same class.
            # (this for-loop is a bit weird, but its to avoid leaking refs to objects)
            cls = objects[0].__class__
            assert all(isinstance(objects[i], cls) for i in range(len(objects)))

            # Test that class matches function name (should prevent a group of copy-paste errors)
            assert ob_name == cls.__name__[3:]

            # Give wgpu some slack to clean up temporary resources
            wgpu.utils.get_default_device()._poll()

            # Measure peak object counts
            counts2 = get_counts()
            more2 = get_excess_counts(counts1, counts2)
            if not TEST_ITERS:
                print("  more after create:", more2)

            # Make sure the actual object has increased
            assert more2  # not empty
            assert more2 == options["expected_counts_after_create"]

            # It's ok if other objects are created too ...

            # ----- Release

            # Delete objects
            del objects
            clear_mem()

            # Measure after-release object counts
            counts3 = get_counts()
            more3 = get_excess_counts(counts1, counts3)
            if not TEST_ITERS:
                print("  more after release:", more3)

            # Check!
            assert more3 == options["expected_counts_after_release"]

            # Print mem usage info
            if TEST_ITERS:
                mem1 = mem3  # initial mem is end-mem of last iter
                mem3 = get_memory_usage()
                mem_info = (int_repr(mem3 - mem1) + "B").rjust(7)
                print(mem_info, end=(" " if (iter + 1) % 10 else "\n"))

    core_test_func.__name__ = create_objects_func.__name__
    return core_test_func
