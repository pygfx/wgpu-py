import os
import sys
import subprocess


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
