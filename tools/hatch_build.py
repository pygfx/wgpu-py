"""
Hook for building wheels with the hatchling build backend.

* Set wheel to being platform-specific (not pure Python).
* Download the wgpu-native library before creating the wheel.
* Support cross-platform wheel building with a custom env var.
* Note that for sdist we go into pure-Python mode.
"""

# Note on an alternative approach:
#
# In pyproject.toml set:
#
#     build-backend = "local_build_backend"
#     backend-path = ["tools"]
#
# In local_build_backend.py define functions like build_wheel and build_sdist
# that simply call the same function from hatchling or flit_core. But first
# download the lib.
#
# I found this approach pretty elegant (it works with any build-backend!) so I
# wanted to write it up. The downside for our use-case, however, is that the
# wheels must be renamed after building, and that the wheels are still marked as
# pure Python.

import os
import sys
import sysconfig
from subprocess import run, PIPE

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

root_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, os.path.join(root_dir, "tools"))

from download_wgpu_native import main as download_lib  # noqa: E402


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # See https://hatch.pypa.io/latest/plugins/builder/wheel/#build-data

        # We only do our thing when this is a wheel build from the repo.
        # If this is an sdist build, or a wheel build from an sdist,
        # we go pure-Python mode, and expect the user to set WGPU_LIB_PATH.
        # We also allow building an arch-agnostic wheel explicitly, using an env var.

        if os.getenv("WGPU_PY_BUILD_NOARCH", "").lower() in ("1", "true"):
            pass  # Explicitly disable including the lib
        elif self.target_name == "wheel" and is_git_repo():
            # Prepare
            check_git_status()
            remove_all_libs()

            # State that the wheel is not cross-platform
            build_data["pure_python"] = False

            # Download and set tag
            platform_info = os.getenv("WGPU_BUILD_PLATFORM_INFO")
            if platform_info:
                # A cross-platform build
                wgpu_native_tag, wheel_tag = platform_info.split()
                opsys, arch = wgpu_native_tag.split("_", 1)
                build_data["tag"] = "py3-none-" + wheel_tag
                download_lib(None, opsys, arch)
            else:
                # A build for this platform, e.g. ``pip install -e .``
                build_data["infer_tag"] = True
                download_lib()

            # Make sure that the download did not bump the wgpu-native version
            check_git_status()

        # Optionally build the (experimental) Dawn backend. This requires Dawn
        # (or Emscripten, for Pyodide), Cython and a C compiler, see tools/build_dawn.py.
        if self.target_name == "wheel" and build_dawn_enabled():
            import build_dawn

            if build_dawn.is_emscripten():
                check_no_native_binaries()
            built = build_dawn.build()
            build_data["pure_python"] = False
            for path in built:
                if path.endswith((".so", ".dll", ".dylib")):
                    continue  # already included as artifacts, see pyproject.toml
                rel = os.path.relpath(path, root_dir).replace(os.sep, "/")
                build_data["force_include"][path] = rel
            if build_dawn.is_emscripten():
                # Pyodide (pyodide build) retags the platform of the wheel
                v = "cp{}{}".format(*sys.version_info[:2])
                plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
                build_data["tag"] = f"{v}-{v}-{plat}"
            else:
                build_data["infer_tag"] = True

    def dependencies(self):
        # Extra build dependencies, see https://hatch.pypa.io/latest/plugins/build-hook/reference/
        if self.target_name == "wheel" and build_dawn_enabled():
            return ["cython>=3.1", "setuptools>=64"]
        return []


def check_no_native_binaries():
    """Binaries in the source tree end up in the wheel; avoid that for Pyodide wheels."""
    found = []
    for dirpath, _, filenames in os.walk(os.path.join(root_dir, "wgpu")):
        for fname in filenames:
            if (
                fname.endswith((".so", ".dll", ".dylib", ".pyd"))
                and "emscripten" not in fname
            ):
                found.append(os.path.relpath(os.path.join(dirpath, fname), root_dir))
    if found:
        raise RuntimeError(
            "Building a Pyodide wheel, but the source tree has native binaries, "
            f"which would be included in the wheel: {found}. Remove them first."
        )


def build_dawn_enabled():
    return os.getenv("WGPU_PY_BUILD_DAWN", "").lower() in ("1", "true")


def is_git_repo():
    return os.path.isdir(os.path.join(root_dir, ".git"))


def check_git_status():
    p = run(
        "git status --porcelain", shell=True, cwd=root_dir, stdout=PIPE, stderr=PIPE
    )
    git_status = p.stdout.decode(errors="ignore")
    # print("Git status:\n" + git_status)
    for line in git_status.splitlines():
        assert not line.strip().startswith("M wgpu/"), "Git has open changes!"


def remove_all_libs():
    dir = os.path.join(root_dir, "wgpu", "resources")
    for fname in os.listdir(dir):
        if fname.endswith((".so", ".dll", ".dylib")):
            os.remove(os.path.join(dir, fname))
            print(f"Removed {fname} from resource dir")
