"""
Hook for building wheels with the hatchling build backend.

* Set wheel to being platform-specific (not pure Python).
* Download the wgpu-native library before creating the wheel.
* Support cross-platform wheel building with a custom env var.
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
from subprocess import getoutput

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from download_wgpu_native import main as download_lib  # noqa


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # See https://hatch.pypa.io/latest/plugins/builder/wheel/#build-data

        if self.target_name == "wheel":

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


def check_git_status():
    git_status = getoutput("git status --porcelain")
    # print("Git status:\n" + git_status)
    for line in git_status.splitlines():
        assert not line.strip().startswith("M wgpu/"), "Git has open changes!"


def remove_all_libs():
    dir = "wgpu/resources/"
    for fname in os.listdir(dir):
        if fname.endswith((".so", ".dll", ".dylib")):
            os.remove(dir + fname)
            print(f"Removed {fname} from resource dir")
