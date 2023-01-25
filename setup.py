import re
import platform

from setuptools import find_packages, setup
from wheel.bdist_wheel import get_platform, bdist_wheel as _bdist_wheel


NAME = "wgpu"
SUMMARY = "Next generation GPU API for Python"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


class bdist_wheel(_bdist_wheel):  # noqa: N801
    def finalize_options(self):
        self.plat_name = get_platform(None)  # force a platform tag
        _bdist_wheel.finalize_options(self)


resources_globs = ["*.h", "*.idl"]
if platform.system() == "Linux":
    resources_globs.append("*-release.so")
elif platform.system() == "Darwin":
    resources_globs.append("*-release.dylib")
elif platform.system() == "Windows":
    resources_globs.append("*-release.dll")
else:
    pass  # don't include binaries; user will have to arrange for the lib

runtime_deps = ["cffi>=1.15.0rc2", "rubicon-objc>=0.4.1; sys_platform == 'darwin'"]
extra_deps = {
    "jupyter": ["jupyter_rfb>=0.3.1"],
    "glfw": ["glfw>=1.9"],
}

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(
        exclude=["codegen", "codegen.*", "tests", "tests.*", "examples", "examples.*"]
    ),
    package_data={f"{NAME}.resources": resources_globs},
    python_requires=">=3.7.0",
    install_requires=runtime_deps,
    extras_require=extra_deps,
    license="BSD 2-Clause",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url="https://github.com/pygfx/wgpu-py",
    cmdclass={"bdist_wheel": bdist_wheel},
    data_files=[("", ["LICENSE"])],
    entry_points={
        "pyinstaller40": [
            "hook-dirs = wgpu.__pyinstaller:get_hook_dirs",
            "tests = wgpu.__pyinstaller:get_test_dirs",
        ],
    },
)
