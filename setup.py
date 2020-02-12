import re
import sys

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from wheel.pep425tags import get_platform


NAME = "wgpu"
SUMMARY = "Next generation GPU API for Python"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


class bdist_wheel(_bdist_wheel):  # noqa: N801
    def finalize_options(self):
        self.plat_name = get_platform(None)  # force a platform tag
        _bdist_wheel.finalize_options(self)


resources_globs = ["*.h", "*.idl", "commit-sha"]
if sys.platform.startswith("win"):
    resources_globs.append("*.dll")
elif sys.platform.startswith("linux"):
    resources_globs.append("*.so")
elif sys.platform.startswith("darwin"):
    resources_globs.append("*.dylib")
else:
    pass  # don't include binaries; user will have to arrange for the lib

runtime_deps = ["cffi>=1.10"]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={f"{NAME}.resources": resources_globs},
    python_requires=">=3.6.0",
    install_requires=runtime_deps,
    license="BSD 2-Clause",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url="https://github.com/almarklein/wgpu-py",
    cmdclass={"bdist_wheel": bdist_wheel},
    data_files=[("", ["LICENSE"])],
)
