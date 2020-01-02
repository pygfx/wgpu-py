import re

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from wheel.pep425tags import get_platform


NAME = "wgpu"
SUMMARY = "Next generation GPU API for Python"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        self.plat_name = get_platform()  # force a platform tag
        _bdist_wheel.finalize_options(self)


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={f"{NAME}.resources": ["*.dll", "*.so", "*.dylib", "*.h", "*.idl", "commit-sha"]},
    python_requires=">=3.6.0",
    license=open("LICENSE").read(),
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Almar Klein",
    author_email="almar.klein@gmail.com",
    url="https://github.com/almarklein/wgpu-py",
    cmdclass={"bdist_wheel": bdist_wheel},
)
