import re

from setuptools import find_packages, setup


NAME = "wgpu"

with open(f"{NAME}/__init__.py") as fh:
    VERSION = re.search(r"__version__ = \"(.*?)\"", fh.read()).group(1)


# the binary components of this library are pre-built
# so setuptools can't tell our wheel should have a platform tag
# therefore we use a custom bdist_wheel command to force it
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        f'{NAME}.resources': ['*.dll', '*.so', '*.dylib', '*.h', '*.idl'],
    },
    python_requires=">=3.6.0",
    license=open("LICENSE").read(),
    description=open("README.md").readlines()[2],
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    author="Almar Klein",
    author_email="almarklein@gmail.com",
    url="https://github.com/almarklein/wgpu-py",
    cmdclass={'bdist_wheel': bdist_wheel},
)
