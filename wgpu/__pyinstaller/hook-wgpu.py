# ruff: noqa: N999

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Init variables that PyInstaller will pick up.
hiddenimports = []
datas = []
binaries = []

# Include our resource data and binaries.
datas += collect_data_files("wgpu", subdir="resources")
binaries += collect_dynamic_libs("wgpu")

# Always include the wgpu-native backend. Since an import is not needed to
# load this (default) backend, PyInstaller does not see it by itself.
hiddenimports += ["wgpu.backends.auto", "wgpu.backends.wgpu_native"]
