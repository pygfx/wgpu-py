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

# For the GUI backends, there always is an import. The auto backend is
# problematic because PyInstaller cannot follow it to a specific
# backend. Also, glfw does not have a hook like this, so it does not
# include the binary when freezing. We can solve both problems with the
# code below. Makes the binaray a bit larger, but only marginally (less
# than 300kb).
try:
    import glfw  # noqa
except ImportError:
    pass
else:
    hiddenimports += ["wgpu.gui.glfw"]
    binaries += collect_dynamic_libs("glfw")
