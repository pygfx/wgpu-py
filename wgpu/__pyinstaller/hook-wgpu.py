from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Init variables that PyInstaller will pick up.
hiddenimports = []
datas = []
binaries = []

# Include the resources subpackage and its contents.
hiddenimports += ["wgpu.resources"]
datas += collect_data_files("wgpu", subdir="resources")
binaries += collect_dynamic_libs("wgpu")

# Include backends. Make sure all our backend code is present,
# and let PyInstaller resolve imports/dependencies for some.
datas += collect_data_files(
    "wgpu", subdir="backends", include_py_files=True, excludes=["__pycache__"]
)
hiddenimports += ["wgpu.backends.auto", "wgpu.backends.rs"]

# Include gui backends. Dito.
collect_data_files(
    "wgpu", subdir="gui", include_py_files=True, excludes=["__pycache__"]
)
hiddenimports += ["wgpu.gui", "wgpu.gui.offscreen"]

# For good measure, we include GLFW if we can, so that code that just
# uses `from wgpu.gui.auto import ..` just works. The glfw library is really
# small, so there is not much harm.
try:
    import glfw  # noqa
except ImportError:
    pass
else:
    hiddenimports += ["wgpu.gui.glfw"]
    binaries += collect_dynamic_libs("glfw")
