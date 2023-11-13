from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Init variables that PyInstaller will pick up.
hiddenimports = []
datas = []
binaries = []

# Include our resource data and binaries
datas += collect_data_files("wgpu", subdir="resources")
binaries += collect_dynamic_libs("wgpu")

# Include the modules that we want to include (PyInstall will trace imports)
hiddenimports += ["wgpu.backends.auto", "wgpu.backends.rs"]
hiddenimports += ["wgpu.gui.auto"]

# Note that the resources, utils, backends, and gui subpackages are imported by default.

# We have multiple subpackages for which the modules are not imported
# by default. We make sure that PyInstaller adds them anyway. Strictly
# speaking this would not be necessary, but it won't hurt, and it covers
# cases like e.g. downstream libs doing dynamic imports of gui backends.
for subpackage in ["utils", "backends", "gui"]:
    datas += collect_data_files(
        "wgpu", subdir=subpackage, include_py_files=True, excludes=["__pycache__"]
    )

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
