from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Include our data and binaires
datas = collect_data_files("wgpu")
binaries = collect_dynamic_libs("wgpu")
hiddenimports = ["wgpu.resources"]

# Include backends that we want to Just Work
hiddenimports += ["wgpu.backends.auto", "wgpu.backends.rs"]

# Include gui backends that we want to Just Work.
# Note that if someone wants to use Qt, both the qt lib and the
# wgpu.gui.qt must be explicitly imported.
hiddenimports += ["wgpu.gui.offscreen", "wgpu.gui.glfw"]

# We also need to help glfw, since it does not have a hook like this one.
binaries += collect_dynamic_libs("glfw")
