from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("wgpu")
binaries = collect_dynamic_libs("wgpu")
hiddenimports = ["wgpu.resources"]
