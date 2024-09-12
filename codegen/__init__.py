import io

from .utils import print, PrintToFile
from . import apiwriter, apipatcher, wgpu_native_patcher, idlparser, hparser
from .files import file_cache


def main():
    """Codegen entry point. This will populate the file cache with the
    new code, but not write it to disk."""

    log = io.StringIO()
    with PrintToFile(log):
        print("# Code generatation report")
        prepare()
        update_api()
        update_wgpu_native()
        file_cache.write("resources/codegen_report.md", log.getvalue())


def prepare():
    """Force parsing (and caching) the IDL and C header."""
    print("## Preparing")
    file_cache.reset()
    idlparser.get_idl_parser(allow_cache=False)
    hparser.get_h_parser(allow_cache=False)


def update_api():
    """Update the public API and patch the public-facing API of the backends."""

    print("## Updating API")

    # Write the simple stuff
    apiwriter.write_flags()
    apiwriter.write_enums()
    apiwriter.write_structs()

    # Patch base API: IDL -> API
    code1 = file_cache.read("_classes.py")
    print("### Patching API for _classes.py")
    code2 = apipatcher.patch_base_api(code1)
    file_cache.write("_classes.py", code2)

    # Patch backend APIs: _classes.py -> API
    for fname in ["backends/wgpu_native/_api.py"]:
        code1 = file_cache.read(fname)
        print(f"### Patching API for {fname}")
        code2 = apipatcher.patch_backend_api(code1)
        file_cache.write(fname, code2)


def update_wgpu_native():
    """Update and check the wgpu-native backend."""

    print("## Validating backends/wgpu_native/_api.py")

    # Write the simple stuff
    wgpu_native_patcher.compare_flags()
    wgpu_native_patcher.write_mappings()

    # Patch wgpu_native api
    code1 = file_cache.read("backends/wgpu_native/_api.py")
    code2 = wgpu_native_patcher.patch_wgpu_native_backend(code1)
    file_cache.write("backends/wgpu_native/_api.py", code2)
