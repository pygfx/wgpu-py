import io

from .utils import print, PrintToFile
from . import apiwriter, apipatcher, rspatcher, idlparser, hparser
from .files import file_cache


def main():
    """Codegen entry point. This will populate the file cache with the
    new code, but not write it to disk."""

    log = io.StringIO()
    with PrintToFile(log):
        print("# Code generatation report")
        prepare()
        update_api()
        update_rs()
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
    code1 = file_cache.read("base.py")
    print("### Patching API for base.py")
    code2 = apipatcher.patch_base_api(code1)
    file_cache.write("base.py", code2)

    # Patch backend APIs: base.py -> API
    for fname in ["backends/rs.py"]:
        code1 = file_cache.read(fname)
        print(f"### Patching API for {fname}")
        code2 = apipatcher.patch_backend_api(code1)
        file_cache.write(fname, code2)


def update_rs():
    """Update and check the rs backend."""

    print("## Validating rs.py")

    # Write the simple stuff
    rspatcher.compare_flags()
    rspatcher.write_mappings()

    # Patch rs.py
    code1 = file_cache.read("backends/rs.py")
    code2 = rspatcher.patch_rs_backend(code1)
    file_cache.write("backends/rs.py", code2)
