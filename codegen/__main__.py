"""
The entrypoint / script to apply automatic patches to the code.
See README.md for more information.
"""

import os
import sys

from codegen.utils import print, lib_dir, add_file_object_to_print_to
from codegen import apiwriter, apipatcher, rspatcher, idlparser, hparser

# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


def update_api():
    """ Update the public API and patch the public-facing API of the backends. """

    print("## Updating API")

    # Write the simple stuff
    apiwriter.write_flags()
    apiwriter.write_enums()
    apiwriter.write_structs()

    # Patch base API: IDL -> API
    fname = "base.py"
    with open(os.path.join(lib_dir, fname), "rb") as f:
        code1 = f.read().decode()
    print(f"### Patching API for {fname}")
    code2 = apipatcher.patch_base_api(code1)
    with open(os.path.join(lib_dir, fname), "wb") as f:
        f.write(code2.encode())

    # Patch backend APIs: base.py -> API
    for fname in ["backends/rs.py"]:
        with open(os.path.join(lib_dir, fname), "rb") as f:
            code1 = f.read().decode()
        print(f"### Patching API for {fname}")
        code2 = apipatcher.patch_backend_api(code1)
        with open(os.path.join(lib_dir, fname), "wb") as f:
            f.write(code2.encode())


def update_rs():
    """ Update and check the rs backend. """

    print("## Validating rs.py")

    # Write the simple stuff
    rspatcher.compare_flags()
    rspatcher.write_mappings()

    # Patch rs.py
    filename = os.path.join(lib_dir, "backends", "rs.py")
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    code2 = rspatcher.patch_rs_backend(code1)
    with open(filename, "wb") as f:
        f.write(code2.encode())


def main():

    f = open(
        os.path.join(lib_dir, "resources", "codegen_report.md"),
        "wt",
        encoding="utf-8",
        newline="\n",
    )
    add_file_object_to_print_to(f)
    print("# Code generatation report")

    print("## Preparing")
    idlparser.get_idl_parser(allow_cache=False)
    hparser.get_h_parser(allow_cache=False)

    update_api()
    update_rs()

    f.close()


if __name__ == "__main__":
    main()
