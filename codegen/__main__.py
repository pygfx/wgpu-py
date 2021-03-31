"""
The entrypoint / script to apply automatic patches to the code.
"""

import os
import sys

from codegen.idlparser import IdlParser
from codegen.apipatcher import patch_base_api, patch_backend_api
from codegen.apiwritersimple import write_flags, write_enums, write_structs
from codegen.utils import lib_dir


# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


def update_api():

    # Load idl
    with open(os.path.join(lib_dir, "resources", "webgpu.idl"), "rb") as f:
        idl = IdlParser(f.read().decode())
    idl.parse(verbose=True)

    # Write the simple stuff
    write_flags(idl)
    write_enums(idl)
    write_structs(idl)

    # Patch base API: IDL -> API
    filename = os.path.join(lib_dir, "base.py")
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    code2 = patch_base_api(code1, idl)
    with open(filename, "wb") as f:
        f.write(code2.encode())

    # Patch backend APIs: base.py -> API
    for filename in [
        os.path.join(lib_dir, "backends", "rs.py"),
    ]:
        with open(filename, "rb") as f:
            code1 = f.read().decode()
        code2 = patch_backend_api(code1)
        with open(filename, "wb") as f:
            f.write(code2.encode())


def main():
    update_api()


if __name__ == "__main__":
    main()
