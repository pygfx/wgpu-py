"""
The entrypoint / script to apply automatic patches to the code.
"""

import os
import sys

from codegen.utils import lib_dir
from codegen.idlparser import IdlParser
from codegen import apiwritersimple
from codegen import apipatcher
from codegen import rsbackend

# Little trick to allow running this file as a script
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))


def update_api():
    """ Update the public API and patch the public-facing API of the backends. """

    # Load idl
    idl = IdlParser()
    idl.parse(verbose=True)

    # Write the simple stuff
    apiwritersimple.write_flags(idl)
    apiwritersimple.write_enums(idl)
    apiwritersimple.write_structs(idl)

    # Patch base API: IDL -> API
    filename = os.path.join(lib_dir, "base.py")
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    code2 = apipatcher.patch_base_api(code1, idl)
    with open(filename, "wb") as f:
        f.write(code2.encode())

    # Patch backend APIs: base.py -> API
    for filename in [
        os.path.join(lib_dir, "backends", "rs.py"),
    ]:
        with open(filename, "rb") as f:
            code1 = f.read().decode()
        code2 = apipatcher.patch_backend_api(code1)
        with open(filename, "wb") as f:
            f.write(code2.encode())


def update_rs_backend():
    """ Update and check the rs backend. """
    rsbackend.write_mappings()


def main():
    update_api()
    update_rs_backend()


if __name__ == "__main__":
    main()
