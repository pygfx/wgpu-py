import os
import sys

from wgpu.codegen.idlparser import IdlParser
from wgpu.codegen.apiwriter import patch_module


# todo: check code coverage and remove old code-paths

lib_dir = os.path.abspath(os.path.join(__file__, "..", ".."))


def patch_api():

    # Obtain the idl definitions
    with open(os.path.join(lib_dir, "resources", "webgpu.idl"), "rb") as f:
        idl = IdlParser(f.read().decode())
    idl.parse(verbose=True)

    for filename in [
        os.path.join(lib_dir, "base.py"),
        # os.path.join(lib_dir, "backends", "rs.py"),
    ]:
        with open(filename, "rb") as f:
            code1 = f.read().decode()

        code2 = patch_module(idl, code1)

        with open(filename, "wb") as f:
            f.write(code2.encode())

def main():
    patch_api()


if __name__ == "__main__":
    main()
