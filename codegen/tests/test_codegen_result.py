import os

from codegen.idlparser import IdlParser
from codegen.apipatcher import patch_base_api, patch_backend_api
from codegen.utils import lib_dir


def test_that_code_is_up_to_date():
    """Test that running the codegen does not introduce changes."""

    # Load idl
    with open(os.path.join(lib_dir, "resources", "webgpu.idl"), "rb") as f:
        idl = IdlParser(f.read().decode())
    idl.parse(verbose=False)

    # Check base API
    filename = os.path.join(lib_dir, "base.py")
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    code2 = patch_base_api(code1, idl)
    assert code1 == code2

    # checks backend API
    for filename in [
        os.path.join(lib_dir, "backends", "rs.py"),
    ]:
        with open(filename, "rb") as f:
            code1 = f.read().decode()
        code2 = patch_backend_api(code1)
        assert code1 == code2


if __name__ == "__main__":
    test_that_code_is_up_to_date()
