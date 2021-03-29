import os

from codegen.apiwriter import patch_base_api, patch_backend_api
from codegen.utils import lib_dir


def test_that_code_is_up_to_date():

    # Check base API
    filename = os.path.join(lib_dir, "base.py")
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    code2 = patch_base_api(code1)
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
