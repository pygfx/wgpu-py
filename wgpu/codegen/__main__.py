import os

from wgpu.codegen import apiwriter
from wgpu.codegen.utils import lib_dir

# todo: check code coverage and remove old code-paths


def patch_api():

    # Patch base API: IDL -> API
    filename = os.path.join(lib_dir, "base.py")
    with open(filename, "rb") as f:
        code1 = f.read().decode()
    code2 = apiwriter.patch_base_api(code1)
    with open(filename, "wb") as f:
        f.write(code2.encode())

    # Patch backend APIs: base.py -> API
    for filename in [
        os.path.join(lib_dir, "backends", "rs.py"),
    ]:
        with open(filename, "rb") as f:
            code1 = f.read().decode()
        code2 = apiwriter.patch_backend_api(code1)
        with open(filename, "wb") as f:
            f.write(code2.encode())


def main():
    patch_api()


if __name__ == "__main__":
    main()
