from apigen_patcher import (
    ApiPatcher,
    CommentRemover,
    detect_implemented_properties_and_methods,
)
from apigen_utils import print, read_file, write_file, dict_deep_merge
from apigen_defs import read_definition_file


def patch_backend_wgpu_native():
    """Patch the api for the wgpu_native backend in backends/wgpu_native/_api.py."""

    api_base = read_definition_file("wgpu_api.toml")

    diff_wgpu_native = read_definition_file("diff_backend_wgpu_native.toml")
    api = dict_deep_merge(api_base, diff_wgpu_native)

    print(f"Found {len(api['classes'])} classes in the wgpu_native backend API")

    implemented = detect_implemented_properties_and_methods(
        read_file("wgpu", "_classes.py")
    )

    filename_parts = "wgpu", "backends", "wgpu_native", "_api.py"
    code = read_file(*filename_parts)

    for patcher in [CommentRemover(), ApiPatcher(api, True, implemented)]:
        patcher.apply(code)
        code = patcher.dumps()

    write_file(code, *filename_parts)


if __name__ == "__main__":
    patch_backend_wgpu_native()
