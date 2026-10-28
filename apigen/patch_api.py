from apigen_patcher import ApiPatcher, CommentRemover
from apigen_utils import read_definition_file, read_file, write_file


def patch_api():
    """Patch the base API in _classes.py."""

    api = read_definition_file("wgpu_api.json")
    code = read_file("wgpu", "_classes.py")

    # code = format_code(code, singleline=True) ??

    # Write __all__
    extra_public_classes = ["GPUPromise", "DrawCancelled"]
    all_public_classes = [*api["classes"].keys(), *extra_public_classes]
    all_public_classes.sort()
    all_public_classes.remove("GPU")
    all_public_classes.insert(0, "GPU")  # ruff RUF022 wants it at the top
    part1, found_all, part2 = code.partition("\n__all__ =")
    if found_all:
        part2 = part2.split("]", 1)[-1]
        line = "\n__all__ = ["
        line += ", ".join(f'"{name}"' for name in all_public_classes)
        line += "]"
        code = part1 + line + part2

    # Patch!
    for patcher in [CommentRemover(), ApiPatcher(api)]:
        patcher.apply(code)
        code = patcher.dumps()

    # Write result back
    write_file(code, "wgpu", "_classes.py")


if __name__ == "__main__":
    patch_api()
