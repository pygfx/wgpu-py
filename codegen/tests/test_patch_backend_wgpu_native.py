"""Test the patchers in patch_backend_wgpu_native.py,
and implicitly tests the HParser.
"""

from wgpu._coreutils import get_header_filename

from patch_backend_wgpu_native import (
    CommentRemover,
    CStructPatcher,
    FunctionPatcher,
    HParser,
    _get_wgpu_header,
)


def dedent(code):
    return code.replace("\n    ", "\n")


# Build a header parser once, for all tests
hparser = HParser(
    _get_wgpu_header(get_header_filename("webgpu.h"), get_header_filename("wgpu.h"))
)
hparser.parse()


def patch_backend(code):
    """Apply the same patchers as patch_backend_api() in patch_backend_wgpu_native.py."""
    for patcher in [
        CommentRemover(("# FIXME: unknown C", "# FIXME: invalid C", "# H:")),
        CStructPatcher(hparser),
        FunctionPatcher(hparser),
    ]:
        patcher.apply(code)
        code = patcher.dumps()
    return code


def test_patch_functions():
    code1 = """
    libf.wgpuAdapterRequestDevice(1, 2, 3)
    libf.wgpuFooBar(1, 2, 3)
    """

    code2 = patch_backend(dedent(code1))

    # All original lines are there
    assert all(line[4:] in code2 for line in code1 if line.strip())

    # But also an annotation
    assert "WGPUAdapter adapter, WGPUDeviceDescriptor" in code2
    # And a notification that foo_bar is unknown
    assert code2.count("# FIXME:") == 1
    assert code2.count("FooBar") == 2

    # The patcher is idempotent: comments do not stack
    assert patch_backend(code2) == code2


def test_patch_structs():
    # Check simple struct
    code1 = """
    struct = new_struct_p(
        "WGPUBufferDescriptor *",
        label=c_label,
        size=size,
        usage=usage,
    )
    """
    code2 = patch_backend(dedent(code1))
    assert all(line[4:] in code2 for line in code1 if line.strip())
    assert "usage: WGPUBufferUsage/int" in code2
    assert "size: int" in code2
    assert "# FIXME:" not in code2
    assert code2 == patch_backend(code2)  # Don't stack comments

    # Check, but now using not-pointer
    code1 = """
    struct = new_struct(
        "WGPUBufferDescriptor",
        label=c_label,
        size=size,
        usage=usage,
    )
    """
    code2 = patch_backend(dedent(code1))
    assert all(line[4:] in code2 for line in code1 if line.strip())
    assert "usage: WGPUBufferUsage/int" in code2
    assert "size: int" in code2
    assert "# FIXME:" not in code2

    # Fail: pointer, but using not-pointer new_struct()
    code1 = 'struct = new_struct("WGPUBufferDescriptor *",label=c_label,size=size,usage=usage,)'
    code2 = patch_backend(dedent(code1))
    assert "# FIXME: invalid C struct, use new_struct_p()" in code2
    assert code2 == patch_backend(code2)  # Don't stack comments

    # Fail: not a pointer, but using new_struct_p()
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor",label=c_label,size=size,usage=usage,)'
    code2 = patch_backend(dedent(code1))
    assert "# FIXME: invalid C struct, use new_struct()" in code2
    assert code2 == patch_backend(code2)  # Don't stack comments

    # Missing values
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor *",label=c_label,size=size,)'
    code2 = patch_backend(dedent(code1))
    assert "usage: WGPUBufferUsage/int" in code2
    assert "# FIXME:" not in code2
    assert "usage" in code2  # comment added
    assert code2 == patch_backend(code2)  # Don't stack comments

    # Too many values
    code1 = 'struct = new_struct_p("WGPUBufferDescriptor *",label=c_label,foo=size,)'
    code2 = patch_backend(dedent(code1))
    assert "usage: WGPUBufferUsage/int" in code2
    assert "# FIXME: unknown" in code2
    assert code2 == patch_backend(code2)  # Don't stack comments


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
